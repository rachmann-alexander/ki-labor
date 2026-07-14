#!/usr/bin/env python3
# file: make_plot_05_latency_vs_throughput_grid.py
# Grid-SVG: 05_throughput_vs_latency_facet.svg (rows=model, cols=precision)
# x = Latency avg [ms], y = Throughput [img/s]
# - Profil = Farbe (farbiger Text rechts)
# - Batch  = Markerform (BS32/64/128)
# - X-Achse global (Default) oder variabel (--vary-x)
# - Optional: Log-Skalen (--xlog/--ylog)
# - Log-Fix: Untergrenzen werden an vmin geclamped (kein 1e-6)

import os, argparse, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D

PROFILE_COLORS = {"slow":"C0","medium":"C1","fast":"C2"}
BATCH_MARKERS  = {32:"o",64:"s",128:"^"}
POINT_SIZE = 50

def fmt_SI(v,_=None):
    if v==0: return "0"
    a=abs(v)
    if a>=1e6:  return f"{v/1e6:.2f}M"
    if a>=1e3:  return f"{v/1e3:.2f}k"
    if a<1e-2:  return f"{v:.2e}"
    return f"{v:.2f}"

def _limits(series: pd.Series, log: bool, pad_frac: float,
            clamp_log_low: float = 0.85, clamp_log_high: float = 1.10):
    """Berechnet sinnvolle Achsenlimits.
       linear: [vmin - pad, vmax + pad]
       log:    [vmin*clamp_low, vmax*clamp_high] (kein 1e-6)
    """
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if vals.size == 0:
        return None
    if log:
        vals = vals[vals > 0]
        if vals.size == 0:
            return None
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmin == vmax:
            return (vmin * 0.9, vmax * 1.1)
        return (vmin * clamp_log_low, vmax * clamp_log_high)
    else:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmin == vmax:
            pad = max(1e-6, pad_frac * max(1.0, vmax))
            return (max(1e-6, vmin - pad), vmax + pad)
        pad = pad_frac * (vmax - vmin)
        return (max(1e-6, vmin - pad), vmax + pad)

def make_facet_plot(df, out_dir, xlog=False, ylog=False, vary_x=False):
    need=["model","precision","profile","batch","throughput_img_s","latency_avg_ms"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Spalte fehlt im Manifest: {c}")

    d = df.dropna(subset=need).copy()
    if d.empty:
        print("[i] Keine Daten für den Plot."); return
    d["batch"] = pd.to_numeric(d["batch"], errors="coerce").astype("Int64")

    precisions=[p for p in ["fp16","fp32","fp64"] if p in d["precision"].unique()]
    models=sorted(d["model"].unique().tolist())
    nrows,ncols=len(models),len(precisions)

    fig,axes=plt.subplots(nrows=nrows, ncols=ncols,
                          figsize=(4.4*ncols, 3.4*nrows),
                          sharex=not vary_x, sharey=True)

    if nrows==1 and ncols==1: axes=[[axes]]
    elif nrows==1:           axes=[axes]
    elif ncols==1:           axes=[[ax] for ax in axes]

    # Globale X-/Y-Limits (X mit log-sicheren Grenzen)
    if not vary_x:
        xlim = _limits(d["latency_avg_ms"], log=xlog, pad_frac=0.06,
                       clamp_log_low=0.85, clamp_log_high=1.10)
        if xlim is None: return
    else:
        xlim = None

    ylim = _limits(d["throughput_img_s"], log=ylog, pad_frac=0.10,
                   clamp_log_low=0.85, clamp_log_high=1.10)
    if ylim is None: return

    for r,m in enumerate(models):
        for c,pz in enumerate(precisions):
            ax=axes[r][c]
            sub=d[(d["model"]==m) & (d["precision"]==pz)]
            if sub.empty:
                ax.set_visible(False); continue

            for prof,gprof in sub.groupby("profile"):
                color=PROFILE_COLORS.get(prof,"C7")
                for bval,gb in gprof.groupby("batch"):
                    try: b=int(bval)
                    except: b=None
                    marker=BATCH_MARKERS.get(b,"o")
                    ax.scatter(gb["latency_avg_ms"], gb["throughput_img_s"],
                               s=POINT_SIZE, marker=marker, color=color,
                               alpha=0.9, edgecolors="none", linewidths=0)

            if r==nrows-1: ax.set_xlabel("Latency avg [ms]")
            if c==0:       ax.set_ylabel("Throughput [img/s]")
            ax.set_title(f"{m} — {pz}")

            ax.xaxis.set_major_formatter(FuncFormatter(fmt_SI))
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_SI))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            if xlog: ax.set_xscale("log")
            if ylog: ax.set_yscale("log")

            # X: global oder panel-spezifisch
            if vary_x:
                xloc = _limits(sub["latency_avg_ms"], log=xlog, pad_frac=0.06,
                               clamp_log_low=0.85, clamp_log_high=1.10)
                if xloc: ax.set_xlim(*xloc)
            else:
                ax.set_xlim(*xlim)

            # Y: immer global (für Vergleichbarkeit)
            ax.set_ylim(*ylim)

            ax.grid(True, which="both", linewidth=0.4, alpha=0.3)

    # Legenden (rechts), wie bei deinen „passt“-Plots
    fig.subplots_adjust(right=0.89, top=0.90, bottom=0.10, wspace=0.28, hspace=0.32)

    prof_order=["slow","medium","fast"]
    prof_handles=[Line2D([],[], linestyle="None", linewidth=0, marker=None, label=lbl)
                  for lbl in prof_order]
    leg_prof=fig.legend(handles=prof_handles, title="profile",
                        loc="center right", bbox_to_anchor=(0.975,0.62), frameon=False)
    for txt in leg_prof.get_texts():
        lbl=txt.get_text()
        txt.set_color(PROFILE_COLORS.get(lbl,"black"))

    batch_handles=[
        Line2D([],[], marker=BATCH_MARKERS[32],  linestyle="None", color="black", label="BS32"),
        Line2D([],[], marker=BATCH_MARKERS[64],  linestyle="None", color="black", label="BS64"),
        Line2D([],[], marker=BATCH_MARKERS[128], linestyle="None", color="black", label="BS128"),
    ]
    fig.legend(handles=batch_handles, title="batch size",
               loc="center right", bbox_to_anchor=(0.975,0.30), frameon=False)

    fig.suptitle("Throughput vs. Latency (avg) — faceted by model × precision", y=0.98, fontsize=11)

    os.makedirs(out_dir, exist_ok=True)
    out_path=os.path.join(out_dir,"05_throughput_vs_latency_facet.svg")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("✔︎ SVG:", out_path)

def main():
    ap=argparse.ArgumentParser(description="Grid-Plot: Throughput vs Latency (SVG)")
    ap.add_argument("--manifest", required=True, help="Pfad zu grid_manifest.csv")
    ap.add_argument("--out", required=True, help="Ausgabeordner")
    ap.add_argument("--xlog", action="store_true")
    ap.add_argument("--ylog", action="store_true")
    ap.add_argument("--vary-x", action="store_true", help="X-Achse (Latency) pro Panel variieren")
    args=ap.parse_args()
    df=pd.read_csv(args.manifest)
    make_facet_plot(df, args.out, xlog=args.xlog, ylog=args.ylog, vary_x=args.vary_x)

if __name__ == "__main__":
    main()
