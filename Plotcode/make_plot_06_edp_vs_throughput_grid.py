#!/usr/bin/env python3
# file: make_plot_06_edp_vs_throughput_grid.py
# Grid-SVG: 06_edp_per_image_vs_throughput_facet.svg (rows=model, cols=precision)
# x = Throughput [img/s], y = EDP per image [J*s]
# - Profil = Farbe (farbiger Text rechts)
# - Batch = Markerform (BS32/64/128)
# - Y-Achse global (Default) oder variabel (--vary-y)
# - Optional: Log-Skalen (--xlog/--ylog)

import os, argparse, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D

PROFILE_COLORS={"slow":"C0","medium":"C1","fast":"C2"}
BATCH_MARKERS ={32:"o",64:"s",128:"^"}
POINT_SIZE=50

def fmt_SI(v,_=None):
    if v==0: return "0"
    a=abs(v)
    if a>=1e6:  return f"{v/1e6:.2f}M"
    if a>=1e3:  return f"{v/1e3:.2f}k"
    if a<1e-2:  return f"{v:.2e}"
    return f"{v:.2f}"

def _limits_from(series, log, pad_frac):
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if log: vals = vals[vals>0]
    if vals.size==0: return None
    vmin,vmax = float(np.min(vals)), float(np.max(vals))
    if vmin==vmax:
        pad=max(1e-6, pad_frac*max(1.0,vmax))
        return (max(1e-6, vmin-pad), vmax+pad)
    pad=pad_frac*(vmax-vmin)
    return (max(1e-6, vmin-pad), vmax+pad)

def _ensure_edp(df: pd.DataFrame) -> pd.DataFrame:
    # energy_per_img_J vorhanden? sonst aus avg_power_W/throughput_img_s ableiten
    if "energy_per_img_J" not in df.columns or df["energy_per_img_J"].isna().all():
        pw = pd.to_numeric(df.get("avg_power_W", np.nan), errors="coerce")
        th = pd.to_numeric(df.get("throughput_img_s", np.nan), errors="coerce")
        df["energy_per_img_J"] = np.where((pw>0)&(th>0), pw/th, np.nan)
    # Latenz in Sekunden
    lat_s = pd.to_numeric(df.get("latency_avg_ms", np.nan), errors="coerce")/1000.0
    df["edp_image_Js"] = pd.to_numeric(df["energy_per_img_J"], errors="coerce") * lat_s
    return df

def make_facet_plot(df, out_dir, xlog=False, ylog=False, vary_y=False):
    need=["model","precision","profile","batch","throughput_img_s","latency_avg_ms"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Spalte fehlt im Manifest: {c}")

    df=_ensure_edp(df.copy())
    d = df.dropna(subset=["model","precision","profile","batch","throughput_img_s","edp_image_Js"]).copy()
    if d.empty:
        print("[i] Keine Daten für den Plot."); return
    d["batch"]=pd.to_numeric(d["batch"], errors="coerce").astype("Int64")

    precisions=[p for p in ["fp16","fp32","fp64"] if p in d["precision"].unique()]
    models=sorted(d["model"].unique().tolist())
    nrows,ncols=len(models),len(precisions)

    fig,axes=plt.subplots(nrows=nrows, ncols=ncols,
                          figsize=(4.4*ncols, 3.4*nrows),
                          sharex=True, sharey=not vary_y)

    if nrows==1 and ncols==1: axes=[[axes]]
    elif nrows==1:           axes=[axes]
    elif ncols==1:           axes=[[ax] for ax in axes]

    # Globale X-/Y-Limits
    xlim=_limits_from(d["throughput_img_s"], log=xlog, pad_frac=0.06)
    if xlim is None: return
    if not vary_y:
        ylim_global=_limits_from(d["edp_image_Js"], log=ylog, pad_frac=0.10)
        if ylim_global is None: return
    else:
        ylim_global=None

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
                    ax.scatter(gb["throughput_img_s"], gb["edp_image_Js"],
                               s=POINT_SIZE, marker=marker, color=color,
                               alpha=0.9, edgecolors="none", linewidths=0)

            if r==nrows-1: ax.set_xlabel("Throughput [img/s]")
            if c==0:       ax.set_ylabel("EDP per image [J·s]")
            ax.set_title(f"{m} — {pz}")

            ax.xaxis.set_major_formatter(FuncFormatter(fmt_SI))
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_SI))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if xlog: ax.set_xscale("log")
            if ylog: ax.set_yscale("log")

            ax.set_xlim(*xlim)
            if vary_y:
                ylim=_limits_from(sub["edp_image_Js"], log=ylog, pad_frac=0.10)
                if ylim: ax.set_ylim(*ylim)
            else:
                ax.set_ylim(*ylim_global)

            ax.grid(True, which="both", linewidth=0.4, alpha=0.3)

    # Legenden wie gehabt (rechts, leicht nach außen)
    fig.subplots_adjust(right=0.89, top=0.90, bottom=0.10, wspace=0.28, hspace=0.32)

    prof_order=["slow","medium","fast"]
    prof_handles=[Line2D([],[], linestyle="None", linewidth=0, marker=None, label=lbl) for lbl in prof_order]
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

    fig.suptitle("EDP (per image) vs. Throughput — faceted by model × precision", y=0.98, fontsize=11)
    os.makedirs(out_dir, exist_ok=True)
    out_path=os.path.join(out_dir,"06_edp_per_image_vs_throughput_facet.svg")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("✔︎ SVG:", out_path)

def main():
    ap=argparse.ArgumentParser(description="Grid-Plot: EDP per image vs Throughput (SVG)")
    ap.add_argument("--manifest", required=True, help="Pfad zu grid_manifest.csv")
    ap.add_argument("--out", required=True, help="Ausgabeordner")
    ap.add_argument("--xlog", action="store_true")
    ap.add_argument("--ylog", action="store_true")
    ap.add_argument("--vary-y", action="store_true")
    args=ap.parse_args()
    df=pd.read_csv(args.manifest)
    make_facet_plot(df, args.out, xlog=args.xlog, ylog=args.ylog, vary_y=args.vary_y)

if __name__ == "__main__":
    main()
