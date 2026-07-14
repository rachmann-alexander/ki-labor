#!/usr/bin/env python3
# file: make_plot_03_images_per_joule_batch_grid.py
# Grid-SVG: 03_images_per_joule_vs_batch_facet.svg (rows=model, cols=precision)
# x = Batch size, y = Images per Joule [img/J]
# - Profil = Farbe (slow/medium/fast; farbiger Text in der Legende)
# - Batch = Markerform (32:o, 64:s, 128:^)
# - Y-Achse global (Default) oder pro Panel variabel (--vary-y)
# - Optional: Log-Skalen (--xlog/--ylog)

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D

PROFILE_COLORS = {"slow": "C0", "medium": "C1", "fast": "C2"}
BATCH_MARKERS  = {32: "o", 64: "s", 128: "^"}
POINT_SIZE = 50

def fmt_SI(v, _pos=None):
    if v == 0: return "0"
    a = abs(v)
    if a >= 1e6:  return f"{v/1e6:.2f}M"
    if a >= 1e3:  return f"{v/1e3:.2f}k"
    if a <  1e-2: return f"{v:.2e}"
    return f"{v:.2f}"

def _limits_from(series: pd.Series, log: bool, pad_frac: float):
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if log:
        vals = vals[vals > 0]
    if vals.size == 0:
        return None
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmin == vmax:
        pad = max(1e-6, pad_frac * max(1.0, vmax))
        return (max(1e-6, vmin - pad), vmax + pad)
    pad = pad_frac * (vmax - vmin)
    return (max(1e-6, vmin - pad), vmax + pad)

def _ensure_images_per_joule(df: pd.DataFrame) -> pd.DataFrame:
    if "images_per_joule" not in df.columns or df["images_per_joule"].isna().all():
        if {"avg_power_W","throughput_img_s"}.issubset(df.columns):
            num = pd.to_numeric(df["throughput_img_s"], errors="coerce")
            den = pd.to_numeric(df["avg_power_W"], errors="coerce")
            df["images_per_joule"] = np.where((num > 0) & (den > 0), num / den, np.nan)
    return df

def make_facet_plot(df: pd.DataFrame, out_dir: str,
                    xlog: bool=False, ylog: bool=False,
                    vary_y: bool=False):
    need = ["model","precision","profile","batch","avg_power_W","throughput_img_s"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Spalte fehlt im Manifest: {c}")

    df = _ensure_images_per_joule(df.copy())
    d = df.dropna(subset=["model","precision","profile","batch","images_per_joule"])
    if d.empty:
        print("[i] Keine Daten für den Plot.")
        return

    # Batch als int
    d["batch"] = pd.to_numeric(d["batch"], errors="coerce").astype("Int64")

    # Facets
    precisions = [p for p in ["fp16","fp32","fp64"] if p in d["precision"].unique()]
    models     = sorted(d["model"].unique().tolist())
    nrows, ncols = len(models), len(precisions)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4.4*ncols, 3.4*nrows),
        sharex=True,
        sharey=not vary_y
    )

    if nrows == 1 and ncols == 1: axes = [[axes]]
    elif nrows == 1:               axes = [axes]
    elif ncols == 1:               axes = [[ax] for ax in axes]

    # Globale X-Limits (Batch ist diskret, aber wir geben etwas Rand)
    xlim = _limits_from(d["batch"], log=xlog, pad_frac=0.05)
    if xlim is None:
        print("[i] Konnte X-Limits nicht bestimmen.")
        return

    # Globale Y-Limits (nur wenn NOT vary_y)
    if not vary_y:
        ylim_global = _limits_from(d["images_per_joule"], log=ylog, pad_frac=0.08)
        if ylim_global is None:
            print("[i] Konnte Y-Limits nicht bestimmen.")
            return
    else:
        ylim_global = None

    for r, m in enumerate(models):
        for c, pz in enumerate(precisions):
            ax = axes[r][c]
            sub = d[(d["model"] == m) & (d["precision"] == pz)]
            if sub.empty:
                ax.set_visible(False)
                continue

            for prof, gprof in sub.groupby("profile"):
                color = PROFILE_COLORS.get(prof, "C7")
                for bval, gb in gprof.groupby("batch"):
                    try: b = int(bval)
                    except: b = None
                    marker = BATCH_MARKERS.get(b, "o")
                    ax.scatter(
                        gb["batch"], gb["images_per_joule"],
                        s=POINT_SIZE, marker=marker, color=color,
                        alpha=0.9, edgecolors="none", linewidths=0
                    )

            if r == nrows - 1: ax.set_xlabel("Batch size")
            if c == 0:         ax.set_ylabel("Images per Joule [img/J]")
            ax.set_title(f"{m} — {pz}")

            # X-Ticks = vorhandene Batches (schön sauber 32/64/128)
            batches_sorted = sorted(sub["batch"].dropna().unique().tolist())
            ax.set_xticks(batches_sorted)
            ax.set_xticklabels([str(int(b)) for b in batches_sorted])

            ax.xaxis.set_major_formatter(FuncFormatter(fmt_SI))
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_SI))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            if xlog: ax.set_xscale("log")
            if ylog: ax.set_yscale("log")

            ax.set_xlim(*xlim)
            if vary_y:
                ylim = _limits_from(sub["images_per_joule"], log=ylog, pad_frac=0.08)
                if ylim: ax.set_ylim(*ylim)
            else:
                ax.set_ylim(*ylim_global)

            ax.grid(True, which="both", linewidth=0.4, alpha=0.3)

    # Legenden rechts – gleiche Positionen wie beim „passt“-Plot
    fig.subplots_adjust(right=0.89, top=0.90, bottom=0.10, wspace=0.28, hspace=0.32)

    # Profil: farbiger Text
    prof_order   = ["slow","medium","fast"]
    prof_handles = [Line2D([], [], linestyle="None", linewidth=0, marker=None, label=lbl)
                    for lbl in prof_order]
    leg_prof = fig.legend(handles=prof_handles, title="profile",
                          loc="center right", bbox_to_anchor=(0.955, 0.62), frameon=False)
    for txt in leg_prof.get_texts():
        lbl = txt.get_text()
        txt.set_color(PROFILE_COLORS.get(lbl, "black"))

    # Batch: Formen
    batch_handles = [
        Line2D([], [], marker=BATCH_MARKERS[32],  linestyle="None", color="black", label="BS32"),
        Line2D([], [], marker=BATCH_MARKERS[64],  linestyle="None", color="black", label="BS64"),
        Line2D([], [], marker=BATCH_MARKERS[128], linestyle="None", color="black", label="BS128"),
    ]
    fig.legend(handles=batch_handles, title="batch size",
               loc="center right", bbox_to_anchor=(0.97, 0.30), frameon=False)

    fig.suptitle("Images per Joule vs. Batch Size — faceted by model × precision", y=0.98, fontsize=11)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "03_images_per_joule_vs_batch_facet.svg")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("✔︎ SVG geschrieben:", out_path)

def main():
    ap = argparse.ArgumentParser(description="Grid-Plot: Images per Joule vs Batch (SVG)")
    ap.add_argument("--manifest", required=True, help="Pfad zu grid_manifest.csv")
    ap.add_argument("--out", required=True, help="Ausgabeordner (wird angelegt)")
    ap.add_argument("--xlog", action="store_true", help="X-Achse logarithmisch")
    ap.add_argument("--ylog", action="store_true", help="Y-Achse logarithmisch")
    ap.add_argument("--vary-y", action="store_true", help="Y-Achse pro Panel variieren")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    make_facet_plot(df, args.out, xlog=args.xlog, ylog=args.ylog, vary_y=args.vary_y)

if __name__ == "__main__":
    main()
