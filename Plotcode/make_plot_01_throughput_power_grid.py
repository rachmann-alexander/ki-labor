#!/usr/bin/env python3
# file: make_plot_01_throughput_power_grid.py
# SVG-Plot: 01_throughput_vs_power_facet.svg (rows=model, cols=precision)
# - Farbe = profile (slow/medium/fast)
# - Markerform = batch size (BS32=Kreis, BS64=Quadrat, BS128=Dreieck)
# - Einheitliche Achsenfenster, keine Punktlabels
# - Legenden rechts: Profil-Text ist eingefärbt (ohne Symbole); Batch-Legende zeigt Formen

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'  # Text editierbar in SVG
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D
import numpy as np

# ------------------- Kodierung -------------------
PROFILE_COLORS = {"slow": "C0", "medium": "C1", "fast": "C2"}  # Farben für Profile
BATCH_MARKERS  = {32: "o", 64: "s", 128: "^"}                  # Markerformen für Batchgrößen
POINT_SIZE = 50

def fmt_SI(x, _pos=None):
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e6:  return f"{x/1e6:.2f}M"
    if ax >= 1e3:  return f"{x/1e3:.2f}k"
    if ax <  1e-2: return f"{x:.2e}"
    return f"{x:.2f}"

def make_facet_plot(df: pd.DataFrame, out_dir: str, xlog: bool = False, ylog: bool = False):
    need = ["model","precision","profile","batch","avg_power_W","throughput_img_s"]
    for col in need:
        if col not in df.columns:
            raise ValueError(f"Spalte fehlt im Manifest: {col}")

    d = df.dropna(subset=need).copy()
    if d.empty:
        print("[i] Keine Daten für den Plot.")
        return

    # feste Präzisions-Reihenfolge; Modelle alphabetisch
    precisions = [p for p in ["fp16","fp32","fp64"] if p in d["precision"].unique()]
    models = sorted(d["model"].unique().tolist())
    nrows, ncols = len(models), len(precisions)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4.4*ncols, 3.4*nrows),
        sharex=True, sharey=True
    )

    # axes in 2D normalisieren
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    # Globale Limits (einheitlich für alle Panels)
    xd = pd.to_numeric(d["avg_power_W"], errors="coerce").to_numpy()
    yd = pd.to_numeric(d["throughput_img_s"], errors="coerce").to_numpy()
    if xlog: xd = xd[xd > 0]
    if ylog: yd = yd[yd > 0]
    if xd.size == 0 or yd.size == 0:
        print("[i] Zu wenige Datenpunkte für globale Limits.")
        return
    xpad = (xd.max() - xd.min()) * 0.06 if xd.max() > xd.min() else max(1e-6, xd.max()*0.06)
    ypad = (yd.max() - yd.min()) * 0.08 if yd.max() > yd.min() else max(1e-6, yd.max()*0.08)
    xlim = (max(1e-6, xd.min() - xpad), xd.max() + xpad)
    ylim = (max(1e-6, yd.min() - ypad), yd.max() + ypad)

    for r, m in enumerate(models):
        for c, pz in enumerate(precisions):
            ax = axes[r][c]
            sub = d[(d["model"] == m) & (d["precision"] == pz)]
            if sub.empty:
                ax.set_visible(False)
                continue

            # Punkte: Farbe=profile, Form=batch
            for prof, gprof in sub.groupby("profile"):
                color = PROFILE_COLORS.get(prof, "C7")
                for bval, gb in gprof.groupby("batch"):
                    try:
                        b = int(bval)
                    except Exception:
                        b = None
                    marker = BATCH_MARKERS.get(b, "o")
                    ax.scatter(
                        gb["avg_power_W"], gb["throughput_img_s"],
                        s=POINT_SIZE, marker=marker, color=color,
                        alpha=0.9, edgecolors="none", linewidths=0
                    )

            # Achsen / Titel / Format
            if r == nrows - 1:
                ax.set_xlabel("Average Power [W]")
            if c == 0:
                ax.set_ylabel("Throughput [img/s]")
            ax.set_title(f"{m} — {pz}")

            ax.xaxis.set_major_formatter(FuncFormatter(fmt_SI))
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_SI))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            if xlog: ax.set_xscale("log")
            if ylog: ax.set_yscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            ax.grid(True, which="both", linewidth=0.4, alpha=0.3)

    # ---- Legenden (rechts, minimal weiter rechts) ----
    fig.subplots_adjust(right=0.89, top=0.90, bottom=0.10, wspace=0.28, hspace=0.32)

    # Profil-Legende: NUR Text, farbig
    profile_labels_order = ["slow", "medium", "fast"]
    profile_handles = [
        Line2D([], [], linestyle="None", linewidth=0, marker=None, label=lbl)
        for lbl in profile_labels_order
    ]
    leg_profile = fig.legend(
        handles=profile_handles, title="profile",
        loc="center right", bbox_to_anchor=(0.955, 0.62),
        frameon=False
    )
    for text in leg_profile.get_texts():
        lbl = text.get_text()
        text.set_color(PROFILE_COLORS.get(lbl, "black"))

    # Batch-Legende (Formen)
    batch_handles = [
        Line2D([], [], marker=BATCH_MARKERS[32],  linestyle="None", color="black", label="BS32"),
        Line2D([], [], marker=BATCH_MARKERS[64],  linestyle="None", color="black", label="BS64"),
        Line2D([], [], marker=BATCH_MARKERS[128], linestyle="None", color="black", label="BS128"),
    ]
    fig.legend(handles=batch_handles, title="batch size",
               loc="center right", bbox_to_anchor=(0.97, 0.30), frameon=False)

    fig.suptitle("Throughput vs. Average Power — faceted by model × precision", y=0.98, fontsize=11)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "01_throughput_vs_power_facet.svg")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("✔︎ SVG geschrieben:", out_path)

def main():
    ap = argparse.ArgumentParser(description="Facet-Plot: Throughput vs Power (SVG)")
    ap.add_argument("--manifest", required=True, help="Pfad zu grid_manifest.csv")
    ap.add_argument("--out", required=True, help="Ausgabeordner (wird angelegt)")
    ap.add_argument("--xlog", action="store_true", help="X-Achse logarithmisch")
    ap.add_argument("--ylog", action="store_true", help="Y-Achse logarithmisch")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    make_facet_plot(df, args.out, xlog=args.xlog, ylog=args.ylog)

if __name__ == "__main__":
    main()
