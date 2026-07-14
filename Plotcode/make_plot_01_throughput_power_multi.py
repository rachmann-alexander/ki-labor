#!/usr/bin/env python3
# file: make_plot_01_throughput_power_multi.py
# Erzeugt EINEN SVG-Plot pro (model, precision):
#   01_throughput_vs_power__{model}__{precision}.svg
#
# - Punkte: Farbe = profile (slow/medium/fast), Marker = batch (BS32 o, BS64 s, BS128 ^)
# - Achsen: global (über alle Plots) ODER pro Plot variabel via --vary-x / --vary-y
# - Profil-Legende je Plot: farbiger Text (ohne Symbole) rechts außen
# - Batch-Legende je Plot: Formen, darunter
# - Optional: Log-Skalen via --xlog / --ylog

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'  # Text editierbar im SVG
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D
import numpy as np

# ------------------- Kodierung -------------------
PROFILE_COLORS = {"slow": "C0", "medium": "C1", "fast": "C2"}  # Profil -> Farbe
BATCH_MARKERS   = {32: "o",  64: "s",   128: "^"}              # Batch  -> Marker
POINT_SIZE = 50

def fmt_SI(x, _pos=None):
    if x == 0: return "0"
    ax = abs(x)
    if ax >= 1e6:  return f"{x/1e6:.2f}M"
    if ax >= 1e3:  return f"{x/1e3:.2f}k"
    if ax <  1e-2: return f"{x:.2e}"
    return f"{x:.2f}"

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

def _plot_single(sub: pd.DataFrame, out_dir: str, model: str, precision: str,
                 xlog: bool, ylog: bool, xlim, ylim):
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    # Punkte: Farbe=profile, Marker=batch
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

    # Achsen / Titel / Formatter
    ax.set_xlabel("Average Power [W]")
    ax.set_ylabel("Throughput [img/s]")
    ax.set_title(f"{model} — {precision}")

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_SI))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_SI))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    if xlog: ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)

    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)

    # Platz rechts für Legenden – wie in deiner Referenzansicht
    fig.subplots_adjust(right=0.82)

    # Profil-Legende (farbiger Text, ohne Symbole), rechts außen
    prof_order   = ["slow","medium","fast"]
    prof_handles = [Line2D([], [], linestyle="None", linewidth=0, marker=None, label=lbl)
                    for lbl in prof_order]
    leg_prof = fig.legend(
        handles=prof_handles, title="profile",
        loc="center right", bbox_to_anchor=(0.985, 0.62), frameon=False
    )
    for txt in leg_prof.get_texts():
        lbl = txt.get_text()
        txt.set_color(PROFILE_COLORS.get(lbl, "black"))

    # Batch-Legende (Formen), direkt darunter
    batch_handles = [
        Line2D([], [], marker=BATCH_MARKERS[32],  linestyle="None", color="black", label="BS32"),
        Line2D([], [], marker=BATCH_MARKERS[64],  linestyle="None", color="black", label="BS64"),
        Line2D([], [], marker=BATCH_MARKERS[128], linestyle="None", color="black", label="BS128"),
    ]
    fig.legend(
        handles=batch_handles, title="batch size",
        loc="center right", bbox_to_anchor=(0.985, 0.30), frameon=False
    )

    # Speichern – tight, damit die außenliegenden Legenden nicht abgeschnitten werden
    os.makedirs(out_dir, exist_ok=True)
    safe_model = str(model).replace("/", "_")
    out_path = os.path.join(out_dir, f"01_throughput_vs_power__{safe_model}__{precision}.svg")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Mehrere Einzelplots: Throughput vs Power (SVG)")
    ap.add_argument("--manifest", required=True, help="Pfad zu grid_manifest.csv")
    ap.add_argument("--out", required=True, help="Ausgabeordner (wird angelegt)")
    ap.add_argument("--xlog", action="store_true", help="X-Achse logarithmisch")
    ap.add_argument("--ylog", action="store_true", help="Y-Achse logarithmisch")
    ap.add_argument("--vary-x", action="store_true", help="X-Achse pro Plot variieren (statt global)")
    ap.add_argument("--vary-y", action="store_true", help="Y-Achse pro Plot variieren (statt global)")
    args = ap.parse_args()

    need = ["model","precision","profile","batch","avg_power_W","throughput_img_s"]
    df = pd.read_csv(args.manifest)
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"Spalte fehlt im Manifest: {c}")

    d = df.dropna(subset=need).copy()
    if d.empty:
        raise SystemExit("Keine Daten im Manifest für diesen Plot.")

    # Reihenfolge: Präzisionen fix, Modelle alphabetisch
    precisions = [p for p in ["fp16","fp32","fp64"] if p in d["precision"].unique()]
    models     = sorted(d["model"].unique().tolist())

    # Globale Limits (nur wenn Achsen NICHT variiert werden)
    xlim_global = None if args.vary_x else _limits_from(d["avg_power_W"], log=args.xlog, pad_frac=0.05)
    ylim_global = None if args.vary_y else _limits_from(d["throughput_img_s"], log=args.ylog, pad_frac=0.08)
    if (not args.vary_x and xlim_global is None) or (not args.vary_y and ylim_global is None):
        raise SystemExit("Konnte globale Limits nicht bestimmen.")

    made = []
    for m in models:
        for pz in precisions:
            sub = d[(d["model"] == m) & (d["precision"] == pz)]
            if sub.empty:
                continue

            # Lokale Limits (falls variiert), sonst globale übernehmen
            xlim = _limits_from(sub["avg_power_W"], log=args.xlog, pad_frac=0.05) if args.vary_x else xlim_global
            ylim = _limits_from(sub["throughput_img_s"], log=args.ylog, pad_frac=0.08) if args.vary_y else ylim_global

            out_path = _plot_single(sub, args.out, m, pz, args.xlog, args.ylog, xlim, ylim)
            print("✔︎ SVG:", out_path)
            made.append(out_path)

    if not made:
        print("Nichts erzeugt (keine passenden Kombinationen).")
    else:
        print(f"Fertig: {len(made)} SVGs.")

if __name__ == "__main__":
    main()
