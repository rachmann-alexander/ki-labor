#!/usr/bin/env python3
import os, sys, json
import numpy as np
import pandas as pd

USAGE = f"Usage: {sys.argv[0]} <grid_manifest.csv> <out_dir>"

NUM_COLS = [
    "avg_power_W","power_samples","throughput_img_s","latency_avg_ms",
    "latency_p50_ms","duration_s","energy_J","energy_per_img_J",
    "energy_per_inference_J","images_per_joule","inferences_per_joule","top1","batch"
]

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def main():
    if len(sys.argv) != 3:
        print(USAGE, file=sys.stderr); sys.exit(2)
    manifest, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(manifest)
    # numerisch casten
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = to_num(df[c])

    # 1) Ranges/NaNs
    rows = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            rows.append({
                "column": c,
                "count": int(s.shape[0]),
                "n_nan": int(df[c].isna().sum()),
                "min": float(s.min()) if not s.empty else np.nan,
                "p25": float(s.quantile(0.25)) if not s.empty else np.nan,
                "median": float(s.median()) if not s.empty else np.nan,
                "p75": float(s.quantile(0.75)) if not s.empty else np.nan,
                "max": float(s.max()) if not s.empty else np.nan,
                "mean": float(s.mean()) if not s.empty else np.nan,
                "std": float(s.std()) if not s.empty else np.nan,
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "diag_ranges.csv"), index=False)

    # 2) Heuristische Unit-Checks
    checks = []
    if "avg_power_W" in df.columns:
        med = df["avg_power_W"].dropna().median()
        checks.append({"check":"avg_power_W_unit",
                       "median": float(med) if pd.notna(med) else np.nan,
                       "suspicious_mW": bool(pd.notna(med) and med>100.0),
                       "note":">100W Median deutet oft auf mW-Skala hin"})
    if "duration_s" in df.columns:
        mx = df["duration_s"].dropna().max()
        checks.append({"check":"duration_s_range",
                       "max": float(mx) if pd.notna(mx) else np.nan,
                       "note":"Sehr große Dauer kann auf Messfenster statt reinen Inferenzlauf hinweisen"})
    if set(["avg_power_W","throughput_img_s"]).issubset(df.columns):
        # Energie pro Bild (rechnerisch)
        epi_calc = df["avg_power_W"] / df["throughput_img_s"]
        epi_med = epi_calc.replace([np.inf,-np.inf], np.nan).dropna().median()
        checks.append({"check":"energy_per_img_from_P_T",
                       "median_J": float(epi_med) if pd.notna(epi_med) else np.nan,
                       "note":"Vergleichswert zu energy_per_img_J (falls vorhanden)"})
    pd.DataFrame(checks).to_csv(os.path.join(out_dir, "diag_unit_checks.csv"), index=False)

    # 3) Korrelationen (Pearson) – einfache Plausibilitäten
    corrs = []
    def add_corr(xc, yc, name):
        x = to_num(df.get(xc, np.nan)); y = to_num(df.get(yc, np.nan))
        s = pd.concat([x, y], axis=1).dropna()
        if s.shape[0] >= 3:
            r = float(s.corr(method="pearson").iloc[0,1])
            corrs.append({"pair": name, "n": int(s.shape[0]), "pearson_r": r})
    add_corr("throughput_img_s","latency_avg_ms","throughput_vs_latency_ms")
    add_corr("avg_power_W","throughput_img_s","power_vs_throughput")
    add_corr("energy_per_img_J","throughput_img_s","epi_vs_throughput")
    pd.DataFrame(corrs).to_csv(os.path.join(out_dir, "diag_correlations.csv"), index=False)

    # 4) Spread je Gruppe (wie stabil sind die Kennzahlen)
    grp_cols = [c for c in ["model","precision","profile","batch"] if c in df.columns]
    if grp_cols:
        agg = df.groupby(grp_cols).agg(
            epi_med = ("energy_per_img_J","median"),
            epi_min = ("energy_per_img_J","min"),
            epi_max = ("energy_per_img_J","max"),
            th_med  = ("throughput_img_s","median"),
            pow_med = ("avg_power_W","median"),
            n=("energy_per_img_J","count")
        ).reset_index()
        agg["epi_rel_spread"] = (agg["epi_max"] - agg["epi_min"]) / agg["epi_med"]
        agg.to_csv(os.path.join(out_dir, "diag_epi_spread_by_group.csv"), index=False)

    # 5) Schnelle Ausreißerhinweise (schwelle = 3*IQR)
    out_rows = []
    for c in ["avg_power_W","throughput_img_s","latency_avg_ms","energy_per_img_J"]:
        if c not in df.columns: continue
        s = df[c].dropna()
        if s.empty: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3*iqr, q3 + 3*iqr
        mask = (df[c] < lo) | (df[c] > hi)
        out_rows.append({"column": c, "n_outliers": int(mask.sum()), "lo": float(lo), "hi": float(hi)})
    pd.DataFrame(out_rows).to_csv(os.path.join(out_dir, "diag_outliers_iqr.csv"), index=False)

    print("✅ Diagnostics written to", out_dir)

if __name__ == "__main__":
    main()
