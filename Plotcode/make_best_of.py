#!/usr/bin/env python3
# make_best_of.py — erzeugt Best-of-Tabellen, Empfehlungsmatrix und Methodik-Text
import argparse, numpy as np, pandas as pd
from pathlib import Path

def to_num(x): return pd.to_numeric(x, errors="coerce")

def load_manifest(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    num_cols = ["batch","avg_power_W","power_samples","throughput_img_s",
                "latency_avg_ms","latency_p50_ms","duration_s",
                "energy_J","energy_per_img_J","energy_per_inference_J","top1"]
    for c in num_cols:
        if c in df.columns: df[c] = to_num(df[c])

    # Ableitungen/Fallbacks
    if ("energy_per_img_J" not in df.columns) or df["energy_per_img_J"].isna().all():
        if {"avg_power_W","throughput_img_s"}.issubset(df.columns):
            df["energy_per_img_J"] = np.where(
                (df["avg_power_W"]>0) & (df["throughput_img_s"]>0),
                df["avg_power_W"]/df["throughput_img_s"], np.nan
            )
    if ("images_per_joule" not in df.columns) and {"avg_power_W","throughput_img_s"}.issubset(df.columns):
        df["images_per_joule"] = np.where(
            (df["avg_power_W"]>0) & (df["throughput_img_s"]>0),
            df["throughput_img_s"]/df["avg_power_W"], np.nan
        )
    if "latency_avg_ms" in df.columns:
        df["edp_image_Js"] = df["energy_per_img_J"] * (df["latency_avg_ms"]/1000.0)
    return df

def best_low_latency(g: pd.DataFrame) -> pd.Series:
    thr_med = g["throughput_img_s"].median()
    cand = g[g["throughput_img_s"] >= thr_med]
    if cand.empty: cand = g
    return cand.loc[cand["latency_avg_ms"].idxmin()]

def best_high_throughput(g: pd.DataFrame) -> pd.Series:
    lat_med = g["latency_avg_ms"].median()
    cand = g[g["latency_avg_ms"] <= lat_med]
    if cand.empty: cand = g
    return cand.loc[cand["throughput_img_s"].idxmax()]

def best_green_edge(g: pd.DataFrame) -> pd.Series:
    return g.loc[g["energy_per_img_J"].idxmin()]

def best_edp(g: pd.DataFrame) -> pd.Series:
    return g.loc[g["edp_image_Js"].idxmin()]

def pack(tag, row, m, pz):
    return {
        "model": m, "precision": pz, "scenario": tag,
        "profile": row.get("profile"),
        "batch": int(row.get("batch")) if pd.notna(row.get("batch")) else "",
        "throughput_img_s": row.get("throughput_img_s"),
        "latency_avg_ms": row.get("latency_avg_ms"),
        "avg_power_W": row.get("avg_power_W"),
        "energy_per_img_J": row.get("energy_per_img_J"),
        "images_per_joule": row.get("images_per_joule"),
        "edp_image_Js": row.get("edp_image_Js"),
        "prefix": row.get("prefix"),
    }

def main():
    ap = argparse.ArgumentParser("Best-of-Auswertung aus grid_manifest.csv")
    ap.add_argument("--manifest", required=True, help="Pfad zu grid_manifest.csv")
    ap.add_argument("--out", required=True, help="Zielordner")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(manifest)
    need = ["model","precision","profile","batch","avg_power_W","throughput_img_s","latency_avg_ms","energy_per_img_J","edp_image_Js"]
    d = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    if d.empty: raise SystemExit("Keine verwertbaren Zeilen im Manifest.")

    # --- Best-of pro (model, precision)
    recs = []
    for (m,pz), g in d.groupby(["model","precision"]):
        if g.empty: continue
        recs += [
            pack("Low-latency (thr ≥ median)",   best_low_latency(g),   m, pz),
            pack("High-throughput (lat ≤ median)", best_high_throughput(g), m, pz),
            pack("Green-Edge (min J/img)",       best_green_edge(g),   m, pz),
            pack("EDP-optimal (min J·s/img)",    best_edp(g),          m, pz),
        ]
    best_per = pd.DataFrame.from_records(recs)
    best_per.to_csv(out_dir/"best_of_per_model_precision.csv", index=False)

    # --- Globale Gewinner über alle M+P
    def pick_global(df_in, scenario):
        if scenario.startswith("Low-latency"):
            thr_med = df_in["throughput_img_s"].median()
            cand = df_in[df_in["throughput_img_s"] >= thr_med]
            if cand.empty: cand = df_in
            return cand.loc[cand["latency_avg_ms"].idxmin()]
        if scenario.startswith("High-throughput"):
            lat_med = df_in["latency_avg_ms"].median()
            cand = df_in[df_in["latency_avg_ms"] <= lat_med]
            if cand.empty: cand = df_in
            return cand.loc[cand["throughput_img_s"].idxmax()]
        if scenario.startswith("Green-Edge"):
            return df_in.loc[df_in["energy_per_img_J"].idxmin()]
        if scenario.startswith("EDP-optimal"):
            return df_in.loc[df_in["edp_image_Js"].idxmin()]
        raise ValueError("unknown scenario")
    scenarios = ["Low-latency (thr ≥ median)", "High-throughput (lat ≤ median)",
                 "Green-Edge (min J/img)", "EDP-optimal (min J·s/img)"]
    glob = []
    for sc in scenarios:
        r = pick_global(d, sc)
        glob.append(pack(sc, r, r.get("model"), r.get("precision")))
    best_global = pd.DataFrame.from_records(glob)
    best_global.to_csv(out_dir/"best_of_global.csv", index=False)

    # --- Empfehlungsmatrix (kompakt)
    def rec_str(row):
        b = row.get("batch"); btxt = f"BS{int(b)}" if b!= "" else ""
        return f"{row.get('model')} / {row.get('precision')} / {btxt} / {row.get('profile')}"
    recommendation = pd.DataFrame({
        "scenario": best_global["scenario"],
        "recommended_config": best_global.apply(rec_str, axis=1),
        "throughput_img_s": best_global["throughput_img_s"].map(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
        "latency_ms": best_global["latency_avg_ms"].map(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
        "energy_per_img_mJ": (best_global["energy_per_img_J"]*1000.0).map(lambda x: f"{x:.2f}" if pd.notna(x) else ""),
        "EDP_Js": best_global["edp_image_Js"].map(lambda x: f"{x:.4f}" if pd.notna(x) else ""),
    })
    recommendation.to_csv(out_dir/"recommendation_matrix.csv", index=False)

    # --- Methodik/Einheiten (Markdown)
    md = """# Methodology & Units (Benchmark-Auswertung)

**Messgrößen & Ableitungen**
- **avg_power_W [W]:** Mittelwert der Stichproben aus `*_tegrastats_power.csv` (mW automatisch normalisiert → W).
- **throughput_img_s [img/s]:** aus Bench-Summary (Gesamtbilder / Summe Batch-Latenzen).
- **latency_avg_ms [ms]:** Mittel der Batch-Latenzen.
- **energy_per_img_J [J/Bild]:** `avg_power_W / throughput_img_s`.
- **images_per_joule [img/J]:** `throughput_img_s / avg_power_W`.
- **EDP per image [J·s]:** `energy_per_img_J × (latency_avg_ms/1000)`.

**Szenario-Definitionen (datenadaptive Schwellen)**
- **Low-latency:** Minimiert `latency_avg_ms` unter der Nebenbedingung `throughput_img_s ≥ Median` (je Gruppe).
- **High-throughput:** Maximiert `throughput_img_s` unter der Nebenbedingung `latency_avg_ms ≤ Median`.
- **Green-Edge:** Minimiert `energy_per_img_J`.
- **EDP-optimal:** Minimiert `EDP per image`.

**Gruppierung & Auswahl**
- Pro **(model, precision)** wird je Szenario die beste Konfiguration (profile, batch) gewählt.
- Zusätzlich werden **globale** Empfehlungen (über alle Modelle/Präzisionen) ermittelt.

**Einheiten**
- Leistung in **Watt [W]**, Energie in **Joule [J]** (für Berichte ggf. **mJ**).
- Latenz in **Millisekunden [ms]**; in EDP-Berechnungen in **Sekunden**.
- In den SVG-Plots: Farbe = **Power-Profil**, Markerform = **Batchgröße**.
"""
    (out_dir/"methodology_units.md").write_text(md, encoding="utf-8")

    print("✅ Geschrieben in:", out_dir)
    print(" - best_of_per_model_precision.csv")
    print(" - best_of_global.csv")
    print(" - recommendation_matrix.csv")
    print(" - methodology_units.md")

if __name__ == "__main__":
    main()
