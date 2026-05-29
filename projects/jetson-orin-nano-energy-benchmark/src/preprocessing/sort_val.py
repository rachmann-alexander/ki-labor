import os, shutil, sys

# mögliche Devkit-Pfade
DEVKIT_PATHS = [
    "/workspace/data/imagenet/ILSVRC2012_devkit_t12",
    "/workspace/data/imagenet/ILSVRC2012_devkit_t3"
]

# Val-Bilder
VAL_DIR = "/workspace/data/imagenet/val"

# Devkit finden
devkit = next((d for d in DEVKIT_PATHS if os.path.exists(os.path.join(d, "data"))), None)
if not devkit:
    print("❌ Kein Devkit gefunden.")
    sys.exit(1)

data_dir = os.path.join(devkit, "data")
gt_txt = os.path.join(data_dir, "ILSVRC2012_validation_ground_truth.txt")
synset_txt = os.path.join(data_dir, "synset_words.txt")

if not (os.path.isfile(gt_txt) and os.path.isfile(synset_txt)):
    print("❌ Erwartete Dateien fehlen:", gt_txt, "oder", synset_txt)
    sys.exit(1)

# Mapping laden
idx_to_syn = {}
with open(synset_txt, "r") as f:
    for i, line in enumerate(f):
        synset = line.strip().split()[0]
        idx_to_syn[i+1] = synset

# Labels laden
with open(gt_txt, "r") as f:
    labels = [int(x.strip()) for x in f.readlines() if x.strip()]

files = sorted([f for f in os.listdir(VAL_DIR) if f.endswith(".JPEG")])
if not files:
    print("❌ Keine JPEG-Dateien gefunden.")
    sys.exit(1)

print(f"📁 Devkit: {devkit}")
print(f"📁 Val: {VAL_DIR}")
print(f"🔢 {len(files)} Dateien, {len(labels)} Labels")

moved = 0
for i, fname in enumerate(files):
    lbl = labels[i] if i < len(labels) else labels[-1]
    syn = idx_to_syn.get(lbl, f"class_{lbl}")
    dst_dir = os.path.join(VAL_DIR, syn)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.move(os.path.join(VAL_DIR, fname), os.path.join(dst_dir, fname))
    moved += 1
    if moved % 5000 == 0:
        print(f"… {moved} / {len(files)} verschoben")

print("✅ Fertig:", moved, "Dateien sortiert.")
print("🔎 Beispiel-Klassenordner:", sorted(os.listdir(VAL_DIR))[:10])


