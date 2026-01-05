# ml/uc1_identity/datasets/build_uc1_dataset.py

from pathlib import Path
import random
import yaml
import csv

# =========================
# CONFIG (LOCKED)
# =========================

PROCESSED_DIR = Path("ml/data/processed/vggface2/train")
OUTPUT_DIR = Path("ml/uc1_identity/datasets")

MIN_IMAGES_PER_ID = 3
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42
NEGATIVES_PER_POSITIVE = 1

# =========================
# HELPERS
# =========================

def list_images(identity_dir: Path):
    return [p.name for p in identity_dir.iterdir() if p.is_file()]

def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)

# =========================
# STEP A — FILTER ELIGIBLE IDENTITIES
# =========================

def get_eligible_identities():
    eligible = []
    for d in PROCESSED_DIR.iterdir():
        if not d.is_dir():
            continue
        imgs = list_images(d)
        if len(imgs) >= MIN_IMAGES_PER_ID:
            eligible.append(d.name)
    return eligible

# =========================
# STEP B — IDENTITY SPLIT
# =========================

def split_identities(identities):
    random.seed(RANDOM_SEED)
    random.shuffle(identities)

    total = len(identities)
    t_end = int(TRAIN_RATIO * total)
    v_end = t_end + int(VAL_RATIO * total)

    return {
        "train_ids": identities[:t_end],
        "val_ids": identities[t_end:v_end],
        "test_ids": identities[v_end:]
    }

# =========================
# STEP C — ENROLLMENT SAMPLING
# =========================

def build_enrollment_map(splits):
    random.seed(RANDOM_SEED)
    enrollment = {
        "TRAIN_IDS": {},
        "VAL_IDS": {},
        "TEST_IDS": {}
    }

    for split_name, key in [
        ("train_ids", "TRAIN_IDS"),
        ("val_ids", "VAL_IDS"),
        ("test_ids", "TEST_IDS")
    ]:
        for identity in splits[split_name]:
            imgs = list_images(PROCESSED_DIR / identity)
            enroll = random.choice(imgs)
            probes = [i for i in imgs if i != enroll]
            enrollment[key][identity] = {
                "enrollment": enroll,
                "probes": probes
            }
    return enrollment

# =========================
# STEP D — TRIPLET GENERATION
# =========================

def generate_triplets(splits, enrollment):
    random.seed(RANDOM_SEED)
    triplets = {}

    for split_name, key in [
        ("train", "TRAIN_IDS"),
        ("val", "VAL_IDS"),
        ("test", "TEST_IDS")
    ]:
        ids = splits[f"{split_name}_ids"]
        rows = []

        for identity in ids:
            data = enrollment[key][identity]
            anchor = PROCESSED_DIR / identity / data["enrollment"]

            for p in data["probes"]:
                positive = PROCESSED_DIR / identity / p

                for _ in range(NEGATIVES_PER_POSITIVE):
                    neg_id = random.choice([i for i in ids if i != identity])
                    neg_data = enrollment[key][neg_id]
                    neg_img = random.choice(neg_data["probes"])
                    negative = PROCESSED_DIR / neg_id / neg_img

                    rows.append(
                        (str(anchor), str(positive), str(negative))
                    )
        triplets[split_name] = rows
    return triplets

# =========================
# STEP E — METADATA CSV
# =========================

def build_metadata(splits, enrollment):
    rows = []

    for split_name, key in [
        ("train", "TRAIN_IDS"),
        ("val", "VAL_IDS"),
        ("test", "TEST_IDS")
    ]:
        for identity in splits[f"{split_name}_ids"]:
            data = enrollment[key][identity]
            rows.append([
                split_name,
                identity,
                "enrollment",
                str(PROCESSED_DIR / identity / data["enrollment"])
            ])
            for p in data["probes"]:
                rows.append([
                    split_name,
                    identity,
                    "probe",
                    str(PROCESSED_DIR / identity / p)
                ])
    return rows

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eligible = get_eligible_identities()
    print(f"[INFO] Eligible identities (>= {MIN_IMAGES_PER_ID} images): {len(eligible)}")

    splits = split_identities(eligible)
    save_yaml(splits, OUTPUT_DIR / "splits.yaml")

    enrollment = build_enrollment_map(splits)
    save_yaml(enrollment, OUTPUT_DIR / "enrollment_map.yaml")

    triplets = generate_triplets(splits, enrollment)
    for split, rows in triplets.items():
        out = OUTPUT_DIR / f"triplets_{split}.txt"
        with open(out, "w") as f:
            for a, p, n in rows:
                f.write(f"{a},{p},{n}\n")
        print(f"[INFO] {split}: {len(rows)} triplets")

    metadata = build_metadata(splits, enrollment)
    with open(OUTPUT_DIR / "uc1_metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "identity_id", "role", "image_path"])
        writer.writerows(metadata)

    print(f"[INFO] Metadata rows: {len(metadata)}")
    print("[SUCCESS] UC1 dataset is now TRAINABLE.")
