# ml/uc1_identity/datasets/enrollment_sampler.py

from pathlib import Path
import random
import yaml

PROCESSED_TRAIN_DIR = Path("ml/data/processed/vggface2/train")

SPLITS_FILE = Path("ml/uc1_identity/datasets/splits.yaml")
OUTPUT_ENROLLMENT_FILE = Path("ml/uc1_identity/datasets/enrollment_map.yaml")

RANDOM_SEED = 42

def load_splits(splits_path: Path):
    with open(splits_path, "r") as f:
        splits = yaml.safe_load(f)
    return splits

def sample_enrollment(identity_dir: Path):
    images = [p.name for p in identity_dir.iterdir() if p.is_file()]

    if len(images) < 2:
        # Cannot form enrollment + probe
        return None

    enrollment_img = random.choice(images)
    probe_imgs = [img for img in images if img != enrollment_img]

    return enrollment_img, probe_imgs

def build_enrollment_map(splits):
    random.seed(RANDOM_SEED)

    enrollment_map = {}

    for split_name, identities in splits.items():
        enrollment_map[split_name] = {}

        for identity in identities:
            identity_dir = PROCESSED_TRAIN_DIR / identity

            if not identity_dir.exists():
                continue

            sampled = sample_enrollment(identity_dir)
            if sampled is None:
                continue

            enrollment_img, probe_imgs = sampled

            enrollment_map[split_name][identity] = {
                "enrollment": enrollment_img,
                "probes": probe_imgs
            }

    return enrollment_map

def save_enrollment_map(enrollment_map, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(enrollment_map, f)

    print("[INFO] Enrollment map saved to:", output_path)

    for split in enrollment_map:
        print(f"[INFO] {split.upper()} identities:", len(enrollment_map[split]))

if __name__ == "__main__":
    splits = load_splits(SPLITS_FILE)
    enrollment_map = build_enrollment_map(splits)
    save_enrollment_map(enrollment_map, OUTPUT_ENROLLMENT_FILE)

