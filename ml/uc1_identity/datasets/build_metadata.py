# ml/uc1_identity/datasets/build_metadata.py

from pathlib import Path
import csv
import yaml

PROCESSED_DIR = Path("ml/data/processed/vggface2/train")

SPLITS_FILE = Path("ml/uc1_identity/datasets/splits.yaml")
ENROLLMENT_FILE = Path("ml/uc1_identity/datasets/enrollment_map.yaml")

OUTPUT_CSV = Path("ml/uc1_identity/datasets/uc1_metadata.csv")

def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_metadata_rows(splits, enrollment_map):
    rows = []

    split_key_map = {
        "train": "TRAIN_IDS",
        "val": "VAL_IDS",
        "test": "TEST_IDS"
    }

    for split_name, enrollment_key in split_key_map.items():
        identity_ids = splits.get(f"{split_name}_ids", [])

        for identity in identity_ids:
            data = enrollment_map.get(enrollment_key, {}).get(identity)
            if data is None:
                continue

            # Enrollment row
            enroll_path = PROCESSED_DIR / identity / data["enrollment"]
            rows.append([
                split_name,
                identity,
                "enrollment",
                str(enroll_path)
            ])

            # Probe rows
            for probe_img in data["probes"]:
                probe_path = PROCESSED_DIR / identity / probe_img
                rows.append([
                    split_name,
                    identity,
                    "probe",
                    str(probe_path)
                ])

    return rows

def save_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "identity_id", "role", "image_path"])
        writer.writerows(rows)

    print(f"[INFO] Metadata CSV saved to: {output_path}")
    print(f"[INFO] Total rows: {len(rows)}")

if __name__ == "__main__":
    splits = load_yaml(SPLITS_FILE)
    enrollment_map = load_yaml(ENROLLMENT_FILE)

    rows = build_metadata_rows(splits, enrollment_map)
    save_csv(rows, OUTPUT_CSV)

