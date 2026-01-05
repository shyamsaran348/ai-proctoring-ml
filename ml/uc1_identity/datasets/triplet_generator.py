# ml/uc1_identity/datasets/triplet_generator.py

from pathlib import Path
import random
import yaml

PROCESSED_TRAIN_DIR = Path("ml/data/processed/vggface2/train")

SPLITS_FILE = Path("ml/uc1_identity/datasets/splits.yaml")
ENROLLMENT_FILE = Path("ml/uc1_identity/datasets/enrollment_map.yaml")

OUTPUT_DIR = Path("ml/uc1_identity/datasets")

RANDOM_SEED = 42
NEGATIVES_PER_POSITIVE = 1


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_triplets_for_split(split_key_upper, split_identities, enrollment_map):
    random.seed(RANDOM_SEED)
    triplets = []

    for identity in split_identities:
        data = enrollment_map.get(split_key_upper, {}).get(identity)
        if data is None:
            continue

        anchor = PROCESSED_TRAIN_DIR / identity / data["enrollment"]
        positives = data["probes"]

        for pos_img in positives:
            positive = PROCESSED_TRAIN_DIR / identity / pos_img

            for _ in range(NEGATIVES_PER_POSITIVE):
                neg_identity = random.choice(
                    [i for i in split_identities if i != identity]
                )

                neg_data = enrollment_map[split_key_upper].get(neg_identity)
                if neg_data is None:
                    continue

                neg_img = random.choice(neg_data["probes"])
                negative = PROCESSED_TRAIN_DIR / neg_identity / neg_img

                triplets.append(
                    (str(anchor), str(positive), str(negative))
                )

    return triplets


def save_triplets(triplets, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for a, p, n in triplets:
            f.write(f"{a},{p},{n}\n")

    print(f"[INFO] Saved {len(triplets)} triplets → {output_path}")


if __name__ == "__main__":
    splits = load_yaml(SPLITS_FILE)
    enrollment_map = load_yaml(ENROLLMENT_FILE)

    # 🔒 Canonical mapping (FINAL)
    SPLIT_MAPPING = {
        "train": "TRAIN_IDS",
        "val": "VAL_IDS",
        "test": "TEST_IDS"
    }

    for split_name, enrollment_key in SPLIT_MAPPING.items():
        split_ids_key = f"{split_name}_ids"
        identities = splits.get(split_ids_key, [])

        triplets = generate_triplets_for_split(
            enrollment_key,
            identities,
            enrollment_map
        )

        output_file = OUTPUT_DIR / f"triplets_{split_name}.txt"
        save_triplets(triplets, output_file)
