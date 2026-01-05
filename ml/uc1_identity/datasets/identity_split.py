# ml/uc1_identity/datasets/identity_split.py

from pathlib import Path
import random
import yaml

PROCESSED_DATA_DIR = Path("ml/data/processed/vggface2/train")
OUTPUT_SPLIT_FILE = Path("ml/uc1_identity/datasets/splits.yaml")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

def get_all_identities(data_dir: Path):
    identities = [
        d.name for d in data_dir.iterdir()
        if d.is_dir()
    ]
    return identities


def split_identities(identities):
    random.seed(RANDOM_SEED)
    random.shuffle(identities)

    total = len(identities)
    train_end = int(TRAIN_RATIO * total)
    val_end = train_end + int(VAL_RATIO * total)

    train_ids = identities[:train_end]
    val_ids = identities[train_end:val_end]
    test_ids = identities[val_end:]

    return train_ids, val_ids, test_ids

def save_splits(train_ids, val_ids, test_ids, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_data = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids
    }

    with open(output_path, "w") as f:
        yaml.dump(split_data, f)

    print("[INFO] Identity splits saved to:", output_path)
    print(f"[INFO] Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

if __name__ == "__main__":
    identities = get_all_identities(PROCESSED_DATA_DIR)
    print(f"[INFO] Total identities found: {len(identities)}")

    train_ids, val_ids, test_ids = split_identities(identities)
    save_splits(train_ids, val_ids, test_ids, OUTPUT_SPLIT_FILE)
