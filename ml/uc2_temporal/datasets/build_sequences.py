# ============================
# PATH SETUP (VERY IMPORTANT)
# ============================
import sys
from pathlib import Path

# ============================
# PATH SETUP (KAGGLE-SAFE)
# ============================
import os
import sys
from pathlib import Path

if os.getcwd().startswith("/kaggle"):
    PROJECT_ROOT = Path("/kaggle/working/ai-proctoring-ml")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.append(str(PROJECT_ROOT))

# ============================
# IMPORTS
# ============================
import torch
import yaml
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from ml.uc1_identity.models.resnet_embedder import ResNetEmbedder

# ============================
# CONFIG
# ============================
BASE_DIR = PROJECT_ROOT / "ml"

UC1_CKPT = BASE_DIR / "uc1_identity/models/checkpoints/uc1_resnet_embedder.pth"
METADATA = BASE_DIR / "uc1_identity/datasets/uc1_metadata.csv"
SPLITS = BASE_DIR / "uc1_identity/datasets/splits.yaml"

SEQ_LEN = 60
SWITCH_RANGE = (15, 45)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# IMAGE TRANSFORM
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_image(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)


# ============================
# MAIN
# ============================
def main():
    # -------- LOAD UC1 DATA --------
    df = pd.read_csv(METADATA)
    splits_raw = yaml.safe_load(open(SPLITS))

    split_map = {
        "train": splits_raw["train_ids"],
        "val": splits_raw["val_ids"],
        "test": splits_raw["test_ids"],
    }

    print("Split sizes:", {k: len(v) for k, v in split_map.items()})

    # -------- LOAD UC1 MODEL --------
    model = ResNetEmbedder(embedding_dim=256)
    state_dict = torch.load(UC1_CKPT, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("✅ UC1 model loaded successfully")

    all_sequences = []
    all_labels = []

    # ============================
    # SEQUENCE GENERATION
    # ============================
    for split in ["train", "val", "test"]:
        identities = split_map[split]

        for identity in identities:
            # -------- ENROLLMENT (SOURCE OF TRUTH) --------
            enroll_rows = df[
                (df.identity_id == identity) &
                (df.role == "enrollment")
            ]

            if len(enroll_rows) != 1:
                continue  # safety guard

            enroll_path = enroll_rows.image_path.values[0]
            enroll_img = load_image(enroll_path).to(device)

            with torch.no_grad():
                enroll_emb = model(enroll_img).cpu().numpy()

            # -------- PROBES --------
            probes = df[
                (df.identity_id == identity) &
                (df.role == "probe")
            ].image_path.tolist()

            if len(probes) < SEQ_LEN:
                continue

            # -------- POSITIVE SEQUENCE --------
            pos_scores = []
            for p in probes[:SEQ_LEN]:
                img = load_image(p).to(device)
                with torch.no_grad():
                    emb = model(img).cpu().numpy()
                pos_scores.append(
                    cosine_similarity(enroll_emb, emb)[0][0]
                )

            all_sequences.append(
                np.array(pos_scores, dtype=np.float32).reshape(SEQ_LEN, 1)
            )
            all_labels.append(0)

            # -------- NEGATIVE SEQUENCE (SINGLE SWITCH) --------
            other_ids = [i for i in identities if i != identity]
            if not other_ids:
                continue

            impostor = np.random.choice(other_ids)

            imp_probes = df[
                (df.identity_id == impostor) &
                (df.role == "probe")
            ].image_path.tolist()

            if len(imp_probes) < SEQ_LEN:
                continue

            switch = np.random.randint(*SWITCH_RANGE)
            neg_scores = []

            for p in probes[:switch]:
                img = load_image(p).to(device)
                with torch.no_grad():
                    emb = model(img).cpu().numpy()
                neg_scores.append(
                    cosine_similarity(enroll_emb, emb)[0][0]
                )

            for p in imp_probes[:SEQ_LEN - switch]:
                img = load_image(p).to(device)
                with torch.no_grad():
                    emb = model(img).cpu().numpy()
                neg_scores.append(
                    cosine_similarity(enroll_emb, emb)[0][0]
                )

            all_sequences.append(
                np.array(neg_scores, dtype=np.float32).reshape(SEQ_LEN, 1)
            )
            all_labels.append(1)

    # ============================
    # SAVE OUTPUTS
    # ============================
    all_sequences = np.array(all_sequences, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    np.save("train_sequences.npy", all_sequences)
    np.save("sequence_labels.npy", all_labels)

    print("✅ UC2 temporal sequences saved successfully")
    print("Sequences shape:", all_sequences.shape)

# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    main()
