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
import random
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

NEGATIVE_MODES = [
    ("abrupt", 0.3),
    ("gradual", 0.4),
    ("multi_switch", 0.3)
]

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
# NEGATIVE GENERATORS
# ============================
def generate_abrupt_switch(genuine, impostor):
    t = random.randint(SEQ_LEN // 4, 3 * SEQ_LEN // 4)
    return np.concatenate([genuine[:t], impostor[t:]])

def generate_gradual_drift(genuine, impostor):
    alpha = np.linspace(0, 1, SEQ_LEN)
    return (1 - alpha) * genuine + alpha * impostor

def generate_multi_switch(genuine, impostor):
    out = []
    switch_every = random.randint(2, 6)
    for i in range(SEQ_LEN):
        if i % switch_every == 0:
            out.append(impostor[i])
        else:
            out.append(genuine[i])
    return np.array(out)

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
    model.load_state_dict(torch.load(UC1_CKPT, map_location=device))
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
            # -------- ENROLLMENT --------
            enroll_row = df[
                (df.identity_id == identity) &
                (df.role == "enrollment")
            ]

            if len(enroll_row) != 1:
                continue

            enroll_img = load_image(enroll_row.image_path.values[0]).to(device)
            with torch.no_grad():
                enroll_emb = model(enroll_img).cpu().numpy()

            # -------- PROBES --------
            probes = df[
                (df.identity_id == identity) &
                (df.role == "probe")
            ].image_path.tolist()

            if len(probes) < SEQ_LEN:
                continue

            # -------- POSITIVE --------
            genuine_scores = []
            for p in probes[:SEQ_LEN]:
                img = load_image(p).to(device)
                with torch.no_grad():
                    emb = model(img).cpu().numpy()
                genuine_scores.append(cosine_similarity(enroll_emb, emb)[0][0])

            genuine_scores = np.array(genuine_scores, dtype=np.float32)

            all_sequences.append(genuine_scores.reshape(SEQ_LEN, 1))
            all_labels.append(0)

            # -------- NEGATIVE --------
            other_ids = [i for i in identities if i != identity]
            if not other_ids:
                continue

            impostor_id = random.choice(other_ids)

            imp_probes = df[
                (df.identity_id == impostor_id) &
                (df.role == "probe")
            ].image_path.tolist()

            if len(imp_probes) < SEQ_LEN:
                continue

            impostor_scores = []
            for p in imp_probes[:SEQ_LEN]:
                img = load_image(p).to(device)
                with torch.no_grad():
                    emb = model(img).cpu().numpy()
                impostor_scores.append(cosine_similarity(enroll_emb, emb)[0][0])

            impostor_scores = np.array(impostor_scores, dtype=np.float32)

            modes, probs = zip(*NEGATIVE_MODES)
            mode = random.choices(modes, probs)[0]

            if mode == "abrupt":
                neg = generate_abrupt_switch(genuine_scores, impostor_scores)
            elif mode == "gradual":
                neg = generate_gradual_drift(genuine_scores, impostor_scores)
            else:
                neg = generate_multi_switch(genuine_scores, impostor_scores)

            all_sequences.append(neg.reshape(SEQ_LEN, 1))
            all_labels.append(1)

    # ============================
    # SAVE OUTPUTS (v2)
    # ============================
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    np.save("train_sequences_v2.npy", X)
    np.save("sequence_labels_v2.npy", y)

    print("✅ UC2 v2 sequences saved")
    print("Shape:", X.shape)
    print("Labels:", np.unique(y, return_counts=True))

# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    main()
