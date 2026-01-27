import os
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from ml.uc1_identity.models.resnet_embedder import ResNetEmbedder

# ============================================================
# CONFIG
# ============================================================

PROCESSED_ROOT = "ml/data/processed/vggface2"
CHECKPOINT_PATH = "ml/uc1_identity/models/checkpoints/uc1_resnet_embedder.pth"
ENROLLMENT_MAP_PATH = "ml/uc1_identity/datasets/enrollment_map.yaml"

OUT_DIR = Path("ml/uc5_risk_fusion/datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODEL LOADING
# ============================================================

def load_frozen_model(device):
    model = ResNetEmbedder(embedding_dim=256)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model

# ============================================================
# TRANSFORM
# ============================================================

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_image(identity_id, filename):
    for split in ["train", "val", "test"]:
        p = os.path.join(PROCESSED_ROOT, split, identity_id, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{identity_id}/{filename}")

# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# ============================================================
# UC1 EXPORT (PHASE 5)
# ============================================================

def export_uc1_scores():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_model(device)
    tfm = get_transform()

    # --------------------------------------------------------
    # LOAD ENROLLMENT MAP (STRUCTURE-AWARE)
    # --------------------------------------------------------
    with open(ENROLLMENT_MAP_PATH, "r") as f:
        enrollment_map = yaml.safe_load(f)

    # Take first split (e.g., TEST_IDS)
    split_key = list(enrollment_map.keys())[0]

    # Take first identity inside that split
    identity_id = list(enrollment_map[split_key].keys())[0]
    data = enrollment_map[split_key][identity_id]

    print(f"[INFO] Using identity '{identity_id}' from split '{split_key}'")

    # --------------------------------------------------------
    # ENROLLMENT EMBEDDING
    # --------------------------------------------------------
    enroll_img = Image.open(
        resolve_image(identity_id, data["enrollment"])
    ).convert("RGB")

    enroll_img = tfm(enroll_img).unsqueeze(0).to(device)

    with torch.no_grad():
        enroll_emb = model(enroll_img).squeeze(0).cpu()

    # --------------------------------------------------------
    # PROBE SIMILARITIES (TEMPORAL ORDER PRESERVED)
    # --------------------------------------------------------
    uc1_scores = []

    for probe_fn in data.get("probes", []):
        img = Image.open(
            resolve_image(identity_id, probe_fn)
        ).convert("RGB")

        img = tfm(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probe_emb = model(img).squeeze(0).cpu()

        uc1_scores.append(cosine_sim(enroll_emb, probe_emb))

    uc1_scores = np.array(uc1_scores, dtype=np.float32)

    # --------------------------------------------------------
    # SAVE FOR UC5
    # --------------------------------------------------------
    np.save(OUT_DIR / "uc1_scores.npy", uc1_scores)

    print(f"[SAVED] uc1_scores.npy | frames = {len(uc1_scores)}")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    export_uc1_scores()
