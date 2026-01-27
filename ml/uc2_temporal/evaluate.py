import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

from temporal_lstm import TemporalLSTM

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"

OUT_DIR = Path("ml/uc5_risk_fusion/datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

X = np.load(DATASET_DIR / "train_sequences_v2.npy")
y = np.load(DATASET_DIR / "sequence_labels_v2.npy")

# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TemporalLSTM().to(device)
model.load_state_dict(torch.load(MODEL_DIR / "lstm_uc2.pth", map_location=device))
model.eval()

# ============================================================
# INFERENCE
# ============================================================

with torch.no_grad():
    logits = model(torch.tensor(X, dtype=torch.float32).to(device))
    probs = torch.sigmoid(logits).cpu().numpy().squeeze()

# ============================================================
# METRIC (OPTIONAL)
# ============================================================

auc = roc_auc_score(y, probs)
print(f"[METRIC] UC2 AUC: {auc:.6f}")

# ============================================================
# EXPORT FOR PHASE 5
# ============================================================

np.save(OUT_DIR / "uc2_probs.npy", probs)
print(f"[SAVED] uc2_probs.npy | windows = {len(probs)}")
