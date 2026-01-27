import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from ml.uc5_risk_fusion.fusion_model import RiskFusionGRU

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"

# ============================================================
# LOAD DATA
# ============================================================

X = np.load(DATASET_DIR / "risk_sequences.npy")   # (B, T, 2)
y = np.load(DATASET_DIR / "risk_labels.npy")      # (B,)

# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RiskFusionGRU(input_dim=2, hidden_dim=32).to(device)
model.load_state_dict(torch.load(MODEL_DIR / "risk_fusion_gru.pth", map_location=device))
model.eval()

# ============================================================
# INFERENCE
# ============================================================

with torch.no_grad():
    risk_traj, final_risk = model(
        torch.tensor(X, dtype=torch.float32).to(device)
    )

risk_traj = risk_traj.cpu().numpy()
final_risk = torch.sigmoid(final_risk).cpu().numpy()

# ============================================================
# VISUALIZATION
# ============================================================

def plot_risk_sequence(idx):
    uc1 = X[idx, :, 0]
    uc2 = X[idx, :, 1]
    risk = risk_traj[idx]
    label = y[idx]

    t = np.arange(len(risk))

    plt.figure(figsize=(10, 5))
    plt.plot(t, uc1, label="UC1 Similarity", alpha=0.6)
    plt.plot(t, uc2, label="UC2 Impersonation Prob", alpha=0.6)
    plt.plot(t, risk, label="Fused Risk (GRU)", linewidth=2)

    plt.title(f"Risk Trajectory | Label = {int(label)}")
    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================================================
# RUN VISUALIZATION
# ============================================================

print("[INFO] Plotting first 3 sequences...")

for i in range(min(3, len(X))):
    plot_risk_sequence(i)

# ============================================================
# FINAL RISK DISTRIBUTION
# ============================================================

plt.figure(figsize=(6, 4))
plt.hist(final_risk[y == 0], bins=20, alpha=0.6, label="Genuine")
plt.hist(final_risk[y == 1], bins=20, alpha=0.6, label="Impersonation")
plt.xlabel("Final Session Risk")
plt.ylabel("Frequency")
plt.title("Final Risk Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
