"""
evaluate_baselines.py
Phase 12 — Baseline Comparisons

Compares the Temporal GRU against 4 increasingly capable baselines.
All models trained/evaluated on the identical noisy Phase 11 dataset split.

Baselines:
  B1 — Threshold Rule Classifier (heuristic, no learning)
  B2 — Mean Signal + Logistic Regression (no temporal structure)
  B3 — Non-Temporal MLP (sees all frames, but not ordered)
  B4 — Last-Frame Logistic Regression (single frame only)
  B5 — Temporal GRU (our system — reference)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ml.uc5_risk_fusion.fusion_model import RiskFusionGRU

# ============================================================
# CONFIG
# ============================================================

BASE_DIR    = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
X_PATH      = DATASET_DIR / "noisy_sequences.npy"
y_PATH      = DATASET_DIR / "noisy_labels.npy"

SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATA LOADING + SPLIT
# ============================================================

X = np.load(X_PATH)   # (5000, 120, 4)
y = np.load(y_PATH)   # (5000,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

print(f"\n{'='*60}")
print(" PHASE 12: BASELINE COMPARISONS")
print(f"{'='*60}")
print(f"  Dataset : {X.shape}")
print(f"  Train   : {len(X_train)}  |  Test: {len(X_test)}")
print(f"  Device  : {DEVICE}\n")

results = {}

def record(name, y_true, y_score):
    roc  = roc_auc_score(y_true, y_score)
    pr   = average_precision_score(y_true, y_score)
    brier = brier_score_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7))
    results[name] = {"roc": roc, "pr": pr, "brier": brier}
    print(f"  ✅ [{name}]  ROC-AUC: {roc:.4f}  |  PR-AUC: {pr:.4f}  |  Brier: {brier:.4f}")
    return roc, pr, brier

# ============================================================
# BASELINE 1 — THRESHOLD RULE CLASSIFIER (Heuristic, No Learning)
# ============================================================
# Flags sessions based on per-frame threshold violations.
# Signal thresholds (centered on mid-point between class means):
#   UC1 < 0.717  (mean genuine=0.745, anomalous=0.690)
#   UC2 > 0.182  (mean genuine=0.153, anomalous=0.210)
#   UC3 < 0.801  (mean genuine=0.826, anomalous=0.776)
#   UC4 > 0.125  (mean genuine=0.084, anomalous=0.166)
# Score = fraction of frames where ANY threshold is violated.

print("─" * 60)
print(" B1: Threshold Rule Classifier (no learning)")

UC1_THR = (0.745 + 0.690) / 2   # 0.7175
UC2_THR = (0.153 + 0.210) / 2   # 0.1815
UC3_THR = (0.826 + 0.776) / 2   # 0.8010
UC4_THR = (0.084 + 0.166) / 2   # 0.1250

def threshold_score(X_data):
    """Fraction of frames tripping any threshold."""
    uc1 = X_data[:, :, 0]
    uc2 = X_data[:, :, 1]
    uc3 = X_data[:, :, 2]
    uc4 = X_data[:, :, 3]
    violations = (
        (uc1 < UC1_THR).astype(float) +
        (uc2 > UC2_THR).astype(float) +
        (uc3 < UC3_THR).astype(float) +
        (uc4 > UC4_THR).astype(float)
    )
    # any-threshold: at least 1 signal tripping
    flagged = (violations > 0).astype(float)
    return flagged.mean(axis=1)   # (N,)

b1_scores_test = threshold_score(X_test)
record("B1: Threshold Rule", y_test, b1_scores_test)

# ============================================================
# BASELINE 2 — MEAN SIGNAL + LOGISTIC REGRESSION
# ============================================================
# Collapses temporal dimension by averaging each signal.
# Input per session: (4,) — no trajectory, just magnitudes.

print("\n─" * 60)
print(" B2: Mean Signal + Logistic Regression")

X_train_mean = X_train.mean(axis=1)   # (N, 4)
X_test_mean  = X_test.mean(axis=1)

scaler_mean = StandardScaler()
X_train_mean_sc = scaler_mean.fit_transform(X_train_mean)
X_test_mean_sc  = scaler_mean.transform(X_test_mean)

lr_mean = LogisticRegression(max_iter=1000, random_state=SEED)
lr_mean.fit(X_train_mean_sc, y_train)
b2_scores = lr_mean.predict_proba(X_test_mean_sc)[:, 1]
record("B2: Mean+LogReg", y_test, b2_scores)

# ============================================================
# BASELINE 3 — NON-TEMPORAL MLP FUSION
# ============================================================
# Flatten (120, 4) → (480,) per session.
# 3-layer MLP. Sees all frames but no temporal ordering.

print("\n─" * 60)
print(" B3: Non-Temporal MLP Fusion (no recurrence)")

class FlatMLP(nn.Module):
    def __init__(self, input_dim=480, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class FlatDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x.reshape(len(x), -1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

torch.manual_seed(SEED)
mlp_model  = FlatMLP(input_dim=120*4).to(DEVICE)
mlp_crit   = nn.BCEWithLogitsLoss()
mlp_opt    = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
mlp_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(mlp_opt, T_max=60)
mlp_loader = DataLoader(FlatDataset(X_train, y_train), batch_size=64, shuffle=True)

for epoch in range(60):
    mlp_model.train()
    for xb, yb in mlp_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        mlp_opt.zero_grad()
        loss = mlp_crit(mlp_model(xb), yb)
        loss.backward()
        mlp_opt.step()
    mlp_sched.step()
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"    Epoch {epoch+1:03d}/60 | Loss: {loss.item():.4f}")

mlp_model.eval()
mlp_probs = []
mlp_test_loader = DataLoader(FlatDataset(X_test, y_test), batch_size=64)
with torch.no_grad():
    for xb, _ in mlp_test_loader:
        logit = mlp_model(xb.to(DEVICE))
        mlp_probs.extend(torch.sigmoid(logit).cpu().numpy())
record("B3: Non-Temporal MLP", y_test, np.array(mlp_probs))

# ============================================================
# BASELINE 4 — LAST-FRAME LOGISTIC REGRESSION
# ============================================================
# Use only the final frame's 4-signal vector.
# Maximally non-temporal.

print("\n─" * 60)
print(" B4: Last-Frame Logistic Regression")

X_train_last = X_train[:, -1, :]   # (N, 4) — last timestep
X_test_last  = X_test[:, -1, :]

scaler_last = StandardScaler()
X_train_last_sc = scaler_last.fit_transform(X_train_last)
X_test_last_sc  = scaler_last.transform(X_test_last)

lr_last = LogisticRegression(max_iter=1000, random_state=SEED)
lr_last.fit(X_train_last_sc, y_train)
b4_scores = lr_last.predict_proba(X_test_last_sc)[:, 1]
record("B4: Last-Frame LogReg", y_test, b4_scores)

# ============================================================
# BASELINE 5 — TEMPORAL GRU (Our System — Reference)
# ============================================================

print("\n─" * 60)
print(" B5: Temporal GRU (Our System — Reference)")

class RiskDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

torch.manual_seed(SEED)
gru_model  = RiskFusionGRU(input_dim=4, hidden_dim=32).to(DEVICE)
gru_crit   = nn.BCEWithLogitsLoss()
gru_opt    = torch.optim.Adam(gru_model.parameters(), lr=1e-3)
gru_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(gru_opt, T_max=60)
gru_loader = DataLoader(RiskDataset(X_train, y_train), batch_size=64, shuffle=True)

for epoch in range(60):
    gru_model.train()
    for xb, yb in gru_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        gru_opt.zero_grad()
        _, final_risk = gru_model(xb)
        loss = gru_crit(final_risk, yb)
        loss.backward()
        gru_opt.step()
    gru_sched.step()
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"    Epoch {epoch+1:03d}/60 | Loss: {loss.item():.4f}")

gru_model.eval()
gru_probs = []
gru_test_loader = DataLoader(RiskDataset(X_test, y_test), batch_size=64)
with torch.no_grad():
    for xb, _ in gru_test_loader:
        _, final_risk = gru_model(xb.to(DEVICE))
        gru_probs.extend(torch.sigmoid(final_risk).cpu().numpy())
record("B5: Temporal GRU (Ours)", y_test, np.array(gru_probs))

# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

ref_roc   = results["B5: Temporal GRU (Ours)"]["roc"]
ref_pr    = results["B5: Temporal GRU (Ours)"]["pr"]
ref_brier = results["B5: Temporal GRU (Ours)"]["brier"]

print(f"\n\n{'='*70}")
print(" PHASE 12: BASELINE COMPARISON RESULTS")
print(f"{'='*70}")
print(f"\n  {'Model':<30} {'ROC-AUC':>8} {'ΔROC':>8} {'PR-AUC':>8} {'ΔPR':>8} {'Brier':>8}")
print(f"  {'─'*68}")

for name, m in results.items():
    delta_roc = ref_roc - m["roc"]
    delta_pr  = ref_pr  - m["pr"]
    is_ours = "(Ours)" in name
    tag_roc  = "—" if is_ours else (f"-{delta_roc:.4f}" if delta_roc > 0 else f"+{abs(delta_roc):.4f}")
    tag_pr   = "—" if is_ours else (f"-{delta_pr:.4f}"  if delta_pr  > 0 else f"+{abs(delta_pr):.4f}")
    marker = " ◀" if is_ours else ""
    print(f"  {name:<30} {m['roc']:>8.4f} {tag_roc:>8} {m['pr']:>8.4f} {tag_pr:>8} {m['brier']:>8.4f}{marker}")

# Export
out_path = BASE_DIR / "baseline_comparison_results.txt"
with open(out_path, "w") as f:
    f.write("Phase 12 — Baseline Comparison Results\n")
    f.write("=" * 50 + "\n\n")
    for name, m in results.items():
        delta_roc = ref_roc - m["roc"]
        delta_pr  = ref_pr  - m["pr"]
        f.write(f"{name}\n")
        f.write(f"  ROC-AUC : {m['roc']:.4f}  (vs GRU: {delta_roc:+.4f})\n")
        f.write(f"  PR-AUC  : {m['pr']:.4f}  (vs GRU: {delta_pr:+.4f})\n")
        f.write(f"  Brier   : {m['brier']:.4f}\n\n")

print(f"\n\n✅ Results exported to {out_path}")
