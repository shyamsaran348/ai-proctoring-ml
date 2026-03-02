"""
evaluate_noisy.py
Phase 11 — Rigorous Ablation Evaluation on Realistic Noisy Dataset

Trains 5 RiskFusionGRU models (1 baseline + 4 ablated).
Evaluates ROC-AUC, PR-AUC, and Brier Score.
Reports ΔAUC tables.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ml.uc5_risk_fusion.fusion_model import RiskFusionGRU

# ============================================================
# CONFIG
# ============================================================

BASE_DIR     = Path(__file__).resolve().parent
DATASET_DIR  = BASE_DIR / "datasets"
X_PATH       = DATASET_DIR / "noisy_sequences.npy"
y_PATH       = DATASET_DIR / "noisy_labels.npy"

EPOCHS       = 60
BATCH_SIZE   = 64
LR           = 1e-3
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATASET
# ============================================================

class RiskDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ============================================================
# TRAIN + EVALUATE
# ============================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, run_name, mask_idx=None):
    """
    Trains RiskFusionGRU(input_dim=4) on data.
    If mask_idx (0-3) given, that signal column is zeroed across all splits.
    Evaluates ROC-AUC, PR-AUC, and Brier Score on test set.
    """
    X_tr = np.copy(X_train)
    X_te = np.copy(X_test)

    if mask_idx is not None:
        X_tr[:, :, mask_idx] = 0.0
        X_te[:, :, mask_idx] = 0.0

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(RiskDataset(X_tr, y_train), batch_size=BATCH_SIZE, shuffle=True)

    model     = RiskFusionGRU(input_dim=4, hidden_dim=32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n{'─'*55}")
    print(f" Training: {run_name}")
    print(f"{'─'*55}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            _, final_risk = model(x_batch)
            loss = criterion(final_risk, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    # ---- Evaluation ----
    model.eval()
    test_loader = DataLoader(RiskDataset(X_te, y_test), batch_size=BATCH_SIZE, shuffle=False)
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            _, final_risk = model(x_batch)
            probs = torch.sigmoid(final_risk).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc  = average_precision_score(all_labels, all_probs)
    brier   = brier_score_loss(all_labels, all_probs)

    print(f"\n  ✅ ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}  |  Brier: {brier:.4f}")
    return roc_auc, pr_auc, brier

# ============================================================
# ABLATION CONFIGURATIONS
# ============================================================

ABLATION_CONFIGS = [
    # (name, mask_idx)
    ("Full 4-Signal Baseline",      None),
    ("Ablate UC1 — No Identity",    0),
    ("Ablate UC2 — No Instability", 1),
    ("Ablate UC3 — No Presence",    2),
    ("Ablate UC4 — No Drift",       3),
]

# ============================================================
# MAIN
# ============================================================

def main():
    print(f"\n{'='*55}")
    print(" PHASE 11: REALISTIC NOISY EVALUATION")
    print(f"{'='*55}")
    print(f"\n[INFO] Loading noisy dataset from {X_PATH}")

    X = np.load(X_PATH)
    y = np.load(y_PATH)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Genuine   : {int(np.sum(y == 0))}")
    print(f"  Anomalous : {int(np.sum(y == 1))}")
    print(f"  Device    : {DEVICE}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")

    results = {}
    for name, mask_idx in ABLATION_CONFIGS:
        roc, pr, brier = train_and_evaluate(X_train, y_train, X_test, y_test, name, mask_idx)
        results[name] = {"roc_auc": roc, "pr_auc": pr, "brier": brier}

    # ---- Summary Table ----
    baseline_roc = results["Full 4-Signal Baseline"]["roc_auc"]
    baseline_pr  = results["Full 4-Signal Baseline"]["pr_auc"]

    print(f"\n\n{'='*55}")
    print(" SIGNAL IMPORTANCE ABLATION RESULTS (NOISY DATA)")
    print(f"{'='*55}")
    print(f"\n  Baseline ROC-AUC : {baseline_roc:.4f}")
    print(f"  Baseline PR-AUC  : {baseline_pr:.4f}\n")

    header = f"  {'Model':<35} {'ROC-AUC':>8} {'ΔROC':>8} {'PR-AUC':>8} {'ΔPR':>8} {'Brier':>8}"
    print(header)
    print(f"  {'─'*75}")

    for name, m in results.items():
        delta_roc = baseline_roc - m["roc_auc"]
        delta_pr  = baseline_pr  - m["pr_auc"]
        tag = "" if name == "Full 4-Signal Baseline" else f"+{delta_roc:.4f}" if delta_roc > 0 else f"{delta_roc:.4f}"
        tag_pr = "" if name == "Full 4-Signal Baseline" else f"+{delta_pr:.4f}" if delta_pr > 0 else f"{delta_pr:.4f}"
        print(f"  {name:<35} {m['roc_auc']:>8.4f} {tag:>8} {m['pr_auc']:>8.4f} {tag_pr:>8} {m['brier']:>8.4f}")

    # ---- Export ----
    out_path = BASE_DIR / "noisy_ablation_results.txt"
    with open(out_path, "w") as f:
        f.write("Phase 11 — Noisy Signal Ablation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Baseline ROC-AUC : {baseline_roc:.4f}\n")
        f.write(f"Baseline PR-AUC  : {baseline_pr:.4f}\n\n")
        for name, m in results.items():
            delta_roc = baseline_roc - m["roc_auc"]
            delta_pr  = baseline_pr  - m["pr_auc"]
            f.write(f"{name}\n")
            f.write(f"  ROC-AUC : {m['roc_auc']:.4f}  (ΔROC = {delta_roc:+.4f})\n")
            f.write(f"  PR-AUC  : {m['pr_auc']:.4f}  (ΔPR  = {delta_pr:+.4f})\n")
            f.write(f"  Brier   : {m['brier']:.4f}\n\n")

    print(f"\n\n✅ Results exported to {out_path}")


if __name__ == "__main__":
    main()
