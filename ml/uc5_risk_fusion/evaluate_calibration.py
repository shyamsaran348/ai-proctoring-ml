"""
evaluate_calibration.py
Phase 13 — Calibration & Reliability

Measures whether predicted risk probabilities are truly calibrated:
i.e., "when the model says risk=0.8, does anomaly actually occur ~80% of the time?"

Computes for GRU, MLP, and Mean+LogReg:
  - Reliability Diagram data (10 equal-width bins)
  - Expected Calibration Error (ECE)
  - Brier Score Decomposition: reliability + resolution + uncertainty
  - Exports calibration_results.txt and reliability_diagram.csv
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from pathlib import Path
import csv
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
N_BINS      = 10

# ============================================================
# DATA
# ============================================================

X = np.load(X_PATH)
y = np.load(y_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

print(f"\n{'='*60}")
print(" PHASE 13: CALIBRATION & RELIABILITY ANALYSIS")
print(f"{'='*60}")
print(f"  Dataset: {X.shape}  |  Test: {len(X_test)}")

# ============================================================
# HELPER: ECE COMPUTATION
# ============================================================

def compute_ece(y_true, y_prob, n_bins=N_BINS):
    """
    Expected Calibration Error (ECE).
    Gaps between mean predicted probability and actual fraction of positives
    per bin, weighted by bin size.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)

    bin_data = []  # For reliability diagram output

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)

        count = mask.sum()
        if count == 0:
            bin_data.append({"bin_lo": lo, "bin_hi": hi, "count": 0,
                             "mean_pred": None, "frac_pos": None, "gap": None})
            continue

        mean_pred = y_prob[mask].mean()
        frac_pos  = y_true[mask].mean()
        gap       = abs(mean_pred - frac_pos)
        ece      += (count / n) * gap
        bin_data.append({
            "bin_lo": round(lo, 2),
            "bin_hi": round(hi, 2),
            "count":  int(count),
            "mean_pred": round(float(mean_pred), 4),
            "frac_pos":  round(float(frac_pos), 4),
            "gap":       round(float(gap), 4)
        })

    return ece, bin_data


def brier_decompose(y_true, y_prob):
    """
    Brier Score = Reliability − Resolution + Uncertainty
    Returns the three components plus total.
    """
    n       = len(y_true)
    o_bar   = y_true.mean()         # Overall climatology
    brier   = brier_score_loss(y_true, y_prob)

    # Reliability: mean squared gap between predicted prob and observed freq per bin
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    rel, res = 0.0, 0.0
    for i in range(N_BINS):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == N_BINS - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        f_k   = y_prob[mask].mean()
        o_k   = y_true[mask].mean()
        n_k   = mask.sum()
        rel  += (n_k / n) * (f_k - o_k) ** 2
        res  += (n_k / n) * (o_k - o_bar) ** 2

    unc = o_bar * (1 - o_bar)
    return {"total": brier, "reliability": rel, "resolution": res, "uncertainty": unc}


# ============================================================
# MODEL TRAINING HELPERS
# ============================================================

class RiskDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


class FlatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(120 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)


def train_gru(X_tr, y_tr, epochs=60):
    torch.manual_seed(SEED)
    model  = RiskFusionGRU(input_dim=4, hidden_dim=32).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit   = nn.BCEWithLogitsLoss()
    loader = DataLoader(RiskDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            _, fr = model(xb)
            crit(fr, yb).backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:03d}/{epochs}")
    return model


def infer_gru(model, X_data):
    model.eval()
    loader = DataLoader(RiskDataset(X_data, np.zeros(len(X_data))), batch_size=64)
    probs  = []
    with torch.no_grad():
        for xb, _ in loader:
            _, fr = model(xb.to(DEVICE))
            probs.extend(torch.sigmoid(fr).cpu().numpy())
    return np.array(probs)


def train_mlp(X_tr, y_tr, epochs=60):
    torch.manual_seed(SEED)
    model  = FlatMLP().to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit   = nn.BCEWithLogitsLoss()

    class FD(Dataset):
        def __init__(self, x, y):
            self.x = torch.tensor(x.reshape(len(x), -1), dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.x)
        def __getitem__(self, i): return self.x[i], self.y[i]

    loader = DataLoader(FD(X_tr, y_tr), batch_size=64, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:03d}/{epochs}")
    return model, FD


def infer_mlp(model, FlatDS, X_data):
    model.eval()
    loader = DataLoader(FlatDS(X_data, np.zeros(len(X_data))), batch_size=64)
    probs  = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    return np.array(probs)


# ============================================================
# RUN ALL MODELS
# ============================================================

model_preds = {}

# --- B5: Temporal GRU ---
print("\n─── Training Temporal GRU ───")
gru = train_gru(X_train, y_train)
model_preds["Temporal GRU"] = infer_gru(gru, X_test)

# --- B3: Non-Temporal MLP ---
print("\n─── Training Non-Temporal MLP ───")
mlp, FlatDS = train_mlp(X_train, y_train)
model_preds["Non-Temporal MLP"] = infer_mlp(mlp, FlatDS, X_test)

# --- B2: Mean + LogReg ---
print("\n─── Fitting Mean+LogReg ───")
scaler = StandardScaler()
X_tr_m = scaler.fit_transform(X_train.mean(axis=1))
X_te_m = scaler.transform(X_test.mean(axis=1))
lr = LogisticRegression(max_iter=1000, random_state=SEED)
lr.fit(X_tr_m, y_train)
model_preds["Mean+LogReg"] = lr.predict_proba(X_te_m)[:, 1]

# ============================================================
# CALIBRATION ANALYSIS
# ============================================================

all_bin_data = {}
summary = {}

print(f"\n\n{'='*60}")
print(" CALIBRATION RESULTS")
print(f"{'='*60}")

for name, probs in model_preds.items():
    ece, bins = compute_ece(y_test, probs)
    brier_d   = brier_decompose(y_test, probs)
    all_bin_data[name] = bins
    summary[name] = {"ece": ece, "brier_decomp": brier_d}

    print(f"\n  ── {name} ──")
    print(f"    ECE    : {ece:.4f}")
    print(f"    Brier  : {brier_d['total']:.4f}  "
          f"(Rel={brier_d['reliability']:.4f} | "
          f"Res={brier_d['resolution']:.4f} | "
          f"Unc={brier_d['uncertainty']:.4f})")
    print(f"    {'Bin Range':<14} {'Count':>6} {'Pred':>8} {'Actual':>8} {'Gap':>8}")
    print(f"    {'─'*48}")
    for b in bins:
        if b["count"] == 0:
            continue
        print(f"    [{b['bin_lo']:.1f}, {b['bin_hi']:.1f})   "
              f"{b['count']:>6}   {b['mean_pred']:>8.4f} {b['frac_pos']:>8.4f} {b['gap']:>8.4f}")

# ============================================================
# EXPORT
# ============================================================

# Calibration summary text
out_txt = BASE_DIR / "calibration_results.txt"
with open(out_txt, "w") as f:
    f.write("Phase 13 — Calibration & Reliability Results\n")
    f.write("=" * 50 + "\n\n")
    for name, s in summary.items():
        bd = s["brier_decomp"]
        f.write(f"{name}\n")
        f.write(f"  ECE         : {s['ece']:.4f}\n")
        f.write(f"  Brier Total : {bd['total']:.4f}\n")
        f.write(f"  Reliability : {bd['reliability']:.4f}\n")
        f.write(f"  Resolution  : {bd['resolution']:.4f}\n")
        f.write(f"  Uncertainty : {bd['uncertainty']:.4f}\n\n")

# Reliability diagram CSV
out_csv = BASE_DIR / "reliability_diagram.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "bin_lo", "bin_hi", "count", "mean_pred", "frac_pos", "gap"])
    for name, bins in all_bin_data.items():
        for b in bins:
            if b["count"] == 0:
                continue
            writer.writerow([
                name, b["bin_lo"], b["bin_hi"], b["count"],
                b["mean_pred"], b["frac_pos"], b["gap"]
            ])

# Summary comparison table
print(f"\n\n{'='*60}")
print(" SUMMARY: ECE & BRIER DECOMPOSITION")
print(f"{'='*60}")
print(f"\n  {'Model':<22} {'ECE':>8} {'Brier':>8} {'Reliab.':>10} {'Resol.':>10}")
print(f"  {'─'*60}")
for name, s in summary.items():
    bd = s["brier_decomp"]
    print(f"  {name:<22} {s['ece']:>8.4f} {bd['total']:>8.4f} {bd['reliability']:>10.4f} {bd['resolution']:>10.4f}")

print(f"\n✅ Exported: {out_txt.name}  |  {out_csv.name}")
