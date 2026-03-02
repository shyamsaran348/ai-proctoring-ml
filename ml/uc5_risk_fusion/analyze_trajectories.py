"""
analyze_trajectories.py
Phase 14 — Risk Trajectory Analysis

Trains the GRU on the noisy Phase 11 dataset, then extracts session-level
risk trajectories (B, T) for 5 representative session archetypes:
  1. Genuine
  2. Abrupt Impersonation
  3. Sophisticated Drift  (the hardest case)
  4. Presence Absence
  5. Flickering Substitution

For each session type, computes:
  - Frame-by-frame risk (from GRU's risk_traj output)
  - Frame-by-frame UC1/UC2/UC3/UC4 signal values
  - Trajectory statistics: mean gradient, smoothness, peak-risk frame

Exports:
  - trajectory_data.csv (frame-level risk + signals per session type)
  - trajectory_stats.txt (gradient, smoothness, peak)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import csv
import os
import sys

# Add root
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
# DATA
# ============================================================

X = np.load(X_PATH)
y = np.load(y_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

print(f"\n{'='*60}")
print(" PHASE 14: RISK TRAJECTORY ANALYSIS")
print(f"{'='*60}")

# ============================================================
# GENERATE ARCHETYPE SESSIONS DIRECTLY
# ============================================================
# We generate fresh archetype sessions using the same
# procedural generators from Phase 11 for clean representation.

import numpy as _np
_RNG = _np.random.default_rng(seed=7)

def ar1(_mean, _std, length=120, phi=0.85):
    noise = _RNG.normal(0, _std * np.sqrt(1 - phi**2), size=length)
    sig = np.zeros(length)
    sig[0] = _RNG.normal(_mean, _std)
    for t in range(1, length):
        sig[t] = phi * sig[t-1] + (1 - phi) * _mean + noise[t]
    return np.clip(sig, 0.0, 1.0)

def inj_drop(sig, start, dur, drop=0.3):
    out = sig.copy()
    out[start:start+dur] = np.clip(out[start:start+dur] - drop, 0, 1)
    return out

# 1. Genuine
def make_genuine():
    uc1 = ar1(0.78, 0.08)
    uc2 = ar1(0.12, 0.07)
    uc3 = ar1(0.88, 0.07)
    uc3 = inj_drop(uc3, 45, 4, 0.35)   # Brief head turn
    uc4 = ar1(0.07, 0.04)
    return np.stack([uc1, uc2, uc3, uc4], axis=1)

# 2. Abrupt Impersonation (switch at frame 60)
def make_abrupt():
    uc1 = np.concatenate([ar1(0.80, 0.08, 60), ar1(0.58, 0.12, 60)])
    uc2 = np.concatenate([ar1(0.11, 0.06, 60), ar1(0.32, 0.10, 60)])
    uc3 = ar1(0.86, 0.08)
    uc4 = np.concatenate([ar1(0.07, 0.04, 60), ar1(0.25, 0.08, 60)])
    return np.stack([uc1, uc2, uc3, uc4], axis=1)

# 3. Sophisticated Drift (designed to fool UC2; only UC4 catches it)
def make_drift():
    uc1 = np.concatenate([
        ar1(0.80, 0.08, 40),
        ar1(0.71, 0.10, 40),
        ar1(0.62, 0.13, 40)
    ])
    uc2 = ar1(0.14, 0.08)          # Stays low — impostor moves slowly
    uc3 = ar1(0.87, 0.08)          # Fully attentive
    uc4_base = ar1(0.07, 0.04)
    uc4 = np.clip(uc4_base + np.linspace(0.03, 0.38, 120), 0, 1)
    return np.stack([uc1, uc2, uc3, uc4], axis=1)

# 4. Presence Absence (student leaves frame 40–80)
def make_absence():
    uc1 = ar1(0.78, 0.08)
    uc2 = ar1(0.13, 0.07)
    uc3 = ar1(0.88, 0.08)
    uc3 = inj_drop(uc3, 40, 40, 0.70)
    uc1[40:80] = np.clip(uc1[40:80] + _RNG.normal(0, 0.25, 40), 0, 1)
    uc4 = ar1(0.07, 0.04)
    return np.stack([uc1, uc2, uc3, uc4], axis=1)

# 5. Flickering (all signals ambiguous throughout)
def make_flicker():
    uc1 = ar1(0.67, 0.18)
    uc2 = ar1(0.21, 0.14)
    uc3 = ar1(0.76, 0.15)
    uc4 = ar1(0.15, 0.10)
    # Add bursts
    for arr in [uc1, uc2, uc3, uc4]:
        mask = _RNG.random(120) < 0.10
        arr[mask] += _RNG.uniform(-0.3, 0.3, mask.sum())
    return np.stack(
        [np.clip(uc1,0,1), np.clip(uc2,0,1), np.clip(uc3,0,1), np.clip(uc4,0,1)],
        axis=1
    )

ARCHETYPES = {
    "Genuine":              make_genuine(),
    "Abrupt Impersonation": make_abrupt(),
    "Sophisticated Drift":  make_drift(),
    "Presence Absence":     make_absence(),
    "Flickering":           make_flicker(),
}

# ============================================================
# TRAIN GRU
# ============================================================

class RiskDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

print("\n─── Training GRU ───")
torch.manual_seed(SEED)
model   = RiskFusionGRU(input_dim=4, hidden_dim=32).to(DEVICE)
opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
crit    = nn.BCEWithLogitsLoss()
loader  = DataLoader(RiskDataset(X_train, y_train), batch_size=64, shuffle=True)

for epoch in range(60):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        _, fr = model(xb)
        loss = crit(fr, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    sched.step()
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:03d}/60 | Loss: {total_loss/len(loader):.4f}")

model.eval()

# ============================================================
# EXTRACT TRAJECTORIES
# ============================================================

print("\n─── Extracting Risk Trajectories ───\n")

trajectory_rows   = []
stats_per_session = {}

SIGNAL_NAMES = ["UC1_sim", "UC2_instab", "UC3_presence", "UC4_drift"]

for session_name, session_data in ARCHETYPES.items():
    seq_tensor = torch.tensor(
        session_data[np.newaxis, :, :],   # (1, 120, 4)
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        risk_traj, final_risk = model(seq_tensor)

    risk_curve  = torch.sigmoid(risk_traj[0]).cpu().numpy()  # (120,)
    final_score = torch.sigmoid(final_risk[0]).item()

    # Trajectory statistics
    gradient    = float(np.diff(risk_curve).mean())         # avg frame-to-frame change
    smoothness  = float(1.0 - np.std(np.diff(risk_curve)))  # 1 = perfectly smooth
    peak_frame  = int(np.argmax(risk_curve))

    stats_per_session[session_name] = {
        "final_risk":  round(final_score, 4),
        "gradient":    round(gradient, 6),
        "smoothness":  round(smoothness, 4),
        "peak_frame":  peak_frame,
    }

    print(f"  {session_name}")
    print(f"    Final Risk  : {final_score:.4f}")
    print(f"    Avg Gradient: {gradient:+.6f} (positive = rising risk)")
    print(f"    Smoothness  : {smoothness:.4f}  (1.0 ideal)")
    print(f"    Peak Frame  : {peak_frame}")
    print()

    # Build CSV rows
    for t in range(120):
        trajectory_rows.append({
            "session_type": session_name,
            "frame":        t,
            "risk":         round(float(risk_curve[t]), 4),
            "UC1_sim":      round(float(session_data[t, 0]), 4),
            "UC2_instab":   round(float(session_data[t, 1]), 4),
            "UC3_presence": round(float(session_data[t, 2]), 4),
            "UC4_drift":    round(float(session_data[t, 3]), 4),
        })

# ============================================================
# EXPORT
# ============================================================

csv_path = BASE_DIR / "trajectory_data.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=trajectory_rows[0].keys())
    writer.writeheader()
    writer.writerows(trajectory_rows)

txt_path = BASE_DIR / "trajectory_stats.txt"
with open(txt_path, "w") as f:
    f.write("Phase 14 — Risk Trajectory Analysis\n")
    f.write("=" * 50 + "\n\n")
    for name, s in stats_per_session.items():
        f.write(f"{name}\n")
        f.write(f"  Final Risk  : {s['final_risk']}\n")
        f.write(f"  Avg Gradient: {s['gradient']:+.6f}\n")
        f.write(f"  Smoothness  : {s['smoothness']}\n")
        f.write(f"  Peak Frame  : {s['peak_frame']}\n\n")

# Summary table
print(f"\n{'='*65}")
print(" TRAJECTORY STATISTICS SUMMARY")
print(f"{'='*65}")
print(f"\n  {'Session Type':<26} {'Final Risk':>10} {'Avg Gradient':>14} {'Smooth':>8} {'Peak':>6}")
print(f"  {'─'*62}")
for name, s in stats_per_session.items():
    print(f"  {name:<26} {s['final_risk']:>10.4f} {s['gradient']:>+14.6f} {s['smoothness']:>8.4f} {s['peak_frame']:>6}")

print(f"\n✅ Exported: {csv_path.name}  |  {txt_path.name}")
