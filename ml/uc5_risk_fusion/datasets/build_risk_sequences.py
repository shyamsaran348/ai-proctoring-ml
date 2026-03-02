import numpy as np
from pathlib import Path

# =========================
# CONFIG (LOCKED DEFAULTS)
# =========================

SEQUENCE_LENGTH = 120
STRIDE = 10
UC2_WINDOW = 60
UC2_STRIDE = 1

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

UC1_PATH = DATA_DIR / "uc1_scores.npy"
UC2_PATH = DATA_DIR / "uc2_probs.npy"
UC3_PATH = DATA_DIR / "uc3_presence.npy"
UC4_PATH = DATA_DIR / "uc4_probs.npy"

OUT_SEQ_PATH = DATA_DIR / "risk_sequences.npy"
OUT_LABEL_PATH = DATA_DIR / "risk_labels.npy"

# =========================
# LOAD INPUT SIGNALS
# =========================

uc1_scores = np.load(UC1_PATH)          # (T,)
uc2_probs = np.load(UC2_PATH)           # (N,)
uc3_presence = np.load(UC3_PATH)        # (T,)
uc4_probs = np.load(UC4_PATH)           # (T,)

T = len(uc1_scores)

print(f"[INFO] UC1 frames: {T}")
print(f"[INFO] UC2 windows: {len(uc2_probs)}")
print(f"[INFO] UC3 frames: {len(uc3_presence)}")
print(f"[INFO] UC4 frames: {len(uc4_probs)}")

assert len(uc3_presence) == T, "UC3 length must match UC1 length"

# =========================
# ALIGN UC2 → FRAME TIMELINE
# =========================

uc2_aligned = np.zeros(T, dtype=np.float32)

for k, prob in enumerate(uc2_probs):
    end = k * UC2_STRIDE
    start = max(0, end - (UC2_WINDOW - 1))

    uc2_aligned[start:end + 1] = np.maximum(
        uc2_aligned[start:end + 1],
        prob
    )

assert uc2_aligned.shape == (T,)

# =========================
# BUILD FEATURE MATRIX (4 SIGNALS)
# =========================

X = np.stack(
    [uc1_scores, uc2_aligned, uc3_presence, uc4_probs],
    axis=1
)
# shape: (T, 4)

# =========================
# BUILD RISK SEQUENCES
# =========================

risk_sequences = []
risk_labels = []

SESSION_LABEL = 1  # <-- CHANGE FOR GENUINE SESSIONS

for start in range(0, T - SEQUENCE_LENGTH + 1, STRIDE):
    seq = X[start:start + SEQUENCE_LENGTH]
    risk_sequences.append(seq)
    risk_labels.append(SESSION_LABEL)

risk_sequences = np.array(risk_sequences, dtype=np.float32)
risk_labels = np.array(risk_labels, dtype=np.int64)

# =========================
# SAVE OUTPUTS
# =========================

np.save(OUT_SEQ_PATH, risk_sequences)
np.save(OUT_LABEL_PATH, risk_labels)

print("[DONE] UC5 risk dataset built (4-signal)")
print(f"  Sequences shape: {risk_sequences.shape}")
print(f"  Labels shape:    {risk_labels.shape}")