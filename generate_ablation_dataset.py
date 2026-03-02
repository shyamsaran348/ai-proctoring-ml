import numpy as np
import os
from sklearn.utils import shuffle

# ============================================================
# CONFIG
# ============================================================

NUM_SAMPLES = 5000
SEQ_LEN = 120
INPUT_DIM = 4  # UC1, UC2, UC3, UC4

OUTPUT_DIR = "ml/uc5_risk_fusion/datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_X = os.path.join(OUTPUT_DIR, "ablation_sequences.npy")
OUT_y = os.path.join(OUTPUT_DIR, "ablation_labels.npy")

# ============================================================
# SIGNAL GENERATORS
# ============================================================
# We model realistic continuous signals. 
# UC1 (Sim): High is good (0.8 - 1.0), Low is bad (0.0 - 0.5)
# UC2 (Instability): Low is good (0.0 - 0.2), High is bad (0.8 - 1.0)
# UC3 (Presence): High is good (0.7 - 1.0), Low is bad (0.0 - 0.4)
# UC4 (Drift): Low is good (0.0 - 0.2), High is bad (0.8 - 1.0)

def generate_base_signal(mean, std, length=SEQ_LEN):
    """Generate a smooth random walk around a mean."""
    walk = np.cumsum(np.random.normal(0, std*0.1, size=length))
    signal = mean + walk
    return np.clip(signal, 0.0, 1.0)

def build_genuine_session():
    """Class 0: Genuine student taking an exam normally."""
    uc1 = generate_base_signal(0.9, 0.05)  # High Sim
    uc2 = generate_base_signal(0.1, 0.05)  # Low Instability
    uc3 = generate_base_signal(0.9, 0.1)   # High Presence
    uc4 = generate_base_signal(0.05, 0.02) # Low Drift
    return np.stack([uc1, uc2, uc3, uc4], axis=1), 0

def build_impersonation_session():
    """Class 1: Someone else sits down (Abrupt Switch)."""
    # Starts genuine, switches halfway
    uc1 = np.concatenate([generate_base_signal(0.9, 0.05, 60), generate_base_signal(0.3, 0.1, 60)])
    uc2 = np.concatenate([generate_base_signal(0.1, 0.05, 60), generate_base_signal(0.9, 0.05, 60)])
    uc3 = generate_base_signal(0.9, 0.1)  # STILL PRESENT
    uc4 = np.concatenate([generate_base_signal(0.05, 0.02, 60), generate_base_signal(0.9, 0.05, 60)])
    return np.stack([uc1, uc2, uc3, uc4], axis=1), 1

def build_absence_session():
    """Class 1: Student leaves the frame."""
    uc1 = np.concatenate([generate_base_signal(0.9, 0.05, 60), generate_base_signal(0.0, 0.0, 60)])
    uc2 = np.concatenate([generate_base_signal(0.1, 0.05, 60), generate_base_signal(0.0, 0.0, 60)])
    uc3 = np.concatenate([generate_base_signal(0.9, 0.1, 60), generate_base_signal(0.1, 0.1, 60)])
    uc4 = np.concatenate([generate_base_signal(0.05, 0.02, 60), generate_base_signal(0.0, 0.0, 60)])
    return np.stack([uc1, uc2, uc3, uc4], axis=1), 1

def build_sophisticated_drift_session():
    """Class 1: Slow, adversarial impersonation over time defeating UC2."""
    uc1 = np.linspace(0.9, 0.6, SEQ_LEN) + np.random.normal(0, 0.05, SEQ_LEN) # Trickles down but not abruptly
    uc1 = np.clip(uc1, 0.0, 1.0)
    uc2 = generate_base_signal(0.2, 0.05)   # UC2 is fooled (stays low)
    uc3 = generate_base_signal(0.9, 0.1)    # Presence is high
    uc4 = np.linspace(0.1, 0.85, SEQ_LEN) + np.random.normal(0, 0.05, SEQ_LEN) # UC4 catches the drift
    uc4 = np.clip(uc4, 0.0, 1.0)
    return np.stack([uc1, uc2, uc3, uc4], axis=1), 1

def build_flicker_session():
    """Class 1: Multiple people popping in and out."""
    uc1 = generate_base_signal(0.5, 0.3)
    uc2 = generate_base_signal(0.85, 0.1)
    uc3 = generate_base_signal(0.8, 0.1)
    uc4 = generate_base_signal(0.6, 0.2)
    return np.stack([uc1, uc2, uc3, uc4], axis=1), 1

# ============================================================
# DATASET GENERATION
# ============================================================

def main():
    print(f"Generating {NUM_SAMPLES} procedural sequences...")
    X, y = [], []
    
    # Distribution: 50% Genuine, 50% distributed anomalous
    half = NUM_SAMPLES // 2
    quarter = half // 4
    
    for _ in range(half):
        seq, label = build_genuine_session()
        X.append(seq)
        y.append(label)
        
    for _ in range(quarter):
        seq, label = build_impersonation_session()
        X.append(seq)
        y.append(label)
        
    for _ in range(quarter):
        seq, label = build_absence_session()
        X.append(seq)
        y.append(label)
        
    for _ in range(quarter):
        seq, label = build_sophisticated_drift_session()
        X.append(seq)
        y.append(label)
        
    # the rest to flicker
    remainder = NUM_SAMPLES - (half + quarter*3)
    for _ in range(remainder):
        seq, label = build_flicker_session()
        X.append(seq)
        y.append(label)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    X, y = shuffle(X, y, random_state=42)
    
    np.save(OUT_X, X)
    np.save(OUT_y, y)
    
    print("\n✅ Ablation Dataset Generated")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class 0 (Genuine): {np.sum(y == 0)}")
    print(f"Class 1 (Anomalous): {np.sum(y == 1)}")

if __name__ == "__main__":
    main()
