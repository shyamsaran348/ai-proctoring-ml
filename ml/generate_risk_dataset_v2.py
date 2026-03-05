import numpy as np
import os
from sklearn.utils import shuffle

# CONFIG
NUM_SAMPLES = 5000
SEQ_LEN = 120
OUTPUT_DIR = "ml/uc5_risk_fusion/datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_X = os.path.join(OUTPUT_DIR, "v2_sequences.npy")
OUT_y = os.path.join(OUTPUT_DIR, "v2_labels.npy")

RNG = np.random.default_rng(seed=43)

def ar1_signal(mean, std, length=SEQ_LEN, phi=0.85):
    noise = RNG.normal(0, std * np.sqrt(1 - phi**2), size=length)
    signal = np.zeros(length)
    signal[0] = RNG.normal(mean, std)
    for t in range(1, length):
        signal[t] = phi * signal[t-1] + (1 - phi) * mean + noise[t]
    return np.clip(signal, 0.0, 1.0)

def build_genuine():
    uc1 = ar1_signal(0.85, 0.1)
    uc2 = ar1_signal(0.1, 0.05)
    uc3 = ar1_signal(0.9, 0.1)
    uc4 = ar1_signal(0.05, 0.03)
    gam = ar1_signal(0.9, 0.05) # Attentive
    return np.stack([uc1, uc2, uc3, uc4, gam], axis=1), 0

def build_impersonation():
    # Abrupt switch at t=60
    uc1 = np.concatenate([ar1_signal(0.85, 0.1, 60), ar1_signal(0.4, 0.15, 60)])
    uc2 = np.concatenate([ar1_signal(0.1, 0.05, 60), ar1_signal(0.8, 0.1, 60)])
    uc3 = ar1_signal(0.9, 0.1)
    uc4 = np.concatenate([ar1_signal(0.05, 0.03, 60), ar1_signal(0.7, 0.15, 60)])
    gam = ar1_signal(0.85, 0.1) # Attentive impostor
    return np.stack([uc1, uc2, uc3, uc4, gam], axis=1), 1

def build_absence():
    # Leaves at t=40, returns at t=100
    uc1 = ar1_signal(0.85, 0.1)
    uc1[40:100] = np.clip(uc1[40:100] - 0.6, 0.0, 1.0)
    uc2 = ar1_signal(0.1, 0.05)
    uc2[40:100] += 0.3
    uc3 = ar1_signal(0.9, 0.1)
    uc3[40:100] = ar1_signal(0.1, 0.1, 60)
    uc4 = ar1_signal(0.05, 0.03)
    gam = ar1_signal(0.9, 0.1)
    gam[40:100] = 0.1 # No gaze
    return np.stack([uc1, uc2, uc3, uc4, np.clip(gam, 0, 1)], axis=1), 1

def build_gaze_anomaly():
    """Gaze only anomaly - phone or scanning."""
    uc1 = ar1_signal(0.85, 0.1)
    uc2 = ar1_signal(0.1, 0.05)
    uc3 = ar1_signal(0.9, 0.1)
    uc4 = ar1_signal(0.05, 0.03)
    # Gaze drops after t=30
    gam = np.concatenate([ar1_signal(0.9, 0.05, 30), ar1_signal(0.2, 0.15, 90)])
    return np.stack([uc1, uc2, uc3, uc4, gam], axis=1), 1

def build_drift():
    uc1 = np.linspace(0.85, 0.6, SEQ_LEN) + RNG.normal(0, 0.05, SEQ_LEN)
    uc2 = ar1_signal(0.15, 0.05)
    uc3 = ar1_signal(0.9, 0.1)
    uc4 = np.linspace(0.05, 0.8, SEQ_LEN) + RNG.normal(0, 0.05, SEQ_LEN)
    gam = ar1_signal(0.85, 0.1)
    return np.stack([np.clip(uc1, 0, 1), uc2, uc3, np.clip(uc4, 0, 1), gam], axis=1), 1

def main():
    print(f"[RFE v2] Generating {NUM_SAMPLES} 5-signal sessions...")
    X, y = [], []
    n_per = NUM_SAMPLES // 5
    
    for _ in range(n_per): X.append(build_genuine()[0]); y.append(0)
    for _ in range(n_per): X.append(build_impersonation()[0]); y.append(1)
    for _ in range(n_per): X.append(build_absence()[0]); y.append(1)
    for _ in range(n_per): X.append(build_gaze_anomaly()[0]); y.append(1)
    for _ in range(NUM_SAMPLES - 4*n_per): X.append(build_drift()[0]); y.append(1)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X, y = shuffle(X, y, random_state=42)
    
    np.save(OUT_X, X)
    np.save(OUT_y, y)
    print(f"✅ V2 Dataset Saved: {X.shape}")

if __name__ == "__main__":
    main()
