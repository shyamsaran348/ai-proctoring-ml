import numpy as np
import os
from sklearn.utils import shuffle

# CONFIG
NUM_SAMPLES = 5000
SEQ_LEN = 120
OUTPUT_DIR = "ml/uc5_risk_fusion/datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_X = os.path.join(OUTPUT_DIR, "gam_features.npy")
OUT_y = os.path.join(OUTPUT_DIR, "gam_labels.npy")

RNG = np.random.default_rng(seed=42)

def ar1_signal(mean, std, length=SEQ_LEN, phi=0.85, clip=None):
    noise = RNG.normal(0, std * np.sqrt(1 - phi**2), size=length)
    signal = np.zeros(length)
    signal[0] = RNG.normal(mean, std)
    for t in range(1, length):
        signal[t] = phi * signal[t-1] + (1 - phi) * mean + noise[t]
    if clip:
        return np.clip(signal, clip[0], clip[1])
    return signal

def build_genuine_gaze():
    """Stable gaze near center, normal blinking, low velocity."""
    yaw = ar1_signal(0, 2, phi=0.9)   # degrees
    pitch = ar1_signal(0, 2, phi=0.9) # degrees
    px = ar1_signal(0, 0.05, phi=0.8) # pupil offset
    py = ar1_signal(0, 0.05, phi=0.8)
    blink = ar1_signal(0.1, 0.05, phi=0.7, clip=(0, 0.5))
    vel = ar1_signal(0.5, 0.2, phi=0.6, clip=(0, 10))
    
    return np.stack([yaw, pitch, px, py, blink, vel], axis=1), 1 # 1 = Attentive

def build_scanning_gaze():
    """Oscillating yaw, higher velocity."""
    t = np.linspace(0, 4 * np.pi, SEQ_LEN)
    yaw = 20 * np.sin(t) + RNG.normal(0, 2, SEQ_LEN)
    pitch = ar1_signal(0, 5, phi=0.85)
    px = 0.5 * np.sin(t) + RNG.normal(0, 0.1, SEQ_LEN)
    py = ar1_signal(0, 0.1, phi=0.8)
    blink = ar1_signal(0.1, 0.05, phi=0.7, clip=(0, 0.5))
    vel = ar1_signal(4.0, 1.5, phi=0.7, clip=(0, 10))
    
    return np.stack([yaw, pitch, px, py, blink, vel], axis=1), 0 # 0 = Suspicious

def build_downward_gaze():
    """Phone usage - negative pitch (downwards)."""
    yaw = ar1_signal(0, 5, phi=0.85)
    pitch = ar1_signal(-25, 5, phi=0.9)
    px = ar1_signal(0, 0.2, phi=0.8)
    py = ar1_signal(-0.6, 0.1, phi=0.85)
    blink = ar1_signal(0.05, 0.03, phi=0.8, clip=(0, 0.5)) # Reduced blinking common during phone focus
    vel = ar1_signal(1.0, 0.5, phi=0.7, clip=(0, 10))
    
    return np.stack([yaw, pitch, px, py, blink, vel], axis=1), 0

def build_offscreen_gaze():
    """Looking away - sustained large offset."""
    yaw = ar1_signal(35, 5, phi=0.95) # Looking far right
    pitch = ar1_signal(10, 5, phi=0.9)
    px = ar1_signal(0.8, 0.1, phi=0.9)
    py = ar1_signal(0.2, 0.1, phi=0.9)
    blink = ar1_signal(0.15, 0.05, phi=0.7, clip=(0, 0.5))
    vel = ar1_signal(0.8, 0.3, phi=0.8, clip=(0, 10))
    
    return np.stack([yaw, pitch, px, py, blink, vel], axis=1), 0

def main():
    print(f"[GAM] Generating {NUM_SAMPLES} gaze feature sequences...")
    X, y = [], []
    
    n_per = NUM_SAMPLES // 4
    
    for _ in range(n_per):
        X.append(build_genuine_gaze()[0]); y.append(1)
    for _ in range(n_per):
        X.append(build_scanning_gaze()[0]); y.append(0)
    for _ in range(n_per):
        X.append(build_downward_gaze()[0]); y.append(0)
    for _ in range(NUM_SAMPLES - 3*n_per):
        X.append(build_offscreen_gaze()[0]); y.append(0)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    X, y = shuffle(X, y, random_state=42)
    
    np.save(OUT_X, X)
    np.save(OUT_y, y)
    
    print(f"✅ GAM Dataset Saved to {OUTPUT_DIR}")
    print(f"   X: {X.shape}, y: {y.shape}")

if __name__ == "__main__":
    main()
