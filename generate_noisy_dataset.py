"""
generate_noisy_dataset.py
Phase 11 — Realistic Noisy Signal Generation

Generates (5000, 120, 4) sequences with overlapping distributions,
temporal autocorrelation, and realistic perturbation events.

Signal layout per timestep: [UC1_sim, UC2_instability, UC3_presence, UC4_drift]
"""

import numpy as np
from sklearn.utils import shuffle
import os

# ============================================================
# CONFIG
# ============================================================

NUM_SAMPLES   = 5000
SEQ_LEN       = 120
OUTPUT_DIR    = "ml/uc5_risk_fusion/datasets"
OUT_X         = os.path.join(OUTPUT_DIR, "noisy_sequences.npy")
OUT_y         = os.path.join(OUTPUT_DIR, "noisy_labels.npy")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(seed=42)

# ============================================================
# CORE PRIMITIVES
# ============================================================

def ar1_signal(mean, std, length=SEQ_LEN, phi=0.85):
    """
    AR(1) Random Walk clamped to [0, 1].
    phi=0.85 gives strong autocorrelation (smooth, slow-moving signal).
    """
    noise = RNG.normal(0, std * np.sqrt(1 - phi**2), size=length)
    signal = np.zeros(length)
    signal[0] = RNG.normal(mean, std)
    for t in range(1, length):
        signal[t] = phi * signal[t-1] + (1 - phi) * mean + noise[t]
    return np.clip(signal, 0.0, 1.0)


def inject_noise_bursts(signal, p=0.05, magnitude=0.3):
    """Randomly spike p-fraction of frames by ±magnitude."""
    out = signal.copy()
    mask = RNG.random(len(out)) < p
    out[mask] += RNG.uniform(-magnitude, magnitude, size=mask.sum())
    return np.clip(out, 0.0, 1.0)


def inject_block_drop(signal, start, duration, drop_by=0.25):
    """Drop a contiguous block of frames (simulates lighting / occlusion)."""
    out = signal.copy()
    end = min(start + duration, len(out))
    out[start:end] = np.clip(out[start:end] - drop_by, 0.0, 1.0)
    return out


def ramp(start_val, end_val, length):
    """Linear ramp array."""
    return np.linspace(start_val, end_val, length)

# ============================================================
# SESSION BUILDERS — GENUINE vs 4 ANOMALY CLASSES
# ============================================================

def build_genuine():
    """
    Class 0: Normal student.
    All distributions biased toward safe values but with realistic noise.
    """
    uc1 = ar1_signal(mean=0.75, std=0.15)
    uc2 = ar1_signal(mean=0.15, std=0.10)
    uc3 = ar1_signal(mean=0.85, std=0.10)
    uc4 = ar1_signal(mean=0.08, std=0.06)

    # Authentic presence flickers: 1-3 short head-turns
    n_flickers = RNG.integers(1, 4)
    for _ in range(n_flickers):
        start = RNG.integers(0, SEQ_LEN - 10)
        uc3 = inject_block_drop(uc3, int(start), RNG.integers(2, 6), drop_by=0.35)

    # Noise bursts on all signals
    uc1 = inject_noise_bursts(uc1, p=0.04, magnitude=0.2)
    uc2 = inject_noise_bursts(uc2, p=0.04, magnitude=0.15)
    uc3 = inject_noise_bursts(uc3, p=0.03, magnitude=0.15)
    uc4 = inject_noise_bursts(uc4, p=0.04, magnitude=0.10)

    return np.stack([uc1, uc2, uc3, uc4], axis=1), 0


def build_abrupt_impersonation():
    """
    Class 1 — Hard impersonation switch at random frame ~[40, 80].
    UC1 drops sharply. UC2 spikes (catching the abrupt pattern). UC4 also rises.
    UC3 stays high (impostor is physically present and attentive).
    """
    switch = int(RNG.integers(40, 80))

    # Before switch: genuine
    uc1_pre = ar1_signal(mean=0.78, std=0.12, length=switch)
    uc2_pre = ar1_signal(mean=0.13, std=0.09, length=switch)
    uc3_pre = ar1_signal(mean=0.86, std=0.09, length=switch)
    uc4_pre = ar1_signal(mean=0.07, std=0.05, length=switch)

    # After switch: impersonator
    post = SEQ_LEN - switch
    uc1_post = ar1_signal(mean=0.60, std=0.15, length=post)  # Overlapping
    uc2_post = ar1_signal(mean=0.30, std=0.12, length=post)  # UC2 catches it
    uc3_post = ar1_signal(mean=0.82, std=0.10, length=post)  # Still present
    uc4_post = ar1_signal(mean=0.22, std=0.10, length=post)  # Drift detected

    uc1 = inject_noise_bursts(np.concatenate([uc1_pre, uc1_post]), p=0.05, magnitude=0.2)
    uc2 = inject_noise_bursts(np.concatenate([uc2_pre, uc2_post]), p=0.04, magnitude=0.12)
    uc3 = inject_noise_bursts(np.concatenate([uc3_pre, uc3_post]), p=0.03, magnitude=0.12)
    uc4 = inject_noise_bursts(np.concatenate([uc4_pre, uc4_post]), p=0.04, magnitude=0.10)

    return np.stack([uc1, uc2, uc3, uc4], axis=1), 1


def build_sophisticated_drift():
    """
    Class 1 — Hardest case. Slow gradual impersonation designed to fool UC2.
    UC1 decays slowly. UC2 stays low (UC2 is short-term; doesn't catch slow drift).
    UC4 is the only signal that catches it (rolling delta over 120 frames ramps up).
    This is the critical test validating UC4's unique discriminative contribution.
    """
    # UC1: starts genuine, slowly decays in the last 2/3 of the session
    uc1_stable = ar1_signal(mean=0.78, std=0.12, length=SEQ_LEN // 3)
    uc1_decay  = ar1_signal(mean=0.70, std=0.14, length=SEQ_LEN // 3)
    uc1_drift  = ar1_signal(mean=0.62, std=0.16, length=SEQ_LEN - 2*(SEQ_LEN // 3))
    uc1_raw    = np.concatenate([uc1_stable, uc1_decay, uc1_drift])
    uc1        = inject_noise_bursts(uc1_raw, p=0.05, magnitude=0.18)

    # UC2: stays LOW intentionally — the impersonator moves slowly so UC2 doesn't fire
    uc2 = ar1_signal(mean=0.17, std=0.10)
    uc2 = inject_noise_bursts(uc2, p=0.04, magnitude=0.12)

    # UC3: attentive throughout (makes it even harder to detect)
    uc3 = ar1_signal(mean=0.84, std=0.10)
    uc3 = inject_noise_bursts(uc3, p=0.03, magnitude=0.13)

    # UC4: slowly ramps up as cumulative drift accumulates — ONLY UC4 catches this
    uc4_base = ar1_signal(mean=0.08, std=0.05)
    drift_ramp = ramp(0.05, 0.35, SEQ_LEN)
    uc4 = np.clip(uc4_base + drift_ramp + RNG.normal(0, 0.04, SEQ_LEN), 0.0, 1.0)

    return np.stack([uc1, uc2, uc3, uc4], axis=1), 1


def build_presence_absence():
    """
    Class 1 — Student leaves camera view for a sustained period.
    UC3 drops hard. UC1 becomes noisy (no face). UC2 slightly rises.
    UC4 stays low (same person just not there).
    """
    out_start = int(RNG.integers(30, 70))
    out_dur   = int(RNG.integers(20, 50))

    uc1 = ar1_signal(mean=0.72, std=0.15)
    uc2 = ar1_signal(mean=0.18, std=0.10)
    uc3 = ar1_signal(mean=0.84, std=0.10)
    uc4 = ar1_signal(mean=0.08, std=0.05)

    # During absence: UC3 drops, UC1 noisy
    out_end = min(out_start + out_dur, SEQ_LEN)
    uc3 = inject_block_drop(uc3, out_start, out_dur, drop_by=0.65)
    uc1[out_start:out_end] = np.clip(
        uc1[out_start:out_end] + RNG.normal(0, 0.3, out_end - out_start), 0.0, 1.0
    )
    uc2[out_start:out_end] += 0.15

    uc1 = inject_noise_bursts(uc1, p=0.05, magnitude=0.2)
    uc2 = np.clip(inject_noise_bursts(uc2, p=0.04, magnitude=0.12), 0.0, 1.0)
    uc3 = inject_noise_bursts(uc3, p=0.04, magnitude=0.12)
    uc4 = inject_noise_bursts(uc4, p=0.04, magnitude=0.08)

    return np.stack([uc1, uc2, uc3, uc4], axis=1), 1


def build_flickering_substitution():
    """
    Class 1 — Multiple identities rapidly alternating.
    All signals oscillate in ambiguous mid-band. No single signal is definitive.
    The full 4-signal model must integrate evidence to classify.
    """
    uc1 = ar1_signal(mean=0.67, std=0.18)  # Overlapping with genuine
    uc2 = ar1_signal(mean=0.22, std=0.14)  # Slightly elevated
    uc3 = ar1_signal(mean=0.78, std=0.14)  # Somewhat present
    uc4 = ar1_signal(mean=0.15, std=0.10)  # Slight drift

    uc1 = inject_noise_bursts(uc1, p=0.10, magnitude=0.28)  # Many bursts
    uc2 = inject_noise_bursts(uc2, p=0.10, magnitude=0.20)
    uc3 = inject_noise_bursts(uc3, p=0.08, magnitude=0.22)
    uc4 = inject_noise_bursts(uc4, p=0.06, magnitude=0.15)

    return np.stack([uc1, uc2, uc3, uc4], axis=1), 1

# ============================================================
# MAIN GENERATION
# ============================================================

def main():
    print(f"[INFO] Generating {NUM_SAMPLES} noisy realistic sessions...")

    X, y = [], []

    # 50% genuine
    n_genuine = NUM_SAMPLES // 2
    for _ in range(n_genuine):
        seq, label = build_genuine()
        X.append(seq); y.append(label)

    # 50% anomalous — split across 4 types (12.5% each)
    n_per_type = (NUM_SAMPLES - n_genuine) // 4

    for _ in range(n_per_type):
        seq, label = build_abrupt_impersonation()
        X.append(seq); y.append(label)

    for _ in range(n_per_type):
        seq, label = build_sophisticated_drift()
        X.append(seq); y.append(label)

    for _ in range(n_per_type):
        seq, label = build_presence_absence()
        X.append(seq); y.append(label)

    # Fill remainder with flickering
    n_flicker = NUM_SAMPLES - n_genuine - 3 * n_per_type
    for _ in range(n_flicker):
        seq, label = build_flickering_substitution()
        X.append(seq); y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X, y = shuffle(X, y, random_state=42)

    np.save(OUT_X, X)
    np.save(OUT_y, y)

    print("\n✅ Noisy Dataset Generated")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Class 0 (Genuine)   : {int(np.sum(y == 0))}")
    print(f"  Class 1 (Anomalous) : {int(np.sum(y == 1))}")
    print(f"\n  Per-signal mean (Genuine) :")
    genuine_mask = (y == 0)
    print(f"    UC1 : {X[genuine_mask, :, 0].mean():.4f}")
    print(f"    UC2 : {X[genuine_mask, :, 1].mean():.4f}")
    print(f"    UC3 : {X[genuine_mask, :, 2].mean():.4f}")
    print(f"    UC4 : {X[genuine_mask, :, 3].mean():.4f}")
    print(f"\n  Per-signal mean (Anomalous) :")
    anom_mask = (y == 1)
    print(f"    UC1 : {X[anom_mask, :, 0].mean():.4f}")
    print(f"    UC2 : {X[anom_mask, :, 1].mean():.4f}")
    print(f"    UC3 : {X[anom_mask, :, 2].mean():.4f}")
    print(f"    UC4 : {X[anom_mask, :, 3].mean():.4f}")


if __name__ == "__main__":
    main()
