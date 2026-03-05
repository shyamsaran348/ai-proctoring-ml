import numpy as np
import os

def ar1_process(mean, std, length, phi=0.85):
    noise = np.random.normal(0, std * np.sqrt(1 - phi**2), length)
    signal = np.zeros(length)
    signal[0] = np.random.normal(mean, std)
    for t in range(1, length):
        signal[t] = phi * signal[t-1] + (1 - phi) * mean + noise[t]
    return signal

def generate_risk_dataset_v3(num_sessions=5000, seq_len=120):
    """
    Generates synthetic 6-signal dataset for Phase 18 RFE training:
    [IdentitySim, Instability, Presence, Drift, Gaze, HGDM]
    """
    X = []
    y = []

    for _ in range(num_sessions):
        is_anomalous = np.random.choice([True, False])
        
        # S_t: Identity Similarity
        s_t = ar1_process(0.75, 0.1, seq_len) if not is_anomalous else ar1_process(0.5, 0.15, seq_len)
        
        # I_t: Instability
        i_t = ar1_process(0.1, 0.05, seq_len) if not is_anomalous else ar1_process(0.4, 0.2, seq_len)
        
        # P_t: Presence
        p_t = ar1_process(0.9, 0.05, seq_len) if not is_anomalous else ar1_process(0.3, 0.3, seq_len)
        
        # D_t: Drift
        d_t = ar1_process(0.1, 0.05, seq_len) if not is_anomalous else ar1_process(0.6, 0.2, seq_len)
        
        # G_t: Gaze
        g_t = ar1_process(0.9, 0.05, seq_len) if not is_anomalous else ar1_process(0.2, 0.3, seq_len)
        
        # H_t: HGDM (Head-Gaze Consistency)
        # In a real anomaly, H_t might stay low even if G_t is okay, or vice versa
        h_t = ar1_process(0.95, 0.02, seq_len) if not is_anomalous else ar1_process(0.3, 0.4, seq_len)

        # Scale and clip
        signals = [s_t, i_t, p_t, d_t, g_t, h_t]
        session = np.stack([np.clip(s, 0, 1) for s in signals], axis=1)
        
        X.append(session)
        y.append(1.0 if is_anomalous else 0.0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    os.makedirs('ml/data', exist_ok=True)
    np.save('ml/data/risk_sequences_v3.npy', X)
    np.save('ml/data/risk_labels_v3.npy', y)
    print(f"Generated RFE v3 dataset: {X.shape}, {y.shape}")

if __name__ == "__main__":
    generate_risk_dataset_v3()
