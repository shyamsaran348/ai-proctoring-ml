import numpy as np
import os

def ar1_process(mean, std, length, phi=0.85):
    noise = np.random.normal(0, std * np.sqrt(1 - phi**2), length)
    signal = np.zeros(length)
    signal[0] = np.random.normal(mean, std)
    for t in range(1, length):
        signal[t] = phi * signal[t-1] + (1 - phi) * mean + noise[t]
    return signal

def generate_hgdm_dataset(num_sessions=1000, seq_len=120):
    """
    Generates 7D features [h_y, h_p, h_r, g_y, g_p, dh_p, dg_p]
    Classes:
    - 0: Decoupled (Anomalous)
    - 1: Coupled (Genuine)
    """
    X = []
    y = []

    for _ in range(num_sessions):
        is_genuine = np.random.choice([True, False])
        
        # Head Poses (AR(1))
        h_yaw = ar1_process(0, 0.1, seq_len)
        h_pitch = ar1_process(0, 0.1, seq_len)
        h_roll = ar1_process(0, 0.05, seq_len)
        
        if is_genuine:
            # Gaze is correlated with head pose
            g_yaw = h_yaw + np.random.normal(0, 0.02, seq_len)
            g_pitch = h_pitch + np.random.normal(0, 0.02, seq_len)
            label = 1.0 # Normal behavior
        else:
            # Anomalous: decoupled
            anomaly_type = np.random.choice(['phone', 'side'])
            if anomaly_type == 'phone':
                # Head stable, gaze drops
                g_yaw = h_yaw + np.random.normal(0, 0.02, seq_len)
                g_pitch = h_pitch - 0.4 + np.random.normal(0, 0.05, seq_len) # Significant drop
            else:
                # Head forward, gaze lateral
                g_yaw = h_yaw + 0.5 * np.random.choice([-1, 1]) + np.random.normal(0, 0.05, seq_len)
                g_pitch = h_pitch + np.random.normal(0, 0.02, seq_len)
            label = 0.0 # Suspicious decoupling behavior

        # Compute velocities (deltas)
        dh_pitch = np.diff(h_pitch, prepend=h_pitch[0])
        dg_pitch = np.diff(g_pitch, prepend=g_pitch[0])
        
        # Stack into 7D session
        session = np.stack([h_yaw, h_pitch, h_roll, g_yaw, g_pitch, dh_pitch, dg_pitch], axis=1)
        X.append(session)
        # We labels the sequence (could be per-step, but for HGDM we'll do sequence-level or per-step)
        # The BiLSTM output is (B, T, 1), so we'll provide per-step labels
        y.append(np.full((seq_len, 1), label))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    os.makedirs('ml/data', exist_ok=True)
    np.save('ml/data/hgdm_x.npy', X)
    np.save('ml/data/hgdm_y.npy', y)
    print(f"Generated HGDM dataset: {X.shape}, {y.shape}")

if __name__ == "__main__":
    generate_hgdm_dataset()
