import numpy as np
import torch
import os

from ml.uc4_drift.train_uc4 import UC4DriftModel

DEVICE = torch.device("cpu")

SEQUENCE_PATH = "ml/uc4_drift/datasets/uc4_drift_sequences.npy"
MODEL_PATH = "ml/uc4_drift/models/uc4_drift_model.pth"
OUTPUT_PATH = "ml/uc5_risk_fusion/datasets/uc4_probs.npy"

def main():
    print("🔄 Loading UC4 sequences...")
    sequences = np.load(SEQUENCE_PATH)
    print("Sequences shape:", sequences.shape)

    print("🔄 Loading UC4 model...")
    model = UC4DriftModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_probs = []

    print("🚀 Generating probabilities...")
    with torch.no_grad():
        for seq in sequences:
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            logit = model(seq_tensor)
            prob = torch.sigmoid(logit).item()

            # Expand session-level probability across 120 frames
            expanded = np.full(120, prob)
            all_probs.append(expanded)

    all_probs = np.array(all_probs)
    print("Final UC4 signal shape:", all_probs.shape)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, all_probs)

    print("✅ UC4 probabilities saved successfully.")

if __name__ == "__main__":
    main()