import numpy as np
import os

DATASET_DIR = "ml/uc5_risk_fusion/datasets"

UC1_PATH = os.path.join(DATASET_DIR, "uc1_scores.npy")
UC2_PATH = os.path.join(DATASET_DIR, "uc2_probs.npy")
UC3_PATH = os.path.join(DATASET_DIR, "uc3_presence.npy")
UC4_PATH = os.path.join(DATASET_DIR, "uc4_probs.npy")
LABELS_PATH = os.path.join(DATASET_DIR, "risk_labels.npy")

OUTPUT_X = os.path.join(DATASET_DIR, "fusion_sequences.npy")
OUTPUT_Y = os.path.join(DATASET_DIR, "fusion_labels.npy")


def main():
    print("🔄 Loading signals...")

    uc1 = np.load(UC1_PATH)
    uc2 = np.load(UC2_PATH)
    uc3 = np.load(UC3_PATH)
    uc4 = np.load(UC4_PATH)
    labels = np.load(LABELS_PATH)

    print("UC1 shape:", uc1.shape)
    print("UC2 shape:", uc2.shape)
    print("UC3 shape:", uc3.shape)
    print("UC4 shape:", uc4.shape)

    # Sanity check
    assert uc1.shape == uc2.shape == uc3.shape == uc4.shape, \
        "Signal shapes do not match!"

    # Stack into (N,120,4)
    fusion = np.stack([uc1, uc2, uc3, uc4], axis=2)

    print("✅ Fusion shape:", fusion.shape)
    print("Labels shape:", labels.shape)

    os.makedirs(DATASET_DIR, exist_ok=True)
    np.save(OUTPUT_X, fusion)
    np.save(OUTPUT_Y, labels)

    print("💾 Fusion dataset saved successfully.")


if __name__ == "__main__":
    main()