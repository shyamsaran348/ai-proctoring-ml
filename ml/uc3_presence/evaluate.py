import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from train import PresenceModel


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.load("ml/uc3_presence/datasets/sequences.npy")
    y = np.load("ml/uc3_presence/datasets/labels.npy")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    model = PresenceModel()
    model.load_state_dict(
        torch.load("ml/uc3_presence/models/presence_model.pth")
    )
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds)

    print("\n=== UC3 Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()