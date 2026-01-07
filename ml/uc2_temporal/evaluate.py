import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from models.temporal_lstm import UC2TemporalLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"

X = np.load("datasets/test_sequences.npy")
y = np.load("datasets/sequence_labels.npy")

X = torch.tensor(X, dtype=torch.float32).to(device)

model = UC2TemporalLSTM().to(device)
model.load_state_dict(torch.load("models/uc2_lstm_temporal.pth"))
model.eval()

with torch.no_grad():
    logits = model(X).squeeze().cpu().numpy()

auc = roc_auc_score(y, logits)
print(f"UC2 ROC-AUC: {auc:.4f}")
