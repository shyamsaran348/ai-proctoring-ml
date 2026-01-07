import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from models.temporal_lstm import UC2TemporalLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"

X = np.load("datasets/train_sequences.npy")
y = np.load("datasets/sequence_labels.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = UC2TemporalLSTM().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    losses = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb).squeeze()
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch} | Loss: {np.mean(losses):.4f}")

torch.save(model.state_dict(), "models/uc2_lstm_temporal.pth")
