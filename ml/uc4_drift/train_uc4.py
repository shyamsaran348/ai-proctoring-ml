import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
import os

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

DATA_PATH = "ml/uc4_drift/datasets/uc4_drift_sequences.npy"
LABEL_PATH = "ml/uc4_drift/datasets/uc4_drift_labels.npy"
MODEL_SAVE_PATH = "ml/uc4_drift/models/uc4_drift_model.pth"

DEVICE = torch.device("cpu")
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

os.makedirs("ml/uc4_drift/models", exist_ok=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

X = np.load(DATA_PATH)
y = np.load(LABEL_PATH)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------
# DATASET CLASS
# ---------------------------------------------------

class DriftDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(DriftDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DriftDataset(X_val, y_val), batch_size=BATCH_SIZE)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

class UC4DriftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=257,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        final_state = out[:, -1, :]
        logit = self.fc(final_state)
        return logit.squeeze(1)

model = UC4DriftModel().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.extend(probs)
            val_targets.extend(yb.numpy())

    auc = roc_auc_score(val_targets, val_preds)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val AUC: {auc:.4f}\n")

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("✅ UC4 Drift Model Saved.")