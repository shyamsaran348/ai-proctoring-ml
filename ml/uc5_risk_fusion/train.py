import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from fusion_model import RiskFusionGRU

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================
# DATASET
# ============================================================

class RiskDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ============================================================
# LOAD DATA
# ============================================================

X = np.load(DATASET_DIR / "risk_sequences.npy")   # (B, 120, 4)
y = np.load(DATASET_DIR / "risk_labels.npy")      # (B,)

dataset = RiskDataset(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ============================================================
# MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RiskFusionGRU(input_dim=4, hidden_dim=32).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================================
# TRAINING LOOP
# ============================================================

EPOCHS = 200

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        _, final_risk = model(x_batch)
        loss = criterion(final_risk, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f}")

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_DIR / "risk_fusion_gru.pth"
)

print("[SAVED] risk_fusion_gru.pth")
