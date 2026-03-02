import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# ---------------- Dataset Loader -----------------------------
# ============================================================

class UC3Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# ---------------- BiLSTM Model -------------------------------
# ============================================================

class PresenceModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze(1)


# ============================================================
# ---------------- Training Function -------------------------
# ============================================================

def train_model(model, train_loader, val_loader, epochs=25, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):

        # ---- Training ----
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f}")

        # ---- Save Best Model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("ml/uc3_presence/models", exist_ok=True)
            torch.save(model.state_dict(),
                       "ml/uc3_presence/models/presence_model.pth")

    print("\nTraining Complete.")
    print("Best model saved at: ml/uc3_presence/models/presence_model.pth")


# ============================================================
# ---------------- Main --------------------------------------
# ============================================================

def main():

    X = np.load("ml/uc3_presence/datasets/sequences.npy")
    y = np.load("ml/uc3_presence/datasets/labels.npy")

    dataset = UC3Dataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = PresenceModel()

    train_model(model, train_loader, val_loader)


if __name__ == "__main__":
    main()