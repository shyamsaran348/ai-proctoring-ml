import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from proctoring_ml_module.models.architectures import RiskFusionGRU

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001

DATA_DIR = "ml/uc5_risk_fusion/datasets"
MODEL_SAVE_PATH = "proctoring_ml_module/models/uc5_risk_gru_v2.pth"

def train():
    print(f"Loading 5-signal data from {DATA_DIR}...")
    X = np.load(os.path.join(DATA_DIR, "v2_sequences.npy"))
    y = np.load(os.path.join(DATA_DIR, "v2_labels.npy"))
    
    # RFE v2 expects (B, T, 5)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # Initialize 5-signal RFE
    model = RiskFusionGRU(input_dim=5, hidden_dim=32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Starting RFE v2 training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            # Forward: returns (traj, final_risk_logits)
            _, final_risk = model(batch_X)
            loss = criterion(final_risk, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                _, final_risk = model(batch_X)
                val_loss += criterion(final_risk, batch_y).item()
                
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ RFE v2 model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
