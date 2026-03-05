import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

# Ensure architectures.py can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from proctoring_ml_module.models.architectures import RiskFusionGRU

def train_rfe_v3():
    print("--- Training RFE v3 (6-Signal Fusion - Phase 18) ---")
    
    # Load Data
    if not os.path.exists('ml/data/risk_sequences_v3.npy'):
        print("Error: RFE v3 dataset not found. Run generate_risk_dataset_v3.py first.")
        return
        
    X = np.load('ml/data/risk_sequences_v3.npy')
    y = np.load('ml/data/risk_labels_v3.npy')
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    
    # Model (6 inputs)
    model = RiskFusionGRU(input_dim=6, hidden_dim=32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            risk_traj, final_logits = model(bx)
            loss = criterion(final_logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                _, out = model(bx)
                val_loss += criterion(out, by).item()
                
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")
        
    # Save Model
    model_dir = 'proctoring_ml_module/models'
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'uc5_risk_gru_v3.pth')
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] RFE v3 Model saved to {save_path}")

if __name__ == "__main__":
    train_rfe_v3()
