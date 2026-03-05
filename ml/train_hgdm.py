import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

# Ensure architectures.py can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from proctoring_ml_module.models.architectures import HGDM

def train_hgdm():
    print("--- Training HGDM (Phase 18) ---")
    
    # Load Data
    if not os.path.exists('ml/data/hgdm_x.npy'):
        print("Error: HGDM dataset not found. Run generate_hgdm_data.py first.")
        return
        
    X = np.load('ml/data/hgdm_x.npy')
    y = np.load('ml/data/hgdm_y.npy')
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # Model
    model = HGDM()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item()
                
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")
        
    # Save Model
    model_dir = 'proctoring_ml_module/models'
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'hgdm_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] HGDM Model saved to {save_path}")

if __name__ == "__main__":
    train_hgdm()
