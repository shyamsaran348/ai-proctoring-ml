import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from ml.models.gam_model import GAM
from ml.preprocessing.gaze_normalizer import normalize_gaze_features

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001

DATA_DIR = "ml/uc5_risk_fusion/datasets"
MODEL_SAVE_PATH = "proctoring_ml_module/models/gam_model.pth"
os.makedirs("proctoring_ml_module/models", exist_ok=True)

def train():
    print(f"Loading data from {DATA_DIR}...")
    X = np.load(os.path.join(DATA_DIR, "gam_features.npy"))
    y = np.load(os.path.join(DATA_DIR, "gam_labels.npy"))
    
    # Normalize features
    X_norm = np.array([normalize_gaze_features(seq) for seq in X])
    
    # GAM expects (B, T, 1) y or session level y?
    # Our GAM outputs (B, T, 1) probability trajectory.
    # For simplicity in synthetic training, if label is 1 (attentive), whole T=1 is target.
    y_seq = np.repeat(y[:, np.newaxis, np.newaxis], X.shape[1], axis=1) # (B, T, 1)
    
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    model = GAM(input_dim=6, hidden_dim=64, num_layers=2, dropout=0.3).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Starting GAM training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ GAM model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
