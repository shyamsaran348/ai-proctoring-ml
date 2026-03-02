import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path
import os
import sys

# Ensure module visibility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ml.uc5_risk_fusion.fusion_model import RiskFusionGRU

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

X_PATH = DATASET_DIR / "ablation_sequences.npy"
y_PATH = DATASET_DIR / "ablation_labels.npy"

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
# TRAIN FUNCTION
# ============================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, run_name, mask_idx=None, device="cpu"):
    """
    Trains RiskFusionGRU.
    If mask_idx is given, that specific signal index (0-3) is completely zeroed out
    across all splits (simulating the model operating without it).
    """
    
    # Apply Ablation Masking BEFORE compiling DataLoaders
    # We copy the arrays so we don't permanently modify the shared dataset
    X_train_run = np.copy(X_train)
    X_test_run = np.copy(X_test)
    
    if mask_idx is not None:
        X_train_run[:, :, mask_idx] = 0.0
        X_test_run[:, :, mask_idx] = 0.0
        
    train_loader = DataLoader(RiskDataset(X_train_run, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(RiskDataset(X_test_run, y_test), batch_size=32, shuffle=False)

    model = RiskFusionGRU(input_dim=4, hidden_dim=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 10
    
    print(f"\n--- Training {run_name} ---")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            _, final_risk = model(x_batch)
            loss = criterion(final_risk, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            _, final_risk = model(x_batch)
            probs = torch.sigmoid(final_risk).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(y_batch.numpy())
            
    auc_score = roc_auc_score(all_labels, all_preds)
    print(f"[{run_name}] Validation ROC-AUC: {auc_score:.4f}")
    return auc_score

# ============================================================
# MAIN ABLATION LOOP
# ============================================================

def main():
    print("[INFO] Loading Synthesized Ablation Dataset...")
    X = np.load(X_PATH)
    y = np.load(y_PATH)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Maintain strict generalization using stratify to ensure perfectly balanced class representation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting device: {device}")
    
    results = {}
    
    # 1. Full 4-Signal Model
    results["Full Structure (4-Signals)"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, 
        "Full Structural Fusion", 
        mask_idx=None, 
        device=device
    )
    
    # 2. Ablate UC1 (Similarity)
    results["No Identity (Sub UC1)"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, 
        "Ablation: Minus UC1", 
        mask_idx=0, 
        device=device
    )
    
    # 3. Ablate UC2 (Short Term Instability)
    results["No Instability (Sub UC2)"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, 
        "Ablation: Minus UC2", 
        mask_idx=1, 
        device=device
    )
    
    # 4. Ablate UC3 (Presence)
    results["No Presence (Sub UC3)"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, 
        "Ablation: Minus UC3", 
        mask_idx=2, 
        device=device
    )
    
    # 5. Ablate UC4 (Long Term Drift)
    results["No Drift Tracking (Sub UC4)"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, 
        "Ablation: Minus UC4", 
        mask_idx=3, 
        device=device
    )
    
    print("\n=============================================")
    print("PHASE 10: SIGNAL IMPORTANCE ABLATION VALIDATION")
    print("=============================================")
    
    baseline = results["Full Structure (4-Signals)"]
    print(f"Baseline (Full 4-Signal Integration) ROC-AUC => {baseline:.4f}\n")
    
    for name, auc in results.items():
        if name == "Full Structure (4-Signals)":
            continue
        delta = baseline - auc
        
        # Format the impact output
        impact = f"+{delta:.4f} loss to global AUC" if delta > 0 else f"{delta:.4f} (redundant)"
        print(f"Ablation: {name: <30} | AUC: {auc:.4f} | Delta: {impact}")

    # Export
    with open("ablation_results.txt", "w") as f:
        f.write("Phase 10 Signal Importance Ablation\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print("\n✅ Matrix exported to ablation_results.txt.")


if __name__ == "__main__":
    main()
