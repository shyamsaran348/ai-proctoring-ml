import torch
import torch.nn as nn
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from proctoring_ml_module.models.architectures import RiskFusionGRU

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "ml/uc5_risk_fusion/datasets"

MODEL_4_PATH = "proctoring_ml_module/models/uc5_risk_gru.pth"
MODEL_5_PATH = "proctoring_ml_module/models/uc5_risk_gru_v2.pth"

def evaluate_ablation():
    print("Loading Phase 17 Evaluation Dataset (v2_sequences)...")
    X = np.load(os.path.join(DATA_DIR, "v2_sequences.npy"))
    y = np.load(os.path.join(DATA_DIR, "v2_labels.npy"))
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    
    # 1. Evaluate 4-Signal Baseline (S, I, P, D)
    print("\n[Baseline] Evaluating 4-Signal Model (S, I, P, D)...")
    # We only take the first 4 columns: UC1, UC2, UC3, UC4
    X_4 = X_tensor[:, :, :4]
    
    model_4 = RiskFusionGRU(input_dim=4, hidden_dim=32).to(DEVICE)
    if os.path.exists(MODEL_4_PATH):
        model_4.load_state_dict(torch.load(MODEL_4_PATH, map_location=DEVICE))
        model_4.eval()
        with torch.no_grad():
            _, risk_logits_4, _, _ = model_4(X_4)
            probs_4 = torch.sigmoid(risk_logits_4).cpu().numpy()
            auc_4 = roc_auc_score(y, probs_4)
            print(f"  4-Signal AUC: {auc_4:.4f}")
    else:
        print(f"  WARNING: Baseline model not found at {MODEL_4_PATH}")
        auc_4 = 0.0

    # 2. Evaluate 5-Signal Model (S, I, P, D, G)
    print("\n[Full Model] Evaluating 5-Signal Model (S, I, P, D, G)...")
    model_5 = RiskFusionGRU(input_dim=5, hidden_dim=32).to(DEVICE)
    if os.path.exists(MODEL_5_PATH):
        model_5.load_state_dict(torch.load(MODEL_5_PATH, map_location=DEVICE))
        model_5.eval()
        with torch.no_grad():
            _, risk_logits_5, _, _ = model_5(X_tensor)
            probs_5 = torch.sigmoid(risk_logits_5).cpu().numpy()
            auc_5 = roc_auc_score(y, probs_5)
            print(f"  5-Signal AUC: {auc_5:.4f}")
    else:
        print(f"  WARNING: Full model not found at {MODEL_5_PATH}")
        auc_5 = 0.0

    print("\n" + "="*40)
    print("           GAM ABLATION SUMMARY           ")
    print("="*40)
    print(f"  Signals: S,I,P,D       | AUC: {auc_4:.4f}")
    print(f"  Signals: S,I,P,D,G     | AUC: {auc_5:.4f}")
    print(f"  Improvement (delta)    | +{(auc_5 - auc_4):.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_ablation()
