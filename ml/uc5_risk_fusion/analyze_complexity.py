"""
analyze_complexity.py
Phase 15 — Complexity & Latency Analysis

Measures for each component model:
  - Trainable parameter count
  - Model file size (MB)
  - Per-frame inference latency (ms): 100 warmup + 1000 timed trials
  - Estimated total pipeline latency per frame

Feasibility threshold: < 33ms per frame for real-time 30fps operation.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from proctoring_ml_module.models.architectures import ResNetEmbedder, TemporalLSTM, RiskFusionGRU

# --------------------------------------------------------
# DEVICE
# --------------------------------------------------------

DEVICE = torch.device("cpu")  # CPU: the deployment target
MODEL_DIR = Path("proctoring_ml_module/models")

print(f"\n{'='*65}")
print(" PHASE 15: COMPLEXITY & LATENCY ANALYSIS")
print(f"{'='*65}")
print(f"  Device : {DEVICE}")
print(f"  Target : Real-time 30fps (< 33ms per frame total)\n")

# --------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def file_size_mb(path):
    if Path(path).exists():
        return Path(path).stat().st_size / (1024 * 1024)
    return None


def measure_latency(fn, n_warmup=100, n_trials=1000):
    """Runs fn n_warmup times then times n_trials calls. Returns mean ± std (ms)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


# --------------------------------------------------------
# MODEL DEFINITIONS & MOCK INFERENCE FUNCTIONS
# --------------------------------------------------------

results = {}

# ----- UC1: ResNet-50 Embedder -----
print("─── UC1: ResNet-50 Identity Embedder ───")

uc1_model = ResNetEmbedder(embedding_dim=256, pretrained=False).to(DEVICE).eval()
uc1_path  = MODEL_DIR / "uc1_resnet_embedder.pth"
dummy_frame = torch.randn(1, 3, 224, 224).to(DEVICE)

mean_ms, std_ms = measure_latency(lambda: uc1_model(dummy_frame))
params = count_params(uc1_model)
fsize  = file_size_mb(uc1_path)

results["UC1 (ResNet-50 Embedder)"] = {
    "params": params, "file_mb": fsize, "latency_ms": mean_ms, "std_ms": std_ms
}
print(f"  Params  : {format_params(params)}")
print(f"  File    : {fsize:.2f} MB" if fsize else "  File    : N/A")
print(f"  Latency : {mean_ms:.3f} ± {std_ms:.3f} ms\n")


# ----- UC2: Temporal LSTM -----
print("─── UC2: Temporal LSTM Instability Detector ───")

uc2_model = TemporalLSTM(input_dim=1, hidden_dim=64, num_layers=2).to(DEVICE).eval()
uc2_path  = MODEL_DIR / "uc2_lstm.pth"
# UC2 input: (1, W, 1) where W = 60-frame rolling window
dummy_window = torch.randn(1, 60, 1).to(DEVICE)

mean_ms, std_ms = measure_latency(lambda: uc2_model(dummy_window))
params = count_params(uc2_model)
fsize  = file_size_mb(uc2_path)

results["UC2 (Temporal LSTM)"] = {
    "params": params, "file_mb": fsize, "latency_ms": mean_ms, "std_ms": std_ms
}
print(f"  Params  : {format_params(params)}")
print(f"  File    : {fsize:.2f} MB" if fsize else "  File    : N/A")
print(f"  Latency : {mean_ms:.3f} ± {std_ms:.3f} ms\n")


# ----- UC3: Bi-LSTM Presence Model -----
print("─── UC3: Bi-LSTM Presence & Attentiveness ───")

# UC3 architecture: Bi-LSTM matches presence_model.pth
class UC3BiLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(out).squeeze(1)

uc3_model = UC3BiLSTM().to(DEVICE).eval()
uc3_path  = MODEL_DIR / "presence_model.pth"
# UC3 input: (1, W, 6) feature sequence
dummy_uc3 = torch.randn(1, 60, 6).to(DEVICE)

mean_ms, std_ms = measure_latency(lambda: uc3_model(dummy_uc3))
params = count_params(uc3_model)
fsize  = file_size_mb(uc3_path)

results["UC3 (Bi-LSTM Presence)"] = {
    "params": params, "file_mb": fsize, "latency_ms": mean_ms, "std_ms": std_ms
}
print(f"  Params  : {format_params(params)}")
print(f"  File    : {fsize:.2f} MB" if fsize else "  File    : N/A")
print(f"  Latency : {mean_ms:.3f} ± {std_ms:.3f} ms\n")


# ----- UC4: Bi-LSTM Drift Detector -----
print("─── UC4: Bi-LSTM Identity Drift Detector ───")

class UC4DriftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=257, hidden_size=128,
                            num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(256, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

uc4_model = UC4DriftModel().to(DEVICE).eval()
uc4_path  = MODEL_DIR / "uc4_drift_model.pth"
# UC4 input: (1, 120, 257) rolling buffer
dummy_uc4 = torch.randn(1, 120, 257).to(DEVICE)

mean_ms, std_ms = measure_latency(lambda: uc4_model(dummy_uc4))
params = count_params(uc4_model)
fsize  = file_size_mb(uc4_path)

results["UC4 (Bi-LSTM Drift)"] = {
    "params": params, "file_mb": fsize, "latency_ms": mean_ms, "std_ms": std_ms
}
print(f"  Params  : {format_params(params)}")
print(f"  File    : {fsize:.2f} MB" if fsize else "  File    : N/A")
print(f"  Latency : {mean_ms:.3f} ± {std_ms:.3f} ms\n")


# ----- UC5: GRU Risk Fusion -----
print("─── UC5: GRU Risk Fusion Engine ───")

uc5_model = RiskFusionGRU(input_dim=4, hidden_dim=32).to(DEVICE).eval()
uc5_path  = MODEL_DIR / "uc5_risk_gru.pth"
# UC5 input: (1, buffer_so_far, 4) — grows frame-by-frame; test with full 120
dummy_uc5 = torch.randn(1, 120, 4).to(DEVICE)

mean_ms, std_ms = measure_latency(lambda: uc5_model(dummy_uc5))
params = count_params(uc5_model)
fsize  = file_size_mb(uc5_path)

results["UC5 (GRU Risk Fusion)"] = {
    "params": params, "file_mb": fsize, "latency_ms": mean_ms, "std_ms": std_ms
}
print(f"  Params  : {format_params(params)}")
print(f"  File    : {fsize:.2f} MB" if fsize else "  File    : N/A")
print(f"  Latency : {mean_ms:.3f} ± {std_ms:.3f} ms\n")


# --------------------------------------------------------
# SUMMARY TABLE
# --------------------------------------------------------

total_ms     = sum(v["latency_ms"] for v in results.values())
total_params = sum(v["params"] for v in results.values())
total_mb     = sum(v["file_mb"] for v in results.values() if v["file_mb"])

print(f"\n{'='*65}")
print(" SUMMARY TABLE")
print(f"{'='*65}")
print(f"\n  {'Component':<28} {'Params':>8} {'File(MB)':>9} {'Latency(ms)':>12} {'Std':>8}")
print(f"  {'─'*65}")

for name, m in results.items():
    fmb = f"{m['file_mb']:.2f}" if m["file_mb"] else "N/A"
    print(f"  {name:<28} {format_params(m['params']):>8} {fmb:>9} {m['latency_ms']:>12.3f} {m['std_ms']:>8.3f}")

print(f"  {'─'*65}")
print(f"  {'TOTAL PIPELINE':<28} {format_params(total_params):>8} {total_mb:>9.2f} {total_ms:>12.3f}")
print(f"\n  Real-time threshold (30fps) : 33.333 ms per frame")
rtf = total_ms / 33.333
print(f"  Pipeline load factor        : {rtf:.3f}x  ", end="")
print("✅ FEASIBLE" if rtf < 1.0 else "⚠️  OVER BUDGET — UC1 (ResNet) dominates")

# --------------------------------------------------------
# EXPORT
# --------------------------------------------------------

out_path = Path(__file__).parent / "complexity_results.txt"
with open(out_path, "w") as f:
    f.write("Phase 15 — Complexity & Latency Analysis\n")
    f.write("=" * 50 + "\n\n")
    for name, m in results.items():
        fmb = f"{m['file_mb']:.2f} MB" if m["file_mb"] else "N/A"
        f.write(f"{name}\n")
        f.write(f"  Parameters : {format_params(m['params'])}\n")
        f.write(f"  File Size  : {fmb}\n")
        f.write(f"  Latency    : {m['latency_ms']:.3f} ± {m['std_ms']:.3f} ms\n\n")
    f.write(f"TOTAL\n")
    f.write(f"  Parameters : {format_params(total_params)}\n")
    f.write(f"  File Size  : {total_mb:.2f} MB\n")
    f.write(f"  Latency    : {total_ms:.3f} ms per frame\n")
    f.write(f"  Load Factor: {rtf:.3f}x (threshold: 1.0 for 30fps)\n")

print(f"\n✅ Exported: {out_path.name}")
