import yaml
import os
import torch
import sys

# Ensure we can import sibling modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from proctoring_ml_module.engines.uc1_engine import UC1Engine
from proctoring_ml_module.engines.uc2_engine import UC2Engine
from proctoring_ml_module.engines.uc5_engine import UC5Engine

class ProctoringEngine:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to config.yaml in module root
            config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
            
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print(f"[ProctoringEngine] Loaded from: {__file__}")
        print("[ProctoringEngine] Initializing Engines...")
        self.uc1 = UC1Engine(self.config)
        self.uc2 = UC2Engine(self.config)
        self.uc5 = UC5Engine(self.config)
        
        self.enrollment_embedding = None
        self.session_active = False
        print("[ProctoringEngine] Ready.")

    def start_session(self, enrollment_image_input):
        """
        Start a new proctoring session.
        Args:
            enrollment_image_input: Path, PIL Image, or Numpy array of the enrollment face.
        """
        # 1. Compute Enrollment Embedding (One-Shot)
        emb = self.uc1.get_embedding(enrollment_image_input)
        if emb is None:
            raise ValueError("Failed to compute enrollment embedding. check image.")
        
        self.enrollment_embedding = emb
        
        # 2. Reset Temporal Engines
        self.uc2.reset()
        self.uc5.reset()
        
        self.session_active = True
        print("[ProctoringEngine] Session Started. Enrollment embedding fixed.")

    def process_frame(self, frame_input):
        """
        Process a live camera frame.
        Args:
            frame_input: Path, PIL Image, or Numpy array.
        Returns:
            dict: {
                "uc1_similarity": float,
                "uc2_instability": float,
                "risk": float
            }
        """
        if not self.session_active or self.enrollment_embedding is None:
            raise RuntimeError("Session not started. Call start_session() first.")

        # UC1: Identity Check
        probe_emb = self.uc1.get_embedding(frame_input)
        if probe_emb is None:
             # If face detection fails or image is bad...
             print("Warning: Could not extract embedding from frame.")
             return {
                 "uc1_similarity": 0.0,
                 "uc2_instability": 1.0, 
                 "risk": 1.0 
             }

        # Compute Similarity vs Enrollment
        uc1_sim = self.uc1.compute_similarity(self.enrollment_embedding, probe_emb)

        # ---------------------------------------------------------
        # CALIBRATION: Standard Mode (Webcam vs Webcam)
        # We now expect high similarity (>0.7), so no massive boost needed.
        # ---------------------------------------------------------
        uc1_sim_calibrated = uc1_sim 

        # UC2: Temporal Instability
        uc2_prob = self.uc2.update(uc1_sim_calibrated)
        
        # UC5: Risk Fusion
        risk = self.uc5.update(uc1_sim_calibrated, uc2_prob)
        
        # ---------------------------------------------------------
        # SAFETY CLAMP: Override GRU if Similarity is High
        # ---------------------------------------------------------
        if uc1_sim_calibrated > 0.65:
            # User is clearly present and matching.
            # Force risk down significantly, but keep slight variance for liveness feel.
            risk = 0.1
            
            # --- ADAPTIVE IDENTITY UPDATE ---
            # If confidence is VERY high, update the reference embedding slightly.
            # This helps the system adapt to changing lighting/angles over time.
            if uc1_sim_calibrated > 0.85:
                self.update_enrollment(probe_emb)

        elif uc1_sim_calibrated > 0.5:
             # Ambiguous zone, dampen the risk
             risk = min(risk, 0.4)
        
        return {
            "uc1_similarity": uc1_sim_calibrated,
            "uc2_instability": uc2_prob,
            "risk": risk
        }

    def update_enrollment(self, new_emb):
        """
        Adapt the enrollment embedding using a running average.
        Ref = (0.98 * Ref) + (0.02 * New)
        """
        if self.enrollment_embedding is None:
            return
            
        # Ensure tensor properties match
        # new_emb is (1, D), self.enrollment_embedding is (1, D)
        alpha = 0.02 # Slow adaptation rate
        
        with torch.no_grad():
            updated = (1.0 - alpha) * self.enrollment_embedding + (alpha * new_emb)
            # Re-normalize to keep it on the hypersphere (Cosine Similarity relies on this)
            updated = torch.nn.functional.normalize(updated, p=2, dim=1)
            self.enrollment_embedding = updated
            # print("[ProctoringEngine] Adaptive Update Applied") # Optional debug

