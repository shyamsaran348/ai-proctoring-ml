import yaml
import os
import torch
import sys

# Ensure we can import sibling modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from proctoring_ml_module.engines.uc1_engine import UC1Engine
from proctoring_ml_module.engines.uc2_engine import UC2Engine
from proctoring_ml_module.engines.uc3_engine import UC3PresenceEngine
from proctoring_ml_module.engines.uc4_engine import UC4Engine
from proctoring_ml_module.engines.uc5_engine import UC5Engine


class ProctoringEngine:
    def __init__(self, config_path=None):

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                '../config.yaml'
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print(f"[ProctoringEngine] Loaded from: {__file__}")
        print("[ProctoringEngine] Initializing Engines...")

        self.uc1 = UC1Engine(self.config)
        self.uc2 = UC2Engine(self.config)
        self.uc3 = UC3PresenceEngine(self.config)
        
        uc4_path = os.path.join(self.config.get('model_dir', 'proctoring_ml_module/models'), 'uc4_drift_model.pth')
        self.uc4 = UC4Engine(uc4_path)
        
        self.uc5 = UC5Engine(self.config)

        self.enrollment_embedding = None
        self.session_active = False

        print("[ProctoringEngine] Ready.")

    # ==========================================================
    # -------------------- SESSION START ------------------------
    # ==========================================================

    def start_session(self, enrollment_image_input):
        """
        Start a new proctoring session (STRICT ONE-SHOT ENROLLMENT).
        """

        # One-shot enrollment (immutable)
        emb = self.uc1.get_embedding(enrollment_image_input)

        if emb is None:
            raise ValueError("Failed to compute enrollment embedding.")

        self.enrollment_embedding = emb

        # Reset temporal models
        self.uc2.reset()
        self.uc3.reset()
        self.uc4.reset()
        self.uc5.reset()

        self.session_active = True

        print("[ProctoringEngine] Session Started.")
        print("[ProctoringEngine] Enrollment embedding fixed (immutable).")

    # ==========================================================
    # -------------------- FRAME PROCESSING --------------------
    # ==========================================================

    def process_frame(self, frame_input, uc3_features=None):
        """
        Process a live camera frame.

        Args:
            frame_input: Path, PIL Image, or Numpy array
            uc3_features: numpy array (6,) for UC3 temporal presence modeling

        Returns:
            dict:
            {
                "uc1_similarity": float,
                "uc2_instability": float,
                "uc3_presence": float or None,
                "uc4_drift": float,
                "risk": float
            }
        """

        if not self.session_active or self.enrollment_embedding is None:
            raise RuntimeError("Session not started.")

        # ------------------------------------------------------
        # UC1 — Identity Embedding
        # ------------------------------------------------------

        probe_emb = self.uc1.get_embedding(frame_input)

        if probe_emb is None:
            print("Warning: Could not extract embedding from frame.")
            return {
                "uc1_similarity": 0.0,
                "uc2_instability": 1.0,
                "uc3_presence": 0.0,
                "uc4_drift": 1.0,
                "risk": 1.0
            }

        uc1_sim = self.uc1.compute_similarity(
            self.enrollment_embedding,
            probe_emb
        )

        # ------------------------------------------------------
        # UC2 — Temporal Identity Instability
        # ------------------------------------------------------

        uc2_prob = self.uc2.update(uc1_sim)

        # ------------------------------------------------------
        # UC3 — Presence & Attentiveness
        # ------------------------------------------------------

        presence_prob = None

        if uc3_features is not None:
            presence_prob = self.uc3.update(uc3_features)

        # If buffer not full yet, treat as neutral signal (do NOT force rules)
        if presence_prob is None:
            presence_prob = 0.5

        # ------------------------------------------------------
        # UC4 — Long-Term Identity Drift
        # ------------------------------------------------------

        probe_vec = probe_emb.cpu().numpy().flatten()
        enroll_vec = self.enrollment_embedding.cpu().numpy().flatten()
        delta_vector = probe_vec - enroll_vec

        uc4_drift = self.uc4.update(delta_vector, uc1_sim)

        # ------------------------------------------------------
        # UC5 — Risk Fusion (Now 4-Signal)
        # ------------------------------------------------------

        risk = self.uc5.update(
            uc1_sim,
            uc2_prob,
            presence_prob,
            uc4_drift
        )

        print(
            f"[Engine] Sim: {uc1_sim:.4f} | "
            f"Instability: {uc2_prob:.4f} | "
            f"Presence: {presence_prob:.4f} | "
            f"Drift: {uc4_drift:.4f} | "
            f"Risk: {risk:.4f}"
        )

        return {
            "uc1_similarity": uc1_sim,
            "uc2_instability": uc2_prob,
            "uc3_presence": presence_prob,
            "uc4_drift": uc4_drift,
            "risk": risk
        }
    