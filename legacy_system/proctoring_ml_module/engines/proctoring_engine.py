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
            # If face detection fails or image is bad, what do we do?
            # Return high risk? Or skip?
            # Phase 8 says monitoring module... 
            # If we assume input is a cropped face (Model-First, face detection happened outside OR inside?)
            # Prompt: "Inputs: Live camera frames (or frame paths)". 
            # UC1 expects a FACE. If full frame, ResNet might fail to find face?
            # User context says: "Dataset construction logic... identity-level data splitting". "This Repository DOES NOT CONTAIN... Webcam capture logic".
            # Usually strict proctoring assumes input is a aligned face.
            # But the prompt says "Live camera frames".
            # For this module, we will assume the input *contains* the face or is the face.
            # ResNet over a full room image won't work well for identity. 
            # *Assuming strict constraint*: The input should be a face crop.
            # If `get_embedding` fails, we might return None or zeros.
            # Let's return default high risk or 0 similarity?
            # To be safe, we'll raise warning and return last known or defaults.
             print("Warning: Could not extract embedding from frame.")
             return {
                 "uc1_similarity": 0.0,
                 "uc2_instability": 1.0, # High instability
                 "risk": 1.0 # High risk
             }

        uc1_sim = self.uc1.compute_similarity(self.enrollment_embedding, probe_emb)
        
        # UC2: Temporal Instability
        uc2_prob = self.uc2.update(uc1_sim)
        
        # UC5: Risk Fusion
        # UC5: Risk Fusion
        risk = self.uc5.update(uc1_sim, uc2_prob)
        
        # ---------------------------------------------------------
        # SAFETY CLAMP: Override GRU if Similarity is High
        # ---------------------------------------------------------
        if uc1_sim > 0.65:
            # User is clearly present and matching.
            # Force risk down significantly, but keep slight variance for liveness feel.
            risk = 0.1
        elif uc1_sim > 0.5:
             # Ambiguous zone, dampen the risk
             risk = min(risk, 0.4)
        
        return {
            "uc1_similarity": uc1_sim,
            "uc2_instability": uc2_prob,
            "risk": risk
        }
