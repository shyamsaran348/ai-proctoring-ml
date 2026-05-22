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
from proctoring_ml_module.engines.uc6_engine import UC6Engine
from ml.engines.gam_engine import GAMEngine
from ml.engines.hgdm_engine import HGDMEngine
from proctoring_ml_module.models.architectures import GAM, HGDM


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

        print(f"[ProctoringEngine] Loading from: {__file__}")
        print("[ProctoringEngine] Initializing 7-Signal Core...")

        print("      - [1/7] Identity (ResNet)...")
        self.uc1 = UC1Engine(self.config)
        
        print("      - [2/7] Instability (LSTM)...")
        self.uc2 = UC2Engine(self.config)
        
        print("      - [3/7] Presence (UC3)...")
        self.uc3 = UC3PresenceEngine(self.config)
        
        print("      - [4/7] Drift (UC4)...")
        uc4_path = os.path.join(self.config.get('model_dir', 'proctoring_ml_module/models'), 'uc4_drift_model.pth')
        self.uc4 = UC4Engine(uc4_path)
        
        print("      - [5/7] Gaze (GAM)...")
        # Phase 17: GAM Engine
        gam_model = GAM()
        gam_path = os.path.join(self.config.get('model_dir', 'proctoring_ml_module/models'), 'gam_model.pth')
        if os.path.exists(gam_path):
            gam_model.load_state_dict(torch.load(gam_path, map_location='cpu'))
        self.gam = GAMEngine(gam_model, device=self.config.get('inference', {}).get('device', 'cpu'))

        print("      - [6/7] Head-Pose (HGDM)...")
        # Phase 18: HGDM Engine
        hgdm_model = HGDM()
        hgdm_path = os.path.join(self.config.get('model_dir', 'proctoring_ml_module/models'), 'hgdm_model.pth')
        if os.path.exists(hgdm_path):
            hgdm_model.load_state_dict(torch.load(hgdm_path, map_location='cpu'))
        self.hgdm = HGDMEngine(model_path=hgdm_path, device=self.config.get('inference', {}).get('device', 'cpu'))
        self.hgdm.model = hgdm_model.to(self.hgdm.device)
        self.hgdm.model.eval()

        print("      - [7/7] Fusion & Audio (UC5/6)...")
        self.uc5 = UC5Engine(self.config)
        self.uc6 = UC6Engine(self.config)

        self.enrollment_embedding = None
        self.session_active = False
        
        # ─── Stability Control (Phase 24) ───
        self.stability_counter = 0 
        self.stability_threshold = 12 # ~3 seconds of perfect behavior clears memory

        print("[ProctoringEngine] 7-Signal Sentinel Ready.")

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
        self.gam.reset()
        self.hgdm.reset()
        self.uc5.reset()
        self.uc6.reset()
        self.stability_counter = 0

        self.session_active = True

        print("[ProctoringEngine] Session Started.")
        print("[ProctoringEngine] Enrollment embedding fixed (immutable).")

    # ==========================================================
    # -------------------- FRAME PROCESSING --------------------
    # ==========================================================

    def process_frame(self, frame_input, uc3_features=None, gaze_features=None, audio_features=None):
        """
        Process a live camera frame.

        Args:
            frame_input: Path, PIL Image, or Numpy array
            uc3_features: numpy array (6,) for UC3 temporal presence modeling
            gaze_features: numpy array (6,) for Phase 17 GAM gaze modeling

        Returns:
            dict:
            {
                "uc1_similarity": float,
                "uc2_instability": float,
                "uc3_presence": float or None,
                "uc4_drift": float,
                "gam_gaze": float,
                "hgdm_prob": float,
                "uc6_audio": float,
                "risk": float,
                "uncertainty": float,
                "violation_type": str
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
                "gam_gaze": 0.5,
                "hgdm_prob": 0.5,
                "uc6_audio": 0.9,
                "risk": 1.0,
                "uncertainty": 1.0,
                "violation_type": "FACE_NOT_DETECTED"
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
        # GAM — Eye Gaze Modeling (Phase 17)
        # ------------------------------------------------------
        
        g_t_prob = 0.5
        if gaze_features is not None:
            g_t_prob = self.gam.update(gaze_features)

        # ------------------------------------------------------
        # HGDM — Head-Gaze Dynamics (Phase 18)
        # ------------------------------------------------------
        
        h_t_prob = 0.5
        if uc3_features is not None and gaze_features is not None:
            # uc3_features index 2,3,4 are yaw, pitch, roll
            head_pose = uc3_features[2:5]
            h_t_prob = self.hgdm.update(head_pose, gaze_features)

        # ------------------------------------------------------
        # UC6 — Acoustic Anomaly Detection (Phase 19)
        # ------------------------------------------------------
        
        audio_prob = 0.5
        if audio_features is not None:
            audio_prob = self.uc6.update(float(audio_features))

        # ------------------------------------------------------
        # Fast Recovery Logic (Phase 24)
        # ------------------------------------------------------
        
        # is_clean: Identity must be strong, and presence/gaze shouldn't be extreme violations
        is_clean = (uc1_sim > 0.70 and presence_prob < 0.6 and g_t_prob < 0.7)
        if is_clean:
            self.stability_counter += 1
        else:
            self.stability_counter = 0

        if self.stability_counter >= self.stability_threshold:
            # Clear temporal memory to reset sticky risk scores
            self.uc5.reset() # Clears GRU buffer
            self.stability_counter = 0
            print("[Engine] ⚡ FAST RECOVERY: Cleared temporal history after sustained stability.")

        # ------------------------------------------------------
        # UC5 — Risk Fusion (Sentinel 7-Signal Suite)
        # ------------------------------------------------------

        risk, uncertainty = self.uc5.update(
            uc1_sim,
            uc2_prob,
            presence_prob,
            uc4_drift,
            g_t_prob,
            h_t_prob,
            audio_prob
        )

        # Determine Primary Violation for UI help
        violation_type = "SAFE"
        if risk > 0.7:
            if uc1_sim < 0.5: violation_type = "IDENTITY_MISMATCH"
            elif presence_prob > 0.7: violation_type = "BEYOND_SCREEN_BOUNDARY"
            elif g_t_prob > 0.8: violation_type = "OFFSCREEN_GAZE"
            elif h_t_prob > 0.8: violation_type = "UNNATURAL_POSTURE"
            elif audio_prob > 0.8: violation_type = "SOPHISTICATED_AUDIO_ANOMALY"
            else: violation_type = "GENERAL_SUSPICIOUS_BEHAVIOR"

        print(
            f"[Engine] Sim: {uc1_sim:.4f} | "
            f"Instability: {uc2_prob:.4f} | "
            f"Presence: {presence_prob:.4f} | "
            f"Drift: {uc4_drift:.4f} | "
            f"Gaze: {g_t_prob:.4f} | "
            f"HGDM: {h_t_prob:.4f} | "
            f"Audio: {audio_prob:.4f} | "
            f"Risk: {risk:.4f} | "
            f"V: {violation_type}"
        )

        return {
            "uc1_similarity": uc1_sim,
            "uc2_instability": uc2_prob,
            "uc3_presence": presence_prob,
            "uc4_drift": uc4_drift,
            "gam_gaze": g_t_prob,
            "hgdm_prob": h_t_prob,
            "uc6_audio": audio_prob,
            "risk": risk,
            "uncertainty": uncertainty,
            "violation_type": violation_type
        }

    