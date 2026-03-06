import cv2
import numpy as np
import threading
import time
import os
import sys
from datetime import datetime
import base64

# Ensure we can find the proctoring module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from proctoring_ml_module.api.inference_interface import create_engine

# UC3 feature extractor (6D presence features)
try:
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    sys.path.append(proj_root)
    from ml.uc3_presence.features.extract_features import UC3FeatureExtractor
    _uc3_available = True
except ImportError:
    _uc3_available = False
    print("[ML Adapter] WARNING: UC3FeatureExtractor not found — presence features will be disabled.")


class MLProctoringAdapter:
    """
    Adapter to replace the Rule-Based ProctoringSystem with the Model-First ProctoringEngine.
    Adheres to the Integration Contract:
    - Input: Enrollment Image + Continuous Frames
    - Output: Continuous Risk Metrics (No binary decisions)
    """

    def __init__(self):
        self.engine = create_engine()  # usage of config.yaml from within the module
        self.monitoring = False
        self.thread = None
        self.cap = None
        self.student_id = None

        # UC3 feature extractor instance
        self.uc3_extractor = UC3FeatureExtractor() if _uc3_available else None

        # State for UI compatibility
        self.latest_status = {
            'is_active': False,
            'uc1_identity_sim': 0.0,
            'uc2_instability': 0.0,
            'uc3_presence': 0.5,
            'uc4_drift': 0.0,
            'risk_score': 0.0,
            'uncertainty': 0.0,

            # Maintaining legacy keys to prevent frontend crashes,
            # but mapping them to ML context where possible.
            'num_faces': 1,  # Logic handled by ML, usually assume 1 if risk is low
            'head_pose': 'Monitoring',
            'face_confidence': 1.0,
            'last_update': time.time(),
            'anomaly_count': 0,  # We do not generate discrete anomalies anymore
            'left_eye_img': '',
            'right_eye_img': '',
            'face_locations': []  # [top, right, bottom, left]
        }
        self.anomalies = []   # Legacy field
        self.history = []     # For risk trajectory logging

    def start_monitoring(self, student_id, image_path=None):
        """
        Initialize the ML session.
        Enforces One-Shot Enrollment using the stored reference image.
        """
        self.student_id = student_id

        # Pre-load Haar Cascades (once at startup)
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            if self.face_cascade.empty() or self.profile_cascade.empty():
                print("[ML Error] Haar Cascades failed to load.")
        except Exception as e:
            print(f"[ML Error] Cascade Load Error: {e}")

        # Resolve enrollment image path
        if image_path and os.path.exists(image_path):
            ref_path = image_path
        else:
            ref_path = f"static/uploads/students/{self.student_id}_reference.jpg"
            if not os.path.exists(ref_path):
                ref_path = f"static/uploads/students/{self.student_id}.jpg"

        if not os.path.exists(ref_path):
            print(f"[ML Adapter] ERROR: Enrollment image not found for {student_id} at {ref_path}")
            return

        print(f"[ML Adapter] Loading enrollment: {ref_path}")
        try:
            self.engine.start_session(ref_path)
            self.monitoring = True
        except Exception as e:
            print(f"[ML Adapter] Failed to start engine session: {e}")
            return

        # Reset UC3 motion state for the new session
        if self.uc3_extractor is not None:
            self.uc3_extractor.prev_gray = None

        print(f"[ML Adapter] Session Ready for {student_id}")

    def stop_monitoring(self):
        self.monitoring = False
        if self.cap:
            self.cap.release()
        return self.history  # Return risk history instead of anomalies list

    def _monitor_loop(self):
        while self.monitoring:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                uc3_features = None
                if self.uc3_extractor is not None:
                    uc3_features = self.uc3_extractor.extract_frame_features(frame)

                metrics = self.engine.process_frame(frame_rgb, uc3_features=uc3_features)

                self._update_status_from_metrics(metrics, num_faces=1, face_locs=[], looking_away=False)

                if len(self.history) % 10 == 0:
                    print(f"[ML Monitor] Risk={metrics['risk']:.4f} | Sim={metrics['uc1_similarity']:.4f} | Instability={metrics['uc2_instability']:.4f}")

                self.history.append({
                    'timestamp': time.time(),
                    'risk': metrics['risk'],
                    'uncertainty': metrics.get('uncertainty', 0.0),
                    'uc1': metrics['uc1_similarity']
                })

                time.sleep(0.1)

            except Exception as e:
                print(f"[ML Adapter] Error in loop: {e}")
                time.sleep(1)

    def process_external_frame(self, frame_data):
        """
        Process a frame sent from the frontend (Base64 or Bytes).
        Extracts UC3 presence features and passes them to the engine.
        """
        try:
            # Decode Base64 if string
            if isinstance(frame_data, str):
                if 'base64,' in frame_data:
                    frame_data = frame_data.split('base64,')[1]
                frame_bytes = base64.b64decode(frame_data)
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                return

            if frame is None:
                return

            # Convert to RGB for the ML engine
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ─── UC3 Feature Extraction ────────────────────────────────────────
            # Extracts 6D vector: [face_conf, area_ratio, yaw, pitch, roll, motion]
            uc3_features = None
            if self.uc3_extractor is not None:
                try:
                    uc3_features = self.uc3_extractor.extract_frame_features(frame)
                except Exception as uc3_e:
                    print(f"[ML Adapter] UC3 feature extraction warning: {uc3_e}")

            # ─── ML Engine: Process Frame ──────────────────────────────────────
            metrics = self.engine.process_frame(frame_rgb, uc3_features=uc3_features)

            # ─── Face Detection (For UI Bounding Boxes & Count) ───────────────
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            if not hasattr(self, 'face_cascade') or self.face_cascade.empty():
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            faces_rects = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(50, 50))

            face_locs = []
            for (x, y, w, h) in faces_rects:
                face_locs.append([int(y), int(x + w), int(y + h), int(x)])

            looking_away = False

            if len(face_locs) == 0:
                # Try profile cascade
                if not hasattr(self, 'profile_cascade') or self.profile_cascade.empty():
                    self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

                profiles = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(50, 50))
                if len(profiles) > 0:
                    looking_away = True
                    for (x, y, w, h) in profiles:
                        face_locs.append([int(y), int(x + w), int(y + h), int(x)])
                else:
                    gray_flipped = cv2.flip(gray, 1)
                    profiles_flipped = self.profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.2, minNeighbors=8, minSize=(50, 50))
                    if len(profiles_flipped) > 0:
                        looking_away = True
                        h_img, w_img = gray.shape
                        for (x, y, w, h) in profiles_flipped:
                            x_orig = w_img - x - w
                            face_locs.append([int(y), int(x_orig + w), int(y + h), int(x_orig)])

            num_faces = len(face_locs)

            # Identity trust override
            uc1_sim = metrics['uc1_similarity']
            if uc1_sim > 0.65:
                if num_faces > 1:
                    print(f"[Override] Forcing 1 face (was {num_faces}) due to High ID Match")
                    num_faces = 1
                    face_locs = sorted(face_locs, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]), reverse=True)[:1]

            self._update_status_from_metrics(metrics, num_faces, face_locs, looking_away)

            if len(self.history) % 10 == 0:
                print(f"[External Frame] Faces={num_faces} | Risk={metrics['risk']:.4f} | Sim={uc1_sim:.4f}")

        except Exception as e:
            print(f"[ML Adapter] Error processing external frame: {e}")

    def _update_status_from_metrics(self, metrics, num_faces, face_locs, looking_away):
        """Shared helper to update latest_status from engine metrics."""
        sim_score   = metrics['uc1_similarity']
        risk_score  = metrics['risk']
        instability = metrics['uc2_instability']

        self.latest_status['uc1_identity_sim'] = sim_score
        self.latest_status['uc2_instability']  = instability
        self.latest_status['uc3_presence']     = metrics.get('uc3_presence', 0.5)
        self.latest_status['uc4_drift']        = metrics.get('uc4_drift', 0.0)
        self.latest_status['gam_gaze']         = metrics.get('gam_gaze', 0.5)
        self.latest_status['hgdm_prob']        = metrics.get('hgdm_prob', 0.5)
        self.latest_status['risk_score']       = risk_score
        self.latest_status['uncertainty']      = metrics.get('uncertainty', 0.0)
        self.latest_status['last_update']      = time.time()
        self.latest_status['is_active']        = True
        self.latest_status['num_faces']        = num_faces
        self.latest_status['face_locations']   = face_locs
        self.latest_status['face_confidence']  = 1.0

        # Build human-readable status text
        if risk_score > 0.7:
            status_text = f"HIGH RISK: {risk_score:.2f} | ID: {sim_score:.2f}"
        elif looking_away:
            status_text = "WARNING: LOOKING AWAY"
            self.latest_status['risk_score'] = max(risk_score, 0.85)
        elif sim_score < 0.50:
            status_text = f"Identity Mismatch: {sim_score:.2f}"
        elif num_faces == 0:
            status_text = "No Face Detected"
        elif num_faces > 1:
            status_text = "Multiple Faces Detected"
        else:
            status_text = f"ID: {sim_score:.2f} | S: {instability:.2f} | R: {risk_score:.2f}"

        self.latest_status['head_pose'] = status_text

        self.history.append({
            'timestamp': time.time(),
            'risk':  risk_score,
            'uc1':   sim_score,
            'uc2':   instability,
            'uc3':   metrics.get('uc3_presence', 0.5),
            'uc4':   metrics.get('uc4_drift', 0.0),
            'gam':   metrics.get('gam_gaze', 0.5),
            'hgdm':  metrics.get('hgdm_prob', 0.5),
            'uncertainty': metrics.get('uncertainty', 0.0),
        })

    def get_live_status(self):
        # Check for timeout (if no external frames received for 5s)
        if time.time() - self.latest_status['last_update'] > 5:
            self.latest_status['is_active'] = False
            self.latest_status['head_pose'] = "Connection Lost"
        return self.latest_status.copy()

    def get_risk_history(self):
        """Return the full risk history for trajectory display."""
        return list(self.history)

    def get_anomalies(self):
        # We replace "anomalies" with the risk trajectory or high-risk events
        return []
