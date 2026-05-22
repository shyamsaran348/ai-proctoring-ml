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

try:
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 3 levels: services -> exams -> legacy_system -> ai-proctoring-ml
    proj_root = os.path.abspath(os.path.join(current_dir, '../../../'))
    
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    
    # --- PROCTORING HARDENING: Protobuf Conflict Bypass ---
    # Mediapipe and Tensorflow 2.16+ have a protobuf version clash.
    # Mediapipe tries to import doc_controls from tensorflow, which fails.
    # We mock it here to allow Mediapipe to load and work for inference.
    from types import ModuleType
    m = ModuleType('doc_controls')
    m.do_not_generate_docs = lambda x: x
    mt = ModuleType('tensorflow.tools.docs')
    mt.doc_controls = m
    sys.modules['tensorflow.tools.docs'] = mt
    # -----------------------------------------------------

    from ml.uc3_presence.features.extract_features import UC3FeatureExtractor
    from ml.engines.gaze_feature_extractor import GazeFeatureExtractor
    _ml_available = True
except ImportError as e:
    import traceback
    traceback.print_exc()
    _ml_available = False
    print(f"[ML Adapter] WARNING: ML Extractors not found or failed to load: {e}")


class MLProctoringAdapter:
    def __init__(self):
        self.engine = create_engine()
        self.monitoring = False
        self.thread = None
        self.cap = None
        self.student_id = None

        self.uc3_extractor = None
        self.gaze_extractor = None
        if _ml_available:
            try:
                self.uc3_extractor = UC3FeatureExtractor()
                self.gaze_extractor = GazeFeatureExtractor()
            except Exception as e:
                print(f"[ML Adapter] Extractor Load Failure: {e}")
                # We stay available but with null local extractors
        
        # Lighting robust pre-processing
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # ─── Concurrency Control ───
        # ─── Concurrency Control ───
        self.lock = threading.RLock()
        self.is_processing = False 
        
        self.latest_status = {
            'is_active': False,
            'uc1_identity_sim': 0.0,
            'uc2_instability': 0.0,
            'uc3_presence': 0.5,
            'uc4_drift': 0.0,
            'gam_gaze': 0.5,
            'hgdm_prob': 0.5,
            'uc6_audio': 0.5,
            'risk_score': 0.0,
            'uncertainty': 0.0,
            'num_faces': 1,
            'head_pose': 'Monitoring',
            'violation_type': 'SAFE',
            'face_confidence': 1.0,
            'last_update': time.time(),
            'anomaly_count': 0,
            'left_eye_img': '',
            'right_eye_img': '',
            'face_locations': [],
            'low_light': False
        }
        self.anomalies = []
        self.history = []
        self.MAX_HISTORY = 1000 # Keep last 1000 frames (~3.3 mins @ 5fps) for sparklines
        
        # ─── Temporal Filtering (Phase 24) ───
        self.looking_away_counter = 0
        self.smoothed_risk = 0.0

    def start_monitoring(self, student_id, image_path=None):
        self.student_id = student_id
        self.monitoring = True
        self.looking_away_counter = 0
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        except Exception as e:
            print(f"[ML Error] Cascade Load Error: {e}")

        if image_path and os.path.exists(image_path):
            ref_path = image_path
        else:
            ref_path = f"legacy_system/media/students/{self.student_id}_reference.jpg"
            if not os.path.exists(ref_path):
                ref_path = f"legacy_system/media/students/{self.student_id}.jpg"

        if os.path.exists(ref_path):
            try:
                self.engine.start_session(ref_path)
            except Exception as e:
                print(f"[ML Adapter] Failed to start engine: {e}")

        if self.uc3_extractor: self.uc3_extractor.prev_gray = None
        print(f"[ML Adapter] Session Ready for {student_id}")

    def stop_monitoring(self):
        self.monitoring = False
        if self.cap: self.cap.release()
        return self.history

    def process_external_frame(self, frame_data, audio_volume=None):
        """Processes a base64 encoded frame + acoustic telemetry from an external source"""
        t_start = time.time()
        try:
            with self.lock:
                if self.is_processing:
                    return # Skip if already in flight
                self.is_processing = True
            
            t_decode_start = time.time()
            if not frame_data or not isinstance(frame_data, str):
                return
            
            # ─── Robust Base64 Decoding ───
            if 'base64,' in frame_data:
                frame_data = frame_data.split('base64,')[1]
            
            try:
                frame_bytes = base64.b64decode(frame_data)
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"[ML Adapter] Decoding Error: {e}")
                return

            if frame is None or frame.size == 0:
                print("[ML Adapter] Empty Frame Decoded")
                return
            t_decode_end = time.time()

            # ─── Lighting Robustness (CLAHE) ──────────────────────────────────
            t_en_start = time.time()
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = self.clahe.apply(l_channel)
            lab = cv2.merge((l_channel, a_channel, b_channel))
            frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            brightness = np.mean(l_channel)
            low_light = brightness < 60  # Threshold [0, 255]
            frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
            t_en_end = time.time()

            # ─── Feature Extraction ──────────────────────────────────────────
            t_feat_start = time.time()
            gaze_features, landmarks = None, None
            if self.gaze_extractor:
                gaze_features, landmarks = self.gaze_extractor.extract_features(frame_enhanced)

            uc3_features = None
            if self.uc3_extractor:
                uc3_features = self.uc3_extractor.extract_frame_features(frame_enhanced, landmarks=landmarks)
            t_feat_end = time.time()

            # ─── ML Engine Processing ────────────────────────────────────────
            t_ml_start = time.time()
            metrics = self.engine.process_frame(
                frame_rgb, 
                uc3_features=uc3_features, 
                gaze_features=gaze_features,
                audio_features=audio_volume
            )
            t_ml_end = time.time()

            # ─── Face Detection (UI) ─────────────────────────────────────────
            gray = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)
            if not hasattr(self, 'face_cascade') or self.face_cascade.empty():
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            faces_rects = self.face_cascade.detectMultiScale(gray, 1.2, 8, minSize=(50, 50))
            face_locs = [[int(y), int(x + w), int(y + h), int(x)] for (x, y, w, h) in faces_rects]

            looking_away = False
            if len(face_locs) == 0:
                if not hasattr(self, 'profile_cascade') or self.profile_cascade.empty():
                    self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
                profiles = self.profile_cascade.detectMultiScale(gray, 1.2, 8, minSize=(50, 50))
                if len(profiles) > 0:
                    looking_away = True
                    face_locs = [[int(y), int(x + w), int(y + h), int(x)] for (x, y, w, h) in profiles]

            num_faces = len(face_locs)
            self._update_status_from_metrics(metrics, num_faces, face_locs, looking_away, low_light)

            t_total = time.time() - t_start
            if t_total > 1.0: # Only log slow frames (>1s) to reduce clutter
                print(f"[Sentinel Telemetry] SLOW FRAME: {t_total:.2f}s | Decode: {t_decode_end-t_decode_start:.3f}s | Enhancement: {t_en_end-t_en_start:.3f}s | Feature: {t_feat_end-t_feat_start:.3f}s | ML: {t_ml_end-t_ml_start:.3f}s")

        except Exception as e:
            print(f"[ML Adapter] Error: {e}")
        finally:
            with self.lock:
                self.is_processing = False

    def _update_status_from_metrics(self, metrics, num_faces, face_locs, looking_away, low_light):
        sim_score = metrics.get('uc1_similarity', metrics.get('uc1_identity_sim', 0.0))
        risk_score = metrics['risk']
        engine_vtype = metrics.get('violation_type', 'SAFE')
        
        with self.lock:
            # ─── Temporal Smoothing (EMA) ───
            self.smoothed_risk = 0.85 * self.smoothed_risk + 0.15 * risk_score

            # ─── Temporal Filtering for Hard Rules ───
            if looking_away:
                self.looking_away_counter += 1
            else:
                self.looking_away_counter = max(0, self.looking_away_counter - 1)
            
            sustained_look_away = self.looking_away_counter >= 5 # ~1.5s at 4fps
            
            self.latest_status.update({
                'uc1_identity_sim': sim_score,
                'uc2_instability': metrics['uc2_instability'],
                'uc3_presence': metrics.get('uc3_presence', 0.5),
                'uc4_drift': metrics.get('uc4_drift', 0.0),
                'gam_gaze': metrics.get('gam_gaze', 0.5),
                'hgdm_prob': metrics.get('hgdm_prob', 0.5),
                'uc6_audio': metrics.get('uc6_audio', 0.5),
                'risk_score': self.smoothed_risk,
                'raw_risk_score': risk_score,
                'uncertainty': metrics.get('uncertainty', 0.0),
                'last_update': time.time(),
                'is_active': True,
                'num_faces': num_faces,
                'face_locations': face_locs,
                'low_light': low_light,
                'violation_type': engine_vtype
            })

            # Status Message Construction
            if engine_vtype != 'SAFE':
                txt = engine_vtype.replace('_', ' ')
                if sustained_look_away:
                    txt = "PLEASE LOOK AT THE SCREEN"
                    self.latest_status['risk_score'] = max(self.smoothed_risk, 0.9)
                    self.latest_status['violation_type'] = 'LOOKING_AWAY'
            elif low_light:
                txt = "Low Light Detected - Please turn on lights"
            elif num_faces == 0:
                txt = "No Face Detected"
            elif num_faces > 1:
                txt = "Multiple Faces Detected"
            else:
                txt = f"ID: {sim_score:.2f} | R: {self.smoothed_risk:.2f}"

            self.latest_status['head_pose'] = txt
            self.history.append({'timestamp': time.time(), 'low_light': low_light, **metrics, 'smoothed_risk': self.smoothed_risk})
        
        # ─── History Capping (Memory Management) ───
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0) # Remove oldest


    def get_live_status(self):
        with self.lock:
            if time.time() - self.latest_status['last_update'] > 5:
                self.latest_status['is_active'] = False
                self.latest_status['head_pose'] = "Connection Lost"
            return self.latest_status.copy()

    def get_risk_history(self):
        with self.lock:
            return list(self.history)
