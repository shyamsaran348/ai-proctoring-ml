import cv2
import numpy as np
import threading
import time
import os
import sys
from datetime import datetime
import base64
# import face_recognition  <-- Removed to avoid complex dlib dependency
# Using OpenCV Haar Cascade instead (already installed)

# Ensure we can find the proctoring module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from proctoring_ml_module.api.inference_interface import create_engine

class MLProctoringAdapter:
    """
    Adapter to replace the Rule-Based ProctoringSystem with the Model-First ProctoringEngine.
    Adheres to the Integration Contract:
    - Input: Enrollment Image + Continuous Frames
    - Output: Continuous Risk Metrics (No binary decisions)
    """

    def __init__(self):
        self.engine = create_engine() # usage of config.yaml from within the module
        self.monitoring = False
        self.thread = None
        self.cap = None
        self.student_id = None
        
        # State for UI compatibility
        self.latest_status = {
            'is_active': False,
            'uc1_identity_sim': 0.0,
            'uc2_instability': 0.0,
            'risk_score': 0.0,
            
            # Maintaining legacy keys to prevent frontend crashes, 
            # but mapping them to ML context where possible.
            'num_faces': 1, # Logic handled by ML, usually assume 1 if risk is low
            'head_pose': 'Monitoring',
            'face_confidence': 1.0,
            'last_update': time.time(),
            'anomaly_count': 0, # We do not generate discrete anomalies anymore
            'left_eye_img': '',
            'right_eye_img': '',
            'face_locations': [] # [top, right, bottom, left]
        }
        self.anomalies = [] # Legacy field
        self.history = [] # For risk trajectory logging

    def start_monitoring(self, student_id, image_path=None):
        """
        Initialize the ML session.
        Enforces One-Shot Enrollment using the stored reference image.
        """
        self.student_id = student_id
        
        # 1. Load Enrollment Image (One-Shot)
        # -------------------------------------------------------------------------
        # Pre-load Haar Cascades (Optimization: Load once at startup)
        # -------------------------------------------------------------------------
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            # Verify they loaded correctly
            if self.face_cascade.empty() or self.profile_cascade.empty():
                print("[ML Error] Haar Cascades failed to load.")
        except Exception as e:
            print(f"[ML Error] Cascade Load Error: {e}")

        # -------------------------------------------------------------------------
        if image_path and os.path.exists(image_path):
            ref_path = image_path
        else:
            # Fallback to old logic
            ref_path = f"static/uploads/students/{self.student_id}_reference.jpg"
            if not os.path.exists(ref_path):
                 ref_path = f"static/uploads/students/{self.student_id}.jpg"
        
        if not os.path.exists(ref_path):
            print(f"[ML Adapter] ERROR: Enrollment image not found for {student_id} at {ref_path}")
            # Failsafe: Cannot start ML session without enrollment
            return
            
        print(f"[ML Adapter] Loading enrollment: {ref_path}")
        try:
            # We pass the path directly to the engine
            self.engine.start_session(ref_path)
            self.monitoring = True # Set to True so we accept external frames
        except Exception as e:
            print(f"[ML Adapter] Failed to start engine session: {e}")
            return

        # 2. Start Capture (Optional - Backend Webcam)
        # DISABLE BACKEND WEBCAM to prevent conflict with Browser
        # try:
        #     self.cap = cv2.VideoCapture(0)
        #     if not self.cap.isOpened():
        #         print("[ML Adapter] Backend webcam unavailable (locked by browser?). Switching to Passive Mode.")
        #         # Do NOT return. We stay in passive mode.
        #     else:
        #          # Only start the loop if we have a webcam
        #         self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        #         self.thread.start()
        #         print(f"[ML Adapter] Active Monitoring started (Webcam)")
        # except Exception as e:
        #      print(f"[ML Adapter] Webcam init failed: {e}. Passive mode.")

        print(f"[ML Adapter] Session Ready for {student_id}")

    def stop_monitoring(self):
        self.monitoring = False
        if self.cap:
            self.cap.release()
        return self.history # Return risk history instead of anomalies list

    def _monitor_loop(self):
        while self.monitoring:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            try:
                # 3. Process Frame with ML Engine
                # Frame resolution can be reduced for speed if needed, but Engine handles resizing
                # Pass numpy array (BGR) directly
                # Convert to RGB if the engine expects it (The engine uses PIL internally which handles array)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                metrics = self.engine.process_frame(frame_rgb)
                
                # 4. Update Status (Continuous Signals)
                # Update status
                self.latest_status['uc1_identity_sim'] = metrics['uc1_similarity']
                self.latest_status['uc2_instability'] = metrics['uc2_instability']
                self.latest_status['risk_score'] = metrics['risk']
                self.latest_status['last_update'] = time.time()
                self.latest_status['is_active'] = True
                
                # VISUAL FEEDBACK MAPPING
                # Map continuous risk to the legacy 'head_pose' field so user sees it in UI
                # UC1 Similarity is also useful to see
                sim_score = metrics['uc1_similarity']
                risk_score = metrics['risk']
                
                if risk_score > 0.7:
                    status_text = f"HIGH RISK: {risk_score:.2f}"
                elif sim_score < 0.4:
                     status_text = f"Identity Mismatch: {sim_score:.2f}"
                else:
                    status_text = f"Monitoring (Risk: {risk_score:.2f})"
                
                self.latest_status['head_pose'] = status_text
                
                # Debug Log (Only every 10 frames to avoid spam)
                if len(self.history) % 10 == 0:
                    print(f"[ML Monitor] Risk={risk_score:.4f} | Sim={sim_score:.4f} | Instability={metrics['uc2_instability']:.4f}")
                
                # Debug Log (Only every 10 frames to avoid spam)
                if len(self.history) % 10 == 0:
                    print(f"[ML Monitor] Risk={risk_score:.4f} | Sim={sim_score:.4f} | Instability={metrics['uc2_instability']:.4f}")
                
                # Store history
                self.history.append({
                    'timestamp': time.time(),
                    'risk': metrics['risk'],
                    'uc1': metrics['uc1_similarity']
                })
                
                # Optional: Sleep to control frame rate (e.g. 5 FPS is enough for monitoring)
                time.sleep(0.1) 
                
            except Exception as e:
                print(f"[ML Adapter] Error in loop: {e}")
                time.sleep(1)

    def process_external_frame(self, frame_data):
        """
        Process a frame sent from the frontend (Base64 or Bytes).
        This bypasses cv2.VideoCapture(0) when browser locks the camera.
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
                # Assume raw bytes or numpy
                return 

            if frame is None:
                return

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with Engine
            metrics = self.engine.process_frame(frame_rgb)
            
            # ---------------------------------------------------------
            # FACE DETECTION (For UI Bounding Boxes & Count)
            # ---------------------------------------------------------
            # Use OpenCV Haar Cascade (Faster, no external dependencies like dlib)
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            
            # Using pre-loaded cascades from __init__
            if not hasattr(self, 'face_cascade') or self.face_cascade.empty():
                 # Fallback if init failed
                 self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # 1. Try Frontal Face First
            faces_rects = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(50, 50))
            
            # Convert to [top, right, bottom, left]
            face_locs = []
            for (x, y, w, h) in faces_rects:
                face_locs.append([int(y), int(x + w), int(y + h), int(x)])
            
            looking_away = False
            
            # 2. If No Frontal Face, Try Profile (Side View) -> "Looking Away"
            if len(face_locs) == 0:
                # Flip gray frame to detect both left and right profiles with one cascade
                # Profile cascade usually detects faces looking mainly to right (or left depending on training)
                # We check normal, if not found, we check flipped.
                
                # Check Normal
                profiles = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(50, 50))
                if len(profiles) > 0:
                    looking_away = True
                    for (x, y, w, h) in profiles:
                        face_locs.append([int(y), int(x + w), int(y + h), int(x)])
                else:
                    # Check Flipped (Mirror)
                    gray_flipped = cv2.flip(gray, 1)
                    profiles_flipped = self.profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.2, minNeighbors=8, minSize=(50, 50))
                    if len(profiles_flipped) > 0:
                        looking_away = True
                        h_img, w_img = gray.shape
                        for (x, y, w, h) in profiles_flipped:
                            # Remap coordinates from flipped string back to original
                            # x_original = width - x_flipped - w
                            x_orig = w_img - x - w
                            face_locs.append([int(y), int(x_orig + w), int(y + h), int(x_orig)])

            num_faces = len(face_locs)
            
            # --- TRUST LOGIC ---
            uc1_sim = metrics['uc1_similarity']
            
                # Print status every frame for debugging
                # print(f"[Debug] Faces: {num_faces} | Sim: {uc1_sim:.4f} | Risk: {metrics['risk']:.4f}")

            if uc1_sim > 0.65:
                # Strong Override: Trust ResNet Identity
                if num_faces > 1:
                     print(f"[Override] Forcing 1 face (was {num_faces}) due to High ID Match")
                     num_faces = 1
                     # Multiple ghosts. Keep biggest.
                     face_locs = sorted(face_locs, key=lambda b: (b[2]-b[0])*(b[1]-b[3]), reverse=True)[:1]
                     
                # NOTE: If num_faces == 0, we DO NOT force it to 1 anymore.
                # Even if ID score is high (false positive), if Haar sees nothing, we should alert "No Face".

            
            # Update Status (Moved AFTER Override Logic)
            self.latest_status['uc1_identity_sim'] = metrics['uc1_similarity']
            self.latest_status['uc2_instability'] = metrics['uc2_instability']
            self.latest_status['risk_score'] = metrics['risk']
            self.latest_status['last_update'] = time.time()
            self.latest_status['is_active'] = True
            
            # Update Face Data (Corrected Values)
            self.latest_status['num_faces'] = num_faces
            self.latest_status['face_locations'] = face_locs
            
            # Visual Feedback Logic
            sim_score = metrics['uc1_similarity']
            risk_score = metrics['risk']
            instability = metrics['uc2_instability']
            
            # Text Status
            if risk_score > 0.7:
                status_text = f"HIGH RISK: {risk_score:.2f} | ID: {sim_score:.2f}"
            elif looking_away:
                 status_text = "WARNING: LOOKING AWAY"
                 # Boost risk artificially if looking away
                 self.latest_status['risk_score'] = max(risk_score, 0.8) 
            elif sim_score < 0.4:
                 status_text = f"Identity Mismatch: {sim_score:.2f}"
            elif num_faces == 0:
                 status_text = "No Face Detected"
            elif num_faces > 1:
                 status_text = "Multiple Faces Detected"
            else:
                # Standard Monitoring Text
                status_text = f"ID: {sim_score:.2f} | S: {instability:.2f} | R: {risk_score:.2f}"
            
            self.latest_status['head_pose'] = status_text
            self.latest_status['face_confidence'] = 1.0 # Active
            
            # Append to history
            self.history.append({
                'timestamp': time.time(),
                'risk': metrics['risk']
            })
            
            # Print occasionally
            if len(self.history) % 10 == 0:
                 print(f"[External Frame] Faces={num_faces} | Risk={risk_score:.4f} | Sim={sim_score:.4f}")

        except Exception as e:
            print(f"[ML Adapter] Error processing external frame: {e}")

    def get_live_status(self):
        # Check for timeout (if no external frames received for 5s)
        if time.time() - self.latest_status['last_update'] > 5:
            self.latest_status['is_active'] = False
            self.latest_status['head_pose'] = "Connection Lost"
            
        return self.latest_status.copy()

    def get_anomalies(self):
        # We replace "anomalies" with the risk trajectory or high-risk events
        # For compatibility, return empty list or a summary
        # The legacy system expects a list of dicts with 'type' and 'timestamp'
        return [] 
