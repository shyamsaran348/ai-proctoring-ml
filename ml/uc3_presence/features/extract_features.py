import os
import cv2
import argparse
import numpy as np

class UC3FeatureExtractor:
    def __init__(self):
        self.prev_gray = None
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def compute_motion_energy(self, frame_gray):
        if self.prev_gray is None or self.prev_gray.shape != frame_gray.shape:
            self.prev_gray = frame_gray
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, frame_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        energy = np.mean(mag)

        self.prev_gray = frame_gray
        return energy

    def extract_frame_features(self, frame, landmarks=None):
        """
        Extracts 6D vector: [face_conf, area_ratio, yaw, pitch, roll, motion]
        If landmarks (MediaPipe) provided, uses them for head pose.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        yaw, pitch, roll = 0.0, 0.0, 0.0
        face_conf = 0.0
        area_ratio = 0.0

        if landmarks is not None:
            # Use MediaPipe landmarks for head pose
            face_conf = 1.0
            # nose, chin, l_eye, r_eye, l_mouth, r_mouth for pnp
            # Simplified for now using relative z/y
            nose = landmarks.landmark[1]
            l_eye = landmarks.landmark[33]
            r_eye = landmarks.landmark[263]
            
            yaw = (l_eye.z - r_eye.z) * 100.0
            pitch = (nose.y - (l_eye.y + r_eye.y)/2.0) * 100.0
            roll = (l_eye.y - r_eye.y) * 100.0
            
            # Simple area_ratio from eye distance
            eye_dist = np.sqrt((l_eye.x - r_eye.x)**2 + (l_eye.y - r_eye.y)**2)
            area_ratio = (eye_dist * 2.5) ** 2 # Approximation
            
        else:
            # Fallback to Haar
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                x, y, bw, bh = faces[0]
                face_conf = 1.0
                area_ratio = (bw * bh) / (w * h)
                yaw, pitch, roll = 0.0, 0.0, 0.0 # Stub fallback

        motion_energy = self.compute_motion_energy(gray)

        return np.array([
            face_conf,
            area_ratio,
            yaw,
            pitch,
            roll,
            motion_energy
        ], dtype=np.float32)

def build_sequences(features, T=60):
    sequences = []
    for i in range(0, len(features) - T + 1, T):
        sequences.append(features[i:i + T])
    if len(sequences) == 0:
        return None
    return np.stack(sequences)