import numpy as np
import cv2
import mediapipe as mp
import math

class GazeFeatureExtractor:
    """
    MediaPipe-based Gaze Feature Extractor.
    Extracts 6D gaze vector: [yaw, pitch, pupil_x, pupil_y, blink, velocity]
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.prev_gaze = None
        
        # Landmark indices for eyes (MediaPipe Refined Landmarks)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_PUPIL = 468
        self.RIGHT_PUPIL = 473

    def _get_pupil_offset(self, landmarks, eye_indices, pupil_index, w, h):
        eye_points = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in eye_indices])
        pupil_point = np.array((landmarks.landmark[pupil_index].x * w, landmarks.landmark[pupil_index].y * h))
        
        # Eye center
        eye_center = np.mean(eye_points, axis=0)
        # Eye width/height for normalization
        eye_w = np.max(eye_points[:, 0]) - np.min(eye_points[:, 0])
        eye_h = np.max(eye_points[:, 1]) - np.min(eye_points[:, 1])
        
        if eye_w == 0 or eye_h == 0:
            return 0.0, 0.0
            
        offset_x = (pupil_point[0] - eye_center[0]) / (eye_w / 2)
        offset_y = (pupil_point[1] - eye_center[1]) / (eye_h / 2)
        
        return np.clip(offset_x, -1.0, 1.0), np.clip(offset_y, -1.0, 1.0)

    def extract_features(self, frame):
        """
        Extracts 6D gaze feature vector and returns landmarks for shared use.
        Returns: (np.ndarray, landmarks)
        """
        if frame is None:
            return np.zeros(6, dtype=np.float32), None

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return np.zeros(6, dtype=np.float32), None

        landmarks = results.multi_face_landmarks[0]
        
        # 1. Pupil Offsets (Average of both eyes)
        lx, ly = self._get_pupil_offset(landmarks, self.LEFT_EYE, self.LEFT_PUPIL, w, h)
        rx, ry = self._get_pupil_offset(landmarks, self.RIGHT_EYE, self.RIGHT_PUPIL, w, h)
        pupil_x = (lx + rx) / 2.0
        pupil_y = (ly + ry) / 2.0

        # 2. Blink Ratio (Eye Aspect Ratio - EAR)
        def get_ear(eye_indices):
            points = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in eye_indices])
            v1 = np.linalg.norm(points[1] - points[5])
            v2 = np.linalg.norm(points[2] - points[4])
            h1 = np.linalg.norm(points[0] - points[3])
            return (v1 + v2) / (2.0 * h1 + 1e-6)
            
        ear = (get_ear(self.LEFT_EYE) + get_ear(self.RIGHT_EYE)) / 2.0
        blink = 1.0 - np.clip(ear / 0.3, 0.0, 1.0)

        # 3. Head Pose (Yaw, Pitch)
        nose = landmarks.landmark[1]
        l_eye = landmarks.landmark[33]
        r_eye = landmarks.landmark[263]
        
        yaw = (l_eye.z - r_eye.z) * 100.0 
        pitch = (nose.y - (l_eye.y + r_eye.y)/2.0) * 100.0
        
        # 4. Velocity
        current_gaze = np.array([yaw, pitch, pupil_x, pupil_y])
        velocity = 0.0
        if self.prev_gaze is not None:
            velocity = np.linalg.norm(current_gaze - self.prev_gaze)
        self.prev_gaze = current_gaze

        return np.array([yaw, pitch, pupil_x, pupil_y, blink, velocity], dtype=np.float32), landmarks
