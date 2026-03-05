import numpy as np

class GazeFeatureExtractor:
    """
    Conceptual stub for MediaPipe FaceMesh gaze feature extraction.
    In a real system, this would process video frames.
    For Phase 17, this provides the interface for computing g_t.
    """
    def __init__(self):
        pass

    def extract_features(self, frame=None):
        """
        Extracts 6D gaze feature vector from a frame.
        
        Returns:
            np.ndarray: [yaw, pitch, pupil_x, pupil_y, blink, velocity]
        """
        # In this synthetic-first phase, this is a stub.
        # Real implementation would use MediaPipe landmarks.
        return np.zeros(6, dtype=np.float32)
