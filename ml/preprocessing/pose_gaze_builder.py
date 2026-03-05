import numpy as np

class PoseGazeBuilder:
    """
    Constructs the 7D feature vector for HGDM:
    [h_yaw, h_pitch, h_roll, g_yaw, g_pitch, delta_h_pitch, delta_g_pitch]
    """
    def __init__(self):
        self.prev_h_pitch = 0.0
        self.prev_g_pitch = 0.0

    def build(self, head_pose, gaze_features):
        """
        head_pose: [yaw, pitch, roll]
        gaze_features: [yaw, pitch, ...] (first two are yaw, pitch)
        """
        h_yaw, h_pitch, h_roll = head_pose
        g_yaw, g_pitch = gaze_features[0], gaze_features[1]
        
        delta_h_pitch = h_pitch - self.prev_h_pitch
        delta_g_pitch = g_pitch - self.prev_g_pitch
        
        self.prev_h_pitch = h_pitch
        self.prev_g_pitch = g_pitch
        
        return np.array([
            h_yaw, h_pitch, h_roll,
            g_yaw, g_pitch,
            delta_h_pitch, delta_g_pitch
        ], dtype=np.float32)

    def reset(self):
        self.prev_h_pitch = 0.0
        self.prev_g_pitch = 0.0
