import numpy as np

def normalize_gaze_features(g):
    """
    Normalizes raw gaze features for BiLSTM stability.
    
    Args:
        g: np.ndarray of shape (N, 6) or (6,)
           [gaze_yaw, gaze_pitch, pupil_x_offset, pupil_y_offset, blink_ratio, gaze_velocity]
    
    Returns:
        Normalized array of the same shape.
    """
    g = np.array(g, dtype=np.float32)
    
    # Normalization factors based on expected ranges
    # yaw/pitch: +/- 30 degrees
    # pupil offsets: +/- 1.0
    # blink ratio: 0 to 0.5
    # velocity: 0 to 10.0
    
    if g.ndim == 1:
        g[0] /= 30.0     # yaw
        g[1] /= 30.0     # pitch
        g[2:4] /= 1.0    # pupil offsets
        g[4] /= 0.5      # blink ratio
        g[5] /= 10.0     # velocity
    else:
        g[:, 0] /= 30.0     # yaw
        g[:, 1] /= 30.0     # pitch
        g[:, 2:4] /= 1.0    # pupil offsets
        g[:, 4] /= 0.5      # blink ratio
        g[:, 5] /= 10.0     # velocity
        
    return g
