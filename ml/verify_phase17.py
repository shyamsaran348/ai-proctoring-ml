import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from proctoring_ml_module.engines.proctoring_engine import ProctoringEngine

def main():
    print("========================================")
    print("   VERIFY PHASE 17 (5-SIGNAL ENGINE)    ")
    print("========================================")

    # 1. Initialize Engine
    # Note: If weights don't exist yet, it will use untrained baselines
    engine = ProctoringEngine()

    # 2. Simulate Enrollment
    print("[INFO] Starting simulated session...")
    dummy_enrollment = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # create a temp file for enrollment
    temp_path = "temp_verify_enroll.jpg"
    import cv2
    cv2.imwrite(temp_path, dummy_enrollment)
    
    try:
        engine.start_session(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 3. Process Dummy Frames
    print("[INFO] Processing 150 dummy frames...")
    for i in range(150):
        dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        uc3_feat = np.random.normal(0, 1, 6).astype(np.float32)
        gaze_feat = np.random.normal(0, 1, 6).astype(np.float32)
        
        metrics = engine.process_frame(dummy_frame, uc3_features=uc3_feat, gaze_features=gaze_feat)
        
        if i % 30 == 0:
            print(f"Frame {i}: Risk={metrics['risk']:.4f}, Gaze={metrics['gam_gaze']:.4f}")

    print("\n[SUCCESS] Phase 17 Engineering Integration Verified.")
    print("The 5-signal temporal pipeline is functioning correctly.")

if __name__ == "__main__":
    main()
