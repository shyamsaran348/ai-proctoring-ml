import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from proctoring_ml_module.engines.proctoring_engine import ProctoringEngine
from ml.uc3_presence.features.extract_features import UC3FeatureExtractor


def main():
    print("========================================")
    print("   GENERATE SESSION SIGNALS (3-SIGNAL)  ")
    print("========================================")

    # 1. Initialize Engine
    engine = ProctoringEngine()

    # 2. Pick a sample enrollment image and a test video
    # First, let's find an image and a video to use
    uc3_raw_dir = PROJECT_ROOT / "ml" / "data" / "uc3_raw"
    present_dir = uc3_raw_dir / "present"
    
    if not present_dir.exists():
        print(f"Error: Could not find present directory at {present_dir}")
        sys.exit(1)
        
    videos = list(present_dir.glob("*.mp4"))
    if not videos:
        print("Error: No test videos found.")
        sys.exit(1)
        
    test_video_path = str(videos[0])
    
    # We need an enrollment image. Let's try to extract the first frame of the video with a face.
    print(f"[INFO] Using video: {test_video_path}")
    cap = cv2.VideoCapture(test_video_path)
    
    enrollment_image = None
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    enrollment_path = str(PROJECT_ROOT / "temp_enrollment.jpg")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            enrollment_image = frame
            # Save for the engine to load
            cv2.imwrite(enrollment_path, enrollment_image)
            print(f"[INFO] Saved enrollment image from video to {enrollment_path}")
            break
            
    if enrollment_image is None:
        print("Error: Could not find a face in the video to use as enrollment.")
        cap.release()
        sys.exit(1)
        
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 3. Start Session
    try:
        # Check if enrollment image is compatible
        # the MLAdapter converts BGR to RGB, ProctoringEngine (via UC1Engine) uses PIL internally
        # we'll save it and pass the path, since start_session expects an input that get_embedding can handle
        # In legacy ML Adapter: self.engine.start_session(ref_path)
        engine.start_session(enrollment_path)
    except Exception as e:
        print(f"Failed to start session: {e}")
        cap.release()
        sys.exit(1)

    # 4. Initialize UC3 Extractor
    extractor = UC3FeatureExtractor()

    # Lists to store sequences
    uc1_scores = []
    uc2_probs = []
    uc3_presence = []
    uc4_probs = []
    uc5_gaze = []
    uc6_hgdm = []
    
    print("[INFO] Starting frame processing...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB for ProctoringEngine (matching ML Adapter)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract UC3 Features
        uc3_features = extractor.extract_frame_features(frame)
        
        # Extract Gaze Features (Stub for Phase 17/18)
        # In a real system, these would come from MediaPipe
        gaze_features = np.zeros(6, dtype=np.float32)
        
        # Process Frame
        metrics = engine.process_frame(frame_rgb, uc3_features=uc3_features, gaze_features=gaze_features)
        
        uc1_scores.append(metrics['uc1_similarity'])
        uc2_probs.append(metrics['uc2_instability'])
        uc3_presence.append(metrics['uc3_presence'])
        uc4_probs.append(metrics['uc4_drift'])
        uc5_gaze.append(metrics['gam_gaze'])
        uc6_hgdm.append(metrics['hgdm_prob'])
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
    cap.release()
    print(f"Finished processing {frame_count} frames.")
    
    # 5. Save Arrays
    out_dir = PROJECT_ROOT / "ml" / "uc5_risk_fusion" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "uc1_scores.npy", np.array(uc1_scores, dtype=np.float32))
    np.save(out_dir / "uc2_probs.npy", np.array(uc2_probs, dtype=np.float32))
    np.save(out_dir / "uc3_presence.npy", np.array(uc3_presence, dtype=np.float32))
    np.save(out_dir / "uc4_probs.npy", np.array(uc4_probs, dtype=np.float32))
    np.save(out_dir / "gam_probs.npy", np.array(uc5_gaze, dtype=np.float32))
    np.save(out_dir / "hgdm_probs.npy", np.array(uc6_hgdm, dtype=np.float32))
    
    print(f"[SUCCESS] Signals saved to {out_dir}")
    print(f"uc1_scores: {len(uc1_scores)}")
    print(f"uc2_probs:  {len(uc2_probs)}")
    print(f"uc3_presence:{len(uc3_presence)}")
    print(f"uc4_probs:   {len(uc4_probs)}")
    print(f"gam_probs:   {len(uc5_gaze)}")
    print(f"hgdm_probs:  {len(uc6_hgdm)}")

    
    # Cleanup
    if os.path.exists(enrollment_path):
        os.remove(enrollment_path)

if __name__ == "__main__":
    main()
