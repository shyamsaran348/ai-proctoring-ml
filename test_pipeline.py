import sys
import os
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from proctoring_ml_module.engines.proctoring_engine import ProctoringEngine
from ml.uc3_presence.features.extract_features import UC3FeatureExtractor

def test_inference():
    print("Testing Engine Initialization...")
    engine = ProctoringEngine()
    
    print("Creating dummy frame...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # create a fake face just in case UC3 fails on black
    cv2.rectangle(dummy_frame, (200, 200), (400, 400), (255, 255, 255), -1)
    
    # Save a dummy enrollment image
    enrollment_path = str(PROJECT_ROOT / "temp_test_enrollment.jpg")
    cv2.imwrite(enrollment_path, dummy_frame)
    
    try:
        print("Starting Session...")
        engine.start_session(enrollment_path)
        
        extractor = UC3FeatureExtractor()
        
        for i in range(5):
            print(f"Processing frame {i+1}...")
            uc3_features = extractor.extract_frame_features(dummy_frame)
            metrics = engine.process_frame(dummy_frame, uc3_features=uc3_features)
            
            print(f"Metrics: {metrics}")
            
            assert 'uc1_similarity' in metrics
            assert 'uc2_instability' in metrics
            assert 'uc3_presence' in metrics
            assert 'risk' in metrics
            
        print("[SUCCESS] Pipeline verification passed.")
    except Exception as e:
        print(f"[ERROR] Pipeline verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if os.path.exists(enrollment_path):
            os.remove(enrollment_path)

if __name__ == "__main__":
    test_inference()
