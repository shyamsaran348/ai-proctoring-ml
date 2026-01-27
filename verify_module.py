import numpy as np
from PIL import Image
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from proctoring_ml_module.api.inference_interface import create_engine

def run_verification():
    print("--- Starting Verification ---")
    
    # 1. Create Engine
    try:
        engine = create_engine()
        print("✅ Engine created")
    except Exception as e:
        print(f"❌ Engine creation failed: {e}")
        return

    # 2. Mock Enrollment Image
    # ResNet expects 3 channels.
    enrollment_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    enrollment_img = Image.fromarray(enrollment_data)
    
    try:
        engine.start_session(enrollment_img)
        print("✅ Session started")
    except Exception as e:
        print(f"❌ Start session failed: {e}")
        return

    # 3. Process Loop
    print("running 100 frames...")
    try:
        for i in range(100):
            # Generate random frame
            frame_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            frame_img = Image.fromarray(frame_data)
            
            result = engine.process_frame(frame_img)
            
            if i % 20 == 0:
                print(f"Frame {i}: {result}")
                
        print("✅ Processed 100 frames successfully")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return

if __name__ == "__main__":
    run_verification()
