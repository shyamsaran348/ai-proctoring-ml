import cv2
import base64
import os
import sys
import numpy as np

# Add legacy_system to path
sys.path.append(os.path.abspath('legacy_system'))

from legacy_system.exams.services.ml_adapter import MLProctoringAdapter

def test_lighting():
    adapter = MLProctoringAdapter()
    img_path = 'legacy_system/media/students/passport_photo.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return

    adapter.start_monitoring('test_user', image_path=img_path)
    
    # 1. Test Normal Light
    print("\n--- Testing Normal Light ---")
    img = cv2.imread(img_path)
    _, buffer = cv2.imencode('.jpg', img)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    adapter.process_external_frame(f"data:image/jpeg;base64,{frame_b64}")
    status = adapter.get_live_status()
    print(f"Low Light Detected: {status['low_light']}")
    print(f"Status Text: {status['head_pose']}")

    # 2. Test Low Light (Simulated)
    print("\n--- Testing Low Light ---")
    dark_img = (img * 0.2).astype(np.uint8) # Drastically darken
    _, buffer = cv2.imencode('.jpg', dark_img)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    adapter.process_external_frame(f"data:image/jpeg;base64,{frame_b64}")
    status = adapter.get_live_status()
    print(f"Low Light Detected: {status['low_light']}")
    print(f"Status Text: {status['head_pose']}")
    print(f"Num Faces: {status['num_faces']}")

    if status['low_light']:
        print("\n✅ Lighting Robustness Verified: Low light detection working.")
    else:
        print("\n❌ Verification Failed: Low light not detected.")

if __name__ == "__main__":
    test_lighting()
