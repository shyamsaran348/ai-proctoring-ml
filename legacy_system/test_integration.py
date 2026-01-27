import os
import time
import sys
import numpy as np
import cv2

# Mock Django settings if needed, or simply test the Adapter isolation
# The adapter uses 'static/uploads/students' path. We need to mock that.

def setup_mock_environment(student_id):
    base_dir = os.path.dirname(__file__)
    target_dir = os.path.join(base_dir, "static/uploads/students")
    os.makedirs(target_dir, exist_ok=True)
    
    # Create a dummy enrollment image
    img_path = os.path.join(target_dir, f"{student_id}_reference.jpg")
    if not os.path.exists(img_path):
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add some noise/features so it's not empty for ResNet
        cv2.randn(dummy_img, 0, 255)
        cv2.imwrite(img_path, dummy_img)
        print(f"Created mock enrollment image at {img_path}")
    else:
        print(f"Using existing mock enrollment at {img_path}")
    
    return img_path

def test_adapter_flow():
    from exams.services.ml_adapter import MLProctoringAdapter
    
    student_id = "test_student_001"
    setup_mock_environment(student_id)
    
    print("\n--- Initializing MLProctoringAdapter ---")
    adapter = MLProctoringAdapter()
    
    print(f"--- Starting Monitoring for {student_id} ---")
    adapter.start_monitoring(student_id)
    
    print("--- Polling Status for 10 seconds ---")
    try:
        for i in range(5):
            time.sleep(2)
            status = adapter.get_live_status()
            print(f"[{i*2}s] Status: Risk={status.get('risk_score', 'N/A'):.4f} | UC1={status.get('uc1_identity_sim', 'N/A'):.4f}")
    except KeyboardInterrupt:
        pass
    finally:
        print("--- Stopping Monitoring ---")
        history = adapter.stop_monitoring()
        print(f"Session History Points: {len(history)}")
        print("✅ Integration Verification Completed")

if __name__ == "__main__":
    test_adapter_flow()
