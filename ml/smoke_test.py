import cv2
import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from legacy_system.exams.services.ml_adapter import MLProctoringAdapter

def smoke_test():
    print("--- Starting ML Adapter Smoke Test ---")
    adapter = MLProctoringAdapter()

    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("Sending 5 dummy frames...")
    for i in range(5):
        adapter.process_external_frame(dummy_frame)
        status = adapter.get_live_status()
        print(f"Frame {i+1}: Risk={status['risk_score']:.4f}, Uncertainty={status.get('uncertainty', 'N/A')}")
        time.sleep(0.1)

    print("--- Smoke Test Complete ---")

if __name__ == "__main__":
    smoke_test()
