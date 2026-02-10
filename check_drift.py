
import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from proctoring_ml_module.api.inference_interface import create_engine

def run_drift_test():
    print("--- 🧪 Starting Face Drift Simulation ---")
    
    # Paths to real images
    person_a_path = "legacy_system/static/uploads/students/shyam.jpeg"
    person_b_path = "legacy_system/static/uploads/students/shanky.jpeg"
    
    if not os.path.exists(person_a_path) or not os.path.exists(person_b_path):
        print(f"❌ Error: Test images not found at {person_a_path} or {person_b_path}")
        return

    # 1. Initialize Engine
    print("⚙️  Initializing Proctoring Engine...")
    engine = create_engine()
    
    # 2. Enroll Person A (Shyam)
    print(f"👤 Enrolling Person A: {person_a_path}")
    engine.start_session(person_a_path)
    
    # 3. Test with Person A (Self-Test)
    # Read image using OpenCV to simulate a frame
    frame_a = cv2.imread(person_a_path)
    frame_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
    
    metrics_a = engine.process_frame(frame_a)
    sim_a = metrics_a['uc1_similarity']
    print(f"✅ Testing Person A vs Person A (Expected High Match):")
    print(f"   Similarity: {sim_a:.4f}")
    if sim_a > 0.6:
        print("   Result: PASS (Identity Verified)")
    else:
        print("   Result: WARNING (Self-match is logically low?)")

    # 4. Test with Person B (Drift / Swap)
    print(f"\n🔁 Swapping to Person B: {person_b_path}")
    frame_b = cv2.imread(person_b_path)
    frame_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
    
    metrics_b = engine.process_frame(frame_b)
    sim_b = metrics_b['uc1_similarity']
    
    print(f"❌ Testing Person A vs Person B (Expected Mismatch):")
    print(f"   Similarity: {sim_b:.4f}")
    
    # Verification Logic
    THRESHOLD = 0.50
    if sim_b < THRESHOLD:
        print(f"   Result: SUCCESS (Drift Detected! {sim_b:.2f} < {THRESHOLD})")
    else:
        print(f"   Result: FAILURE (Drift NOT Detected! {sim_b:.2f} >= {THRESHOLD})")
        print("   ⚠️  Threshold might be too loose for these two faces.")

if __name__ == "__main__":
    run_drift_test()
