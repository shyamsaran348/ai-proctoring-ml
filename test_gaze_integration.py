import cv2
import base64
import os
import sys

# Add legacy_system to path to import MLProctoringAdapter
sys.path.append(os.path.abspath('legacy_system'))

from legacy_system.exams.services.ml_adapter import MLProctoringAdapter

def test_integration():
    adapter = MLProctoringAdapter()
    
    # Test image
    img_path = 'legacy_system/media/students/passport_photo.jpg'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return

    # Start monitoring (enroll with same image for high sim)
    print(f"Starting monitoring for 'test_user' with {img_path}")
    adapter.start_monitoring('test_user', image_path=img_path)
    
    # Encode frame
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    frame_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Process frame
    print("Processing frame...")
    adapter.process_external_frame(f"data:image/jpeg;base64,{frame_b64}")
    
    # Check status
    status = adapter.get_live_status()
    print("\n--- Proctoring Status ---")
    print(f"Is Active: {status['is_active']}")
    print(f"UC1 Identity Sim: {status['uc1_identity_sim']:.4f}")
    print(f"UC3 Presence: {status['uc3_presence']:.4f}")
    print(f"GAM Gaze: {status['gam_gaze']:.4f}")
    print(f"HGDM Prob: {status['hgdm_prob']:.4f}")
    print(f"Risk Score: {status['risk_score']:.4f}")
    print(f"Head Pose/Status: {status['head_pose']}")
    print(f"Num Faces: {status['num_faces']}")
    print(f"Face Locs: {status['face_locations']}")
    
    # Verification
    if status['is_active'] and status['num_faces'] > 0:
        print("\n✅ Integration Verified: Signals are flowing.")
    else:
        print("\n❌ Verification Failed: No face detected or inactive.")

if __name__ == "__main__":
    test_integration()
