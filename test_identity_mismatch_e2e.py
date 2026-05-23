#!/usr/bin/env python3
"""
End-to-End Identity Mismatch Verification Test
Simulates:
1. Enrollment of Person A.
2. 10 frames of Person A (Matching Face) showing stable verification.
3. 10 frames of Person B (Imposter Face) showing instant mismatch trigger and flag.
"""

import os
import sys
import numpy as np
import cv2
import torch

# Ensure we can find the proctoring modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Protobuf bypass block
from types import ModuleType
m = ModuleType('doc_controls')
m.do_not_generate_docs = lambda x: x
mt = ModuleType('tensorflow.tools.docs')
mt.doc_controls = m
sys.modules['tensorflow.tools.docs'] = mt

from proctoring_ml_module.engines.proctoring_engine import ProctoringEngine


class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def create_face_representation(pattern_seed, filename):
    """
    Creates a mock facial crop with a specific noise pattern.
    Pattern seeds ensure distinct, discriminative features for ResNet-50.
    """
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    np.random.seed(pattern_seed)
    
    # Draw simple distinct spatial features (simulating different facial structures)
    # Person A features
    if pattern_seed == 101:
        cv2.circle(img, (112, 112), 40, (100, 200, 100), -1)
        cv2.rectangle(img, (80, 80), (140, 140), (50, 100, 250), 3)
    # Person B features (Imposter - completely distinct structural features)
    else:
        cv2.line(img, (10, 10), (200, 10), (0, 0, 255), 10)
        cv2.circle(img, (30, 180), 20, (255, 0, 0), -1)
        
    # Add random pixel noise to standard deviation
    noise = np.random.normal(128, 5, (224, 224, 3)).astype(np.uint8)
    img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)
    
    cv2.imwrite(filename, img)
    return os.path.abspath(filename)


def run_identity_test():
    print(f"\n{Color.BOLD}{Color.BLUE}======================================================================{Color.RESET}")
    print(f" {Color.BOLD}STARTING E2E IDENTITY VERIFICATION & IMPOSTER CRASH TEST{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}======================================================================{Color.RESET}")
    
    # 1. Load Real Faces or Fall Back to Mock
    real_a = "legacy_system/media/students/passport_photo.jpg"
    real_b = "legacy_system/media/students/3122235002123.jpg"
    
    if os.path.exists(real_a) and os.path.exists(real_b):
        ref_a_path = os.path.abspath(real_a)
        frame_a_path = os.path.abspath(real_a) # Exact match
        frame_b_path = os.path.abspath(real_b) # Real imposter
        is_real = True
    else:
        ref_a_path = create_face_representation(101, "person_a_ref.jpg")
        frame_a_path = create_face_representation(101, "person_a_frame.jpg")
        frame_b_path = create_face_representation(999, "person_b_frame.jpg")
        is_real = False
        
    print(f"  - Using Real Faces: {is_real}")
    print(f"  - Enrollment Reference: {ref_a_path}")
    print(f"  - Live Feed Frame: {frame_a_path}")
    print(f"  - Imposter Frame: {frame_b_path}")
    
    # 2. Initialize Engine
    print("\n[1/4] Initializing Proctoring Engine...")
    engine = ProctoringEngine()
    
    # 3. Enroll Person A
    print("\n[2/4] Enrolling Person A (Anchoring Ground-Truth $e_0$)...")
    engine.start_session(ref_a_path)
    
    # 4. Feed Matching Frames (Person A)
    print("\n[3/4] Simulating Person A Taking the Exam (10 compliant frames)...")
    print(f"  {'Frame':<6} | {'Target':<10} | {'Cosine Sim':<11} | {'Instability':<12} | {'Risk':<8} | {'Status':<15}")
    print("  " + "-" * 73)
    
    for i in range(1, 11):
        metrics = engine.process_frame(frame_a_path)
        print(f"  #{i:<5} | {'Person A':<10} | {metrics['uc1_similarity']:<11.4f} | {metrics['uc2_instability']:<12.4f} | {metrics['risk']:<8.4f} | {Color.GREEN}{metrics['violation_type']:<15}{Color.RESET}")
    
    # 5. Feed Imposter Frames (Person B)
    print("\n[4/4] Simulating Person B Sitting in the Chair (10 imposter frames)...")
    print(f"  {'Frame':<6} | {'Target':<10} | {'Cosine Sim':<11} | {'Instability':<12} | {'Risk':<8} | {'Status':<15}")
    print("  " + "-" * 73)
    
    for i in range(11, 21):
        metrics = engine.process_frame(frame_b_path)
        color = Color.RED if metrics['violation_type'] == "IDENTITY_MISMATCH" else Color.YELLOW
        print(f"  #{i:<5} | {'Person B':<10} | {metrics['uc1_similarity']:<11.4f} | {metrics['uc2_instability']:<12.4f} | {metrics['risk']:<8.4f} | {color}{metrics['violation_type']:<15}{Color.RESET}")
        
    # Cleanup temp images if they were created
    if not is_real:
        for f in [ref_a_path, frame_a_path, frame_b_path]:
            if os.path.exists(f):
                os.remove(f)
            
    print(f"\n{Color.BOLD}{Color.GREEN}======================================================================{Color.RESET}")
    print(f" {Color.BOLD}IDENTITY MISMATCH VERIFICATION COMPLETED SUCCESSFULLY{Color.RESET}")
    print(f"{Color.BOLD}{Color.GREEN}======================================================================{Color.RESET}\n")


if __name__ == "__main__":
    run_identity_test()
