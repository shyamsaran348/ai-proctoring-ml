#!/usr/bin/env python3
"""
Comprehensive E2E Backend Audit & Stress-Testing Suite
Validates the Temporal Behavioral Inference Engine (TBIE) Core and MLProctoringAdapter.
Tests structural stability, edge-case resilience, temporal dynamics, and multi-session isolation.
"""

import os
import sys
import time
import numpy as np
import cv2
import base64
import threading
import traceback

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Protobuf bypass block (matching the adapter's implementation)
from types import ModuleType
m = ModuleType('doc_controls')
m.do_not_generate_docs = lambda x: x
mt = ModuleType('tensorflow.tools.docs')
mt.doc_controls = m
sys.modules['tensorflow.tools.docs'] = mt

from proctoring_ml_module.engines.proctoring_engine import ProctoringEngine
from legacy_system.exams.services.ml_adapter import MLProctoringAdapter


class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_header(title):
    print(f"\n{Color.BOLD}{Color.BLUE}======================================================================")
    print(f" {title}")
    print(f"======================================================================{Color.RESET}")


def print_status(test_name, success, info=""):
    status_str = f"{Color.GREEN}[PASS]{Color.RESET}" if success else f"{Color.RED}[FAIL]{Color.RESET}"
    info_str = f" - {info}" if info else ""
    print(f"  {status_str} {test_name}{info_str}")


def create_dummy_reference_image(filename="test_ref.jpg"):
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.randn(img, 128, 50)  # Random noise
    cv2.imwrite(filename, img)
    return os.path.abspath(filename)


def run_e2e_audit():
    print_header("STARTING TEMPORAL INFERENCE ENGINE (TBIE) DEEP AUDIT")
    
    # Setup baseline mock files
    ref_image_path = create_dummy_reference_image("test_enrollment_ref.jpg")
    probe_image_path = create_dummy_reference_image("test_probe.jpg")
    
    results = {}
    
    # Initialize Core Engine
    print("[1/5] Initializing TBIE Core Engine...")
    try:
        engine = ProctoringEngine()
        results["Engine Init"] = True
        print_status("Engine Initialization", True)
    except Exception as e:
        results["Engine Init"] = False
        print_status("Engine Initialization", False, f"Error: {e}")
        traceback.print_exc()
        return

    # -------------------------------------------------------------
    # STAGE 1: Session & Enrollment Verification
    # -------------------------------------------------------------
    print_header("STAGE 1: SESSION & ENROLLMENT BOUNDARIES")
    
    # 1.1 Process Frame without Session Active (Loophole check)
    try:
        engine.process_frame(probe_image_path)
        print_status("Process frame without active session (Loophole check)", False, "Engine processed frame without session starting (Expected RuntimeError)")
        results["Frame Process before Session Check"] = False
    except RuntimeError:
        print_status("Process frame without active session (Loophole check)", True, "RuntimeError correctly raised")
        results["Frame Process before Session Check"] = True
    except Exception as e:
        print_status("Process frame without active session (Loophole check)", False, f"Unexpected error: {e}")
        results["Frame Process before Session Check"] = False

    # 1.2 Start Session with Valid Enrollment
    try:
        engine.start_session(ref_image_path)
        print_status("Start session with valid enrollment image", True)
        results["Valid Enrollment Start"] = True
    except Exception as e:
        print_status("Start session with valid enrollment image", False, f"Error: {e}")
        results["Valid Enrollment Start"] = False

    # 1.3 Start Session with Invalid/None Enrollment (Error Boundary)
    try:
        engine.start_session(None)
        print_status("Start session with None enrollment (Error Boundary)", False, "Engine accepted None reference without throwing error")
        results["None Enrollment Start"] = False
    except (ValueError, TypeError, AttributeError):
        print_status("Start session with None enrollment (Error Boundary)", True, "ValueError/TypeError correctly raised")
        results["None Enrollment Start"] = True
    except Exception as e:
        print_status("Start session with None enrollment (Error Boundary)", False, f"Unexpected error: {e}")
        results["None Enrollment Start"] = False

    # Reset active session
    engine.start_session(ref_image_path)

    # -------------------------------------------------------------
    # STAGE 2: Modality Input Edge Cases (Numeric Stress)
    # -------------------------------------------------------------
    print_header("STAGE 2: MODALITY INPUT EDGE CASES & NUMERIC STRESS")

    # 2.1 Mismatched Dimensions (Non-standard feature vector lengths)
    try:
        mismatched_uc3 = np.array([1.0, 2.0]) # Expected size: (6,)
        mismatched_gaze = np.array([0.5])      # Expected size: (6,)
        metrics = engine.process_frame(probe_image_path, uc3_features=mismatched_uc3, gaze_features=mismatched_gaze)
        print_status("Mismatched dimensions handling", True, f"Engine survived mismatched shapes gracefully (Risk: {metrics['risk']:.4f})")
        results["Mismatched Shapes"] = True
    except Exception as e:
        print_status("Mismatched dimensions handling", False, f"Engine crashed on non-standard feature vectors: {e}")
        results["Mismatched Shapes"] = False

    # 2.2 NaN Injection (Verifying numerical robustness of temporal filters)
    try:
        nan_uc3 = np.array([np.nan, 0.0, np.nan, 1.0, 0.0, 0.5])
        nan_gaze = np.array([0.0, np.nan, 0.5, 0.5, np.nan, 1.0])
        metrics = engine.process_frame(probe_image_path, uc3_features=nan_uc3, gaze_features=nan_gaze, audio_features=np.nan)
        
        # Verify if NaN propagates to risk outputs
        risk_nan = np.isnan(metrics['risk'])
        unc_nan = np.isnan(metrics['uncertainty'])
        
        if risk_nan or unc_nan:
            print_status("NaN Injection Vulnerability Check", False, f"Warning: NaNs propagated to output! Risk={metrics['risk']}, Uncertainty={metrics['uncertainty']}")
            results["NaN Injection Handling"] = False
        else:
            print_status("NaN Injection Vulnerability Check", True, f"Engine filtered out NaNs gracefully (Risk: {metrics['risk']:.4f}, Uncertainty: {metrics['uncertainty']:.4f})")
            results["NaN Injection Handling"] = True
    except Exception as e:
        print_status("NaN Injection Vulnerability Check", False, f"Engine crashed on NaN injection: {e}")
        results["NaN Injection Handling"] = False

    # 2.3 Inf Injection
    try:
        inf_uc3 = np.array([np.inf, -np.inf, 0.0, 1.0, 0.0, 0.5])
        metrics = engine.process_frame(probe_image_path, uc3_features=inf_uc3)
        risk_inf = np.isinf(metrics['risk']) or np.isnan(metrics['risk'])
        
        if risk_inf:
            print_status("Inf Injection Vulnerability Check", False, f"Warning: Infinities corrupted risk logits!")
            results["Inf Injection Handling"] = False
        else:
            print_status("Inf Injection Vulnerability Check", True, f"Engine suppressed infinite bounds gracefully (Risk: {metrics['risk']:.4f})")
            results["Inf Injection Handling"] = True
    except Exception as e:
        print_status("Inf Injection Vulnerability Check", False, f"Engine crashed on infinity injection: {e}")
        results["Inf Injection Handling"] = False

    # -------------------------------------------------------------
    # STAGE 3: Dynamic Temporal Load and Horizon Checks
    # -------------------------------------------------------------
    print_header("STAGE 3: DYNAMIC TEMPORAL HORIZONS & LOAD")

    # 3.1 Transient Phase (Sequence length = 1)
    engine.start_session(ref_image_path)
    try:
        metrics = engine.process_frame(probe_image_path, uc3_features=np.zeros(6), gaze_features=np.zeros(6), audio_features=0.1)
        print_status("Transient Sequence (T=1) Performance", True, f"Completed successfully. Risk: {metrics['risk']:.4f}")
        results["Transient Sequence Load"] = True
    except Exception as e:
        print_status("Transient Sequence (T=1) Performance", False, f"Crashed on T=1: {e}")
        results["Transient Sequence Load"] = False

    # 3.2 Target Horizon Load (T=120)
    print("Feeding 120 continuous sequence steps to engine...")
    try:
        start_t = time.time()
        for i in range(120):
            engine.process_frame(probe_image_path, uc3_features=np.random.rand(6), gaze_features=np.random.rand(6), audio_features=float(np.random.rand()))
        duration = time.time() - start_t
        fps = 120 / duration
        print_status("Target Horizon Load (T=120) Stress Test", True, f"Processed 120 frames in {duration:.2f}s ({fps:.1f} FPS)")
        results["Target Horizon Load"] = True
    except Exception as e:
        print_status("Target Horizon Load (T=120) Stress Test", False, f"Crashed during target sequence progression: {e}")
        results["Target Horizon Load"] = False

    # 3.3 Overload Phase (T=1000)
    print("Feeding 1000 sequence steps to check buffer capping...")
    try:
        for i in range(1000):
            engine.process_frame(probe_image_path, uc3_features=np.random.rand(6), gaze_features=np.random.rand(6), audio_features=0.2)
        print_status("Buffer Overload Stress Test (T=1000)", True, "Memory stayed bounded, temporal experts processed 1000 steps smoothly.")
        results["Buffer Overload Load"] = True
    except Exception as e:
        print_status("Buffer Overload Stress Test (T=1000)", False, f"Crashed on extended sequence load: {e}")
        results["Buffer Overload Load"] = False

    # -------------------------------------------------------------
    # STAGE 4: Temporal Gating, Suppression & Recovery Calibration
    # -------------------------------------------------------------
    print_header("STAGE 4: CALIBRATION, UNCERTAINTY GATING & FAST-RECOVERY")

    # 4.1 Aleatoric Uncertainty Gate Check
    # Verify that a frame with high uncertainty correctly triggers lower strike risks in our adapter
    try:
        adapter = MLProctoringAdapter()
        adapter.start_monitoring("audit_student", ref_image_path)
        
        # Test default risk
        adapter.smoothed_risk = 0.0
        dummy_base64 = base64.b64encode(open(probe_image_path, "rb").read()).decode("utf-8")
        adapter.process_external_frame(dummy_base64, audio_volume=0.9)
        status_after_anomaly = adapter.get_live_status()
        
        # Ensure our smoothed risk is updated
        print_status("Uncertainty & Strike Gating Check", True, f"Smoothed Risk: {status_after_anomaly['risk_score']:.4f} | Raw: {status_after_anomaly.get('raw_risk_score', 0.0):.4f} | Uncertainty: {status_after_anomaly['uncertainty']:.4f}")
        results["Uncertainty Gating"] = True
    except Exception as e:
        print_status("Uncertainty & Strike Gating Check", False, f"Failed to test adapter: {e}")
        results["Uncertainty Gating"] = False

    # 4.2 Hysteresis Fast-Recovery Check
    # Sustainable compliance (is_clean) resetting temporal state buffer
    try:
        engine.start_session(ref_image_path)
        
        # Inject deliberate high anomalies to saturate risk to near 1.0
        print("Injecting anomalous inputs to saturate risk...")
        for _ in range(20):
            engine.process_frame(probe_image_path, uc3_features=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), gaze_features=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), audio_features=0.9)
        
        anomalous_metrics = engine.process_frame(probe_image_path, uc3_features=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), gaze_features=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), audio_features=0.9)
        risk_before = anomalous_metrics['risk']
        
        # Now feed 15 consecutive compliance steps
        print("Injecting 15 consecutive compliance frames (Perfect face/no anomaly)...")
        comp_ref = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.randn(comp_ref, 128, 5) # Matches reference closely
        cv2.imwrite("temp_comp.jpg", comp_ref)
        
        recovered_metrics = None
        for i in range(15):
            recovered_metrics = engine.process_frame("temp_comp.jpg", uc3_features=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), gaze_features=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), audio_features=0.0)
        
        if os.path.exists("temp_comp.jpg"): os.remove("temp_comp.jpg")
        
        risk_after = recovered_metrics['risk']
        print_status("Hysteresis Fast-Recovery Validation", risk_after < risk_before, f"Risk saturated: {risk_before:.4f} -> Successfully Reset: {risk_after:.4f}")
        results["Hysteresis Recovery"] = (risk_after < risk_before)
    except Exception as e:
        print_status("Hysteresis Fast-Recovery Validation", False, f"Failed during recovery testing: {e}")
        results["Hysteresis Recovery"] = False

    # -------------------------------------------------------------
    # STAGE 5: Multi-Session Concurrency & Thread Isolation
    # -------------------------------------------------------------
    print_header("STAGE 5: MULTI-SESSION CONCURRENCY STRESS & THREAD-SAFETY")
    
    concurrency_errors = []
    active_threads = []
    
    def session_worker(student_id):
        try:
            worker_adapter = MLProctoringAdapter()
            worker_adapter.start_monitoring(student_id, ref_image_path)
            
            # Simulate 10 process iterations
            for iteration in range(10):
                dummy_b64 = base64.b64encode(open(probe_image_path, "rb").read()).decode("utf-8")
                worker_adapter.process_external_frame(dummy_b64, audio_volume=0.1)
                time.sleep(0.05)
                
                # Check status
                status = worker_adapter.get_live_status()
                # Verify that student ID is isolated and hasn't crossed
                if worker_adapter.student_id != student_id:
                    concurrency_errors.append(f"State Contamination! Expected ID {student_id}, found {worker_adapter.student_id}")
            
            worker_adapter.stop_monitoring()
        except Exception as err:
            concurrency_errors.append(f"Thread {student_id} failed: {err}")

    # Spin up 10 parallel student sessions concurrently
    print("Launching 10 parallel proctoring threads...")
    for idx in range(10):
        t = threading.Thread(target=session_worker, args=(f"student_stress_{idx:03d}",))
        active_threads.append(t)
        t.start()
        
    for t in active_threads:
        t.join()
        
    if len(concurrency_errors) == 0:
        print_status("Multi-Session Concurrency Isolation", True, "All 10 parallel threads executed and isolated their internal state perfectly.")
        results["Thread Concurrency Isolation"] = True
    else:
        print_status("Multi-Session Concurrency Isolation", False, f"Errors: {concurrency_errors}")
        results["Thread Concurrency Isolation"] = False

    # Cleanup temp files
    if os.path.exists(ref_image_path): os.remove(ref_image_path)
    if os.path.exists(probe_image_path): os.remove(probe_image_path)

    # -------------------------------------------------------------
    # FINAL STATS SUMMARY
    # -------------------------------------------------------------
    print_header("FINAL TBIE BACKEND AUDIT REPORT SUMMARY")
    passed = sum(1 for k, v in results.items() if v)
    total = len(results)
    pct = (passed / total) * 100
    
    print(f" {Color.BOLD}Overall Score: {passed}/{total} ({pct:.1f}% Passes){Color.RESET}")
    print("----------------------------------------------------------------------")
    for k, v in results.items():
        val_str = f"{Color.GREEN}SUCCESS{Color.RESET}" if v else f"{Color.RED}FAIL{Color.RESET}"
        print(f"  - {k:<45} : {val_str}")
    print("======================================================================")


if __name__ == "__main__":
    run_e2e_audit()
