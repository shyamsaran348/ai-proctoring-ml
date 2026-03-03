"""
webcam_demo.py
--------------
Standalone real-webcam live proctoring demo using the full UC1-UC5 pipeline.

Controls:
  E  — Enroll (capture reference from current webcam frame)
  Q  — Quit (saves risk log to webcam_session_log.csv)

Display:
  Top-left overlay: UC1 similarity, UC2 instability, UC3 presence, UC4 drift, Risk
  Color-coded risk bar: green (<0.3) → amber (0.3-0.7) → red (>0.7)
  Bottom of window: session status banner
"""

import os
import sys
import cv2
import csv
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from proctoring_ml_module.engines.proctoring_engine import ProctoringEngine
from ml.uc3_presence.features.extract_features import UC3FeatureExtractor

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_NAME  = "DevProctor — Live Session"
LOG_FILE     = str(PROJECT_ROOT / "webcam_session_log.csv")
FRAME_RATE   = 5        # Process every Nth frame for ML (display at full rate)
SIGNAL_W     = 300      # Width of the sidebar overlay
SIGNAL_H     = 180      # Height of the signal panel

COLOR_BG     = (30,  41,  59)   # Slate 800
COLOR_GREEN  = (16, 185, 129)   # Emerald 500
COLOR_AMBER  = (245, 158, 11)   # Amber 500
COLOR_RED    = (239,  68,  68)  # Red 500
COLOR_BLUE   = (99, 102, 241)   # Indigo 500
COLOR_WHITE  = (241, 245, 249)
COLOR_MUTED  = (100, 116, 139)


def risk_color(risk: float):
    if risk < 0.3:
        return COLOR_GREEN
    elif risk < 0.7:
        return COLOR_AMBER
    return COLOR_RED


def draw_signal_bar(img, x, y, label, value, bar_color, value_fmt='.3f'):
    """Draw a labelled progress bar for a signal value ∈ [0,1]."""
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, COLOR_MUTED, 1, cv2.LINE_AA)
    val_str = format(value, value_fmt)
    cv2.putText(img, val_str, (x + 195, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, COLOR_WHITE, 1, cv2.LINE_AA)
    # Background bar
    bar_h = 6; bar_y = y + 5; bar_w = 230
    cv2.rectangle(img, (x, bar_y), (x + bar_w, bar_y + bar_h), (55, 65, 81), -1)
    # Filled bar
    fill = max(1, int(bar_w * np.clip(value, 0.0, 1.0)))
    cv2.rectangle(img, (x, bar_y), (x + fill, bar_y + bar_h), bar_color, -1)


def overlay_panel(frame, metrics, enrolled: bool, frame_idx: int):
    """Composite the signal overlay panel onto the camera frame in place."""
    h, w = frame.shape[:2]

    # Semi-transparent sidebar background
    panel_x = w - SIGNAL_W - 12
    panel_y = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 10, panel_y - 5),
                  (w - 8, panel_y + SIGNAL_H + 10), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    if not enrolled:
        cv2.putText(frame, "Press  E  to Enroll",
                    (panel_x, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, COLOR_AMBER, 1, cv2.LINE_AA)
        cv2.putText(frame, "Press  Q  to Quit",
                    (panel_x, panel_y + 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, COLOR_MUTED, 1, cv2.LINE_AA)
        return

    risk = metrics.get('risk', 0.0)
    rc   = risk_color(risk)

    # Title
    cv2.putText(frame, "DEVPROCTOR  LIVE", (panel_x, panel_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (panel_x - 10, panel_y + 20),
                  (w - 8, panel_y + 21), COLOR_MUTED, -1)

    y = panel_y + 38
    draw_signal_bar(frame, panel_x, y, "UC1  Identity Sim",
                    metrics.get('uc1_similarity', 0), COLOR_BLUE)
    y += 26
    draw_signal_bar(frame, panel_x, y, "UC2  Instability",
                    metrics.get('uc2_instability', 0), COLOR_AMBER)
    y += 26
    draw_signal_bar(frame, panel_x, y, "UC3  Presence",
                    metrics.get('uc3_presence', 0.5), COLOR_GREEN)
    y += 26
    draw_signal_bar(frame, panel_x, y, "UC4  Drift",
                    metrics.get('uc4_drift', 0), COLOR_RED)
    y += 30

    # Risk headline
    cv2.putText(frame, f"RISK SCORE", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, COLOR_MUTED, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{risk:.4f}", (panel_x + 100, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, rc, 2, cv2.LINE_AA)
    y += 14
    # Risk bar (wider)
    cv2.rectangle(frame, (panel_x, y), (panel_x + 230, y + 10), (55, 65, 81), -1)
    fill = max(1, int(230 * np.clip(risk, 0, 1)))
    cv2.rectangle(frame, (panel_x, y), (panel_x + fill, y + 10), rc, -1)

    # Bottom: frame counter
    cv2.putText(frame, f"Frame {frame_idx}", (panel_x, panel_y + SIGNAL_H + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_MUTED, 1, cv2.LINE_AA)


def draw_status_banner(frame, text: str, color):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 32), (w, h), (15, 23, 42), -1)
    cv2.putText(frame, text, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    print("[WebcamDemo] Initializing ProctoringEngine...")
    engine    = ProctoringEngine()
    extractor = UC3FeatureExtractor()

    print("[WebcamDemo] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Exiting.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    enrolled   = False
    metrics    = {}
    risk_log   = []
    frame_idx  = 0
    ml_frame   = 0

    print(f"\n{'='*55}")
    print("  DEVPROCTOR — LIVE PROCTORING DEMO")
    print("  Press E to enroll | Press Q to quit")
    print(f"{'='*55}\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 900, 520)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame.")
                time.sleep(0.05)
                continue

            display = frame.copy()
            frame_idx += 1

            # ── ML Processing (throttled)
            if enrolled and frame_idx % FRAME_RATE == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # UC3 features (6D) from BGR frame
                    uc3_feats = extractor.extract_frame_features(frame)
                    metrics = engine.process_frame(frame_rgb, uc3_features=uc3_feats)
                    ml_frame += 1

                    risk_log.append({
                        'timestamp':       time.time(),
                        'frame':           frame_idx,
                        'uc1_similarity':  metrics.get('uc1_similarity', 0),
                        'uc2_instability': metrics.get('uc2_instability', 0),
                        'uc3_presence':    metrics.get('uc3_presence', 0.5),
                        'uc4_drift':       metrics.get('uc4_drift', 0),
                        'risk':            metrics.get('risk', 0),
                    })
                except Exception as e:
                    print(f"[WARN] Frame processing error: {e}")

            # ── Draw overlay
            overlay_panel(display, metrics, enrolled, ml_frame)

            # ── Status banner
            if not enrolled:
                draw_status_banner(display,
                    "Not enrolled — press E to capture reference frame", COLOR_AMBER)
            else:
                risk = metrics.get('risk', 0)
                if risk > 0.7:
                    draw_status_banner(display,
                        f"HIGH RISK DETECTED  ({risk:.3f}) — Possible integrity violation", COLOR_RED)
                elif risk > 0.3:
                    draw_status_banner(display,
                        f"Warning: Elevated risk  ({risk:.3f}) — monitoring closely", COLOR_AMBER)
                else:
                    draw_status_banner(display,
                        f"Monitoring active — all signals nominal  (Risk: {risk:.3f})", COLOR_GREEN)

            cv2.imshow(WINDOW_NAME, display)

            # ── Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e') or key == ord('E'):
                if not enrolled:
                    print("[WebcamDemo] Capturing enrollment frame...")
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        engine.start_session(frame_rgb)
                        extractor.prev_gray = None  # reset motion state
                        enrolled = True
                        metrics = {}
                        print("[WebcamDemo] ✅ Enrolled. Monitoring started.")
                    except Exception as e:
                        print(f"[ERROR] Enrollment failed: {e}")
                else:
                    print("[WebcamDemo] Already enrolled. Press Q to quit.")

            elif key == ord('q') or key == ord('Q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ── Save risk log
        if risk_log:
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=risk_log[0].keys())
                writer.writeheader()
                writer.writerows(risk_log)
            print(f"\n[WebcamDemo] Risk log saved → {LOG_FILE}")
            print(f"[WebcamDemo] Total ML frames processed: {ml_frame}")

            # Print summary
            risks = [r['risk'] for r in risk_log]
            print(f"\nSession Summary:")
            print(f"  Frames analysed : {ml_frame}")
            print(f"  Final risk      : {risks[-1]:.4f}")
            print(f"  Mean risk       : {np.mean(risks):.4f}")
            print(f"  Peak risk       : {max(risks):.4f}")
        else:
            print("[WebcamDemo] No frames processed (session not started).")

        print("[WebcamDemo] Goodbye.")


if __name__ == '__main__':
    main()
