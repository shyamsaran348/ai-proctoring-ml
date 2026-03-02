import os
import cv2
import argparse
import numpy as np


# ============================================================
# ---------------- Feature Extraction -------------------------
# ============================================================

class UC3FeatureExtractor:
    def __init__(self):
        self.prev_gray = None
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def compute_motion_energy(self, frame_gray):
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, frame_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        energy = np.mean(mag)

        self.prev_gray = frame_gray
        return energy

    def simple_pose_stub(self):
        # Placeholder — replace later with real head pose model
        return 0.0, 0.0, 0.0

    def extract_frame_features(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, bw, bh = faces[0]
            face_conf = 1.0
            area_ratio = (bw * bh) / (w * h)
            yaw, pitch, roll = self.simple_pose_stub()
        else:
            face_conf = 0.0
            area_ratio = 0.0
            yaw, pitch, roll = 0.0, 0.0, 0.0

        motion_energy = self.compute_motion_energy(gray)

        return np.array([
            face_conf,
            area_ratio,
            yaw,
            pitch,
            roll,
            motion_energy
        ], dtype=np.float32)


# ============================================================
# ---------------- Sequence Builder ---------------------------
# ============================================================

def build_sequences(features, T=60):
    sequences = []
    for i in range(0, len(features) - T + 1, T):
        sequences.append(features[i:i + T])
    if len(sequences) == 0:
        return None
    return np.stack(sequences)


# ============================================================
# ---------------- Main Dataset Builder ----------------------
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="UC3 Dataset Builder")
    parser.add_argument("--input", required=True, help="Path to uc3_raw directory")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--sequence-length", type=int, default=60)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    T = args.sequence_length

    os.makedirs(output_dir, exist_ok=True)

    extractor = UC3FeatureExtractor()

    label_map = {
        "present": 1,
        "absent": 0
    }

    all_sequences = []
    all_labels = []

    for label_name, label_value in label_map.items():
        class_folder = os.path.join(input_dir, label_name)

        if not os.path.exists(class_folder):
            print(f"[WARNING] Folder not found: {class_folder}")
            continue

        for video_file in os.listdir(class_folder):
            if not video_file.endswith(".mp4"):
                continue

            video_path = os.path.join(class_folder, video_file)
            print(f"[INFO] Processing {video_path}")

            cap = cv2.VideoCapture(video_path)
            features = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                feat = extractor.extract_frame_features(frame)
                features.append(feat)

            cap.release()

            features = np.array(features)

            if len(features) < T:
                print(f"[SKIP] Too short: {video_file}")
                continue

            sequences = build_sequences(features, T)

            if sequences is not None:
                all_sequences.append(sequences)
                all_labels.extend([label_value] * len(sequences))

    if len(all_sequences) == 0:
        print("No sequences generated. Check videos.")
        return

    X = np.concatenate(all_sequences, axis=0)
    y = np.array(all_labels)

    # ======================================================
    # -------- Normalize Features Across Dataset ----------
    # ======================================================

    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1)) + 1e-6

    X = (X - mean) / std

    # ======================================================
    # ---------------- Save Dataset ------------------------
    # ======================================================

    np.save(os.path.join(output_dir, "sequences.npy"), X)
    np.save(os.path.join(output_dir, "labels.npy"), y)
    np.save(os.path.join(output_dir, "feature_mean.npy"), mean)
    np.save(os.path.join(output_dir, "feature_std.npy"), std)

    print("\n==============================")
    print(f"Dataset Saved to: {output_dir}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("==============================\n")


if __name__ == "__main__":
    main()