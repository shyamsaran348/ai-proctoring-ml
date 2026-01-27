import cv2
import numpy as np

class UC3FeatureExtractor:
    def __init__(self, face_detector, pose_estimator):
        self.face_detector = face_detector
        self.pose_estimator = pose_estimator
        self.prev_gray = None

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

    def extract_frame_features(self, frame):
        h, w = frame.shape[:2]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Face detection (measurement only)
        face_conf, bbox = self.face_detector(frame)

        if bbox is not None:
            x, y, bw, bh = bbox
            area_ratio = (bw * bh) / (w * h)
            yaw, pitch, roll = self.pose_estimator(frame, bbox)
        else:
            area_ratio = 0.0
            yaw, pitch, roll = 0.0, 0.0, 0.0

        motion_energy = self.compute_motion_energy(frame_gray)

        return np.array([
            face_conf,
            area_ratio,
            yaw,
            pitch,
            roll,
            motion_energy
        ], dtype=np.float32)


import argparse
import os
import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser(description="UC3 Feature Extraction")
    parser.add_argument("--input", required=True, help="Path to UC3 raw data")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--sequence-length", type=int, default=60)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    T = args.sequence_length

    os.makedirs(output_dir, exist_ok=True)

    all_sequences = []
    all_labels = []

    # Example: folder-based labeling
    label_map = {
        "present": 1,
        "absent": 0
    }

    for label_name, label_value in label_map.items():
        folder = os.path.join(input_dir, label_name)
        if not os.path.exists(folder):
            continue

        for video_file in os.listdir(folder):
            if not video_file.endswith(".mp4"):
                continue

            video_path = os.path.join(folder, video_file)
            print(f"[INFO] Processing {video_path}")

            cap = cv2.VideoCapture(video_path)
            features = []

            extractor = UC3FeatureExtractor(
                face_detector=your_face_detector,
                pose_estimator=your_pose_estimator
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                x_t = extractor.extract_frame_features(frame)
                features.append(x_t)

            cap.release()

            features = np.array(features)
            if len(features) < T:
                continue

            sequences = build_sequences(features, T)
            all_sequences.append(sequences)
            all_labels.extend([label_value] * len(sequences))

    X = np.concatenate(all_sequences, axis=0)
    y = np.array(all_labels)

    np.save(os.path.join(output_dir, "sequences.npy"), X)
    np.save(os.path.join(output_dir, "labels.npy"), y)

    print(f"[DONE] Saved {X.shape[0]} sequences")

if __name__ == "__main__":
    main()
