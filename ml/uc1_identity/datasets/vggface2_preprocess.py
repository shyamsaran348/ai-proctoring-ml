import time
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from facenet_pytorch import MTCNN

# =========================
# CONFIG
# =========================

RAW_DATA_DIR = Path("ml/data/raw/vggface2")
PROCESSED_DATA_DIR = Path("ml/data/processed/vggface2")

SPLITS = ["train", "val"]   # you can temporarily use ["train"]
IMAGE_SIZE = 224

PRINT_EVERY = 100  # print progress every N images

# =========================
# DEVICE & DETECTOR
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,
    device=device
)

# =========================
# UTILS
# =========================

def load_image(image_path: Path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot load image: {image_path} | {e}")
        return None


def detect_face(img: Image.Image):
    try:
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    except RuntimeError as e:
        # Known MTCNN failure case: empty detections
        return None

    if boxes is None or len(boxes) == 0:
        return None

    # Select largest face
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = np.argmax(areas)

    return {
        "box": boxes[idx],
        "landmarks": landmarks[idx]
    }


def crop_face(img: Image.Image, box):
    x1, y1, x2, y2 = map(int, box)
    return img.crop((x1, y1, x2, y2))


def align_face(img: Image.Image, landmarks):
    left_eye, right_eye = landmarks[0], landmarks[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(np.arctan2(dy, dx))

    return img.rotate(angle, resample=Image.BICUBIC, expand=True)


def process_image(image_path: Path, output_path: Path):
    # Resume support: skip if already processed
    if output_path.exists():
        return "skipped"

    img = load_image(image_path)
    if img is None:
        return "failed"

    detection = detect_face(img)
    if detection is None:
        return "failed"

    # 1️⃣ Crop FIRST
    cropped = crop_face(img, detection["box"])

    # 2️⃣ Align cropped face
    aligned = align_face(cropped, detection["landmarks"])

    # 3️⃣ Resize
    resized = aligned.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path)

    return "processed"


def count_images(input_dir: Path):
    total = 0
    for identity_dir in input_dir.iterdir():
        if identity_dir.is_dir():
            total += len(list(identity_dir.iterdir()))
    return total


def format_time(seconds: float):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def preprocess_split(split: str):
    input_dir = RAW_DATA_DIR / split
    output_dir = PROCESSED_DATA_DIR / split

    total_images = count_images(input_dir)
    processed_count = 0

    print(f"\n[INFO] {split.upper()} split")
    print(f"[INFO] Total images found: {total_images}")

    start_time = time.time()

    for identity_dir in input_dir.iterdir():
        if not identity_dir.is_dir():
            continue

        for img_path in identity_dir.iterdir():
            out_path = output_dir / identity_dir.name / img_path.name
            result = process_image(img_path, out_path)

            if result != "skipped":
                processed_count += 1

            elapsed = time.time() - start_time
            avg_time = elapsed / max(processed_count, 1)
            remaining = (total_images - processed_count) * avg_time

            if processed_count % PRINT_EVERY == 0 or processed_count == total_images:
                percent = (processed_count / total_images) * 100
                print(
                    f"[PROGRESS] {split}: "
                    f"{processed_count}/{total_images} "
                    f"({percent:.2f}%) | "
                    f"Elapsed: {format_time(elapsed)} | "
                    f"ETA: {format_time(remaining)}"
                )


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    for split in SPLITS:
        preprocess_split(split)
