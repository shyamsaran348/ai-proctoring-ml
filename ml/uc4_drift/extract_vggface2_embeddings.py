import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

DEVICE = torch.device("cpu")  # Force CPU (MacBook safe)

UC1_CHECKPOINT = "ml/uc1_identity/models/checkpoints/uc1_resnet_embedder.pth"
VGGFACE2_ROOT = "ml/data/raw/vggface2/train"
OUTPUT_DIR = "ml/uc4_drift/datasets"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# LOAD UC1 MODEL (USE EXACT TRAINED ARCHITECTURE)
# ---------------------------------------------------

from ml.uc1_identity.models.resnet_embedder import ResNetEmbedder

model = ResNetEmbedder()
checkpoint = torch.load(UC1_CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

print("✅ UC1 model loaded successfully.")

# ---------------------------------------------------
# IMAGE TRANSFORM (Match UC1 training)
# ---------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------------------------------
# EXTRACTION
# ---------------------------------------------------

all_embeddings = []
all_ids = []

identity_folders = sorted(os.listdir(VGGFACE2_ROOT))

print("🚀 Starting embedding extraction...")

for identity_idx, identity_name in enumerate(tqdm(identity_folders)):

    identity_path = os.path.join(VGGFACE2_ROOT, identity_name)

    if not os.path.isdir(identity_path):
        continue

    image_files = os.listdir(identity_path)

    for img_name in image_files:
        img_path = os.path.join(identity_path, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                embedding = model(img)

            embedding = embedding.cpu().numpy().squeeze()

            all_embeddings.append(embedding)
            all_ids.append(identity_idx)

        except Exception as e:
            print(f"⚠️ Skipping {img_path} due to error: {e}")
            continue

# ---------------------------------------------------
# SAVE
# ---------------------------------------------------

embeddings_array = np.array(all_embeddings)
identity_array = np.array(all_ids)

np.save(os.path.join(OUTPUT_DIR, "vggface2_embeddings.npy"), embeddings_array)
np.save(os.path.join(OUTPUT_DIR, "vggface2_identity_ids.npy"), identity_array)

print("\n✅ Extraction Complete.")
print("Saved embeddings shape:", embeddings_array.shape)
print("Saved identity ids shape:", identity_array.shape)
print("Unique identities:", len(set(identity_array)))