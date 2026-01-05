# ml/uc1_identity/train.py

import os
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from tqdm import tqdm

from ml.uc1_identity.models.resnet_embedder import ResNetEmbedder
from ml.uc1_identity.datasets.uc1_triplet_dataset import UC1TripletDataset
from ml.uc1_identity.losses.triplet_loss import TripletLoss


# -----------------------------
# CONFIG (LOCKED FOR NOW)
# -----------------------------
TRIPLET_FILE = "ml/uc1_identity/datasets/triplets_train.txt"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SANITY_OVERFIT = True          # MUST RUN FIRST
SANITY_TRIPLETS = 1000         # Small subset


def main():
    print(f"Using device: {DEVICE}")

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = UC1TripletDataset(TRIPLET_FILE)

    if SANITY_OVERFIT:
        print("⚠️ SANITY OVERFIT MODE ENABLED")
        dataset = Subset(dataset, range(SANITY_TRIPLETS))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = ResNetEmbedder(
        embedding_dim=EMBEDDING_DIM,
        pretrained=True
    ).to(DEVICE)

    # -----------------------------
    # Loss & Optimizer
    # -----------------------------
    criterion = TripletLoss(margin=0.2)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # -----------------------------
    # Training Loop
    # -----------------------------
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for anchor, positive, negative in progress:
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            optimizer.zero_grad()

            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            loss = criterion(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    # -----------------------------
    # Save checkpoint
    # -----------------------------
    os.makedirs("ml/uc1_identity/models/checkpoints", exist_ok=True)
    torch.save(
        model.state_dict(),
        "ml/uc1_identity/models/checkpoints/uc1_resnet_embedder.pth"
    )

    print("✅ Training complete. Model checkpoint saved.")


if __name__ == "__main__":
    main()
