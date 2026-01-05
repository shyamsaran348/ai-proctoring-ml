# ml/uc1_identity/datasets/uc1_triplet_dataset.py

import os
from typing import Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UC1TripletDataset(Dataset):
    """
    UC1 Triplet Dataset
    ------------------
    Loads (anchor, positive, negative) image triplets
    generated during Phase 1.

    This dataset:
    - Does NOT know identities
    - Does NOT know enrollment logic
    - Simply trusts the triplet files
    """

    def __init__(
        self,
        triplet_file: str,
        image_size: int = 224
    ):
        """
        Args:
            triplet_file: Path to triplets_train.txt / val / test
            image_size: Input size expected by the model
        """
        self.triplets = self._load_triplets(triplet_file)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_triplets(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Triplet file not found: {path}")

        triplets = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Phase-1 triplets are comma-separated
                parts = line.split(",")

                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid triplet format (expected 3 paths): {line}"
                    )

                triplets.append(tuple(parts))

        if len(triplets) == 0:
            raise RuntimeError("No valid triplets found.")

        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)

        return anchor, positive, negative
