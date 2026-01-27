import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys

# Add root to path to find models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from proctoring_ml_module.models.architectures import ResNetEmbedder

class UC1Engine:
    def __init__(self, config):
        """
        Args:
            config: Dictionary containing 'uc1' model config and 'inference' settings.
        """
        self.device = torch.device(config.get('inference', {}).get('device', 'cpu'))
        self.model_path = config['models']['uc1']['path']
        self.embedding_dim = config['models']['uc1'].get('embedding_dim', 256)
        
        # Initialize model
        self.model = ResNetEmbedder(embedding_dim=self.embedding_dim, pretrained=False)
        self.load_weights()
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing (Standard ImageNet)
        # Note: Preprocessing should match training (VGGFace2 usually uses ImageNet stats or similar)
        # We assume config provides these or we use defaults.
        mean = config.get('inference', {}).get('enrollment_transform', {}).get('mean', [0.485, 0.456, 0.406])
        std = config.get('inference', {}).get('enrollment_transform', {}).get('std', [0.229, 0.224, 0.225])
        size = config.get('inference', {}).get('enrollment_transform', {}).get('resize', 224)

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def load_weights(self):
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            # Handle potential DataParallel wrapping or key mismatches if necessary
            # For now, assume direct match or 'state_dict' key
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Simple fix for 'module.' prefix if trained with DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"[UC1] Loaded weights from {self.model_path}")
        else:
            print(f"[UC1] WARNING: Checkpoint not found at {self.model_path}. Using random weights.")

    def get_embedding(self, image_input):
        """
        Compute embedding for a single image.
        Args:
            image_input: PIL Image, or numpy array (RGB), or path to image.
        Returns:
            embedding: torch.Tensor of shape (1, embedding_dim)
        """
        img = self._prepare_image(image_input)
        if img is None:
            return None # Handle loading error

        img_tensor = self.transform(img).unsqueeze(0).to(self.device) # (1, 3, 224, 224)
        
        with torch.no_grad():
            emb = self.model(img_tensor) # (1, 256) normalized
        
        return emb

    def compute_similarity(self, emb1, emb2):
        """
        Compute Cosine Similarity between two embeddings.
        Args:
            emb1: (1, D) tensor
            emb2: (1, D) tensor
        Returns:
            similarity: float (-1 to 1)
        """
        if emb1 is None or emb2 is None:
            return 0.0
            
        # Since embeddings are L2 normalized by the model:
        # Cosine Similarity = Dot Product
        return torch.mm(emb1, emb2.T).item()

    def _prepare_image(self, image_input):
        try:
            if isinstance(image_input, str):
                return Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                return Image.fromarray(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                return image_input.convert('RGB')
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
        except Exception as e:
            print(f"[UC1] Error processing image: {e}")
            return None
