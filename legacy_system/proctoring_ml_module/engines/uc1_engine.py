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

class ScoreNormalizer:
    """
    Applies T-Norm (Target Normalization) to calibrate similarity scores.
    Assumes an impostor distribution N(mean, std).
    """
    def __init__(self, mean=0.3, std=0.1):
        self.mean = mean
        self.std = std

    def normalize(self, raw_score):
        # Z-Norm
        z_score = (raw_score - self.mean) / (self.std + 1e-6)
        
        # Sigmoid squash to keep in 0-1 range for downstream LSTM
        # Centering: decision boundary approx 0.6 in raw score -> Z=3.0 -> Sigmoid ~0.95
        # We want 0.3 (mean impostor) -> Z=0 -> Sigmoid 0.5? No, impostor should be low.
        # Let's shift Z. If Z=0 (typical impostor), we want score ~0.1.
        # If we use raw sigmoid: 1 / (1 + exp(-z)). Z=0 -> 0.5.
        # Maybe simply return the Z-score and let UC2/UC5 handle it?
        # But UC2 expects 0-1 likely (based on previous logs/code).
        # Let's map: 
        # range [0.0, 1.0].
        # Clip Z to [-3, 3] then map to [0,1]?
        # Or just use raw score? The prompt asks for T-Norm.
        # "Used before feeding UC2".
        # Let's use a calibrated sigmoid that pushes impostors down.
        # calibrated = 1 / (1 + exp(-(z_score - 2.0))) 
        # If raw=0.3 (meam), z=0, exp(2) ~7.3, 1/8.3 ~ 0.12 (Low, Good).
        # If raw=0.7 (match), z=4, exp(-2) ~0.13, 1/1.13 ~ 0.88 (High, Good).
        normalized = 1.0 / (1.0 + np.exp(-(z_score - 2.0)))
        return float(normalized)

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
        
        # FP16 Efficiency check
        if self.device.type != 'cpu':
             self.model.half()

        # Score Normalizer
        self.normalizer = ScoreNormalizer()

        # Caching
        self.enrollment_cache = {}

        # Preprocessing (Standard ImageNet)
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
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
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
        # 1. Check Cache (if input is path)
        if isinstance(image_input, str) and image_input in self.enrollment_cache:
            return self.enrollment_cache[image_input]

        img = self._prepare_image(image_input)
        if img is None:
            return None 

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        if self.device.type != 'cpu':
            img_tensor = img_tensor.half()
        
        # 2. Optimized Inference
        with torch.inference_mode():
            emb = self.model(img_tensor)
        
        # Cache Result (Small RAM footprint for embeddings)
        if isinstance(image_input, str):
            self.enrollment_cache[image_input] = emb

        return emb

    def compute_similarity(self, emb1, emb2):
        """
        Compute Normalized Cosine Similarity.
        Args:
            emb1: (1, D) tensor
            emb2: (1, D) tensor
        Returns:
            similarity: float (0 to 1, calibrated)
        """
        if emb1 is None or emb2 is None:
            return 0.0
            
        raw_sim = torch.mm(emb1, emb2.T).item()
        
        # Apply T-Norm Calibration
        norm_sim = self.normalizer.normalize(raw_sim)
        
        return norm_sim

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
