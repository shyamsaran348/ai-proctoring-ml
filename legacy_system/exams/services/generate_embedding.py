# from deepface import DeepFace <-- DEPRECATED
import pickle
import os
import sys
import torch
import numpy as np
from django.conf import settings

# Integration: Import UC1 Engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from proctoring_ml_module.engines.uc1_engine import UC1Engine

# Singleton Engine to avoid reloading
_UC1_ENGINE = None

def get_engine():
    global _UC1_ENGINE
    if _UC1_ENGINE is None:
        # Load config from module root
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../proctoring_ml_module/config.yaml'))
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        _UC1_ENGINE = UC1Engine(config)
    return _UC1_ENGINE

def generate_and_store_embedding(student_id: str) -> str:
    """Generate face embedding for student's reference image and store under media/embeddings.
    Returns absolute path to stored embedding pickle file.
    """
    engine = get_engine()
    
    # Store embeddings under static/uploads/embeddings
    emb_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads', 'embeddings')
    os.makedirs(emb_dir, exist_ok=True)
    
    # Read reference image from static/uploads/students
    # Try multiple extensions or suffixes
    base_student_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads', 'students')
    possible_paths = [
        f'{student_id}_reference.jpg',
        f'{student_id}.jpg'
    ]
    
    img_path = None
    for p in possible_paths:
        full = os.path.join(base_student_dir, p)
        if os.path.exists(full):
            img_path = full
            break
            
    if not img_path:
        raise FileNotFoundError(f"Reference image for {student_id} not found.")

    # Compute Embedding using UC1 ResNet
    # Returns (1, 256) Tensor
    emb_tensor = engine.get_embedding(img_path)
    
    if emb_tensor is None:
        raise RuntimeError("UC1 Engine failed to extract embedding from image.")

    # Store as simple list or numpy for portability
    embedding = emb_tensor.cpu().detach().numpy().flatten()
    
    pickle_path = os.path.join(emb_dir, f'{student_id}.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(embedding, f)

    return pickle_path


__all__ = ['generate_and_store_embedding']
