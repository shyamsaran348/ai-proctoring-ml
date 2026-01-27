import cv2
import pickle
import numpy as np
import os
import sys
import torch
from django.conf import settings

# Integration: Import UC1 Engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from proctoring_ml_module.engines.uc1_engine import UC1Engine

# Singleton Engine reuse
_UC1_ENGINE = None

def get_engine():
    global _UC1_ENGINE
    if _UC1_ENGINE is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../proctoring_ml_module/config.yaml'))
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        _UC1_ENGINE = UC1Engine(config)
    return _UC1_ENGINE

def verify_identity(student_id: str, threshold: float = 0.6) -> bool:
    """Capture one live frame and compare against stored embedding. Returns True if verified."""
    
    # Load embedding
    pickle_path = os.path.join(settings.BASE_DIR, 'static', 'uploads', 'embeddings', f'{student_id}.pkl')
    if not os.path.exists(pickle_path):
        # Fallback: Try to generate on the fly if reference image exists
        try:
            from .generate_embedding import generate_and_store_embedding
            pickle_path = generate_and_store_embedding(student_id)
        except Exception as e:
            print(f"Error auto-generating embedding: {e}")
            raise FileNotFoundError('Stored embedding not found.')

    with open(pickle_path, 'rb') as f:
        stored_embedding_np = pickle.load(f)
        # stored_embedding_np is (256,) numpy array
    
    if stored_embedding_np is None:
        return False
        
    engine = get_engine()
    
    # Convert stored to Tensor for computation
    # (1, 256)
    stored_tensor = torch.tensor(stored_embedding_np).unsqueeze(0).to(engine.device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam')
        
    print('[INFO] Press SPACE to capture face for verification.')
    verified = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Show Feed
            cv2.imshow('Verify - Press SPACE', frame)
            key = cv2.waitKey(1)
            
            if key % 256 == 32:  # SPACE
                try:
                    # Capture Frame -> UC1 Embedding
                    # Frame is BGR from OpenCV, convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    live_emb = engine.get_embedding(frame_rgb) # (1, 256)
                    
                    if live_emb is not None:
                        # Compute Cosine Sim
                        similarity = engine.compute_similarity(stored_tensor, live_emb)
                        print(f"[Verification] Similarity Score: {similarity:.4f} (Threshold: {threshold})")
                        verified = similarity > threshold
                    else:
                        print("[Verification] Could not extract embedding from live frame.")
                        verified = False
                        
                except Exception as e:
                    print(f"Verification Error: {e}")
                    verified = False
                break
            elif key % 256 == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    return verified


__all__ = ['verify_identity']
