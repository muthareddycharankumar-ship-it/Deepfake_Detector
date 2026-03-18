import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import (
    AutoImageProcessor,
    SiglipForImageClassification,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

_face_model     = None
_face_processor = None
_ai_model       = None
_ai_processor   = None

def load_model():
    global _face_model, _face_processor
    if _face_model is None:
        print("Loading face deepfake model (prithivMLmods/deepfake-detector-model-v1)...")
        _face_processor = AutoImageProcessor.from_pretrained('prithivMLmods/deepfake-detector-model-v1')
        _face_model     = SiglipForImageClassification.from_pretrained('prithivMLmods/deepfake-detector-model-v1').to(device)
        _face_model.eval()
        print(f"Face model labels: {_face_model.config.id2label}")
    return _face_model, _face_processor

def load_ai_model():
    global _ai_model, _ai_processor
    if _ai_model is None:
        print("Loading AI-vs-real model (dima806/ai_vs_real_image_detection)...")
        _ai_processor = AutoFeatureExtractor.from_pretrained('dima806/ai_vs_real_image_detection')
        _ai_model     = AutoModelForImageClassification.from_pretrained('dima806/ai_vs_real_image_detection').to(device)
        _ai_model.eval()
        print(f"AI model labels: {_ai_model.config.id2label}")
    return _ai_model, _ai_processor
