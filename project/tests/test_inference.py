from __future__ import annotations

from pathlib import Path
import torch

from  trashnet.models.resnet18 import load_weights, build_resnet18_raw
from trashnet.models.inference import inference
from trashnet.data.transforms import get_inference_transforms, load_single_image


ROOT_DIR = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_single_image_prediction():

    weights = load_weights(ROOT_DIR/"artifacts"/"best_classifier.pt")
    
    model = build_resnet18_raw().to(DEVICE)
    model.load_state_dict(weights)
    transform = get_inference_transforms()
    tensor = load_single_image(ROOT_DIR /"data"/"test.jpg", transform).to(DEVICE)
    result = inference(model,tensor)

    assert 0<=result[0]<=5


