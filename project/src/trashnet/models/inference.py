import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassRecall, 
    MulticlassF1Score, 
    MulticlassPrecision, 
    MulticlassAccuracy)
from torch.utils.data import DataLoader
from typing import Dict

def inference(model: nn.Module, tensor: torch.Tensor):
    model.eval()
    #model.to(device)
    logits = model(tensor)
    return (torch.argmax(logits, dim=1).item(),logits[0].tolist())


@torch.no_grad()
def metrics_on_loader(model: nn.Module,
                    loader: DataLoader, 
                    num_classes: int,
                    device: torch.device = None,
                    )-> Dict[str, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    recall = MulticlassRecall(num_classes=num_classes,average="macro").to(device)
    f1score = MulticlassF1Score(num_classes=num_classes,average="macro").to(device)
    precision = MulticlassPrecision(num_classes=num_classes,average="macro").to(device)
    accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)

    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        recall.update(preds,labels)
        f1score.update(preds,labels)
        precision.update(preds,labels)
        accuracy.update(preds,labels)

    metrics = {
        'accuracy': accuracy.compute().item(),
        'recall_macro': recall.compute().item(),
        'precision_macro': precision.compute().item(),
        'f1_macro': f1score.compute().item(),
    }
    
    return metrics

