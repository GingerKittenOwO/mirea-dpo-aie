import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from pathlib import Path
from trashnet.utils.settings import PROJECT_ROOT


def load_weights(path):
    try:
        return torch.load(path, weights_only=True)
    except Exception as e:
        print(f"Не удалось загрузить веса из файла по пути: {path}")

def get_resnet18_weights():
    # Пытаемся взять предобученные веса. Если не получилось – вернем None.
    try:
        w = ResNet18_Weights.DEFAULT
        # иногда ошибка возникает не здесь, а при фактической загрузке весов;
        # но на практике этого достаточно как "правильный путь".
        return w
    except Exception as e:
        print("Не удалось получить веса ResNet18_Weights.DEFAULT. Причина:", repr(e))
        return None

def build_resnet18_raw(num_classes: int = 6) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_resnet18(weights, num_classes: int = 6) -> nn.Module:
    # Внимание: реальная загрузка весов может требовать интернет.
    # Если не получается – используйте weights=None.
    try:
        model = resnet18(weights=weights)
    except Exception as e:
        print("Не удалось загрузить предобученные веса. Переходим на weights=None. Причина:", repr(e))
        model = resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model




