from __future__ import annotations

import os
import tempfile
from pathlib import Path
import torch
from torch import nn

from trashnet.data.prepair_train import make_loaders, set_seed
from  trashnet.models.resnet18 import build_resnet18, get_resnet18_weights
from trashnet.data.transforms import get_augmented_train_transforms, get_inference_transforms
from trashnet.models.train import fit, set_requires_grad


ROOT_DIR = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_train_resnet18():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RANDOM_STATE = 42

    set_seed(RANDOM_STATE)

    # Загрузка данных
    tf_train = get_augmented_train_transforms()
    tf_test = get_inference_transforms()
    train_loader, val_loader, test_loader, num_classes = make_loaders(
        path=ROOT_DIR / "data"/"trashnet_sample",
        tf_train=tf_train,
        tf_test=tf_test,
        val_size=0.5,
        test_size=None,
        random_state=RANDOM_STATE,
        )

    # Загрузка ResNet18
    weights = get_resnet18_weights()
    model = build_resnet18(weights=weights, num_classes=num_classes).to(DEVICE)

    #------------------
    #-----ОБУЧЕНИЕ-----
    #------------------
    # Сначала обучаем только "голову"
    set_seed()
    # freeze всё
    set_requires_grad(model, False)
    # размораживаем только голову
    set_requires_grad(model.fc, True)
    optimizer_head_only = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    hist_head = fit(model,
                     train_loader,
                       val_loader, 
                       optimizer_head_only, 
                       criterion, 
                       epochs=1, 
                       verbose=False,)



    # Дообучаем модель
    set_seed()
    criterion = nn.CrossEntropyLoss()
    set_requires_grad(model.layer4, True)
    set_requires_grad(model.fc, True)

    # param groups: backbone меньше, head больше
    params = [
        {"params": model.layer4.parameters(), "lr": 1e-3},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ]
    optimizer_ft = torch.optim.Adam(params, weight_decay=0.0)

    
    hist_ft = fit(model,
                train_loader, 
                val_loader,
                optimizer_ft, 
                criterion, 
                epochs=1, 
                verbose=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "model_test.pt")
        torch.save(model.state_dict(), file_path)
        assert os.path.isfile(file_path)