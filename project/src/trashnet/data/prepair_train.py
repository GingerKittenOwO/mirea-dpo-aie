from pathlib import Path
from typing import Tuple

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

import numpy as np
import random





def make_loaders(
        path: Path,
        tf_train,
        tf_test,
        val_size: float = 0.1,
        test_size: float | None = 0.1,
        random_state: int = 42,
        batch_size: int = 128,
        use_pin_mem: bool = torch.cuda.is_available()
        ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    
    train_full = ImageFolder(path,transform=tf_train)
    test_full = ImageFolder(path,transform=tf_test)
    num_classes = len(train_full.classes)
    targets = np.array(train_full.targets)

    train_idx, temp_idx = train_test_split(
        range(len(train_full.targets)),
        test_size=val_size+(0.0 if test_size == None else test_size),
        stratify=targets,
        random_state=random_state,
        shuffle=True
    )
    if test_size!=None and test_size!=0.0:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_size/(val_size+test_size),              
            stratify=targets[temp_idx], 
            random_state=random_state,
            shuffle=True
        )
    else:
        val_idx = temp_idx
        test_idx = []

    train_dataset = Subset(train_full, train_idx)
    val_dataset   = Subset(test_full, val_idx)
    test_dataset  = Subset(test_full, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_pin_mem, generator=torch.Generator().manual_seed(random_state))
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_pin_mem)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_pin_mem)

    return (train_loader, val_loader, test_loader, num_classes)


def set_seed(seed: int = 42) -> None:
    # Фиксируем seed для воспроизводимости (насколько это возможно).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Более детерминированное поведение (может чуть замедлить).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False