from torchvision import transforms
from PIL import Image
from pathlib import Path
from torch import Tensor
from fastapi import UploadFile

IMAGENET_IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_single_image(path: Path, transform)-> Tensor:
    try:
        image = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Не удалось загрузить изображение по пути: {path}")
    tensor = transform(image)
    return tensor.unsqueeze(0)

def load_single_image_from_file(file: UploadFile, transform)-> Tensor:
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        print(f"Не удалось загрузить изображение из файла: {file}")
    tensor = transform(image)
    return tensor.unsqueeze(0)
    


def get_inference_transforms() -> transforms.transforms.Compose:
    return transforms.Compose([
        transforms.Resize(IMAGENET_IMG_SIZE+32),
        transforms.CenterCrop(IMAGENET_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)
    ])

def get_augmented_train_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGENET_IMG_SIZE+32),
        transforms.CenterCrop(IMAGENET_IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.4,             # Probability of applying the transform
            scale=(0.02, 0.20),# Range of proportion of erased area against the input image
            ratio=(0.3, 3.3),  # Range of aspect ratio of erased area
            value=0,           # Value to fill erased pixels (0 for black, or a tuple for RGB)
            inplace=False
        ),
        transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
    ])


