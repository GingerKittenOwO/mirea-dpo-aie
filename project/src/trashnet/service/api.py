from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from time import perf_counter
from pathlib import Path
import torch

from fastapi import FastAPI, File,Form, Depends, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from torch import nn

from trashnet.data.prepair_train import make_loaders, set_seed
from trashnet.models.train import fit, set_requires_grad

from trashnet.data.transforms import get_augmented_train_transforms, get_inference_transforms, load_single_image_from_file
from trashnet.models.inference import inference
from trashnet.models.resnet18 import build_resnet18, build_resnet18_raw, get_resnet18_weights, load_weights

from trashnet.utils.settings import PROJECT_ROOT


# В общем... часть с запросом мне помог решить ИИ
#
# Из-за ограничения multipart/form-data и Swagger UI
# модель дополнительно преобразуется через as_form(),
# чтобы параметры отображались отдельными полями
# вместе с UploadFile.
#
# Насколько я понял эта проблема возникает
# сугубо из-за попытки отправлять файл и request одновременно,
# поэтому я решил использовать такой workaround

class TrainRequest(BaseModel):
    """Конфигурация для дообучения ResNet18."""

    random_state: int = Field(
        42,
        ge=0,
        description="Сид для воспроизводимости"
    )

    val_size: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Размер валидационного датасета"
    )

    test_size: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Размер тестового датасета"
    )

    epochs_head: int = Field(
        4,
        ge=0,
        description='Количество эпох обучения "головы" ResNet18'
    )

    lr_head: float = Field(
        1e-3,
        ge=0.0,
        description='Скорость обучения "головы" ResNet18'
    )

    epochs_ft: int = Field(
        8,
        ge=0,
        description="Количество эпох дообучения ResNet18"
    )

    lr_ft: float = Field(
        1e-4,
        ge=0.0,
        description='Скорость обучения "головы" ResNet18'
    )

    @classmethod
    def as_form(
        cls,
        random_state: int = Form(
            42,
            ge=0,
            description="Сид для воспроизводимости"
        ),
        val_size: float = Form(
            0.1,
            ge=0.0,
            le=1.0,
            description="Размер валидационного датасета"
        ),
        test_size: float = Form(
            0.1,
            ge=0.0,
            le=1.0,
            description="Размер тестового датасета"
        ),
        epochs_head: int = Form(
            4,
            ge=0,
            description='Количество эпох обучения "головы" ResNet18'
        ),
        lr_head: float = Form(
            1e-3,
            ge=0.0,
            description='Скорость обучения "головы" ResNet18'
        ),
        epochs_ft: int = Form(
            8,
            ge=0,
            description="Количество эпох дообучения ResNet18"
        ),
        lr_ft: float = Form(
            1e-4,
            ge=0.0,
            description='Скорость обучения "головы" ResNet18'
        ),
    ):
        return cls(
            random_state=random_state,
            val_size=val_size,
            test_size=test_size,
            epochs_head=epochs_head,
            lr_head=lr_head,
            epochs_ft=epochs_ft,
            lr_ft=lr_ft,
        )


# Примечание: этот блок кода мне помог написать ИИ
# Нужен он для того, чтобы сделать своего рода cache систему
# Lifespan - You can define this startup and shutdown logic using the lifespan parameter of the FastAPI
# То, что before yield - startup
# После - shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ LOAD ONCE AT STARTUP
    app.root_dir = PROJECT_ROOT
    print(app.root_dir)
    app.my_weights = load_weights(app.root_dir / "artifacts"/"best_classifier.pt")
    app.my_model = build_resnet18_raw(num_classes=6)
    app.my_model.load_state_dict(app.my_weights)
    app.my_model.eval()
    app.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.my_model.to(app.device)
    yield

app = FastAPI(
    title="AIE final project",
    version="0.1.0",
    description=(
        "FastAPI обертка для модели классификации мусора"
    ),
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "trash-classification",
        "version": "0.1.0",
    }



@app.post(
    "/predict",
    tags=["inference"],
    summary="Предсказание типа мусора по картинке",
)
async def predict_from_image(file: UploadFile = File(...)) -> dict:
    """
    Принимает файл изображения и возвращает предсказанный класс (число и название) и время выполнения
    """

    start = perf_counter()

    if file.content_type not in ("image/png", "image/jpeg"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        raise HTTPException(status_code=400, detail="Ожидается файл изображения (content-type image/png, image/jpeg).")
    
    transform = get_inference_transforms()
    try:
        # FastAPI даёт file.file как file-like объект, который можно читать 
        tensor = load_single_image_from_file(file,transform).to(app.device)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать изобраение: {exc}")

    if tensor.numel() == 0:
        raise HTTPException(status_code=400, detail="Изображение не содержит данных (файл пустой).")

    prediction = inference(app.my_model,tensor)
    print(prediction)
    latency_ms = (perf_counter()-start) * 1000.0
    class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


    return {'class': prediction[0],
            'class_name': class_names[prediction[0]],
            'latency': latency_ms,
            }


def run_training_pipeline(data_path: Path, 
                          save_path: Path, 
                          random_state: int,
                          val_size: float,
                          test_size: float,
                          epochs_head:int,
                          lr_head: float,
                          epochs_ft: int,
                          lr_ft: float) -> None:
    """
    Обучить ResNet18 на изображениях из DATA_PATH и сохранить модель в OUTPUT_PATH.
    (Датасет в DATA_PATH должен быть совместим с ImageFolder)
    """

    DEVICE = torch.device(app.device)

    set_seed(random_state)

    # Загрузка данных
    if val_size+test_size>= 1:
        raise ValueError("Размер валидациионного и тестового датасета не может быть больше или равным 1")
    set_seed()
    tf_train = get_augmented_train_transforms()
    tf_test = get_inference_transforms()
    train_loader, val_loader, test_loader, num_classes = make_loaders(
        data_path,
        tf_train=tf_train,
        tf_test=tf_test,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        )

    # Загрузка ResNet18
    weights = get_resnet18_weights()
    model = build_resnet18(weights=weights, num_classes=num_classes).to(DEVICE)

    

    #------------------
    #-----ОБУЧЕНИЕ-----
    #------------------
    if epochs_ft <0 or epochs_head<0:
        raise ValueError("Количество эпох должно быть неотрицательным")
    
    
    # Сначала обучаем только "голову"
    set_seed()
    # freeze всё
    set_requires_grad(model, False)
    # размораживаем только голову
    set_requires_grad(model.fc, True)
    optimizer_head_only = torch.optim.Adam(model.fc.parameters(), lr=lr_head)
    criterion = nn.CrossEntropyLoss()
    print("\n" + "=" * 80)
    print("Phase 1: head-only training")
    hist_head = fit(model,
                     train_loader,
                       val_loader, 
                       optimizer_head_only, 
                       criterion, 
                       epochs=epochs_head, 
                       verbose=True,
                       output_func=print,
                       device=DEVICE)



    # Дообучаем модель
    set_seed()
    criterion = nn.CrossEntropyLoss()
    set_requires_grad(model.layer4, True)
    set_requires_grad(model.fc, True)

    # param groups: backbone меньше, head больше
    params = [
        {"params": model.layer4.parameters(), "lr": lr_ft},
        {"params": model.fc.parameters(), "lr": lr_head},
    ]
    optimizer_ft = torch.optim.Adam(params, weight_decay=1e-4)

    print("\n" + "=" * 80)
    print("Phase 2: fine-tuning layer4 + fc")
    hist_ft = fit(model, 
                  train_loader, 
                  val_loader, 
                  optimizer_ft, 
                  criterion, 
                  epochs=epochs_ft, 
                  verbose=True,
                  output_func=print,
                  device=DEVICE
                  )
    torch.save(model.state_dict(), save_path)



@app.post("/train", tags=["train"])
async def train_from_zip(
    background_tasks: BackgroundTasks,
    req: TrainRequest = Depends(TrainRequest.as_form),
    dataset: UploadFile = File(...),
    )->FileResponse:
    tmp_dir = Path(tempfile.mkdtemp())
    zip_path = tmp_dir / "dataset.zip"
    extract_dir = tmp_dir / "extracted"
    model_path = tmp_dir / "model.pt"
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Путь к архиву: {zip_path}")
    print(f"Путь к распакованным данным: {extract_dir}")
    print(f"Путь к итоговой модели: {model_path}")
    def cleanup():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        # 2. Save uploaded ZIP to disk
        with open(zip_path, "wb") as f:
            f.write(await dataset.read())

        # 3. Extract & validate ImageFolder structure
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)  
        
        if not any(p.is_dir() for p in Path(extract_dir).iterdir()):
            raise HTTPException(400, "Invalid dataset: ZIP must contain class folders (ImageFolder format)")

        # 4. Run training (blocks until finished)
        run_training_pipeline(
            data_path = Path(extract_dir), 
            save_path = Path(model_path),
            random_state=req.random_state,
            val_size=req.val_size,
            test_size=req.test_size,
            epochs_head=req.epochs_head,
            lr_head=req.lr_head,
            epochs_ft=req.epochs_ft,
            lr_ft=req.lr_ft,)

        # 5. Schedule cleanup AFTER the file is fully sent to the client
        background_tasks.add_task(cleanup)

        # 6. Stream the model file to the client
        return FileResponse(model_path, filename="trained_model.pt")

    except Exception as e:
        cleanup()
        raise HTTPException(500, f"Training failed: {str(e)}")