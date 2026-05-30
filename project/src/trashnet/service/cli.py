from __future__ import annotations
import typer
import os
import numpy as np
from pathlib import Path
import torch
from torch import nn
from typing import Annotated


from ..models.resnet18 import load_weights, get_resnet18_weights, build_resnet18_raw, build_resnet18
from ..models.inference import inference, metrics_on_loader
from ..data.transforms import get_inference_transforms,get_augmented_train_transforms, load_single_image
from ..data.prepair_train import make_loaders, set_seed
from ..models.train import set_requires_grad, fit
from trashnet.utils.settings import PROJECT_ROOT


app = typer.Typer(help="Классификация мусора")

@app.command()
def check(
    msg: Annotated[str,
                   typer.Option(help="Какое сообщение вывести")
                   ]="ok"
)->None:
    """
    Проверить работоспособность cli
    """
    typer.echo(msg)

@app.command()
def predict(
    img_path: Annotated[Path,
                    typer.Argument(help="Путь к файлу изображения, которое надо классифицирвоать")
                    ],
    weights_path: Annotated[Path,
                          typer.Option(help="Путь к файлу с весами (.pt)")
                          ] = PROJECT_ROOT /"artifacts"/"best_classifier.pt",
    device: Annotated[str,
                      typer.Option(help="Устройство на котором будут происходить вычисления")
                      ]="cpu",
)-> None:
    """
    Предсказать класс мусора на изображении через предобученную модель
    """
    DEVICE = torch.device(device)
    weights = load_weights(weights_path)
    model = build_resnet18_raw()
    model.load_state_dict(weights)
    model.to(DEVICE)
    transform = get_inference_transforms()
    tensor = load_single_image(img_path,transform).to(DEVICE)
    prediction = inference(model,tensor)
    class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    typer.echo(f"Модель предсказала тип мусора: {class_names[prediction[0]]}")


@app.command()
def train_resnet18(
    data_path: Annotated[Path, 
                         typer.Argument(
                                     exists=True,
                                     file_okay = False,
                                     dir_okay=True,
                                     readable=True,
                                     resolve_path = True,
                                     help="Путь к изображениям для обучения. Каждый класс должен быть в своей папке с соответсвующим названием")
                                     ],
    
    output_path: Annotated[Path,
                           typer.Argument( 
                                       file_okay = False,
                                       dir_okay=True,
                                       resolve_path=True,
                                       help="Путь, куда будет сохранена обученная модель")
                                       ],
    verbose: Annotated[bool, 
                       typer.Option(
                           "--verbose/--quiet",
                           "-v/-q",
                           help="Выводить ли процесс обучения"),
                       ]=True, 
    device: Annotated[str,
                      typer.Option(
                          "--device","-d",
                          help="Устройство на котором будет происходить обучение")
                      ]="cpu",
    random_state: Annotated[int,
                            typer.Option(
                                "--random-state","-r",
                                min=0,
                                help="Сид для воспроизводимости")
                            ]=42,
    val_size: Annotated[float,
                        typer.Option(
                            "--val-size",
                            min=0.0,
                            max=1.0,
                            help="Размер валидационного датасета")
                        ]=0.1,
    test_size: Annotated[float,
                         typer.Option(
                             "--test-size",
                             min=0.0,
                             max=1.0,
                             help="Размер тестового датасета")
                         ]=0.1,
    epochs_head: Annotated[int,
                           typer.Option(
                               "--epochs-head",
                               min=0,
                               help='Количество эпох обучения "головы" ResNet18')
                           ]=4,
    lr_head: Annotated[float,
                       typer.Option(
                           "--lr-head",
                            min=0.0,
                            help='Скорость обучения "головы" ResNet18')
                       ]=1e-3,
    epochs_ft: Annotated[int,
                         typer.Option(
                             "--epochs-ft",
                             min=0,
                             help='Количество эпох дообучения ResNet18')
                         ]=4,
    lr_ft: Annotated[float,
                     typer.Option(
                         "--lr-ft",
                         min=0.0,
                         help='Скорость дообучения ResNet18')
                     ]=1e-4,
    compute_metrics: Annotated[bool, 
                               typer.Option(
                                   "--compute-metrics","-m",
                                    help="Вывести ли метрики итоговой модели на test")
                                ]=False,
)->None:
    """
    Обучить ResNet18 на изображениях из DATA_PATH и сохранить модель в OUTPUT_PATH.
    (Датасет в DATA_PATH должен быть совместим с ImageFolder)
    """

    DEVICE = torch.device(device)


    try:
        # Create parent directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Now check if we can write to it
        if not os.access(output_path.parent, os.W_OK):
            typer.echo(f"Error: Cannot write to directory {output_path.parent}", err=True)
            raise typer.Exit(code=1)
            
    except PermissionError as e:
        typer.echo(f"Error: Permission denied - {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: Cannot create output directory - {e}", err=True)
        raise typer.Exit(code=1)

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
    if verbose:
        typer.echo("\n" + "=" * 80)
        typer.echo("Phase 1: head-only training")
    hist_head = fit(model,
                     train_loader,
                       val_loader, 
                       optimizer_head_only, 
                       criterion, 
                       epochs=epochs_head, 
                       verbose=verbose,
                       output_func=typer.echo,
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

    if verbose:
        typer.echo("\n" + "=" * 80)
        typer.echo("Phase 2: fine-tuning layer4 + fc")
    hist_ft = fit(model, 
                  train_loader, 
                  val_loader, 
                  optimizer_ft, 
                  criterion, 
                  epochs=epochs_ft, 
                  verbose=verbose,
                  output_func=typer.echo,
                  device=DEVICE)

    torch.save(model.state_dict(), output_path/"model.pt")
    typer.echo(f"Модель была сохранена: {output_path/'model.pt'}")

    if compute_metrics and test_size>0.0 and test_size:
        metrics = metrics_on_loader(
            model,
            test_loader,
            num_classes=num_classes,
            device=DEVICE
        )
        for k,v in metrics.items():
            typer.echo(f"{k}: {v}")


if __name__ == "__main__":
    app()