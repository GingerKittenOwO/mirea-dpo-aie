import torch
from torch import nn


from typing import Dict, Tuple, List, Optional, Callable
import time
import math


def accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y_true).float().mean().item()





def train_one_epoch(model,
                    loader,
                    optimizer,
                    criterion,
                    device=None,
                    )-> Tuple[float, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            return float("nan"), float("nan")

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
        total_seen += bs

    return total_loss / total_seen, total_correct / total_seen

@torch.no_grad()
def evaluate(model,
            loader,
            criterion,
            device=None
            ) -> Tuple[float,float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            return float("nan"), float("nan")

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
        total_seen += bs

    return total_loss / total_seen, total_correct / total_seen

def fit(model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion, 
    epochs: int, 
    verbose: bool = True,
    output_func: Optional[Callable[[str], None]] = None,
    device=None,
    )->Dict[str, List[float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if output_func is None:
        output_func = print
    if verbose:
        output_func(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion,device=device)
        va_loss, va_acc = evaluate(model, val_loader, criterion,device=device)
        dt = time.time() - t0

        if verbose:
            output_func(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {tr_loss:.4f}, acc {tr_acc:.3f} | "
                f"val loss {va_loss:.4f}, acc {va_acc:.3f} | {dt:.1f}s"
            )


        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        
        

        if (not math.isfinite(tr_loss)) or (not math.isfinite(va_loss)):
            print("NaN/Inf в loss – обычно это признак проблем с LR/стабильностью. Останавливаем обучение.")
            break

    return history


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag

# def plot_history(hist: Dict[str, List[float]],save="", title: str = "") -> None:
#     epochs = list(range(1, len(hist["train_loss"]) + 1))

#     plt.figure(figsize=(10, 4))
#     plt.plot(epochs, hist["train_loss"], label="train loss")
#     plt.plot(epochs, hist["val_loss"], label="val loss")
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.title(title + " | loss")
#     plt.grid(True)
#     plt.legend()
#     if save:
#         plt.tight_layout()
#         plt.savefig(ARTIFACTS_DIR+"/figures/"+save+"_1.png")
#     plt.show()

#     plt.figure(figsize=(10, 4))
#     plt.plot(epochs, hist["train_acc"], label="train acc")
#     plt.plot(epochs, hist["val_acc"], label="val acc")
#     plt.xlabel("epoch")
#     plt.ylabel("accuracy")
#     plt.title(title + " | accuracy")
#     plt.grid(True)
#     plt.legend()
#     if save:
#         plt.tight_layout()
#         plt.savefig(ARTIFACTS_DIR+"/figures/"+save+"_2.png")
#     plt.show()