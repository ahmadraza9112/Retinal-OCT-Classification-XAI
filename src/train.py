import random

import numpy as np
import torch
from tqdm.auto import tqdm

from .evaluate import calculate_metrics


def set_all_seeds(seed=42):
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch and return loss plus classification metrics."""
    model.train()
    use_amp = device.type == "cuda"

    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    running_loss = 0.0
    true_labels, predicted_labels = [], []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        true_labels.extend(labels.detach().cpu().numpy())
        predicted_labels.extend(predictions.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, calculate_metrics(true_labels, predicted_labels)
