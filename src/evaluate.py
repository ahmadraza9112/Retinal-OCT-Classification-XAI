import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm


def calculate_metrics(true_labels, predicted_labels):
    """Return accuracy and macro-averaged classification metrics."""
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_precision": precision_score(
            true_labels, predicted_labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            true_labels, predicted_labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            true_labels, predicted_labels, average="macro", zero_division=0
        ),
    }


def evaluate_model(model, loader, criterion, device):
    """Evaluate a model and return loss, metrics, true labels, and predictions."""
    model.eval()
    use_amp = device.type == "cuda"
    running_loss = 0.0
    true_labels, predicted_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluation", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            true_labels.extend(labels.detach().cpu().numpy())
            predicted_labels.extend(predictions.detach().cpu().numpy())

    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    epoch_loss = running_loss / len(loader.dataset)

    return (
        epoch_loss,
        calculate_metrics(true_labels, predicted_labels),
        true_labels,
        predicted_labels,
    )


def collect_prediction_probabilities(model, loader, device):
    """Collect labels, predictions, and softmax probabilities."""
    model.eval()
    use_amp = device.type == "cuda"
    all_true, all_pred, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Predictions", leave=False):
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)

            probabilities = torch.softmax(logits.float(), dim=1)
            predictions = probabilities.argmax(dim=1)

            all_true.extend(labels.numpy())
            all_pred.extend(predictions.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())

    return (
        np.asarray(all_true),
        np.asarray(all_pred),
        np.concatenate(all_probs, axis=0),
    )


def expected_calibration_error(confidences, correct, n_bins=15):
    """Compute expected calibration error and per-bin calibration statistics."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, rows = 0.0, []

    for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (
            (confidences >= lower) & (confidences <= upper)
            if i == 0
            else (confidences > lower) & (confidences <= upper)
        )
        count = int(mask.sum())
        if count == 0:
            continue

        mean_confidence = float(confidences[mask].mean())
        accuracy = float(correct[mask].mean())
        fraction = count / len(confidences)
        ece += fraction * abs(accuracy - mean_confidence)

        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )

    return ece, pd.DataFrame(rows)
