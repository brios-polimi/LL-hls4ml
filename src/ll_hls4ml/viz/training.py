"""Training evaluation plots."""

import matplotlib.pyplot as plt
import torch

from ll_hls4ml.training.targets import to_luts


def plot_loss_curves(training_history, title="Training Loss"):
    plt.figure()
    plt.plot(training_history["train_loss"], label="Train Loss")
    plt.plot(training_history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_predictions_vs_targets(
    model,
    val_loader,
    device,
    y_mean,
    y_std,
    figsize=(12, 6),
):
    """Sorted prediction vs target curves on validation set (LUT scale)."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = to_luts(
                model(batch).squeeze(-1),
                y_mean.to(device),
                y_std.to(device),
            )
            all_preds.append(preds.cpu())
            all_targets.append(batch.y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    sort_idx = all_targets.argsort()
    sorted_targets = all_targets[sort_idx]
    sorted_preds = all_preds[sort_idx]

    plt.figure(figsize=figsize)
    plt.plot(sorted_preds, label="Prediction")
    plt.plot(sorted_targets, label="Target")
    plt.xlabel("Samples (sorted by target)")
    plt.ylabel("LUTs")
    plt.title("Predictions vs Targets")
    plt.legend()
    plt.show()
