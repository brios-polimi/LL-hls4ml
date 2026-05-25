"""Training and validation loops."""

from __future__ import annotations

from pathlib import Path

import torch
from scipy.stats import pearsonr

from ll_hls4ml.training.targets import normalize_target


def train_one_epoch(model, train_loader, criterion, optimizer, device, y_mean, y_std):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        target = normalize_target(batch.y, y_mean.to(device), y_std.to(device))
        pred = model(batch).squeeze(-1)
        loss = criterion(pred, target)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

    return running_loss / len(train_loader)


def validate_one_epoch(model, val_loader, criterion, device, y_mean, y_std):
    model.eval()
    all_preds, all_targets = [], []
    running_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            target = normalize_target(batch.y, y_mean.to(device), y_std.to(device))
            pred = model(batch).squeeze(-1)
            loss = criterion(pred, target)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            running_loss += loss.item()

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    r, _ = pearsonr(preds.numpy(), targets.numpy())
    epoch_loss = running_loss / len(val_loader)
    return epoch_loss, r, preds.std(), targets.std()


def fit(
    model,
    train_loader,
    val_loader,
    epochs,
    criterion,
    optimizer,
    device,
    y_mean,
    y_std,
    patience=0,
    evaluation_metric="val_loss",
    mode="min",
    restore_best_weights=True,
    writer=None,
    verbose=10,
    experiment_name="",
    checkpoint_dir: str | Path | None = None,
):
    """
    Train model with optional early stopping.

    Checkpoints are saved to ``checkpoint_dir / {experiment_name}_model.pt``.
    """
    checkpoint_dir = Path(checkpoint_dir or "artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{experiment_name}_model.pt"

    training_history = {"train_loss": [], "val_loss": []}

    if patience > 0:
        patience_counter = 0
        best_metric = float("-inf") if mode == "max" else float("inf")
        best_epoch = 0

    print(f"Training {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, y_mean, y_std
        )
        val_loss, r, preds_std, targets_std = validate_one_epoch(
            model, val_loader, criterion, device, y_mean, y_std
        )

        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)

        if verbose > 0 and (epoch % verbose == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: Loss={train_loss:.4f} | "
                f"Val: Loss={val_loss:.4f}, R={r:.4f}, "
                f"pred std={preds_std:.4f}, target std={targets_std:.4f}"
            )

        if patience > 0:
            current_metric = training_history[evaluation_metric][-1]
            is_improvement = (
                current_metric > best_metric if mode == "max" else current_metric < best_metric
            )

            if is_improvement:
                best_metric = current_metric
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    if restore_best_weights and patience > 0 and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print(
            f"Best model restored from epoch {best_epoch} "
            f"with {evaluation_metric} {best_metric:.4f}"
        )

    if patience == 0:
        torch.save(model.state_dict(), ckpt_path)

    return model, training_history
