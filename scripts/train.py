#!/usr/bin/env python3
"""Train R-GCN surrogate model (minimal CLI)."""

import argparse

import torch
import torch.nn as nn

from ll_hls4ml.config import load_config
from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.data.splits import compute_target_stats, random_train_val_split
from ll_hls4ml.models.registry import build
from ll_hls4ml.training import fit, make_loader


def main():
    parser = argparse.ArgumentParser(description="Train CDFG R-GCN")
    parser.add_argument("--config", default=None)
    parser.add_argument("--types", nargs="*", default=["2layer", "exemplar"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--name", default="cdfg_rgcn")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HeteroGraphDataset(cfg.tensor_dir, types=args.types)
    train_ds, val_ds = random_train_val_split(dataset)
    y_mean, y_std = compute_target_stats(dataset, train_ds.indices)

    node_vocab_sizes = {nt: 5000 for nt in ["instruction", "variable", "constant"]}
    model = build(
        "rgcn",
        node_vocab_sizes=node_vocab_sizes,
        edge_pos_vocab_size=32,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    fit(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        experiment_name=args.name,
        checkpoint_dir=cfg.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
