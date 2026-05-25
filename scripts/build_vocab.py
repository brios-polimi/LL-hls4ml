#!/usr/bin/env python3
"""Scan CDFG JSON graphs and save vocabulary to artifacts."""

import argparse

from ll_hls4ml.config import load_config
from ll_hls4ml.data.vocab import save_vocab, vocab_scan


def main():
    parser = argparse.ArgumentParser(description="Build CDFG vocabulary from JSON graphs")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument(
        "--kernels",
        nargs="*",
        default=None,
        help="Kernel types to scan (default: all)",
    )
    parser.add_argument("--max-archives", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab, max_pos, _counts = vocab_scan(
        cfg.graph_dir,
        kernel_subset=args.kernels,
        max_archives=args.max_archives,
    )
    save_vocab(vocab, cfg.vocab_path, max_pos=max_pos)
    print(f"Saved vocab to {cfg.vocab_path} (max_pos={max_pos})")


if __name__ == "__main__":
    main()
