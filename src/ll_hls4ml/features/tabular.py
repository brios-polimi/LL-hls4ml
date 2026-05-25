"""Build tabular feature DataFrames from CDFG JSON graphs."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import tqdm

from ll_hls4ml.features.graph_stats import extract_graph_features
from ll_hls4ml.io.discovery import iter_graph_paths
from ll_hls4ml.io.load_json import load_graph_json


def build_feature_row(graph_path):
    """Load a graph JSON file and extract its feature vector."""
    graph_data = load_graph_json(graph_path)
    return extract_graph_features(graph_data)


def build_feature_dataframe(
    graph_dir,
    kernel_subset=None,
    max_archives=None,
    num_workers=None,
):
    """
    Walk graph_dir and create a dataframe of graph features.

    Returns a DataFrame with a ``kernel_type`` column per graph.
    """
    graph_dir = Path(graph_dir)
    rows = []

    paths = list(iter_graph_paths(graph_dir, kernel_subset, max_archives))
    if not paths:
        return pd.DataFrame()

    if num_workers is None:
        num_workers = min(32, (os.cpu_count() or 1))

    by_kernel: dict[str, list] = {}
    for ks, path in paths:
        by_kernel.setdefault(ks, []).append(path)

    for ks, graph_paths in by_kernel.items():
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for features in tqdm.tqdm(
                executor.map(build_feature_row, graph_paths),
                total=len(graph_paths),
                desc=f"Parsing graph files for kernel subset '{ks}'",
            ):
                features["kernel_type"] = ks
                rows.append(features)

    return pd.DataFrame(rows)
