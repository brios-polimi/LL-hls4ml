"""Build and persist instruction/variable/constant vocabularies from CDFG JSON."""

from __future__ import annotations

import json
from pathlib import Path

import tqdm

from ll_hls4ml.io.discovery import iter_graph_paths
from ll_hls4ml.io.load_json import load_graph_json
from ll_hls4ml.io.schema import NODE_CONSTANT, NODE_INSTRUCTION, NODE_VARIABLE


def vocab_scan(
    graph_dir: str | Path,
    kernel_subset: str | list[str] | None = None,
    max_archives: int | None = None,
):
    """
    Walk graph_dir and collect vocabularies of instructions, variables, and constants.

    Returns vocab dict, max edge position, and per-token counts.
    """
    vocab_sets = {
        "instruction": set(),
        "variable": set(),
        "constant": set(),
    }
    vocab_counts = {
        "instruction": {},
        "variable": {},
        "constant": {},
    }
    max_pos = 0

    paths = list(iter_graph_paths(graph_dir, kernel_subset, max_archives))
    for _ks, graph_path in tqdm.tqdm(paths, desc="Parsing graph files for building vocab"):
        graph_data = load_graph_json(graph_path)
        nodes = graph_data.get("nodes") or []
        for n in nodes:
            node_type = n.get("type", -1)
            term = n.get("text", "")
            if node_type == NODE_INSTRUCTION:
                vocab_sets["instruction"].add(term)
                vocab_counts["instruction"][term] = vocab_counts["instruction"].get(term, 0) + 1
            elif node_type == NODE_VARIABLE:
                vocab_sets["variable"].add(term)
                vocab_counts["variable"][term] = vocab_counts["variable"].get(term, 0) + 1
            elif node_type == NODE_CONSTANT:
                vocab_sets["constant"].add(term)
                vocab_counts["constant"][term] = vocab_counts["constant"].get(term, 0) + 1

        for link in graph_data.get("links") or []:
            position = link.get("position", 0)
            if position > max_pos:
                max_pos = position

    vocab = {k: {t: i for i, t in enumerate(sorted(v))} for k, v in vocab_sets.items()}
    return vocab, max_pos, vocab_counts


def save_vocab(vocab: dict, path: str | Path, max_pos: int | None = None, vocab_counts: dict | None = None) -> None:
    """Persist vocab mapping to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"vocab": vocab}
    if max_pos is not None:
        payload["max_pos"] = max_pos
    if vocab_counts is not None:
        payload["vocab_counts"] = vocab_counts
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_vocab(path: str | Path) -> tuple[dict, int, dict]:
    """Load vocab from JSON. Returns (vocab, max_pos, vocab_counts)."""
    with Path(path).open() as f:
        payload = json.load(f)
    return payload["vocab"], payload.get("max_pos", 0), payload.get("vocab_counts", {})
