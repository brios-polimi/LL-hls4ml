"""Filesystem discovery for CDFG JSON graphs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def _normalize_kernel_subset(kernel_subset: str | list[str] | None) -> list[str]:
    if kernel_subset is None:
        return [""]
    if isinstance(kernel_subset, str):
        return [kernel_subset]
    return list(kernel_subset)


def iter_graph_paths(
    graph_dir: str | Path,
    kernel_subset: str | list[str] | None = None,
    max_archives: int | None = None,
) -> Iterator[tuple[str, Path]]:
    """
    Yield ``(kernel_type, path)`` for each CDFG JSON under ``graph_dir``.

    When ``kernel_subset`` is empty string, walks the root directly.
    """
    graph_dir = Path(graph_dir)

    for ks in _normalize_kernel_subset(kernel_subset):
        subset_dir = graph_dir / ks if ks else graph_dir
        if ks and not subset_dir.exists():
            raise ValueError(f"Subset directory not found: {subset_dir}")

        if max_archives is not None:
            archive_dirs = sorted(p for p in subset_dir.iterdir() if p.is_dir())
            archive_dirs = archive_dirs[:max_archives]
            graph_paths = []
            for archive_dir in archive_dirs:
                graph_paths.extend(sorted(archive_dir.rglob("*.json")))
        else:
            graph_paths = sorted(subset_dir.rglob("*.json"))

        for path in graph_paths:
            yield ks, path


def collect_graph_paths(
    graph_dir: str | Path,
    kernel_subset: str | list[str] | None = None,
    max_archives: int | None = None,
) -> list[tuple[str, Path]]:
    """Return all ``(kernel_type, path)`` pairs from ``iter_graph_paths``."""
    return list(iter_graph_paths(graph_dir, kernel_subset, max_archives))
