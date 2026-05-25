"""Convert CDFG JSON graphs to PyG HeteroData tensors."""

from __future__ import annotations

from pathlib import Path

import torch
import tqdm
from torch_geometric.data import HeteroData

from ll_hls4ml.io.discovery import iter_graph_paths
from ll_hls4ml.io.load_json import load_graph_json
from ll_hls4ml.io.schema import (
    FLOW_CALL,
    FLOW_CONTROL,
    FLOW_DATA,
    NODE_CONSTANT,
    NODE_INSTRUCTION,
    NODE_VARIABLE,
    EDGE_TYPES,
    EDGE_TYPES_WITH_ATTR,
    LABEL_KEYS,
    safe_int,
)


def create_graph_tensors(
    graph_dir: str | Path,
    pt_dir: str | Path,
    vocab: dict,
    kernel_subset: str | list[str] | None = None,
    max_archives: int | None = None,
    target_labels: str | list[str] | None = None,
):
    """
    Walk graph_dir and convert JSON files to PyG HeteroData, mirroring structure in pt_dir.

    Example: graphs/exemplar/archive_1/*.json → tensors/exemplar/archive_1/*.pt
    """
    graph_dir = Path(graph_dir)
    pt_dir = Path(pt_dir)
    pt_dir.mkdir(parents=True, exist_ok=True)

    graph_dir_original = graph_dir
    kernel_subsets = [kernel_subset] if kernel_subset else [None]

    for ks in kernel_subsets:
        paths = list(iter_graph_paths(graph_dir, ks, max_archives))
        if ks:
            graph_dir_original = graph_dir

        for _kernel_type, graph_path in tqdm.tqdm(
            paths, desc="Processing graph files into PyTorch tensors"
        ):
            rel_path = graph_path.relative_to(graph_dir_original)
            out_subdir = pt_dir / rel_path.parent
            out_subdir.mkdir(parents=True, exist_ok=True)

            graph_data = load_graph_json(graph_path)
            data = _json_to_hetero(graph_data, vocab, target_labels)
            out_path = out_subdir / (graph_path.stem + ".pt")
            torch.save(data, out_path)


def _json_to_hetero(graph_data: dict, vocab: dict, target_labels: str | list[str] | None) -> HeteroData:
    data = HeteroData()
    inst_map = {}
    var_map = {}
    const_map = {}

    features = {
        "instruction": [],
        "variable": [],
        "constant": [],
    }
    nodes = graph_data.get("nodes") or []
    for n in nodes:
        node_id = safe_int(n.get("id", -1))
        if node_id == -1:
            raise ValueError(f"Missing node id in node {n}")

        node_type = safe_int(n.get("type", -1))
        if node_type not in [NODE_INSTRUCTION, NODE_VARIABLE, NODE_CONSTANT]:
            raise ValueError(f"Invalid node type: {node_type} in node {n}")        

        # Map node text to vocabulary index (0 is unknown token)
        # Create global index map for each node type
        node_term = n.get("text", None)
        if node_term is None:
            raise ValueError(f"Missing text field in node {n}")
        if node_type == NODE_INSTRUCTION:
            text_idx = vocab["instruction"].get(node_term, -1) + 1
            features["instruction"].append([text_idx])
            inst_map[node_id] = len(inst_map)
        elif node_type == NODE_VARIABLE:
            text_idx = vocab["variable"].get(node_term, -1) + 1
            features["variable"].append([text_idx])
            var_map[node_id] = len(var_map)
        elif node_type == NODE_CONSTANT:
            text_idx = vocab["constant"].get(node_term, -1) + 1
            features["constant"].append([text_idx])
            const_map[node_id] = len(const_map)

    for k, v in features.items():
        if v:
            data[k].x = torch.tensor(v, dtype=torch.long)


    edge_index = { k: [] for k in EDGE_TYPES }
    edge_attrs = { k: [] for k in EDGE_TYPES_WITH_ATTR }
    edges = graph_data.get("links") or []
    for edge in edges:
        flow = safe_int(edge.get("flow", -1))
        source = safe_int(edge.get("source", -1))
        target = safe_int(edge.get("target", -1))
        if source <= 0 or source >= len(nodes) or target <= 0 or target >= len(nodes) or flow not in [FLOW_CONTROL, FLOW_DATA, FLOW_CALL]: # this should fire next run, i know theres an off by one error here, i just dont know which direction
            raise ValueError(f"Invalid edge with invalid source/target/flow: {edge}")
            
        position = safe_int(edge.get("position", 0))
        local_idx_source = None
        local_idx_target = None

        if flow == FLOW_CONTROL:
            local_idx_source = inst_map.get(source)
            local_idx_target = inst_map.get(target)
            edge_index[("instruction", "control", "instruction")].append([local_idx_source, local_idx_target])
            edge_attrs[("instruction", "control", "instruction")].append([position])
        elif flow == FLOW_DATA:
            if nodes[source].get("type") == NODE_INSTRUCTION:
                local_idx_source = inst_map.get(source)
                local_idx_target = var_map.get(target)
                edge_index[("instruction", "data", "variable")].append([local_idx_source, local_idx_target])
            elif nodes[source].get("type") == NODE_VARIABLE:
                local_idx_source = var_map.get(source)
                local_idx_target = inst_map.get(target)
                edge_index[("variable", "data", "instruction")].append([local_idx_source, local_idx_target])
                edge_attrs[("variable", "data", "instruction")].append([position])
            elif nodes[source].get("type") == NODE_CONSTANT:
                local_idx_source = const_map.get(source)
                local_idx_target = inst_map.get(target)
                edge_index[("constant", "data", "instruction")].append([local_idx_source, local_idx_target])
                edge_attrs[("constant", "data", "instruction")].append([position])
        elif flow == FLOW_CALL:
            local_idx_source = inst_map.get(source)
            local_idx_target = inst_map.get(target)
            edge_index[("instruction", "call", "instruction")].append([local_idx_source, local_idx_target])

        if local_idx_source is None or local_idx_target is None:
            raise ValueError(
                f"Invalid edge indices: {local_idx_source=}, {local_idx_target=}, "
                f"original source={source}, target={target}"
            )

    for et, v in edge_index.items():
        if v:
            data[et].edge_index = torch.tensor(v, dtype=torch.long).t().contiguous()

    for et, v in edge_attrs.items():
        if v:
            data[et].edge_attr = torch.tensor(v, dtype=torch.long)

    # If target_label is None, default to all labels
    if target_labels is None:
        target_labels = LABEL_KEYS
    elif isinstance(target_labels, str):
        target_labels = [target_labels]

    ys = []
    labels = graph_data.get("labels", {})
    for tl in target_labels:
        if tl not in labels:
            raise ValueError(f"Target label {tl} not found in graph data")
        ys.append(labels[tl])

    data.y = torch.tensor(ys, dtype=torch.float)
    data.y_names = target_labels

    return data
