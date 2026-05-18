# preprocessing.py

import json
from pprint import pprint
import torch
import tqdm
from torch_geometric.data import HeteroData
from pathlib import Path

def vocab_scan(graph_dir, kernel_subset=None, max_archives=None):
    """
    Walk graph_dir and collect fixed vocabularies of instructions,
    variables, and constants. If kernel_subset is provided, only scan those subdirectories. 
    If max_archives is provided, only scan that many archive subdirectories.

    Returns a dict of vocabularies and the maximum position value found in edges.
    """
    vocab_sets = {
        "instruction": set(),
        "variable": set(),
        "constant": set(),
    }
    
    graph_dir = Path(graph_dir)
    if isinstance(kernel_subset, str):
        kernel_subset = [kernel_subset]

    for ks in kernel_subset or [""]:
        subset_dir = graph_dir / ks
        if ks != "" and not subset_dir.exists():
            raise ValueError(f"Subset directory not found: {subset_dir}")

        if max_archives is not None:
            archive_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir()])
            archive_dirs = archive_dirs[:max_archives]
            graph_paths = []
            for archive_dir in archive_dirs:
                graph_paths.extend(sorted(archive_dir.rglob('*.json')))
        else:
            graph_paths = sorted(subset_dir.rglob('*.json'))

        # Walk all selected graph files
        for graph_path in tqdm.tqdm(graph_paths, desc="Parsing graph files for building vocab"):
            # Load and convert
            with graph_path.open('r') as f:
                graph_data = json.load(f)

            # Accumulate vocab from nodes
            nodes = graph_data.get('nodes') or []
            for n in nodes:
                node_type = n.get('type', -1)
                term = n.get('text', '')
                if node_type == 0:
                    vocab_sets["instruction"].add(term)
                elif node_type == 1:
                    vocab_sets["variable"].add(term)
                elif node_type == 2:
                    vocab_sets["constant"].add(term)

            max_pos = 0
            links = graph_data.get('links') or []
            for l in links:
                position = l.get('position', 0)
                if position > max_pos:
                    max_pos = position
            

    # Create final vocab mapping
    vocab = {k: {t: i for i, t in enumerate(sorted(v))} for k, v in vocab_sets.items()}
    return vocab, max_pos


def _safe_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return -1


def create_graph_tensors(graph_dir, pt_dir, vocab, kernel_subset=None, max_archives=None):
    """
    Walk graph_dir and convert JSON files to PyG HeteroData objects,
    preserving the directory structure in pt_dir.

    Use pre-generated vocab to convert instruction/variable/constant text to numeric indices.

    If kernel_subset is provided, only process graphs under that
    subdirectory (for example, "exemplar"). If max_archives is provided,
    only process the first N archive subdirectories inside that subset.

    Example: /data/graphs/exemplar/archive_1/*.json → /data/tensors/exemplar/archive_1/*.pt
    """
    graph_dir = Path(graph_dir)
    if kernel_subset is not None:
        graph_dir = graph_dir / kernel_subset
    pt_dir = Path(pt_dir)
    pt_dir.mkdir(parents=True, exist_ok=True)

    if kernel_subset is not None and not graph_dir.exists():
        raise ValueError(f"Subset directory not found: {graph_dir}")

    if max_archives is not None:
        archive_dirs = sorted([p for p in graph_dir.iterdir() if p.is_dir()])
        archive_dirs = archive_dirs[:max_archives]
        graph_paths = []
        for archive_dir in archive_dirs:
            graph_paths.extend(sorted(archive_dir.rglob('*.json')))
    else:
        graph_paths = sorted(graph_dir.rglob('*.json'))

    # Walk all selected graph files
    graph_dir_original = graph_dir if kernel_subset is None else graph_dir.parent
    for graph_path in tqdm.tqdm(graph_paths, desc="Processing graph files into PyTorch tensors"):
        # Compute relative path from original root (preserves kernel_subset in path)
        rel_path = graph_path.relative_to(graph_dir_original)
        
        # Create mirrored output directory
        out_subdir = pt_dir / rel_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        
        # Load and convert
        with graph_path.open('r') as f:
            graph_data = json.load(f)

        data = HeteroData()

        # Global node indices must be turned into typed node indices
        inst_map = {}
        var_map = {}
        const_map = {}

        # Build a simple numeric feature per typed node: [text (fixed-size vocab)]
        features = {
            "instruction": [],
            "variable": [],
            "constant": []
        }
        nodes = graph_data.get('nodes') or []
        for n in nodes:
            # why did i add these in the first place?
            # block = _safe_int(n.get('block', -1))
            # function = _safe_int(n.get('function', -1))
            node_type = _safe_int(n.get('type', -1))
            node_id = _safe_int(n.get('id', -1))
            if node_type not in [-1, 0, 1, 2]:
                raise ValueError(f"Invalid node type: {node_type} or id: {node_id} in node {n}")
            
            t = n.get('text', '')
            text_idx = int(vocab.get(t, -1) + 1) # offset by 1 to reserve 0 for unknown tokens
            if node_type == 0:
                features["instruction"].append([text_idx])
                inst_map[node_id] = len(inst_map)
            elif node_type == 1:
                features["variable"].append([text_idx])
                var_map[node_id] = len(var_map)
            elif node_type == 2:
                features["constant"].append([text_idx])
                const_map[node_id] = len(const_map)
        for k, v in features.items():
            if v:
                data[k].x = torch.tensor(v, dtype=torch.long)

        # Build a simple numeric feature per typed edge: [source, target, optional(position)]
        # `position` only applies to Control edges
        links = graph_data.get('links') or []
        edge_index = {
            "control": [],
            "inst_data_var": [],
            "var_data_inst": [],
            "const_data_inst": [],
            "call": []
        }
        edge_attrs = {
            "control": [],
            "var_data_inst": [],
            "const_data_inst": [],
        }
        for l in links:
            flow = _safe_int(l.get('flow', -1))
            source = _safe_int(l.get('source', -1))
            target = _safe_int(l.get('target', -1))
            #source, target = target, source # backwards in json?
            if source== -1 or target == -1 or flow == -1:
                raise ValueError("Invalid edge with missing source/target/flow")

            position = _safe_int(l.get('position', 0))
            if flow == 0: # instruction -> control -> instruction
                local_idx_source = inst_map.get(source)
                local_idx_target = inst_map.get(target)
                edge_index["control"].append([local_idx_source, local_idx_target])
                edge_attrs["control"].append([position])
            elif flow == 1: # data edges
                if nodes[source].get('type') == 0:  # instruction -> data -> variable
                    local_idx_source = inst_map.get(source)
                    local_idx_target = var_map.get(target)
                    edge_index["inst_data_var"].append([local_idx_source, local_idx_target])
                elif nodes[source].get('type') == 1:  # variable -> data -> instruction
                    local_idx_source = var_map.get(source)
                    local_idx_target = inst_map.get(target)
                    edge_index["var_data_inst"].append([local_idx_source, local_idx_target])
                    edge_attrs["var_data_inst"].append([position])
                elif nodes[source].get('type') == 2:  # constant -> data -> instruction
                    local_idx_source = const_map.get(source)
                    local_idx_target = inst_map.get(target)
                    edge_index["const_data_inst"].append([local_idx_source, local_idx_target])
                    edge_attrs["const_data_inst"].append([position])
            elif flow == 2: # instruction -> call -> instruction
                local_idx_source = inst_map.get(source)
                local_idx_target = inst_map.get(target)
                edge_index["call"].append([local_idx_source, local_idx_target])

            if local_idx_source is None or local_idx_target is None:
                print(f"{l=}")
                print(f"{len(inst_map)=}, {len(var_map)=}, {len(const_map)=}")
                print(inst_map, '\n', var_map, '\n', const_map)
                print(f"{nodes[source]=}, {nodes[target]=}")
                raise ValueError(f"Invalid edge indices: {local_idx_source=}, {local_idx_target=}, original source={source}, target={target}")
                

        # Save edge tensors
        for k, v in edge_index.items():
            if v:
                if k == "control":
                    data["instruction", "control", "instruction"].edge_index = torch.tensor(edge_index["control"], dtype=torch.long).t().contiguous()
                elif k == "inst_data_var":
                    data["instruction", "data", "variable"].edge_index = torch.tensor(edge_index["inst_data_var"], dtype=torch.long).t().contiguous()
                elif k == "var_data_inst":
                    data["variable", "data", "instruction"].edge_index = torch.tensor(edge_index["var_data_inst"], dtype=torch.long).t().contiguous()
                elif k == "const_data_inst":
                    data["constant", "data", "instruction"].edge_index = torch.tensor(edge_index["const_data_inst"], dtype=torch.long).t().contiguous()
                elif k == "call":
                    data["instruction", "call", "instruction"].edge_index = torch.tensor(edge_index["call"], dtype=torch.long).t().contiguous()

        # Save edge attributes
        for attr_k, attr_v in edge_attrs.items():
            if attr_v:
                if attr_k == "control":
                    data["instruction", "control", "instruction"].edge_attr = torch.tensor(attr_v, dtype=torch.long)
                elif attr_k == "var_data_inst":
                    data["variable", "data", "instruction"].edge_attr = torch.tensor(attr_v, dtype=torch.long)
                elif attr_k == "const_data_inst":
                    data["constant", "data", "instruction"].edge_attr = torch.tensor(attr_v, dtype=torch.long)

        # Save labels
        labels = graph_data.get('labels') or {}
        if labels:
            if "lut" in labels.keys(): # just add LUTs for now
                data.y = torch.tensor([labels["lut"]], dtype=torch.float)

        out_path = out_subdir / (graph_path.stem + '.pt')
        torch.save(data, out_path)