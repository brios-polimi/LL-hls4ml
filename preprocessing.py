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

    vocab_counts = {
        "instruction": {},
        "variable": {},
        "constant": {},
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
                    vocab_counts["instruction"][term] = vocab_counts["instruction"].get(term, 0) + 1
                elif node_type == 1:
                    vocab_sets["variable"].add(term)
                    vocab_counts["variable"][term] = vocab_counts["variable"].get(term, 0) + 1
                elif node_type == 2:
                    vocab_sets["constant"].add(term)
                    vocab_counts["constant"][term] = vocab_counts["constant"].get(term, 0) + 1

            max_pos = 0
            links = graph_data.get('links') or []
            for l in links:
                position = l.get('position', 0)
                if position > max_pos:
                    max_pos = position
            

    # Create final vocab mapping
    vocab = {k: {t: i for i, t in enumerate(sorted(v))} for k, v in vocab_sets.items()}
    return vocab, max_pos, vocab_counts


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
            
            t = n.get('text', None)
            if t is None:
                raise ValueError(f"Missing text field in node {n}")
            if node_type == 0:
                text_idx = int(vocab["instruction"].get(t, -1) + 1) # offset by 1 to reserve 0 for unknown tokens
                features["instruction"].append([text_idx])
                inst_map[node_id] = len(inst_map)
            elif node_type == 1:
                text_idx = int(vocab["variable"].get(t, -1) + 1) # offset by 1 to reserve 0 for unknown tokens
                features["variable"].append([text_idx])
                var_map[node_id] = len(var_map)
            elif node_type == 2:
                text_idx = int(vocab["constant"].get(t, -1) + 1) # offset by 1 to reserve 0 for unknown tokens
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





from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

def analyze_graph_dataset(pt_dir, max_files=None, log_scale=True):
    pt_dir = Path(pt_dir)
    files = list(pt_dir.rglob("*.pt"))
    if max_files:
        files = files[:max_files]

    node_counts = defaultdict(list)
    edge_counts = defaultdict(list)

    inst_degrees, var_degrees, const_degrees = [], [], []
    control_degrees, data_degrees = [], []
    edge_type_entropy, node_type_entropy = [], []
    control_positions, var_data_positions, const_data_positions = [], [], []
    isolated_graphs = 0
    labels = []

    for f in tqdm.tqdm(files, desc="Analyzing graphs"):
        g = torch.load(f, weights_only=False)

        # ---- node counts ----
        inst_n  = g["instruction"].x.shape[0] if "instruction" in g.node_types and g["instruction"].x is not None else 0
        var_n   = g["variable"].x.shape[0]    if "variable"    in g.node_types and g["variable"].x is not None else 0
        const_n = g["constant"].x.shape[0]    if "constant"    in g.node_types and g["constant"].x is not None else 0

        node_counts["instruction"].append(inst_n)
        node_counts["variable"].append(var_n)
        node_counts["constant"].append(const_n)

        n_vec = np.array([inst_n, var_n, const_n], dtype=float)
        if n_vec.sum() > 0:
            p = n_vec / n_vec.sum()
            node_type_entropy.append(-np.sum(p * np.log(p + 1e-12)))

        # ---- node feature stats ----
        if "instruction" in g.node_types and g["instruction"].x is not None:
            x = g["instruction"].x.float()
            node_counts["inst_feat_mean"].append(x.mean().item())
            node_counts["inst_feat_sparsity"].append((x == 0).float().mean().item())

        # ---- edges ----
        def get_edge(etype):
            return g[etype].edge_index if etype in g.edge_types else None

        edges = {
            "control":       get_edge(("instruction", "control", "instruction")),
            "call":          get_edge(("instruction", "call",    "instruction")),
            "inst_data_var": get_edge(("instruction", "data",    "variable")),
            "var_data_inst": get_edge(("variable",    "data",    "instruction")),
            "const_data_inst": get_edge(("constant",  "data",    "instruction")),
        }

        e_counts = {k: (v.shape[1] if v is not None else 0) for k, v in edges.items()}
        for k, v in e_counts.items():
            edge_counts[k].append(v)

        e_vec = np.array(list(e_counts.values()), dtype=float)
        if e_vec.sum() > 0:
            p = e_vec / e_vec.sum()
            edge_type_entropy.append(-np.sum(p * np.log(p + 1e-12)))

        # control vs data ratio
        ctrl = e_counts["control"] + e_counts["call"]
        data = e_counts["inst_data_var"] + e_counts["var_data_inst"] + e_counts["const_data_inst"]
        edge_counts["ctrl_data_ratio"].append(ctrl / data if data > 0 else float("nan"))

        # ---- per-graph degrees (fixed: no double-counting) ----
        inst_deg, var_deg, const_deg = 0, 0, 0

        for k, e in edges.items():
            if e is None:
                continue
            n_edges = e.shape[1]
            if k in ["control", "call"]:
                inst_deg += n_edges          # each edge touches inst on both sides
                control_degrees.append(n_edges)
            elif k == "inst_data_var":
                inst_deg += n_edges
                var_deg  += n_edges
                data_degrees.append(n_edges)
            elif k == "var_data_inst":
                var_deg  += n_edges
                inst_deg += n_edges
                data_degrees.append(n_edges)
            elif k == "const_data_inst":
                const_deg += n_edges
                inst_deg  += n_edges
                data_degrees.append(n_edges)

        inst_degrees.append(inst_deg)
        var_degrees.append(var_deg)
        const_degrees.append(const_deg)

        if inst_deg + var_deg + const_deg == 0:
            isolated_graphs += 1

        # ---- edge attributes (safe squeeze) ----
        ctrl_key = ("instruction", "control", "instruction")
        if ctrl_key in g.edge_types and g[ctrl_key].edge_attr is not None:
            control_positions.extend(g[ctrl_key].edge_attr.reshape(-1).tolist())

        vdi_key = ("variable", "data", "instruction")
        if vdi_key in g.edge_types and g[vdi_key].edge_attr is not None:
            var_data_positions.extend(g[vdi_key].edge_attr.reshape(-1).tolist())

        cdi_key = ("constant", "data", "instruction")
        if cdi_key in g.edge_types and g[cdi_key].edge_attr is not None:
            const_data_positions.extend(g[cdi_key].edge_attr.reshape(-1).tolist())

        # ---- label distribution ----
        if hasattr(g, "y") and g.y is not None:
            labels.extend(g.y.reshape(-1).tolist())

    print("Sample node counts:", node_counts["instruction"][:10])
    print("Sample edge counts:", edge_counts["control"][:10])
    print("Total graphs processed:", len(node_counts["instruction"]))
    print("instruction" in g.node_types)          # should be True
    print(g["instruction"].x.shape)    # should be [N, 1]

    NODE_COLORS = {
        "instruction": "steelblue",
        "variable": "orange",
        "constant": "green",
    }

    EDGE_COLORS = {
        "control": "crimson",
        "call": "darkorchid",
        "inst_data_var": "teal",
        "var_data_inst": "darkorange",
        "const_data_inst": "olive",
        "ctrl_data_ratio": "slategray",
    }

    def plot_hist(data, title, bins=50, color="steelblue"):
        data = np.array([x for x in data if x == x and x > 0])
        if not len(data):
            return
        plt.figure()
        log_bins = np.logspace(np.log10(data.min()), np.log10(data.max()), bins)
        plt.hist(data, bins=log_bins, color=color, alpha=0.8)
        if log_scale:
            plt.yscale("log")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # singular plots
    for key in ["instruction", "variable", "constant"]:
        plot_hist(node_counts[key], f"Distribution of {key.capitalize()} Nodes", color=NODE_COLORS[key])

    for key in ["control", "call", "inst_data_var", "var_data_inst", "const_data_inst"]:
        plot_hist(edge_counts[key], f"Distribution of {key} Edges", color=EDGE_COLORS[key])

    # combined plot
    plt.figure()
    for ntype, color in NODE_COLORS.items():
        plt.hist(node_counts[ntype], bins=50, alpha=0.5, label=ntype, color=color)
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Node counts by type")
    plt.show()

    plot_hist(edge_counts["ctrl_data_ratio"], "Control/data edge ratio per graph")
    plot_hist(inst_degrees,  "Instruction total degree")
    plot_hist(var_degrees,   "Variable total degree")
    plot_hist(const_degrees, "Constant total degree")
    plot_hist(edge_type_entropy, "Edge-type entropy per graph")
    plot_hist(node_type_entropy, "Node-type entropy per graph")
    plot_hist(node_counts["inst_feat_sparsity"], "Instruction feature sparsity")

    if control_positions:   plot_hist(control_positions,   "Control edge position attribute")
    if var_data_positions:  plot_hist(var_data_positions,  "Var→Inst position attribute")
    if const_data_positions: plot_hist(const_data_positions, "Const→Inst position attribute")

    if labels:
        plot_hist(labels, "Label distribution")

    # ---- summary ----
    print(f"Graphs:               {len(files)}")
    print(f"Isolated graphs:      {isolated_graphs} ({100*isolated_graphs/len(files):.1f}%)")
    print(f"Avg instruction nodes:{np.mean(node_counts['instruction']):.1f}")
    print(f"Avg variable nodes:   {np.mean(node_counts['variable']):.1f}")
    print(f"Avg constant nodes:   {np.mean(node_counts['constant']):.1f}")
    print(f"Avg control edges:    {np.mean(edge_counts['control']):.1f}")
    avg_data = (np.mean(edge_counts['inst_data_var']) +
                np.mean(edge_counts['var_data_inst']) +
                np.mean(edge_counts['const_data_inst']))
    print(f"Avg data edges:       {avg_data:.1f}")
    print(f"Avg ctrl/data ratio:  {np.nanmean(edge_counts['ctrl_data_ratio']):.2f}")
    if labels:
        unique, cnts = np.unique(labels, return_counts=True)
        print("Label distribution:", dict(zip(unique.tolist(), cnts.tolist())))