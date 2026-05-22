from pathlib import Path
import json
import os
from collections import Counter, deque

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tqdm
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#import umap.umap_ as umap

import matplotlib.pyplot as plt


# ============================================================
# GRAPH FEATURE EXTRACTION
# ============================================================
def dag_level_stats(G):
    # assign each node its longest path from any source
    in_deg = dict(G.in_degree())
    level = {n: 0 for n in G.nodes()}
    queue = deque([n for n, d in in_deg.items() if d == 0])
    
    while queue:
        node = queue.popleft()
        for succ in G.successors(node):
            level[succ] = max(level[succ], level[node] + 1)
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)
    
    level_counts = Counter(level.values())
    widths = list(level_counts.values())
    
    return {
        "critical_path_depth": max(level.values()),
        "max_width": max(widths),          # peak parallelism
        "mean_width": np.mean(widths),     # average parallelism
        "total_levels": len(widths),
    }

def extract_graph_features(graph_data):
    """
    Return a flat dict of handcrafted graph features.
    """

    nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", [])

    G = nx.DiGraph()

    num_nodes = len(nodes)
    num_edges = len(links)
    num_llvm_blocks     = 0
    num_llvm_functions  = 0

    # --------------------------------------------------------
    # Node counts
    # --------------------------------------------------------

    node_type_counts = Counter()
    node_text_counts = Counter()

    for n in nodes:
        nid = n.get("id")
        G.add_node(nid)

        node_type = n.get("type", -1)
        node_type_counts[node_type] += 1

        node_text = n.get("text", "")
        node_text_counts[node_text] += 1

        block = n.get("block", -1)
        function = n.get("function", -1)

        num_llvm_blocks = block if block > num_llvm_blocks else num_llvm_blocks
        num_llvm_functions = function if function > num_llvm_functions else num_llvm_functions

    num_instruction_nodes = node_type_counts[0]
    num_variable_nodes = node_type_counts[1]
    num_constant_nodes = node_type_counts[2]

    instruction_ratio = (num_instruction_nodes / num_nodes if num_nodes > 0 else 0.0)
    variable_ratio = (num_variable_nodes / num_nodes if num_nodes > 0 else 0.0)
    constant_ratio = (num_constant_nodes / num_nodes if num_nodes > 0 else 0.0)

    # --------------------------------------------------------
    # Edge counts
    # --------------------------------------------------------

    flow_counts = Counter()
    in_degree = Counter()
    out_degree = Counter()

    FLOW_TYPES = [
        ("instruction", "control", "instruction"),
        ("instruction", "data",     "variable"),
        ("variable",    "data",     "instruction"),
        ("constant",    "data",     "instruction"),
        ("instruction", "call",     "instruction"),
    ]

    for e in links:
        flow = e.get("flow", -1)
        source = e.get("source", -1)
        target = e.get("target", -1)
        
        out_degree[source] += 1
        in_degree[target] += 1

        G.add_edge(source, target)

        source_type = nodes[source].get("type", -1)
        target_type = nodes[target].get("type", -1)

        known_edge_type = False
        if flow == 0:   # control
            if source_type == 0 and target_type == 0: # instruction -> control -> instruction
                flow_counts[FLOW_TYPES[0]] += 1
                known_edge_type = True
        elif flow == 1: # data
            if source_type == 0 and target_type == 1: # instruction -> data -> variable
                flow_counts[FLOW_TYPES[1]] += 1
                known_edge_type = True
            elif source_type == 1 and target_type == 0: # variable -> data -> instruction
                flow_counts[FLOW_TYPES[2]] += 1
                known_edge_type = True
            elif source_type == 2 and target_type == 0: # constant -> data -> instruction
                flow_counts[FLOW_TYPES[3]] += 1
                known_edge_type = True
        elif flow == 2: # call
            if source_type == 0 and target_type == 0: # instruction -> call -> instruction
                flow_counts[FLOW_TYPES[4]] += 1
                known_edge_type = True

        if not known_edge_type:
            raise ValueError(f"Unknown edge type with flow={flow}, source_type={source_type}, target_type={target_type}")

    num_inst_control_inst_edges = flow_counts[FLOW_TYPES[0]]
    num_inst_data_var_edges     = flow_counts[FLOW_TYPES[1]]
    num_var_data_inst_edges     = flow_counts[FLOW_TYPES[2]]
    num_const_data_inst_edges   = flow_counts[FLOW_TYPES[3]]
    num_inst_call_inst_edges    = flow_counts[FLOW_TYPES[4]]

    inst_control_inst_ratio =   (num_inst_control_inst_edges / num_edges if num_edges > 0 else 0.0)
    inst_data_var_ratio =       (num_inst_data_var_edges / num_edges if num_edges > 0 else 0.0)
    var_data_inst_ratio =       (num_var_data_inst_edges / num_edges if num_edges > 0 else 0.0)
    const_data_inst_ratio =     (num_const_data_inst_edges / num_edges if num_edges > 0 else 0.0)
    inst_call_inst_ratio =      (num_inst_call_inst_edges / num_edges if num_edges > 0 else 0.0)

    # --------------------------------------------------------
    # Basic graph statistics
    # --------------------------------------------------------

    density = (
        num_edges / (num_nodes * (num_nodes - 1))
        if num_nodes > 1 else 0.0
    )

    condensed = nx.condensation(G)  # always a DAG
    geometry_features = dag_level_stats(condensed)

    # --------------------------------------------------------
    # Degree statistics
    # --------------------------------------------------------        

    all_node_ids = [n["id"] for n in nodes]
    in_degree    = [in_degree[nid]  for nid in all_node_ids]
    out_degree   = [out_degree[nid] for nid in all_node_ids]

    mean_in_degree = np.mean(in_degree) if in_degree else 0.0
    max_in_degree  = np.max(in_degree) if in_degree else 0.0
    std_in_degree  = np.std(in_degree) if in_degree else 0.0

    mean_out_degree = np.mean(out_degree) if out_degree else 0.0
    max_out_degree  = np.max(out_degree) if out_degree else 0.0
    std_out_degree  = np.std(out_degree) if out_degree else 0.0

    # ========================================================
    # FEATURE VECTOR
    # ========================================================

    features = {
        # global
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,

        # node types
        "instruction_ratio": instruction_ratio,
        "variable_ratio": variable_ratio,
        "constant_ratio": constant_ratio,

        # edge types
        "inst_control_inst_ratio": inst_control_inst_ratio,
        "inst_data_var_ratio": inst_data_var_ratio,
        "var_data_inst_ratio": var_data_inst_ratio,
        "const_data_inst_ratio": const_data_inst_ratio,
        "inst_call_inst_ratio": inst_call_inst_ratio,

        # degree stats
        "mean_in_degree": mean_in_degree,
        "max_in_degree": max_in_degree,
        "std_in_degree": std_in_degree,
        "mean_out_degree": mean_out_degree,
        "max_out_degree": max_out_degree,
        "std_out_degree": std_out_degree,
    }

    features.update(geometry_features)

    for term, count in node_text_counts.items():
        features[f"op_{term}"] = count
    if node_text_counts.get("load", 0) != 0 and node_text_counts.get("store") != 0:
        features["load_store_ratio"] = node_text_counts.get("load") / node_text_counts.get("store")

    # labels
    labels = graph_data.get("labels", {})
    for k, v in labels.items():
        features[k] = v

    return features

def build_feature_row(graph_path):
    """
    Load a graph JSON file and extract its feature vector.
    """
    graph_path = Path(graph_path)
    with graph_path.open("r") as f:
        graph_data = json.load(f)

    features = extract_graph_features(graph_data)
    return features


def build_feature_dataframe(graph_dir, kernel_subset=None, max_archives=None, num_workers=None):
    """
    Walk graph_dir and create a dataframe of graph features.
    If kernel_subset is provided, only scan those subdirectories.
    If max_archives is provided, only scan that many archive subdirectories.

    Returns a dataframe of custom graph features
    """

    graph_dir = Path(graph_dir)
    if isinstance(kernel_subset, str):
        kernel_subset = [kernel_subset]

    rows = []

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

        if not graph_paths:
            continue

        if num_workers is None:
            num_workers = min(32, (os.cpu_count() or 1))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for features in tqdm.tqdm(
                executor.map(build_feature_row, graph_paths),
                total=len(graph_paths),
                desc=f"Parsing graph files for kernel subset '{ks}'",
            ):
                features["kernel_type"] = ks
                rows.append(features)

    df = pd.DataFrame(rows)
    return df