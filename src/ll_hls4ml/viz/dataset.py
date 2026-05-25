"""Distribution plots for preprocessed tensor datasets."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm


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

        inst_n = (
            g["instruction"].x.shape[0]
            if "instruction" in g.node_types and g["instruction"].x is not None
            else 0
        )
        var_n = (
            g["variable"].x.shape[0]
            if "variable" in g.node_types and g["variable"].x is not None
            else 0
        )
        const_n = (
            g["constant"].x.shape[0]
            if "constant" in g.node_types and g["constant"].x is not None
            else 0
        )

        node_counts["instruction"].append(inst_n)
        node_counts["variable"].append(var_n)
        node_counts["constant"].append(const_n)

        n_vec = np.array([inst_n, var_n, const_n], dtype=float)
        if n_vec.sum() > 0:
            p = n_vec / n_vec.sum()
            node_type_entropy.append(-np.sum(p * np.log(p + 1e-12)))

        if "instruction" in g.node_types and g["instruction"].x is not None:
            x = g["instruction"].x.float()
            node_counts["inst_feat_mean"].append(x.mean().item())
            node_counts["inst_feat_sparsity"].append((x == 0).float().mean().item())

        def get_edge(etype):
            return g[etype].edge_index if etype in g.edge_types else None

        edges = {
            "control": get_edge(("instruction", "control", "instruction")),
            "call": get_edge(("instruction", "call", "instruction")),
            "inst_data_var": get_edge(("instruction", "data", "variable")),
            "var_data_inst": get_edge(("variable", "data", "instruction")),
            "const_data_inst": get_edge(("constant", "data", "instruction")),
        }

        e_counts = {k: (v.shape[1] if v is not None else 0) for k, v in edges.items()}
        for k, v in e_counts.items():
            edge_counts[k].append(v)

        e_vec = np.array(list(e_counts.values()), dtype=float)
        if e_vec.sum() > 0:
            p = e_vec / e_vec.sum()
            edge_type_entropy.append(-np.sum(p * np.log(p + 1e-12)))

        ctrl = e_counts["control"] + e_counts["call"]
        data = e_counts["inst_data_var"] + e_counts["var_data_inst"] + e_counts["const_data_inst"]
        edge_counts["ctrl_data_ratio"].append(ctrl / data if data > 0 else float("nan"))

        inst_deg, var_deg, const_deg = 0, 0, 0
        for k, e in edges.items():
            if e is None:
                continue
            n_edges = e.shape[1]
            if k in ["control", "call"]:
                inst_deg += n_edges
                control_degrees.append(n_edges)
            elif k == "inst_data_var":
                inst_deg += n_edges
                var_deg += n_edges
                data_degrees.append(n_edges)
            elif k == "var_data_inst":
                var_deg += n_edges
                inst_deg += n_edges
                data_degrees.append(n_edges)
            elif k == "const_data_inst":
                const_deg += n_edges
                inst_deg += n_edges
                data_degrees.append(n_edges)

        inst_degrees.append(inst_deg)
        var_degrees.append(var_deg)
        const_degrees.append(const_deg)

        if inst_deg + var_deg + const_deg == 0:
            isolated_graphs += 1

        ctrl_key = ("instruction", "control", "instruction")
        if ctrl_key in g.edge_types and g[ctrl_key].edge_attr is not None:
            control_positions.extend(g[ctrl_key].edge_attr.reshape(-1).tolist())

        vdi_key = ("variable", "data", "instruction")
        if vdi_key in g.edge_types and g[vdi_key].edge_attr is not None:
            var_data_positions.extend(g[vdi_key].edge_attr.reshape(-1).tolist())

        cdi_key = ("constant", "data", "instruction")
        if cdi_key in g.edge_types and g[cdi_key].edge_attr is not None:
            const_data_positions.extend(g[cdi_key].edge_attr.reshape(-1).tolist())

        if hasattr(g, "y") and g.y is not None:
            labels.extend(g.y.reshape(-1).tolist())

    _plot_distributions(
        node_counts,
        edge_counts,
        inst_degrees,
        var_degrees,
        const_degrees,
        edge_type_entropy,
        node_type_entropy,
        control_positions,
        var_data_positions,
        const_data_positions,
        labels,
        log_scale,
    )

    print(f"Graphs:               {len(files)}")
    print(f"Isolated graphs:      {isolated_graphs} ({100 * isolated_graphs / len(files):.1f}%)")
    print(f"Avg instruction nodes:{np.mean(node_counts['instruction']):.1f}")
    print(f"Avg variable nodes:   {np.mean(node_counts['variable']):.1f}")
    print(f"Avg constant nodes:   {np.mean(node_counts['constant']):.1f}")
    print(f"Avg control edges:    {np.mean(edge_counts['control']):.1f}")
    avg_data = (
        np.mean(edge_counts["inst_data_var"])
        + np.mean(edge_counts["var_data_inst"])
        + np.mean(edge_counts["const_data_inst"])
    )
    print(f"Avg data edges:       {avg_data:.1f}")
    print(f"Avg ctrl/data ratio:  {np.nanmean(edge_counts['ctrl_data_ratio']):.2f}")
    if labels:
        unique, cnts = np.unique(labels, return_counts=True)
        print("Label distribution:", dict(zip(unique.tolist(), cnts.tolist())))


def _plot_distributions(
    node_counts,
    edge_counts,
    inst_degrees,
    var_degrees,
    const_degrees,
    edge_type_entropy,
    node_type_entropy,
    control_positions,
    var_data_positions,
    const_data_positions,
    labels,
    log_scale,
):
    node_colors = {
        "instruction": "steelblue",
        "variable": "orange",
        "constant": "green",
    }
    edge_colors = {
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

    for key in ["instruction", "variable", "constant"]:
        plot_hist(
            node_counts[key],
            f"Distribution of {key.capitalize()} Nodes",
            color=node_colors[key],
        )

    for key in ["control", "call", "inst_data_var", "var_data_inst", "const_data_inst"]:
        plot_hist(edge_counts[key], f"Distribution of {key} Edges", color=edge_colors[key])

    plt.figure()
    for ntype, color in node_colors.items():
        plt.hist(node_counts[ntype], bins=50, alpha=0.5, label=ntype, color=color)
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Node counts by type")
    plt.show()

    plot_hist(edge_counts["ctrl_data_ratio"], "Control/data edge ratio per graph")
    plot_hist(inst_degrees, "Instruction total degree")
    plot_hist(var_degrees, "Variable total degree")
    plot_hist(const_degrees, "Constant total degree")
    plot_hist(edge_type_entropy, "Edge-type entropy per graph")
    plot_hist(node_type_entropy, "Node-type entropy per graph")
    plot_hist(node_counts["inst_feat_sparsity"], "Instruction feature sparsity")

    if control_positions:
        plot_hist(control_positions, "Control edge position attribute")
    if var_data_positions:
        plot_hist(var_data_positions, "Var→Inst position attribute")
    if const_data_positions:
        plot_hist(const_data_positions, "Const→Inst position attribute")
    if labels:
        plot_hist(labels, "Label distribution")
