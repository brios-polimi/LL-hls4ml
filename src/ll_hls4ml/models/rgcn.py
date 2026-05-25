"""Heterogeneous R-GCN for LLVM CDFG graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from ll_hls4ml.io.schema import EDGE_TYPES, EDGE_TYPES_WITH_ATTR, NODE_TYPES


class CDFGInputProjection(nn.Module):
    """
    Embeds opcode vocab per node type and positional arg encoding for edges.

    Node feature layout: [vocab_id]
    Edge attr layout:    [arg_position]
    """

    def __init__(
        self,
        node_vocab_sizes: dict[str, int],
        edge_pos_vocab_size: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.node_emb = nn.ModuleDict({
            nt: nn.Embedding(vocab_size + 1, hidden_dim, padding_idx=0)
            for nt, vocab_size in node_vocab_sizes.items()
        })
        self.edge_pos_emb = nn.Embedding(edge_pos_vocab_size + 1, hidden_dim, padding_idx=0)

    def forward(self, x_dict, edge_attr_dict):
        h_dict = {nt: self.node_emb[nt](x[:, 0]) for nt, x in x_dict.items()}
        edge_emb_dict = {
            et: self.edge_pos_emb(attr[:, 0]) for et, attr in edge_attr_dict.items()
        }
        return h_dict, edge_emb_dict


class CDFGConvLayer(nn.Module):
    """One heterogeneous message-passing step over all edge types."""

    def __init__(self, hidden_dim: int, aggr: str = "mean"):
        super().__init__()
        self.conv = HeteroConv(
            {
                et: SAGEConv(
                    in_channels=(hidden_dim, hidden_dim),
                    out_channels=hidden_dim,
                    aggr=aggr,
                )
                for et in EDGE_TYPES
            },
            aggr="mean",
        )
        self.norm = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in NODE_TYPES})

    def forward(self, h_dict, edge_index_dict, edge_emb_dict):
        h = {nt: h_dict[nt] for nt in NODE_TYPES}
        out = self.conv(h, edge_index_dict)
        for nt in NODE_TYPES:
            if nt not in out:
                out[nt] = h_dict[nt]
        return {nt: self.norm[nt](F.relu(h_out)) for nt, h_out in out.items()}


class CDFGRGCN(nn.Module):
    """Heterogeneous R-GCN for LLVM CDFG graphs with graph-level regression head."""

    def __init__(
        self,
        node_vocab_sizes: dict[str, int],
        edge_pos_vocab_size: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        pool: str = "mean",
        aggr: str = "mean",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pool = pool

        self.input_proj = CDFGInputProjection(
            node_vocab_sizes, edge_pos_vocab_size, hidden_dim
        )
        self.layers = nn.ModuleList([
            CDFGConvLayer(hidden_dim, aggr=aggr) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(NODE_TYPES), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: HeteroData):
        x_dict = {nt: data[nt].x.long() for nt in NODE_TYPES}
        edge_index_dict = {et: data[et].edge_index for et in EDGE_TYPES}
        edge_attr_dict = {
            et: data[et].edge_attr.long()
            for et in EDGE_TYPES_WITH_ATTR
            if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
        }

        h_dict, edge_emb_dict = self.input_proj(x_dict, edge_attr_dict)

        for i, layer in enumerate(self.layers):
            h_dict = layer(h_dict, edge_index_dict, edge_emb_dict)
            if i < len(self.layers) - 1:
                h_dict = {nt: self.dropout(h) for nt, h in h_dict.items()}

        pool_fn = {
            "mean": global_mean_pool,
            "sum": global_add_pool,
            "max": global_max_pool,
        }[self.pool]

        pooled = torch.cat(
            [pool_fn(h_dict[nt], data[nt].batch) for nt in NODE_TYPES],
            dim=-1,
        )
        return self.classifier(pooled)
