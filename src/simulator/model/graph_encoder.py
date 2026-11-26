"""
model/graph_encoder.py

A small, heterogeneous (typed) graph transformer encoder: 2 layers,
typed Q/K/V projections per node type (USV/AUV/IoUT), attention over the
candidate-edge structure with a physics-biased logit term.

Design decisions
-----------------
1. Heterogeneous, not homogeneous: USV/AUV/IoUT play structurally different
   roles, so separate per-type Q/K/V linear projections let attention
   specialize by node type without adding a second architectural module.
2. Time encoding is folded into the node encoder via a small GRU over the
   k-step window, rather than a separate memory-network stack, per the
   architecture design's anti-stacking principle.
3. Physics-biased attention is configurable: when enabled, edge attention
   logits add a learned function of the edge's physics hazard vector to the
   standard dot-product logit. It can be disabled for a genuinely
   physics-free neural ablation.
4. Two layers, fixed, matching the architecture design's explicit warning
   against over-stacking on these small acoustic graphs.

Data structures
----------------
NodeTemporalEncoder, GraphTransformerLayer, GraphEncoder (the full stack).

Storage implications
---------------------
None -- model code only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_NODE_TYPES = 3  # USV, AUV, IoUT
N_EDGE_KINDS = 2  # usv_auv, auv_iout


class NodeTemporalEncoder(nn.Module):
    """Summarizes a node's [k, F_NODE] window plus its type into one
    [hidden_dim] vector via a GRU followed by a type-embedding concat."""

    def __init__(self, f_node: int, hidden_dim: int, type_embed_dim: int = 16) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=f_node, hidden_size=hidden_dim, batch_first=True)
        self.type_embedding = nn.Embedding(N_NODE_TYPES, type_embed_dim)
        self.project = nn.Linear(hidden_dim + type_embed_dim, hidden_dim)

    def forward(self, node_features: torch.Tensor, node_type_ids: torch.Tensor) -> torch.Tensor:
        b, k, v, f_node = node_features.shape
        flat = node_features.permute(0, 2, 1, 3).reshape(b * v, k, f_node)
        _, h_n = self.gru(flat)
        h = h_n.squeeze(0).view(b, v, -1)
        type_embed = self.type_embedding(node_type_ids)
        combined = torch.cat([h, type_embed], dim=-1)
        return self.project(combined)


class GraphTransformerLayer(nn.Module):
    """One heterogeneous multi-head attention + feedforward block.

    ``use_physics_attention`` selects whether the learned physics bias is
    added to attention logits.
    block, operating on a flattened [N, hidden_dim] node tensor and an
    undirected edge list (aggregated in both directions internally)."""

    def __init__(self, hidden_dim: int, n_heads: int = 4, physics_dim: int = 4, use_physics_attention: bool = True) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_physics_attention = use_physics_attention
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(N_NODE_TYPES)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(N_NODE_TYPES)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(N_NODE_TYPES)])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.physics_logit_mlp = nn.Sequential(
            nn.Linear(physics_dim, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, n_heads)
        )
        self.physics_gate = nn.Parameter(torch.tensor(0.5))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def _typed_projection(self, x: torch.Tensor, node_type_ids: torch.Tensor, proj_list: nn.ModuleList) -> torch.Tensor:
        out = torch.zeros_like(x)
        for type_id in range(N_NODE_TYPES):
            mask = node_type_ids == type_id
            if mask.any():
                out[mask] = proj_list[type_id](x[mask])
        return out

    def forward(
        self,
        node_embed: torch.Tensor,
        node_type_ids_flat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_physics: torch.Tensor,
        node_mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        n = node_embed.shape[0]
        q = self._typed_projection(node_embed, node_type_ids_flat, self.q_proj).view(n, self.n_heads, self.head_dim)
        k = self._typed_projection(node_embed, node_type_ids_flat, self.k_proj).view(n, self.n_heads, self.head_dim)
        v = self._typed_projection(node_embed, node_type_ids_flat, self.v_proj).view(n, self.n_heads, self.head_dim)

        if edge_index.shape[1] == 0:
            attn_out = torch.zeros_like(node_embed)
        else:
            src, dst = edge_index[0], edge_index[1]
            src_full = torch.cat([src, dst])
            dst_full = torch.cat([dst, src])
            physics_full = torch.cat([edge_physics, edge_physics], dim=0)

            q_dst = q[dst_full]
            k_src = k[src_full]
            v_src = v[src_full]

            dot = (q_dst * k_src).sum(-1) / (self.head_dim ** 0.5)
            if self.use_physics_attention:
                physics_bias = self.physics_logit_mlp(physics_full)
                logits = dot + torch.sigmoid(self.physics_gate) * physics_bias
            else:
                logits = dot

            attn_out = self._segment_softmax_aggregate(logits, dst_full, v_src, n)

        attn_out = self.out_proj(attn_out.reshape(n, self.hidden_dim))
        node_embed = self.norm1(node_embed + attn_out)
        node_embed = self.norm2(node_embed + self.ffn(node_embed))
        return node_embed * node_mask_flat.unsqueeze(-1)

    @staticmethod
    def _segment_softmax_aggregate(
        logits: torch.Tensor, dst: torch.Tensor, values: torch.Tensor, n_nodes: int
    ) -> torch.Tensor:
        heads, head_dim = values.shape[1], values.shape[2]

        max_per_dst = torch.full((n_nodes, heads), float("-inf"), device=logits.device)
        max_per_dst = max_per_dst.scatter_reduce(
            0, dst.unsqueeze(-1).expand(-1, heads), logits, reduce="amax", include_self=True
        )
        max_per_dst = torch.nan_to_num(max_per_dst, neginf=0.0)

        shifted = logits - max_per_dst[dst]
        exp_logits = shifted.exp()

        sum_per_dst = torch.zeros((n_nodes, heads), device=logits.device)
        sum_per_dst = sum_per_dst.scatter_add(0, dst.unsqueeze(-1).expand(-1, heads), exp_logits)
        sum_per_dst = sum_per_dst.clamp_min(1e-12)

        weights = exp_logits / sum_per_dst[dst]

        weighted_values = weights.unsqueeze(-1) * values
        out = torch.zeros((n_nodes, heads, head_dim), device=logits.device)
        out = out.scatter_add(0, dst.view(-1, 1, 1).expand(-1, heads, head_dim), weighted_values)
        return out


class GraphEncoder(nn.Module):
    """Full model encoder: temporal node summarization + 2 physics-biased
    heterogeneous graph transformer layers."""

    def __init__(self, f_node: int, hidden_dim: int = 64, n_heads: int = 4, physics_dim: int = 4, use_physics_attention: bool = True) -> None:
        super().__init__()
        self.node_encoder = NodeTemporalEncoder(f_node, hidden_dim)
        self.layer1 = GraphTransformerLayer(hidden_dim, n_heads, physics_dim, use_physics_attention=use_physics_attention)
        self.layer2 = GraphTransformerLayer(hidden_dim, n_heads, physics_dim, use_physics_attention=use_physics_attention)

    def forward(
        self,
        node_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_physics: torch.Tensor,
    ) -> torch.Tensor:
        """Returns flattened node context embeddings [B*V_max, hidden_dim]."""
        b, k, v_max, f_node = node_features.shape
        node_embed = self.node_encoder(node_features, node_type_ids)

        node_embed_flat = node_embed.reshape(b * v_max, -1)
        node_type_flat = node_type_ids.reshape(b * v_max)
        node_mask_flat = node_mask.reshape(b * v_max)

        node_embed_flat = self.layer1(node_embed_flat, node_type_flat, edge_index, edge_physics, node_mask_flat)
        node_embed_flat = self.layer2(node_embed_flat, node_type_flat, edge_index, edge_physics, node_mask_flat)
        return node_embed_flat
