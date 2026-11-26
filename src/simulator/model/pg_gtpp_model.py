"""
model/pg_gtpp_model.py

Assembles the three components into the full PG-GTPP model: graph encoder
(spatial + lightweight temporal), trust-gated fusion (physics + neural per
cause), and the mark decoder. No new architectural machinery is introduced
here -- this module only wires the pieces together and defines how an
edge's context embedding is derived from its two endpoint node embeddings.

Design decisions
-----------------
1. Edge context = concatenation of endpoint node embeddings -- the
   simplest way to derive an edge-level representation with no additional
   learned parameters beyond what's already inside the fusion/mark heads'
   own trunks, avoiding a fourth "edge embedding" module.
2. One forward pass, three outputs -- the encoder is not run twice for the
   two downstream heads (fusion, marks).
3. Total parameter count reported at construction so it's trivially
   checkable against the architecture design's 1-3M parameter target.

Data structures
----------------
PGGTPPOutput -- fusion outputs plus mark prediction.
PGGTPPModel -- the full model.

Storage implications
---------------------
None -- model code only. Checkpointing is a training-phase concern.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .fusion import TrustGatedFusion
from .graph_encoder import GraphEncoder
from .mark_decoder import MarkDecoder, MarkPrediction


class OperationalEdgeTemporalEncoder(nn.Module):
    """Dedicated pair-history encoder for observable operational state.

    It consumes only raw trajectory history at each endpoint (position,
    velocity, battery and FIM uncertainty).  No truth-only latent variable or
    simulator event information is used.  The encoder is deliberately small
    and only feeds the fusion/residual pathway, leaving the mark head and the
    original graph representation unchanged.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Relative x/y, endpoint velocities, battery and FIM for both ends.
        # Positions are scaled before the GRU to avoid raw metre magnitudes.
        self.input_proj = nn.Linear(10, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        b, k, v, _ = node_features.shape
        if edge_index.shape[1] == 0:
            return node_features.new_zeros((0, self.gru.hidden_size))
        nodes = node_features.permute(0, 2, 1, 3)  # [B,V,K,F]
        src, dst = edge_index[0], edge_index[1]
        sb, sl = src // v, src % v
        db, dl = dst // v, dst % v
        si = nodes[sb, sl]
        sj = nodes[db, dl]
        # [E,K,8], all features are observable.
        rel_xy = torch.tanh((si[..., 0:2] - sj[..., 0:2]) / 1000.0)
        pair = torch.cat([
            rel_xy, si[..., 3:5], sj[..., 3:5],
            si[..., 6:8], sj[..., 6:8],
        ], dim=-1)
        x = torch.tanh(self.input_proj(pair))
        _, h = self.gru(x)
        return self.norm(h[-1])


@dataclass
class PGGTPPOutput:
    fusion: dict[str, dict[str, torch.Tensor]]
    marks: MarkPrediction

    def total_intensity(self) -> torch.Tensor:
        return TrustGatedFusion.total_intensity(self.fusion)

    def cause_intensities(self) -> dict[str, torch.Tensor]:
        """Return fused cause-specific intensities for competing-risk likelihoods."""
        return {name: self.fusion[name]["fused"] for name in self.fusion}


class PGGTPPModel(nn.Module):
    """The full PG-GTPP model: GraphEncoder -> edge context -> TrustGatedFusion + MarkDecoder."""

    def __init__(
        self,
        f_node: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        physics_dim: int = 4,
        fusion_mlp_width: int = 32,
        mark_trunk_width: int = 32,
        learned_correction: bool = True,
        use_physics_intensity: bool = True,
        use_fusion_gate: bool = True,
        fixed_fusion_gate: float | None = None,
        decouple_mark_head: bool = False,
        mark_physics_skip: bool = False,
        use_physics_attention: bool = True,
        use_operational_temporal: bool = False,
    ) -> None:
        """
        ``learned_correction`` / ``use_physics_intensity`` / ``use_fusion_gate``
        reproduce the three ``configs/paper1_baselines.yaml`` ablations
        (defaults = ``pg_gtpp``, the full model):

            physics_only: learned_correction=False, use_fusion_gate=False
            neural_only:  use_physics_intensity=False, use_fusion_gate=False
            pg_gtpp:      all True (default)

        These flags only gate ``TrustGatedFusion``/``CauseFusionHead``
        construction (see ``fusion.py``) -- the graph encoder's own
        attention-level physics bias (``GraphEncoder``'s per-layer
        ``physics_gate`` parameter) is unaffected and present in all three,
        matching what the originally-uploaded checkpoints show: that
        parameter exists in physics_only/neural_only/pg_gtpp alike, so it is
        not one of the three things this ablation set is meant to toggle.

        ``decouple_mark_head`` (default False -- OFF, so existing results
        are unaffected unless explicitly requested): every baseline in
        ``model/baselines/`` has exactly ONE intensity output feeding back
        into its shared encoder; PG-GTPP's TrustGatedFusion has FOUR
        cause-specific heads, each with its own correction_mlp AND gate_mlp
        (8 sub-modules total), all consuming gradient signal from the exact
        same edge_context tensor the mark_decoder also reads unmodified.
        Empirically (see docs/BASELINES.md, 'Improving PG-GTPP's mark
        accuracy'), PG-GTPP's mark loss is consistently higher than
        similar-hidden-dim baselines even when its TPP loss is comparable or
        better -- consistent with the shared encoder representation being
        pulled harder toward what the 8-head fusion module needs, at the
        mark decoder's expense. When True, this inserts one small
        Linear+ReLU adapter between edge_context and the mark decoder,
        giving marks a task-specific transformation instead of the literal
        same tensor every fusion head reads -- standard multi-task-learning
        practice (shared trunk, per-task head), not a new capability, and
        adds only hidden_dim*2 -> hidden_dim*2 parameters (a few thousand at
        default hidden_dim=64). This is a hypothesis to A/B against the
        default (False), not a replacement for it -- report both if used,
        per the anti-stacking principle in the brutal-review roadmap
        document (Phase 1, item D): don't silently swap the headline model,
        ablate the change and show the delta.

        ``use_physics_attention`` (default True): controls the encoder-level
        physics bias in graph attention. The full PG-GTPP and physics-only
        ablation keep this True; the new ``neural_only`` ablation sets it
        False so that neural-only is genuinely independent of the physics
        covariates throughout the forward path.

        ``mark_physics_skip`` (default False -- OFF): a second, independent
        hypothesis for the same mark-accuracy gap, orthogonal to
        ``decouple_mark_head`` (this flag does not touch the encoder-sharing
        question at all, and the two can in principle be combined, though
        this project deliberately tests them one at a time so a result is
        interpretable -- see docs/BASELINES.md, 'Does raw physics help
        marks directly'). Hypothesis: DySAT-style -- the strongest baseline
        on every mark metric in the full comparison -- has no physics
        covariates at all, yet beats PG-GTPP by a wide margin on marks. If
        PG-GTPP's mark decoder is bottlenecked by having to infer
        physics-relevant signal (SNR margin, wave state, etc.) purely
        through the 2-layer graph encoder's abstraction rather than reading
        it directly, giving the mark decoder a direct skip connection to
        the raw ``edge_physics`` vector (geo/wave/doppler/current, the same
        4 values already fed to the fusion module) should help if that
        bottleneck is real. When True, ``edge_physics`` is concatenated to
        the mark decoder's input (after the optional ``decouple_mark_head``
        adapter, if both are enabled) -- adds only
        ``physics_dim`` (4) extra input features, i.e. a few hundred extra
        parameters in the mark decoder's first layer, not a new module.
        Like ``decouple_mark_head``, this is a hypothesis to A/B against
        the default and report honestly either way, not a guaranteed fix.
        """
        super().__init__()
        self.encoder = GraphEncoder(
            f_node=f_node, hidden_dim=hidden_dim, n_heads=n_heads, physics_dim=physics_dim,
            use_physics_attention=use_physics_attention,
        )
        self.use_operational_temporal = use_operational_temporal
        self.operational_temporal = (
            OperationalEdgeTemporalEncoder(hidden_dim) if use_operational_temporal else None
        )
        fusion_context_dim = hidden_dim * (3 if use_operational_temporal else 2)
        self.fusion = TrustGatedFusion(
            hidden_dim=fusion_context_dim,
            mlp_width=fusion_mlp_width,
            learned_correction=learned_correction,
            use_physics_intensity=use_physics_intensity,
            use_fusion_gate=use_fusion_gate,
            fixed_gate=fixed_fusion_gate,
        )
        self.mark_physics_skip = mark_physics_skip
        mark_input_dim = hidden_dim * 2 + (physics_dim if mark_physics_skip else 0)
        self.mark_decoder = MarkDecoder(hidden_dim=mark_input_dim, trunk_width=mark_trunk_width)
        self.decouple_mark_head = decouple_mark_head
        self.mark_adapter = (
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU())
            if decouple_mark_head
            else None
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_physics: torch.Tensor,
    ) -> PGGTPPOutput:
        """
        node_features: [B, k, V_max, F_NODE]
        node_type_ids: [B, V_max]
        node_mask: [B, V_max]
        edge_index: [2, E] -- indices into the flattened [B*V_max] node axis
        edge_physics: [E, 4] -- (geo, wave, doppler, current)
        """
        node_embed = self.encoder(node_features, node_type_ids, node_mask, edge_index, edge_physics)

        src, dst = edge_index[0], edge_index[1]
        edge_context = torch.cat([node_embed[src], node_embed[dst]], dim=-1)
        if self.operational_temporal is not None:
            operational_context = self.operational_temporal(node_features, edge_index)
            edge_context_fusion = torch.cat([edge_context, operational_context], dim=-1)
        else:
            edge_context_fusion = edge_context

        fusion_outputs = self.fusion(edge_context_fusion, edge_physics)
        mark_input = self.mark_adapter(edge_context) if self.mark_adapter is not None else edge_context
        if self.mark_physics_skip:
            mark_input = torch.cat([mark_input, edge_physics], dim=-1)
        marks = self.mark_decoder(mark_input)

        return PGGTPPOutput(fusion=fusion_outputs, marks=marks)


    def predict_intensity_path(
        self,
        base_output: PGGTPPOutput,
        future_cause_intensities: torch.Tensor | None = None,
        scale_factors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create a time-varying competing-risk intensity path.

        Preferred input is ``future_cause_intensities`` with shape [K,E,4],
        generated from future physics states. The returned tensor has shape
        [K,E] and is the sum over causes. ``scale_factors`` remains only as a
        backwards-compatible fallback.
        """
        if future_cause_intensities is not None:
            return future_cause_intensities.clamp_min(1e-8).sum(dim=-1)
        lam = base_output.total_intensity().clamp_min(1e-8)
        if scale_factors is None:
            return lam.unsqueeze(0)
        factors = scale_factors.reshape(-1, 1).to(lam.device)
        return lam.unsqueeze(0) * factors

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
