"""
model/fusion.py

Per-cause neural correction MLPs and the trust-gated fusion combining each
cause's closed-form physics intensity with a learned correction.

Design decisions
-----------------
1. One correction MLP and one gate MLP per cause, not a single shared
   module across causes -- the architecture design specifies four small
   per-cause MLPs, not one multi-output head, so trust weights stay
   individually interpretable per cause.
2. Fusion formula for the full model:
   lambda_c = lambda_phys_c * exp((1 - g_c) * delta_c), where
   g_c = sigmoid(gate_mlp_c(h, phys_c)) and delta_c is a bounded learned
   log-residual.  This is a physics-preserving multiplicative correction:
   g_c=1 gives exactly the physics hazard, delta_c=0 gives exactly the
   physics hazard, and delta_c may correct either upward or downward.
   The gate remains directly interpretable as how strongly the learned
   correction is suppressed.  The physics-only and neural-only ablations
   retain their exact definitions.

   This replaces the old convex blend of an absolute softplus intensity.
   The old parameterization made the learned branch live on a different
   scale from the physics hazard and, in the observed runs, reduced the
   learned contribution to only a few percent even under deliberate
   physics misspecification.  A bounded log-residual is numerically stable
   and is the natural parameterization for positive hazard rates.
3. Named TrustGatedFusion, not EvidentialFusion -- per the earlier brutal-
   review finding that "evidential fusion" overclaims for a plain sigmoid
   gate; this mechanism is a learned convex combination, described as such.
4. Physics intensities are fixed inputs, not backpropagated through --
   they come from the deterministic HazardEvaluator (Phase 3); only the
   correction and gate MLPs are learned parameters.

Data structures
----------------
CauseFusionHead -- one cause's correction + gate MLPs.
TrustGatedFusion -- all four cause heads plus combination logic.

Storage implications
---------------------
None -- model code only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CAUSE_NAMES = ("geo", "wave", "doppler", "current")


class CauseFusionHead(nn.Module):
    """One cause's neural correction MLP and trust-gate MLP.

    Three construction flags reproduce the three ``configs/paper1_baselines.yaml``
    ablations from *inside* this one class, so ``physics_only`` / ``neural_only``
    / ``pg_gtpp`` differ only in which sub-modules exist -- not in a separate
    code path -- and so parameter counts for the reduced ablations are
    honestly smaller (no correction_mlp/gate_mlp weights are allocated for a
    variant that doesn't use them):

    - ``learned_correction=False``: no ``correction_mlp`` is created; the
      neural correction term is a constant zero.
    - ``use_physics_intensity=False``: the physics intensity input is
      ignored (treated as zero) in the fused output.
    - ``use_fusion_gate=False``: no ``gate_mlp`` is created; the gate is a
      fixed constant instead of learned -- 1.0 if physics is the only term
      present (``physics_only``), 0.0 if the neural correction is the only
      term present (``neural_only``). Requesting ``use_fusion_gate=False``
      while both other flags are ``True`` is not one of the three named
      ablations and falls back to a fixed 0.5/0.5 blend (documented, not
      silently defaulted) since no config in this project asks for it.
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_width: int = 32,
        learned_correction: bool = True,
        use_physics_intensity: bool = True,
        use_fusion_gate: bool = True,
        fixed_gate: float | None = None,
    ) -> None:
        super().__init__()
        self.learned_correction = learned_correction
        self.use_physics_intensity = use_physics_intensity
        self.use_fusion_gate = use_fusion_gate
        if fixed_gate is not None and not (0.0 <= float(fixed_gate) <= 1.0):
            raise ValueError("fixed_gate must be in [0, 1]")
        self.fixed_gate = fixed_gate

        # The correction head must see the physics scale explicitly.  Without
        # this, the residual MLP only received the learned graph context and
        # had to infer which cause/intensity scale it was correcting.  We use
        # log1p(physics) as a numerically stable, positive-scale feature.
        correction_input_dim = hidden_dim + 1
        self.correction_mlp = (
            nn.Sequential(nn.Linear(correction_input_dim, mlp_width), nn.ReLU(), nn.Linear(mlp_width, 1))
            if learned_correction
            else None
        )
        # Start the learned log-residual at zero so the full model starts
        # exactly at the physics prior rather than injecting an arbitrary
        # learned hazard at initialization.
        if self.correction_mlp is not None:
            nn.init.zeros_(self.correction_mlp[-1].weight)
            nn.init.zeros_(self.correction_mlp[-1].bias)
        self.gate_mlp = (
            nn.Sequential(nn.Linear(hidden_dim + 1, mlp_width), nn.ReLU(), nn.Linear(mlp_width, 1))
            if use_fusion_gate
            else None
        )

    def forward(
        self, context: torch.Tensor, physics_intensity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        context: [E, hidden_dim]
        physics_intensity: [E]
        returns: (fused_intensity [E], trust_gate [E], neural_correction [E])
        """
        if self.learned_correction:
            physics_feature = torch.log1p(physics_intensity.clamp_min(0.0)).unsqueeze(-1)
            correction_input = torch.cat([context, physics_feature], dim=-1)
            correction_raw = self.correction_mlp(correction_input).squeeze(-1)
            # Bound the log-residual to keep hazards finite and prevent a
            # small number of windows from exploding the TPP integral.
            log_correction = 2.0 * torch.tanh(correction_raw)
        else:
            log_correction = torch.zeros_like(physics_intensity)

        physics_term = physics_intensity if self.use_physics_intensity else torch.zeros_like(physics_intensity)

        if self.use_fusion_gate:
            physics_feature = torch.log1p(physics_intensity.clamp_min(0.0)).unsqueeze(-1)
            gate_input = torch.cat([context, physics_feature], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(gate_input).squeeze(-1))
        elif self.use_physics_intensity and not self.learned_correction:
            gate = torch.ones_like(physics_intensity)      # physics_only: fused == physics
        elif self.learned_correction and not self.use_physics_intensity:
            gate = torch.zeros_like(physics_intensity)      # neural_only: fused == neural branch
        else:
            # When both physics and learned correction are active, the legacy
            # fallback is 0.5.  A diagnostic may explicitly set fixed_gate=0
            # to test the ungated residual path without changing the learned
            # representation or exposing the hidden operational state.
            fixed = 0.5 if self.fixed_gate is None else float(self.fixed_gate)
            gate = torch.full_like(physics_intensity, fixed)

        if self.use_physics_intensity and self.learned_correction:
            # Physics-preserving multiplicative correction.  The neural
            # branch is expressed relative to the physics scale, so it can
            # learn both under- and over-prediction without having to learn
            # the absolute hazard scale from scratch.
            fused = physics_term * torch.exp((1.0 - gate) * log_correction)
            # Keep the diagnostic "correction" in intensity units.
            # Preserve the actual physics scale; do not impose an absolute
            # hazard floor here because that would destroy the multiplicative
            # physics-preserving parameterization for tiny hazards.
            neural_correction = physics_term * torch.expm1(log_correction)
        elif self.learned_correction and not self.use_physics_intensity:
            # Genuine physics-free neural ablation.
            zero_physics_feature = torch.zeros_like(physics_intensity).unsqueeze(-1)
            neural_input = torch.cat([context, zero_physics_feature], dim=-1)
            neural_correction = F.softplus(self.correction_mlp(neural_input).squeeze(-1))
            fused = neural_correction
        else:
            neural_correction = torch.zeros_like(physics_intensity)
            fused = physics_term

        # Do not clamp the physics-guided fused hazard to an absolute floor.
        # Downstream likelihood code applies numerical floors only where
        # logarithms are evaluated.
        return fused, gate, neural_correction


class TrustGatedFusion(nn.Module):
    """All four cause-specific fusion heads, combined."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_width: int = 32,
        learned_correction: bool = True,
        use_physics_intensity: bool = True,
        use_fusion_gate: bool = True,
        fixed_gate: float | None = None,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                name: CauseFusionHead(
                    hidden_dim,
                    mlp_width,
                    learned_correction=learned_correction,
                    use_physics_intensity=use_physics_intensity,
                    use_fusion_gate=use_fusion_gate,
                    fixed_gate=fixed_gate,
                )
                for name in CAUSE_NAMES
            }
        )

    def forward(
        self, edge_context: torch.Tensor, physics_intensities: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        edge_context: [E, hidden_dim]
        physics_intensities: [E, 4] in CAUSE_NAMES column order
        returns: {cause_name: {"fused": [E], "gate": [E], "correction": [E]}}
        """
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for i, name in enumerate(CAUSE_NAMES):
            fused, gate, correction = self.heads[name](edge_context, physics_intensities[:, i])
            outputs[name] = {"fused": fused, "gate": gate, "correction": correction}
        return outputs

    @staticmethod
    def total_intensity(fusion_outputs: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        """Sum of the four fused cause-specific intensities."""
        return sum(fusion_outputs[name]["fused"] for name in CAUSE_NAMES)
