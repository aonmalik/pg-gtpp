"""
tests/test_baselines.py

Pytest suite for the 7 literature baselines in ``model/baselines/``.
Mirrors ``tests/test_phase5_model.py``'s fixture pattern (real simulator
pipeline, not synthetic tensors) and ``tests/test_phase6_training.py``'s
loss/gradient checks, so every baseline is held to the same bar as
PGGTPPModel itself before being trusted for the full comparison run
(``scripts/run_paper1_full_comparison.py``).

What is (and is not) checked here
----------------------------------
- Forward pass runs and produces correctly-shaped, finite, non-negative
  intensities and valid mark-distribution parameters, on a real window
  batch built from the actual simulator/window_builder pipeline.
- Gradients flow into every parameter (nothing accidentally detached).
- ``compute_loss`` (the SAME loss function PG-GTPP itself trains against)
  accepts each baseline's output unchanged.
- ``training.eval.evaluate`` (the SAME function that produced
  auc/ece/pdr_mae/etc. for the 3 existing ablations) runs unchanged
  against each baseline.
- A short (a few steps) real ``train()`` call does not raise and produces
  finite losses -- NOT a claim that any baseline has converged; that is
  the job of the full run in ``scripts/run_paper1_full_comparison.py``.
- ``physics_soft_reg``'s auxiliary loss term specifically: it is checked
  to be finite, non-negative, and to actually change the total loss
  relative to plain ``compute_loss`` (i.e. the regularizer is not a
  silent no-op).

This suite does not check accuracy/convergence -- that is an empirical
question for the full run, not a unit-test concern.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from simulator.dataset.window_builder import build_window, collate_windows
from simulator.event_generator import EventGenerator
from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.mission_generator import MissionGenerator
from simulator.model.baselines import BASELINE_REGISTRY
from simulator.model.baselines.physics_soft_reg import physics_regularized_loss
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.physics_intensity import HazardEvaluator
from simulator.random_manager import RandomManager
from simulator.trajectory_generator import TrajectoryGenerator
from simulator.trajectory_storage import F_NODE
from simulator.training.eval import evaluate
from simulator.training.loss import compute_loss
from simulator.training.train import MissionCache, TrainConfig, train


# --------------------------------------------------------------------- #
# Fixtures -- identical pattern to test_phase5_model.py, so baselines are
# exercised against the same real pipeline PGGTPPModel already is.
# --------------------------------------------------------------------- #
@pytest.fixture
def fast_config():
    base = default_config(master_seed=5)
    small_fleet = FleetConfig(
        usv_count=1, n_auv_min=3, n_auv_max=4, m_iout_min=6, m_iout_max=8,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    short_envelope = MissionEnvelopeConfig(
        duration_s=300.0, node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m, depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state,
    )
    config = MissionConfig(
        master_seed=5, split="train", wave=base.wave, current=base.current,
        acoustic=base.acoustic, fleet=small_fleet, envelope=short_envelope,
        hazard=base.hazard, graph=base.graph,
    )
    config.validate()
    return config


@pytest.fixture
def mission_artifacts(fast_config):
    rm = RandomManager(master_seed=5)
    mission_id = "baseline_fixture"
    state = MissionGenerator(fast_config, rm).generate(mission_id)
    array, node_ids = TrajectoryGenerator(fast_config, rm).propagate(state)
    wf = WaveField.sample(fast_config.wave, rm.get_numpy_rng(mission_id, "wave"))
    cf = CurrentField.sample(fast_config.current, rm.get_numpy_rng(mission_id, "current"))
    ch = AcousticChannel(fast_config.acoustic)
    node_types = {n.node_id: n.node_type for n in state.nodes}
    timestamps = np.arange(array.shape[0], dtype=np.float64) / fast_config.envelope.node_sample_rate_hz
    events = EventGenerator(fast_config, rm, ch, wf, cf).generate(mission_id, array, node_ids, node_types, timestamps)
    evaluator = HazardEvaluator(ch, wf, cf, fast_config.hazard)
    return {
        "config": fast_config, "mission_id": mission_id, "array": array, "node_ids": node_ids,
        "node_types": node_types, "timestamps": timestamps, "events": events, "evaluator": evaluator,
    }


@pytest.fixture
def sample_batch(mission_artifacts):
    m = mission_artifacts
    samples = [
        build_window(
            m["mission_id"], m["array"], m["node_ids"], m["node_types"], m["timestamps"], m["events"],
            m["evaluator"], m["config"].graph.candidate_range_factor,
            reference_t_idx=t_idx, window_length=8, lookahead_s=30.0,
        )
        for t_idx in [40, 80, 120, 160]
    ]
    return collate_windows(samples)


@pytest.fixture
def mission_cache(fast_config):
    return MissionCache.generate(fast_config, master_seed=5, n_missions=3)


# --------------------------------------------------------------------- #
# Per-baseline forward-pass, gradient, loss, and eval checks
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(BASELINE_REGISTRY.keys()))
def test_forward_pass_shapes_and_validity(name, sample_batch):
    model_cls = BASELINE_REGISTRY[name]
    model = model_cls(f_node=F_NODE, hidden_dim=16)  # small hidden_dim for test speed
    out = model(
        sample_batch.node_features, sample_batch.node_type_ids, sample_batch.node_mask,
        sample_batch.edge_index, sample_batch.edge_physics,
    )
    n_edges = sample_batch.edge_index.shape[1]

    intensity = out.total_intensity()
    assert intensity.shape == (n_edges,), f"{name}: intensity shape {intensity.shape} != ({n_edges},)"
    assert torch.isfinite(intensity).all(), f"{name}: non-finite intensity"
    assert (intensity >= 0).all(), f"{name}: negative intensity (must be softplus'd)"

    assert out.marks.pdr_mean().shape == (n_edges,)
    assert out.marks.throughput_mean().shape == (n_edges,)
    assert out.marks.delay_mean().shape == (n_edges,)
    assert torch.isfinite(out.marks.pdr_mean()).all()
    assert (out.marks.pdr_mean() >= 0).all() and (out.marks.pdr_mean() <= 1).all(), \
        f"{name}: PDR mean outside [0,1] -- Beta head misconfigured"


@pytest.mark.parametrize("name", list(BASELINE_REGISTRY.keys()))
def test_gradients_flow_into_every_parameter(name, sample_batch):
    model_cls = BASELINE_REGISTRY[name]
    model = model_cls(f_node=F_NODE, hidden_dim=16)
    out = model(
        sample_batch.node_features, sample_batch.node_type_ids, sample_batch.node_mask,
        sample_batch.edge_index, sample_batch.edge_physics,
    )
    loss_out = compute_loss(out, sample_batch)
    loss_out.total_loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"{name}: no gradient reached parameters: {missing}"


@pytest.mark.parametrize("name", list(BASELINE_REGISTRY.keys()))
def test_evaluate_runs_unchanged(name, sample_batch):
    """The SAME eval.py::evaluate() used for the 3 existing ablations must
    accept every baseline's output with zero special-casing."""
    model_cls = BASELINE_REGISTRY[name]
    model = model_cls(f_node=F_NODE, hidden_dim=16)
    metrics = evaluate(model, sample_batch)
    assert np.isfinite(metrics.val_loss)
    assert metrics.n_edges == sample_batch.edge_index.shape[1]
    # AUC/ECE may be nan on this tiny fixture batch if it happens to contain
    # only one class of outcome -- that is a fixture-scale artifact, not a
    # baseline-correctness failure, so it is not asserted here.


@pytest.mark.parametrize("name", list(BASELINE_REGISTRY.keys()))
def test_short_real_training_run_is_finite(name, mission_cache):
    """A few real optimizer steps against MissionCache -- not a
    convergence claim, just "does the full train() loop run cleanly"."""
    model_cls = BASELINE_REGISTRY[name]
    torch.manual_seed(0)
    model = model_cls(f_node=F_NODE, hidden_dim=16)
    config = TrainConfig(n_steps=5, batch_size=4, window_length=6, lookahead_s=30.0, log_every=0)
    history = train(model, mission_cache, config)
    assert len(history.total_losses) > 0
    assert all(np.isfinite(v) for v in history.total_losses)


# --------------------------------------------------------------------- #
# physics_soft_reg-specific: the auxiliary term must be real, not a no-op
# --------------------------------------------------------------------- #
def test_physics_soft_reg_auxiliary_term_is_active(sample_batch):
    model_cls = BASELINE_REGISTRY["physics_soft_reg"]
    model = model_cls(f_node=F_NODE, hidden_dim=16)
    out = model(
        sample_batch.node_features, sample_batch.node_type_ids, sample_batch.node_mask,
        sample_batch.edge_index, sample_batch.edge_physics,
    )
    base = compute_loss(out, sample_batch)
    reg_out = physics_regularized_loss(
        out, sample_batch, sample_batch.edge_physics, physics_reg_weight=0.1,
    )
    assert torch.isfinite(reg_out.physics_reg_loss)
    assert reg_out.physics_reg_loss >= 0
    # With a nonzero weight, the regularized total must differ from the
    # unregularized total unless the term happens to be exactly zero
    # (astronomically unlikely at random init) -- guards against a wiring
    # bug that silently drops the auxiliary term.
    assert not torch.isclose(reg_out.total_loss, base.total_loss), \
        "physics_reg term appears to be a no-op -- check physics_regularized_loss wiring"


def test_baseline_registry_has_seven_entries():
    assert len(BASELINE_REGISTRY) == 7, (
        f"expected 7 baselines (one per literature-gap row, see baselines/__init__.py), "
        f"got {len(BASELINE_REGISTRY)}: {list(BASELINE_REGISTRY.keys())}"
    )


# --------------------------------------------------------------------- #
# PGGTPPModel ablation flags (learned_correction / use_physics_intensity /
# use_fusion_gate) -- added so scripts/run_paper1_full_comparison.py can
# reproduce physics_only/neural_only/pg_gtpp on the same corpus as the 7
# baselines. Not baseline-registry tests, but live in this file since they
# exist for the same end-to-end comparison run.
# --------------------------------------------------------------------- #
from simulator.model.fusion import CAUSE_NAMES, TrustGatedFusion
from simulator.model.pg_gtpp_model import PGGTPPModel


def test_physics_only_flags_disable_correction_and_gate_params():
    """physics_only must have NO correction_mlp/gate_mlp parameters at all
    (not just zeroed-out ones) -- an honest ablation, not a masked one."""
    fusion = TrustGatedFusion(hidden_dim=16, learned_correction=False, use_fusion_gate=False)
    for name in CAUSE_NAMES:
        head = fusion.heads[name]
        assert head.correction_mlp is None
        assert head.gate_mlp is None


def test_physics_only_fused_output_equals_physics_intensity():
    torch.manual_seed(0)
    fusion = TrustGatedFusion(hidden_dim=16, learned_correction=False, use_fusion_gate=False)
    context = torch.randn(5, 16)
    physics = torch.rand(5, 4).clamp_min(1e-3)
    out = fusion(context, physics)
    for i, name in enumerate(CAUSE_NAMES):
        assert torch.allclose(out[name]["fused"], physics[:, i]), (
            f"physics_only cause={name}: fused output must equal the physics intensity exactly "
            f"(gate forced to 1, no learned correction contributes)"
        )


def test_neural_only_flags_disable_physics_and_gate_params():
    fusion = TrustGatedFusion(hidden_dim=16, use_physics_intensity=False, use_fusion_gate=False)
    for name in CAUSE_NAMES:
        assert fusion.heads[name].gate_mlp is None
        assert fusion.heads[name].correction_mlp is not None  # learned_correction stays True


def test_neural_only_fused_output_ignores_physics_intensity():
    """Changing physics_intensity must not change neural_only's fused output at all."""
    torch.manual_seed(0)
    fusion = TrustGatedFusion(hidden_dim=16, use_physics_intensity=False, use_fusion_gate=False)
    context = torch.randn(5, 16)
    physics_a = torch.rand(5, 4).clamp_min(1e-3)
    physics_b = physics_a * 1000.0 + 50.0  # wildly different physics input
    out_a = fusion(context, physics_a)
    out_b = fusion(context, physics_b)
    for name in CAUSE_NAMES:
        assert torch.allclose(out_a[name]["fused"], out_b[name]["fused"]), (
            f"neural_only cause={name}: fused output changed when only physics_intensity changed "
            f"-- physics is supposed to be fully ignored in this ablation"
        )


def test_pg_gtpp_default_flags_reproduce_original_unflagged_behavior():
    """Defaults (all True) must be numerically identical to the pre-flag
    TrustGatedFusion -- guards against accidentally changing pg_gtpp's own
    behavior while adding the ablation flags."""
    torch.manual_seed(0)
    context = torch.randn(5, 16)
    physics = torch.rand(5, 4).clamp_min(1e-3)

    torch.manual_seed(1)
    fusion_default = TrustGatedFusion(hidden_dim=16)
    torch.manual_seed(1)
    fusion_explicit = TrustGatedFusion(
        hidden_dim=16, learned_correction=True, use_physics_intensity=True, use_fusion_gate=True
    )
    out_default = fusion_default(context, physics)
    out_explicit = fusion_explicit(context, physics)
    for name in CAUSE_NAMES:
        assert torch.allclose(out_default[name]["fused"], out_explicit[name]["fused"])


@pytest.mark.parametrize(
    "flags,expected_smaller_than_full",
    [
        (dict(learned_correction=False, use_fusion_gate=False), True),   # physics_only
        (dict(use_physics_intensity=False, use_fusion_gate=False), True),  # neural_only
    ],
)
def test_ablation_variants_have_fewer_parameters_than_full_model(flags, expected_smaller_than_full):
    full_model = PGGTPPModel(f_node=F_NODE, hidden_dim=16)
    ablated_model = PGGTPPModel(f_node=F_NODE, hidden_dim=16, **flags)
    if expected_smaller_than_full:
        assert ablated_model.parameter_count() < full_model.parameter_count()


def test_pg_gtpp_model_forward_runs_for_all_three_ablation_flag_sets(mission_cache):
    """End-to-end (not just the fusion module in isolation): construct all
    three flag combinations through PGGTPPModel itself and run one real
    forward pass through the actual window-builder pipeline."""
    rng = np.random.default_rng(0)
    batch = mission_cache.sample_batch(rng, batch_size=4, window_length=6, lookahead_s=30.0)
    for flags in (
        dict(learned_correction=False, use_physics_intensity=True, use_fusion_gate=False),  # physics_only
        dict(learned_correction=True, use_physics_intensity=False, use_fusion_gate=False),   # neural_only
        dict(learned_correction=True, use_physics_intensity=True, use_fusion_gate=True),     # pg_gtpp
    ):
        torch.manual_seed(0)
        model = PGGTPPModel(f_node=F_NODE, hidden_dim=16, **flags)
        out = model(batch.node_features, batch.node_type_ids, batch.node_mask, batch.edge_index, batch.edge_physics)
        intensity = out.total_intensity()
        assert torch.isfinite(intensity).all()
        assert (intensity >= 0).all()
        loss_out = compute_loss(out, batch)
        assert torch.isfinite(loss_out.total_loss)


# --------------------------------------------------------------------- #
# decouple_mark_head -- opt-in mark-decoder adapter added to investigate
# why PG-GTPP's mark loss runs consistently higher than similar-sized
# baselines despite comparable TPP loss (see docs/BASELINES.md).
# --------------------------------------------------------------------- #

def test_decouple_mark_head_off_by_default_reproduces_old_behavior():
    """Default (False) must be numerically identical to a model built
    before this flag existed -- guards against accidentally changing
    pg_gtpp's own default architecture while adding the opt-in adapter."""
    torch.manual_seed(0)
    model_default = PGGTPPModel(f_node=F_NODE, hidden_dim=16)
    torch.manual_seed(0)
    model_explicit_false = PGGTPPModel(f_node=F_NODE, hidden_dim=16, decouple_mark_head=False)
    assert model_default.mark_adapter is None
    assert model_explicit_false.mark_adapter is None
    sd1 = model_default.state_dict()
    sd2 = model_explicit_false.state_dict()
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k])


def test_decouple_mark_head_true_adds_adapter_module():
    model = PGGTPPModel(f_node=F_NODE, hidden_dim=16, decouple_mark_head=True)
    assert model.mark_adapter is not None
    assert isinstance(model.mark_adapter, torch.nn.Sequential)
    # Adapter parameters must be included in the model's own parameter count
    # (i.e. actually registered, not a dangling unused module).
    adapter_params = sum(p.numel() for p in model.mark_adapter.parameters())
    assert adapter_params > 0
    assert model.parameter_count() > PGGTPPModel(f_node=F_NODE, hidden_dim=16).parameter_count()


def test_decouple_mark_head_forward_runs_and_changes_mark_predictions(mission_cache):
    """End-to-end: the adapter must actually sit in the forward path (mark
    predictions should differ from the non-decoupled model given the same
    seed/inputs), not be constructed-but-unused."""
    rng = np.random.default_rng(0)
    batch = mission_cache.sample_batch(rng, batch_size=4, window_length=6, lookahead_s=30.0)

    torch.manual_seed(0)
    model_shared = PGGTPPModel(f_node=F_NODE, hidden_dim=16, decouple_mark_head=False)
    torch.manual_seed(0)
    model_decoupled = PGGTPPModel(f_node=F_NODE, hidden_dim=16, decouple_mark_head=True)

    out_shared = model_shared(batch.node_features, batch.node_type_ids, batch.node_mask,
                               batch.edge_index, batch.edge_physics)
    out_decoupled = model_decoupled(batch.node_features, batch.node_type_ids, batch.node_mask,
                                     batch.edge_index, batch.edge_physics)

    assert torch.isfinite(out_decoupled.total_intensity()).all()
    loss_out = compute_loss(out_decoupled, batch)
    assert torch.isfinite(loss_out.total_loss)

    # Fusion/intensity path must be UNCHANGED by this flag (it only touches
    # the mark decoder's input) -- same seed => identical encoder+fusion
    # weights => identical total_intensity.
    assert torch.allclose(out_shared.total_intensity(), out_decoupled.total_intensity())
    # Mark predictions SHOULD differ -- the adapter is actually in the path.
    assert not torch.allclose(out_shared.marks.pdr_mean(), out_decoupled.marks.pdr_mean())


# --------------------------------------------------------------------- #
# mark_physics_skip -- second, independent hypothesis for the same
# mark-accuracy gap: does giving the mark decoder direct access to raw
# edge_physics (bypassing the encoder's abstraction) help, the way it
# seems to for DySAT-style (no physics at all, still much better marks)?
# --------------------------------------------------------------------- #

def test_mark_physics_skip_off_by_default_reproduces_old_behavior():
    torch.manual_seed(0)
    model_default = PGGTPPModel(f_node=F_NODE, hidden_dim=16)
    torch.manual_seed(0)
    model_explicit_false = PGGTPPModel(f_node=F_NODE, hidden_dim=16, mark_physics_skip=False)
    assert model_default.mark_decoder.trunk[0].in_features == 32  # hidden_dim*2, no physics_dim added
    assert model_explicit_false.mark_decoder.trunk[0].in_features == 32
    sd1 = model_default.state_dict()
    sd2 = model_explicit_false.state_dict()
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k])


def test_mark_physics_skip_true_widens_mark_decoder_input():
    model = PGGTPPModel(f_node=F_NODE, hidden_dim=16, mark_physics_skip=True)
    # hidden_dim*2 (32) + physics_dim (4) = 36
    assert model.mark_decoder.trunk[0].in_features == 36
    assert model.parameter_count() > PGGTPPModel(f_node=F_NODE, hidden_dim=16).parameter_count()


def test_mark_physics_skip_forward_runs_and_changes_mark_predictions(mission_cache):
    """End-to-end: raw edge_physics must actually reach the mark decoder
    (mark predictions differ from the baseline model), while the
    intensity/fusion path stays completely unaffected -- this flag only
    touches the mark decoder's input, never the TPP path."""
    rng = np.random.default_rng(0)
    batch = mission_cache.sample_batch(rng, batch_size=4, window_length=6, lookahead_s=30.0)

    torch.manual_seed(0)
    model_shared = PGGTPPModel(f_node=F_NODE, hidden_dim=16, mark_physics_skip=False)
    torch.manual_seed(0)
    model_skip = PGGTPPModel(f_node=F_NODE, hidden_dim=16, mark_physics_skip=True)

    out_shared = model_shared(batch.node_features, batch.node_type_ids, batch.node_mask,
                               batch.edge_index, batch.edge_physics)
    out_skip = model_skip(batch.node_features, batch.node_type_ids, batch.node_mask,
                           batch.edge_index, batch.edge_physics)

    assert torch.isfinite(out_skip.total_intensity()).all()
    loss_out = compute_loss(out_skip, batch)
    assert torch.isfinite(loss_out.total_loss)

    assert torch.allclose(out_shared.total_intensity(), out_skip.total_intensity())
    assert not torch.allclose(out_shared.marks.pdr_mean(), out_skip.marks.pdr_mean())


def test_mark_physics_skip_and_decouple_mark_head_compose_cleanly(mission_cache):
    """The two mark-accuracy experiments are independent flags -- verify
    they can be combined without shape errors, even though the project's
    own experiments run them one at a time for interpretability."""
    rng = np.random.default_rng(0)
    batch = mission_cache.sample_batch(rng, batch_size=4, window_length=6, lookahead_s=30.0)
    torch.manual_seed(0)
    model = PGGTPPModel(f_node=F_NODE, hidden_dim=16, decouple_mark_head=True, mark_physics_skip=True)
    assert model.mark_decoder.trunk[0].in_features == 36  # adapter preserves 32, +4 physics
    out = model(batch.node_features, batch.node_type_ids, batch.node_mask,
                batch.edge_index, batch.edge_physics)
    assert torch.isfinite(out.total_intensity()).all()
    loss_out = compute_loss(out, batch)
    assert torch.isfinite(loss_out.total_loss)
