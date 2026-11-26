"""
tests/test_phase5_model.py

Pytest suite for the Phase 5 model foundation: the window-tensor pipeline
(dataset/) and the model architecture (model/) -- graph encoder, trust-gated
fusion, mark decoder, and the assembled PGGTPPModel.

Uses the same small/fast mission config pattern as Phase 3/4's test suites
to keep runtime low while still exercising the real simulator-to-tensor
pipeline, not just synthetic tensors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simulator.dataset.window_builder import build_window, collate_windows
from simulator.event_generator import EventGenerator
from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.mission_generator import MissionGenerator
from simulator.model.fusion import CAUSE_NAMES, TrustGatedFusion
from simulator.model.graph_encoder import GraphEncoder
from simulator.model.mark_decoder import MarkDecoder
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.physics_intensity import HazardEvaluator
from simulator.random_manager import RandomManager
from simulator.trajectory_generator import TrajectoryGenerator
from simulator.trajectory_storage import F_NODE


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
    mission_id = "model_fixture"
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


# ---------------------------------------------------------------------- #
# Window builder
# ---------------------------------------------------------------------- #
def test_build_window_shapes(mission_artifacts):
    m = mission_artifacts
    sample = build_window(
        m["mission_id"], m["array"], m["node_ids"], m["node_types"], m["timestamps"], m["events"],
        m["evaluator"], m["config"].graph.candidate_range_factor,
        reference_t_idx=100, window_length=10, lookahead_s=30.0,
    )
    assert sample.node_features.shape == (10, len(m["node_ids"]), F_NODE)
    assert sample.edge_index.shape[0] == 2
    assert sample.edge_physics.shape == (sample.edge_index.shape[1], 4)
    assert sample.edge_link_down_target.shape == (sample.edge_index.shape[1],)


def test_build_window_left_pads_near_mission_start(mission_artifacts):
    m = mission_artifacts
    sample = build_window(
        m["mission_id"], m["array"], m["node_ids"], m["node_types"], m["timestamps"], m["events"],
        m["evaluator"], m["config"].graph.candidate_range_factor,
        reference_t_idx=2, window_length=10, lookahead_s=30.0,
    )
    assert sample.node_features.shape[0] == 10
    assert torch.allclose(sample.node_features[0], sample.node_features[1])


def test_build_window_rejects_out_of_range_reference(mission_artifacts):
    m = mission_artifacts
    with pytest.raises(ValueError):
        build_window(
            m["mission_id"], m["array"], m["node_ids"], m["node_types"], m["timestamps"], m["events"],
            m["evaluator"], m["config"].graph.candidate_range_factor,
            reference_t_idx=len(m["timestamps"]), window_length=10, lookahead_s=30.0,
        )


def test_collate_windows_pads_and_offsets_correctly(mission_artifacts):
    m = mission_artifacts
    s1 = build_window(
        m["mission_id"], m["array"], m["node_ids"], m["node_types"], m["timestamps"], m["events"],
        m["evaluator"], m["config"].graph.candidate_range_factor, 50, 8, 30.0,
    )
    s2 = build_window(
        m["mission_id"], m["array"], m["node_ids"], m["node_types"], m["timestamps"], m["events"],
        m["evaluator"], m["config"].graph.candidate_range_factor, 150, 8, 30.0,
    )
    batch = collate_windows([s1, s2])
    v_max = batch.node_features.shape[2]
    assert batch.node_features.shape[0] == 2
    if batch.edge_index.shape[1] > 0:
        assert int(batch.edge_index.max()) < 2 * v_max
    assert set(batch.edge_batch.tolist()) <= {0, 1}


def test_collate_windows_requires_at_least_one_sample():
    with pytest.raises(ValueError):
        collate_windows([])


# ---------------------------------------------------------------------- #
# Graph encoder
# ---------------------------------------------------------------------- #
def test_graph_encoder_forward_shape_and_gradients():
    torch.manual_seed(0)
    B, k, V, f_node = 2, 5, 8, 11
    enc = GraphEncoder(f_node=f_node, hidden_dim=32)
    node_features = torch.randn(B, k, V, f_node)
    node_type_ids = torch.randint(0, 3, (B, V))
    node_mask = torch.ones(B, V)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    edge_physics = torch.rand(2, 4)

    out = enc(node_features, node_type_ids, node_mask, edge_index, edge_physics)
    assert out.shape == (B * V, 32)
    out.sum().backward()
    assert all(p.grad is not None for p in enc.parameters())


def test_graph_encoder_masks_padded_nodes():
    torch.manual_seed(0)
    enc = GraphEncoder(f_node=11, hidden_dim=16)
    node_features = torch.randn(1, 4, 6, 11)
    node_type_ids = torch.randint(0, 3, (1, 6))
    node_mask = torch.ones(1, 6)
    node_mask[0, 4:] = 0
    edge_index = torch.zeros((2, 0), dtype=torch.int64)
    edge_physics = torch.zeros((0, 4))

    out = enc(node_features, node_type_ids, node_mask, edge_index, edge_physics)
    assert torch.allclose(out.view(1, 6, 16)[0, 4:], torch.zeros(2, 16))


def test_graph_encoder_handles_zero_edges_without_nan():
    enc = GraphEncoder(f_node=11, hidden_dim=16)
    node_features = torch.randn(1, 4, 5, 11)
    node_type_ids = torch.randint(0, 3, (1, 5))
    node_mask = torch.ones(1, 5)
    out = enc(node_features, node_type_ids, node_mask, torch.zeros((2, 0), dtype=torch.int64), torch.zeros((0, 4)))
    assert torch.isfinite(out).all()


def test_graph_encoder_real_pipeline_forward(sample_batch):
    enc = GraphEncoder(f_node=F_NODE, hidden_dim=32)
    out = enc(sample_batch.node_features, sample_batch.node_type_ids, sample_batch.node_mask,
               sample_batch.edge_index, sample_batch.edge_physics)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------- #
# Trust-gated fusion
# ---------------------------------------------------------------------- #
def test_fusion_output_structure_and_nonnegativity():
    torch.manual_seed(0)
    fusion = TrustGatedFusion(hidden_dim=16)
    context = torch.randn(5, 16)
    physics = torch.rand(5, 4)
    out = fusion(context, physics)
    assert set(out.keys()) == set(CAUSE_NAMES)
    for name in CAUSE_NAMES:
        assert (out[name]["fused"] >= 0).all()
        assert ((out[name]["gate"] >= 0) & (out[name]["gate"] <= 1)).all()
    total = TrustGatedFusion.total_intensity(out)
    assert (total >= 0).all()


def test_fusion_gradients_flow():
    fusion = TrustGatedFusion(hidden_dim=16)
    context = torch.randn(5, 16)
    physics = torch.rand(5, 4)
    out = fusion(context, physics)
    TrustGatedFusion.total_intensity(out).sum().backward()
    assert all(p.grad is not None for p in fusion.parameters())


def test_fusion_g_equals_one_limit_is_exactly_physics():
    """The learned correction must disappear exactly at full physics trust."""
    g = torch.ones(7)
    physics = torch.rand(7)
    log_delta = torch.randn(7)
    fused = physics * torch.exp((1.0 - g) * log_delta)
    assert torch.allclose(fused, physics)


def test_fusion_zero_log_residual_is_exactly_physics():
    """A zero learned log-residual must preserve the physics prior."""
    physics = torch.rand(7)
    g = torch.rand(7)
    fused = physics * torch.exp((1.0 - g) * torch.zeros_like(physics))
    assert torch.allclose(fused, physics)


# ---------------------------------------------------------------------- #
# Mark decoder
# ---------------------------------------------------------------------- #
def test_mark_decoder_pdr_bounded_and_positive_marks():
    torch.manual_seed(0)
    decoder = MarkDecoder(hidden_dim=16)
    context = torch.randn(5, 16)
    pred = decoder(context)
    assert ((pred.pdr_mean() >= 0) & (pred.pdr_mean() <= 1)).all()
    assert (pred.throughput_mean() > 0).all()
    assert (pred.delay_mean() > 0).all()


def test_mark_decoder_nll_handles_boundary_observations():
    decoder = MarkDecoder(hidden_dim=16)
    context = torch.randn(5, 16)
    pred = decoder(context)
    nll = pred.negative_log_likelihood(
        torch.tensor([0.0, 1.0, 0.5, 0.0, 1.0]), torch.zeros(5), torch.zeros(5)
    )
    assert torch.isfinite(nll).all()


def test_mark_decoder_gradients_flow():
    decoder = MarkDecoder(hidden_dim=16)
    context = torch.randn(5, 16)
    pred = decoder(context)
    nll = pred.negative_log_likelihood(torch.rand(5), torch.rand(5) * 1000, torch.rand(5))
    nll.sum().backward()
    assert all(p.grad is not None for p in decoder.parameters())


# ---------------------------------------------------------------------- #
# Full assembled model
# ---------------------------------------------------------------------- #
def test_pg_gtpp_model_forward_on_real_pipeline_data(sample_batch):
    model = PGGTPPModel(f_node=F_NODE, hidden_dim=32)
    out = model(sample_batch.node_features, sample_batch.node_type_ids, sample_batch.node_mask,
                sample_batch.edge_index, sample_batch.edge_physics)
    total = out.total_intensity()
    assert torch.isfinite(total).all()
    assert (total >= 0).all()
    assert out.marks.pdr_mean().shape == total.shape


def test_pg_gtpp_model_end_to_end_backward_pass(sample_batch):
    model = PGGTPPModel(f_node=F_NODE, hidden_dim=32)
    out = model(sample_batch.node_features, sample_batch.node_type_ids, sample_batch.node_mask,
                sample_batch.edge_index, sample_batch.edge_physics)

    total_intensity = out.total_intensity()
    link_prob = 1 - torch.exp(-total_intensity * 30.0)
    bce = torch.nn.functional.binary_cross_entropy(
        link_prob.clamp(1e-6, 1 - 1e-6), sample_batch.edge_link_down_target
    )
    mark_nll = out.marks.negative_log_likelihood(
        torch.rand_like(total_intensity), torch.rand_like(total_intensity) * 1000, torch.rand_like(total_intensity)
    ).mean()
    loss = bce + mark_nll

    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_pg_gtpp_model_parameter_count_is_reasonable():
    model = PGGTPPModel(f_node=F_NODE, hidden_dim=64)
    n_params = model.parameter_count()
    assert 10_000 < n_params < 5_000_000


def test_pg_gtpp_model_deterministic_with_fixed_seed(sample_batch):
    torch.manual_seed(42)
    model_a = PGGTPPModel(f_node=F_NODE, hidden_dim=32)
    torch.manual_seed(42)
    model_b = PGGTPPModel(f_node=F_NODE, hidden_dim=32)

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(sample_batch.node_features, sample_batch.node_type_ids, sample_batch.node_mask,
                         sample_batch.edge_index, sample_batch.edge_physics)
        out_b = model_b(sample_batch.node_features, sample_batch.node_type_ids, sample_batch.node_mask,
                         sample_batch.edge_index, sample_batch.edge_physics)
    assert torch.allclose(out_a.total_intensity(), out_b.total_intensity())
