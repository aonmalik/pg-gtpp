"""
tests/test_phase6_training.py

Pytest suite for the Phase 6 training pipeline: the TPP+mark loss function
and the training loop (including checkpointing), exercised against a real,
freshly generated small mission corpus.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simulator.dataset.window_builder import build_window, collate_windows
from simulator.event_generator import EventGenerator
from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.mission_generator import MissionGenerator
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.physics import AcousticChannel, CurrentField, WaveField
from simulator.physics_intensity import HazardEvaluator
from simulator.random_manager import RandomManager
from simulator.trajectory_generator import TrajectoryGenerator
from simulator.training.loss import compute_loss, tpp_negative_log_likelihood
from simulator.training.train import MissionCache, TrainConfig, load_checkpoint, save_checkpoint, train


@pytest.fixture
def fast_config():
    base = default_config(master_seed=7)
    small_fleet = FleetConfig(
        usv_count=1, n_auv_min=3, n_auv_max=5, m_iout_min=6, m_iout_max=10,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    short_envelope = MissionEnvelopeConfig(
        duration_s=200.0, node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m, depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state,
    )
    config = MissionConfig(
        master_seed=7, split="train", wave=base.wave, current=base.current,
        acoustic=base.acoustic, fleet=small_fleet, envelope=short_envelope,
        hazard=base.hazard, graph=base.graph,
    )
    config.validate()
    return config


@pytest.fixture
def mission_artifacts(fast_config):
    rm = RandomManager(master_seed=7)
    mission_id = "loss_fixture"
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
        for t_idx in range(20, 180, 15)
    ]
    return collate_windows(samples)


# ---------------------------------------------------------------------- #
# TPP NLL formula
# ---------------------------------------------------------------------- #
def test_tpp_nll_matches_hand_calculation():
    lam = torch.tensor([0.1, 0.5, 2.0])
    occurred = torch.tensor([0.0, 1.0, 1.0])
    nll = tpp_negative_log_likelihood(lam, occurred, dt=10.0)
    expected = torch.tensor([
        0.1 * 10.0,
        0.5 * 10.0 - torch.log(torch.tensor(0.5)),
        2.0 * 10.0 - torch.log(torch.tensor(2.0)),
    ])
    assert torch.allclose(nll, expected, atol=1e-5)


def test_tpp_nll_no_event_is_pure_survival_term():
    lam = torch.tensor([0.3, 1.5])
    occurred = torch.zeros(2)
    nll = tpp_negative_log_likelihood(lam, occurred, dt=5.0)
    assert torch.allclose(nll, lam * 5.0)


def test_tpp_nll_handles_near_zero_intensity_without_nan():
    lam = torch.tensor([0.0, 1e-10])
    occurred = torch.tensor([1.0, 0.0])
    nll = tpp_negative_log_likelihood(lam, occurred, dt=10.0)
    assert torch.isfinite(nll).all()


# ---------------------------------------------------------------------- #
# Combined loss, on real pipeline data
# ---------------------------------------------------------------------- #
def test_compute_loss_on_real_data(sample_batch):
    model = PGGTPPModel(f_node=11, hidden_dim=32)
    out = model(sample_batch.node_features, sample_batch.node_type_ids, sample_batch.node_mask,
                sample_batch.edge_index, sample_batch.edge_physics)
    loss_out = compute_loss(out, sample_batch)
    assert torch.isfinite(loss_out.total_loss)
    assert loss_out.n_events == int(sample_batch.edge_has_event_marks.sum().item())
    assert loss_out.n_edges == sample_batch.edge_index.shape[1]


def test_compute_loss_gradients_flow(sample_batch):
    model = PGGTPPModel(f_node=11, hidden_dim=32)
    out = model(sample_batch.node_features, sample_batch.node_type_ids, sample_batch.node_mask,
                sample_batch.edge_index, sample_batch.edge_physics)
    loss_out = compute_loss(out, sample_batch)
    loss_out.total_loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_compute_loss_zero_events_skips_mark_loss(mission_artifacts):
    """When no edge in the batch has an event, mark_loss should be exactly
    zero (no fabricated supervision signal)."""
    m = mission_artifacts
    sample = build_window(
        m["mission_id"], m["array"], m["node_ids"], m["node_types"], m["timestamps"], [],  # no events at all
        m["evaluator"], m["config"].graph.candidate_range_factor, 50, 8, 30.0,
    )
    batch = collate_windows([sample])
    model = PGGTPPModel(f_node=11, hidden_dim=16)
    out = model(batch.node_features, batch.node_type_ids, batch.node_mask, batch.edge_index, batch.edge_physics)
    loss_out = compute_loss(out, batch)
    assert loss_out.n_events == 0
    assert loss_out.mark_loss.item() == 0.0


# ---------------------------------------------------------------------- #
# MissionCache
# ---------------------------------------------------------------------- #
def test_mission_cache_generates_missions(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=3)
    assert len(cache.missions) == 3
    assert all(len(m.node_ids) > 0 for m in cache.missions)


def test_mission_cache_sample_batch_produces_valid_batch(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=3)
    rng = np.random.default_rng(0)
    batch = cache.sample_batch(rng, batch_size=4, window_length=6, lookahead_s=20.0)
    assert batch.node_features.shape[0] == 4
    assert batch.lookahead_s == 20.0


# ---------------------------------------------------------------------- #
# Training loop -- the key regression test
# ---------------------------------------------------------------------- #
def test_training_loop_loss_decreases(fast_config):
    """Regression test: on a small real corpus, average loss over the last
    portion of training should be lower than over the first portion. This
    is the test that should start failing if a future change breaks the
    loss function, the optimizer wiring, or the model's differentiability."""
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=5)
    model = PGGTPPModel(f_node=11, hidden_dim=32)
    train_config = TrainConfig(n_steps=50, batch_size=8, window_length=8, lookahead_s=30.0,
                                learning_rate=1e-2, log_every=0, seed=1)

    history = train(model, cache, train_config)
    assert len(history.total_losses) > 0

    first_avg = np.mean(history.total_losses[:5])
    last_avg = np.mean(history.total_losses[-5:])
    assert last_avg < first_avg, (
        f"Training loss did not decrease: first-5 avg={first_avg:.3f}, last-5 avg={last_avg:.3f}"
    )


def test_training_loop_rejects_empty_cache():
    from simulator.training.train import MissionCache as EmptyCache

    model = PGGTPPModel(f_node=11, hidden_dim=16)
    with pytest.raises(ValueError):
        train(model, EmptyCache(), TrainConfig(n_steps=1))


# ---------------------------------------------------------------------- #
# Checkpointing
# ---------------------------------------------------------------------- #
def test_checkpoint_roundtrip_preserves_weights(tmp_path):
    model = PGGTPPModel(f_node=11, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, step=7)

    model2 = PGGTPPModel(f_node=11, hidden_dim=16)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    loaded_step = load_checkpoint(ckpt_path, model2, optimizer2)

    assert loaded_step == 7
    for (_, p1), (_, p2) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert torch.allclose(p1, p2)


def test_training_loop_with_checkpointing(fast_config, tmp_path):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=3)
    model = PGGTPPModel(f_node=11, hidden_dim=16)
    train_config = TrainConfig(
        n_steps=10, batch_size=4, window_length=6, lookahead_s=20.0,
        learning_rate=1e-2, log_every=0, checkpoint_every=5, checkpoint_dir=tmp_path, seed=2,
    )
    train(model, cache, train_config)
    assert (tmp_path / "checkpoint_step5.pt").exists()
    assert (tmp_path / "checkpoint_step10.pt").exists()


# ---------------------------------------------------------------------- #
# End-to-end reproducibility (the pattern scripts/train_demo.py relies on)
# ---------------------------------------------------------------------- #
def test_full_pipeline_reproducible_when_torch_seed_is_fixed(fast_config):
    """Regression test for a real bug found via manual cross-machine
    verification: MissionCache.generate() is fully seeded via
    RandomManager, but PGGTPPModel's weight initialization uses PyTorch's
    own global RNG, which is NOT seeded by anything in this codebase. Two
    runs with identical data but no torch.manual_seed() call produced
    different trained models and different loss trajectories -- silently,
    since nothing raised an error. This test locks in the fix: calling
    torch.manual_seed() before model construction, with already-seeded
    data generation, must yield bit-identical training results end to end.
    """
    def run_once():
        cache = MissionCache.generate(fast_config, master_seed=7, n_missions=3)
        torch.manual_seed(1234)
        model = PGGTPPModel(f_node=11, hidden_dim=16)
        train_config = TrainConfig(n_steps=15, batch_size=4, window_length=6, lookahead_s=20.0,
                                    learning_rate=1e-2, log_every=0, seed=3)
        history = train(model, cache, train_config)
        return history.total_losses

    losses_a = run_once()
    losses_b = run_once()
    assert losses_a == losses_b, "Training was not reproducible despite fixed seeds for both data and model init"
