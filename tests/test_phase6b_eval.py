"""
tests/test_phase6b_eval.py

Pytest suite for the Phase 6 extension: mission-level train/val split,
evaluation metrics (AUC, calibration error, mark MAE), and the LR schedule
integration in the training loop.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.training.eval import compute_auc, compute_calibration_error, evaluate
from simulator.training.train import MissionCache, TrainConfig, train


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


# ---------------------------------------------------------------------- #
# compute_auc -- verified against a brute-force O(n^2) reference
# ---------------------------------------------------------------------- #
def _brute_force_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    count = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                count += 1.0
            elif p == n:
                count += 0.5
    return count / (len(pos) * len(neg))


def test_auc_matches_brute_force_reference():
    torch.manual_seed(0)
    scores = torch.rand(40)
    labels = (torch.rand(40) > 0.5).float()
    assert abs(compute_auc(scores, labels) - _brute_force_auc(scores, labels)) < 1e-9


def test_auc_perfect_separation_is_one():
    scores = torch.tensor([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert compute_auc(scores, labels) == pytest.approx(1.0)


def test_auc_perfect_inversion_is_zero():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    labels = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert compute_auc(scores, labels) == pytest.approx(0.0)


def test_auc_tied_scores_is_half():
    scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert compute_auc(scores, labels) == pytest.approx(0.5)


def test_auc_single_class_is_nan():
    result = compute_auc(torch.rand(5), torch.zeros(5))
    assert np.isnan(result)


# ---------------------------------------------------------------------- #
# compute_calibration_error
# ---------------------------------------------------------------------- #
def test_calibration_error_near_zero_when_well_calibrated():
    p = torch.tensor([0.1] * 10 + [0.9] * 10)
    y = torch.cat([torch.zeros(9), torch.ones(1), torch.zeros(1), torch.ones(9)])
    ece = compute_calibration_error(p, y, n_bins=10)
    assert ece < 0.05


def test_calibration_error_large_when_badly_miscalibrated():
    p = torch.full((20,), 0.9)
    y = torch.cat([torch.ones(2), torch.zeros(18)])
    ece = compute_calibration_error(p, y, n_bins=10)
    assert ece > 0.5


def test_calibration_error_empty_is_nan():
    assert np.isnan(compute_calibration_error(torch.tensor([]), torch.tensor([])))


# ---------------------------------------------------------------------- #
# MissionCache.split -- mission-level, not window-level
# ---------------------------------------------------------------------- #
def test_split_produces_disjoint_missions(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=8)
    train_cache, val_cache = cache.split(val_fraction=0.25, seed=0)
    train_ids = {m.mission_id for m in train_cache.missions}
    val_ids = {m.mission_id for m in val_cache.missions}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {m.mission_id for m in cache.missions}


def test_split_is_reproducible(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=8)
    _, val_a = cache.split(val_fraction=0.25, seed=5)
    _, val_b = cache.split(val_fraction=0.25, seed=5)
    assert {m.mission_id for m in val_a.missions} == {m.mission_id for m in val_b.missions}


def test_split_rejects_invalid_fraction(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=4)
    with pytest.raises(ValueError):
        cache.split(val_fraction=1.5, seed=0)
    with pytest.raises(ValueError):
        cache.split(val_fraction=0.0, seed=0)


def test_split_rejects_too_few_missions(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=1)
    with pytest.raises(ValueError):
        cache.split(val_fraction=0.5, seed=0)


# ---------------------------------------------------------------------- #
# evaluate() on real pipeline data
# ---------------------------------------------------------------------- #
def test_evaluate_on_real_held_out_data(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=6)
    _, val_cache = cache.split(val_fraction=0.34, seed=0)
    val_batch = val_cache.all_windows(window_length=8, lookahead_s=30.0, stride=10)

    torch.manual_seed(1)
    model = PGGTPPModel(f_node=11, hidden_dim=16)
    metrics = evaluate(model, val_batch)

    assert np.isfinite(metrics.val_loss)
    assert metrics.n_edges > 0
    if not np.isnan(metrics.event_auc):
        assert 0.0 <= metrics.event_auc <= 1.0
    if not np.isnan(metrics.calibration_error):
        assert 0.0 <= metrics.calibration_error <= 1.0


def test_evaluate_leaves_model_in_eval_mode(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=6)
    _, val_cache = cache.split(val_fraction=0.34, seed=0)
    val_batch = val_cache.all_windows(window_length=8, lookahead_s=30.0, stride=10)

    model = PGGTPPModel(f_node=11, hidden_dim=16)
    model.train()
    evaluate(model, val_batch)
    assert not model.training


# ---------------------------------------------------------------------- #
# Training loop with LR schedule + validation integration
# ---------------------------------------------------------------------- #
def test_lr_schedule_cosine_decays_to_near_zero(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=5)
    torch.manual_seed(1)
    model = PGGTPPModel(f_node=11, hidden_dim=16)
    train_config = TrainConfig(n_steps=20, batch_size=4, window_length=6, lookahead_s=20.0,
                                learning_rate=1e-2, log_every=0, seed=1, lr_schedule="cosine")
    history = train(model, cache, train_config)
    assert history.learning_rates[0] > history.learning_rates[-1]
    assert history.learning_rates[-1] < 1e-4


def test_lr_schedule_none_keeps_lr_constant(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=5)
    model = PGGTPPModel(f_node=11, hidden_dim=16)
    train_config = TrainConfig(n_steps=15, batch_size=4, window_length=6, lookahead_s=20.0,
                                learning_rate=1e-2, log_every=0, seed=1, lr_schedule="none")
    history = train(model, cache, train_config)
    assert all(lr == pytest.approx(1e-2) for lr in history.learning_rates)


def test_lr_schedule_rejects_unknown_name(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=3)
    model = PGGTPPModel(f_node=11, hidden_dim=16)
    train_config = TrainConfig(n_steps=5, batch_size=4, lr_schedule="not_a_real_schedule")
    with pytest.raises(ValueError):
        train(model, cache, train_config)


def test_train_with_validation_records_val_history(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=8)
    train_cache, val_cache = cache.split(val_fraction=0.25, seed=0)

    model = PGGTPPModel(f_node=11, hidden_dim=16)
    train_config = TrainConfig(n_steps=20, batch_size=4, window_length=6, lookahead_s=20.0,
                                learning_rate=1e-2, log_every=0, seed=1, val_every=10, val_stride=15)
    history = train(model, train_cache, train_config, val_cache=val_cache)

    assert history.val_steps == [10, 20]
    assert len(history.val_losses) == 2
    assert all(np.isfinite(v) for v in history.val_losses)


def test_train_val_every_without_val_cache_raises(fast_config):
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=5)
    model = PGGTPPModel(f_node=11, hidden_dim=16)
    train_config = TrainConfig(n_steps=5, batch_size=4, val_every=2)
    with pytest.raises(ValueError):
        train(model, cache, train_config)


def test_train_resumes_train_mode_after_validation(fast_config):
    """evaluate() puts the model in eval() mode; the training loop must
    switch back to train() mode afterward or subsequent training steps
    would silently run with e.g. any future dropout/batchnorm disabled."""
    cache = MissionCache.generate(fast_config, master_seed=7, n_missions=8)
    train_cache, val_cache = cache.split(val_fraction=0.25, seed=0)

    model = PGGTPPModel(f_node=11, hidden_dim=16)
    train_config = TrainConfig(n_steps=12, batch_size=4, window_length=6, lookahead_s=20.0,
                                learning_rate=1e-2, log_every=0, seed=1, val_every=5, val_stride=15)
    train(model, train_cache, train_config, val_cache=val_cache)
    assert model.training
