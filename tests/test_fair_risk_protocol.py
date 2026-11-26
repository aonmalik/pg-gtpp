import torch

from simulator.dataset.window_builder import WindowBatch
from simulator.training.loss import compute_loss, competing_risk_nll


def _batch():
    return WindowBatch(
        node_features=torch.zeros(1, 2, 2, 11),
        node_type_ids=torch.zeros(1, 2, dtype=torch.long),
        node_mask=torch.ones(1, 2),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_batch=torch.zeros(1, dtype=torch.long),
        edge_kind_ids=torch.zeros(1, dtype=torch.long),
        edge_physics=torch.ones(1, 4),
        edge_link_down_target=torch.tensor([1.0]),
        edge_link_up_target=torch.tensor([0.0]),
        edge_any_event_target=torch.tensor([1.0]),
        edge_at_risk=torch.tensor([1.0]),
        edge_event_time_delta=torch.tensor([2.0]),
        edge_event_cause_id=torch.tensor([1], dtype=torch.long),
        edge_event_pdr=torch.tensor([0.5]),
        edge_event_throughput=torch.tensor([10.0]),
        edge_event_delay=torch.tensor([0.1]),
        edge_has_event_marks=torch.tensor([1.0]),
        lookahead_s=10.0,
        mission_ids=["m"],
    )


class _Marks:
    def pdr_mean(self): return torch.tensor([0.5])
    def throughput_mean(self): return torch.tensor([10.0])
    def delay_mean(self): return torch.tensor([0.1])
    def negative_log_likelihood(self, pdr, throughput, delay): return torch.zeros(1)


class _PGOutput:
    marks = _Marks()

    def total_intensity(self):
        return torch.tensor([1.0])

    def cause_intensities(self):
        return {
            "geo": torch.tensor([0.1]),
            "wave": torch.tensor([0.2]),
            "doppler": torch.tensor([0.3]),
            "current": torch.tensor([0.4]),
        }


class _AggregateOutput:
    marks = _Marks()

    def total_intensity(self):
        return torch.tensor([1.0])

    def cause_intensities(self):
        return {"aggregate": torch.tensor([1.0])}


def test_competing_risk_uses_observed_cause():
    # total hazard = 1, horizon = 10, event cause is wave (lambda=0.2)
    nll = competing_risk_nll(
        _PGOutput().cause_intensities(),
        torch.tensor([1.0]),
        torch.tensor([1]),
        10.0,
        torch.tensor([1.0]),
    )
    expected = torch.tensor([10.0 - torch.log(torch.tensor(0.2))])
    assert torch.allclose(nll, expected, atol=1e-6)


def test_compute_loss_excludes_non_risk_edges():
    batch = _batch()
    batch.edge_at_risk[:] = 0.0
    out = _PGOutput()
    loss = compute_loss(out, batch)
    assert loss.n_at_risk == 0
    assert loss.n_events == 0
    assert float(loss.total_loss) == 0.0


def test_aggregate_baseline_has_valid_survival_likelihood():
    loss = compute_loss(_AggregateOutput(), _batch())
    assert torch.isfinite(loss.total_loss)
    assert loss.n_at_risk == 1
    assert loss.n_events == 1
