import torch

from simulator.training.loss import competing_risk_nll, tpp_negative_log_likelihood


def test_constant_hazard_first_event_nll():
    lam = torch.tensor([2.0])
    event = torch.tensor([1.0])
    nll = tpp_negative_log_likelihood(lam, event, 1.0, event_intensity=lam)
    assert torch.allclose(nll, torch.tensor([2.0 - torch.log(torch.tensor(2.0))]))


def test_constant_hazard_censored_nll():
    lam = torch.tensor([2.0])
    nll = tpp_negative_log_likelihood(lam, torch.tensor([0.0]), 1.0, event_intensity=lam)
    assert torch.allclose(nll, torch.tensor([2.0]))


def test_competing_risk_uses_observed_cause_intensity():
    causes = {
        "geo": torch.tensor([0.1]),
        "wave": torch.tensor([0.2]),
        "doppler": torch.tensor([0.3]),
        "current": torch.tensor([0.4]),
    }
    # total = 1.0; observed cause = wave => event term is -log(0.2)
    nll = competing_risk_nll(causes, torch.tensor([1.0]), torch.tensor([1]), 10.0, torch.tensor([1.0]))
    expected = torch.tensor([10.0 - torch.log(torch.tensor(0.2))])
    assert torch.allclose(nll, expected, atol=1e-6)
