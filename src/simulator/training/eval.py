"""
training/eval.py

Publication-grade evaluation for first-link-down risk forecasting.

Primary metrics:
  - time-dependent AUROC and AUPRC
  - Brier score and ECE at multiple horizons
  - integrated Brier score (IBS)
  - Harrell concordance index for censored time-to-failure
  - mark MAE/RMSE for PDR, throughput and delay

Secondary competing-risk diagnostics are reported when the model exposes
four named cause-specific hazards.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import torch
from scipy.stats import rankdata

from ..dataset.window_builder import WindowBatch
from .loss import compute_loss

CAUSE_NAMES = ("geo", "wave", "doppler", "current")


@dataclass
class EvalMetrics:
    val_loss: float
    val_tpp_loss: float
    val_mark_loss: float
    event_auc: float
    event_auprc: float
    brier_score: float
    integrated_brier_score: float
    calibration_error: float
    survival_calibration_error: float
    c_index: float
    failure_time_mae_s: float
    pdr_mae: float
    pdr_rmse: float
    throughput_mae: float
    throughput_rmse: float
    delay_mae: float
    delay_rmse: float
    cause_macro_f1: float
    cause_brier: float
    n_edges: int
    n_at_risk: int
    n_events: int
    horizon_metrics: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _valid_risk(batch: WindowBatch) -> torch.Tensor:
    return batch.edge_at_risk > 0.5


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    y = labels.detach().cpu().numpy().astype(int)
    s = scores.detach().cpu().numpy()
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Average precision via the standard precision-at-positive definition."""
    y = labels.detach().cpu().numpy().astype(int)
    s = scores.detach().cpu().numpy()
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = s.argsort()[::-1]
    ys = y[order]
    tp = ys.cumsum()
    precision = tp / (torch.arange(len(ys), dtype=torch.float64).numpy() + 1.0)
    return float((precision * ys).sum() / n_pos)


def compute_calibration_error(predicted_prob: torch.Tensor, observed: torch.Tensor, n_bins: int = 10) -> float:
    if predicted_prob.numel() == 0:
        return float("nan")
    p = predicted_prob.detach().cpu().clamp(0, 1)
    y = observed.detach().cpu().float()
    edges = torch.linspace(0, 1, n_bins + 1)
    err = total = 0.0
    for i in range(n_bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        n = int(mask.sum())
        if n:
            err += n * abs(float(p[mask].mean()) - float(y[mask].mean()))
            total += n
    return err / total if total else float("nan")


def brier_score(prob: torch.Tensor, labels: torch.Tensor) -> float:
    if prob.numel() == 0:
        return float("nan")
    return float(((prob.clamp(0, 1) - labels.float()) ** 2).mean().item())


def harrell_c_index(time: torch.Tensor, event: torch.Tensor, risk_score: torch.Tensor) -> float:
    """Harrell's C for right-censored observations.

    Higher risk_score means shorter expected failure time.
    """
    t = time.detach().cpu().numpy()
    e = event.detach().cpu().numpy().astype(bool)
    r = risk_score.detach().cpu().numpy()
    concordant = tied = comparable = 0.0
    n = len(t)
    for i in range(n):
        if not e[i]:
            continue
        for j in range(n):
            if i == j or t[i] >= t[j]:
                continue
            comparable += 1.0
            if r[i] > r[j]:
                concordant += 1.0
            elif r[i] == r[j]:
                tied += 1.0
    return float((concordant + 0.5 * tied) / comparable) if comparable else float("nan")


def _cause_metrics(output: Any, batch: WindowBatch, risk_mask: torch.Tensor) -> tuple[float, float]:
    if not hasattr(output, "cause_intensities"):
        return float("nan"), float("nan")
    causes = output.cause_intensities()
    if set(causes.keys()) != set(CAUSE_NAMES):
        return float("nan"), float("nan")
    event_mask = risk_mask & (batch.edge_link_down_target > 0.5)
    if not event_mask.any():
        return float("nan"), float("nan")

    stacked = torch.stack([causes[c] for c in CAUSE_NAMES], dim=-1)
    stacked = stacked[event_mask].clamp_min(1e-8)
    probs = stacked / stacked.sum(dim=-1, keepdim=True)
    true = batch.edge_event_cause_id[event_mask].long()

    pred = probs.argmax(dim=-1)
    f1s = []
    for c in range(len(CAUSE_NAMES)):
        tp = ((pred == c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    one_hot = torch.nn.functional.one_hot(true, num_classes=len(CAUSE_NAMES)).float()
    return float(sum(f1s) / len(f1s)), float(((probs - one_hot) ** 2).mean().item())


def evaluate(model: Any, batch: WindowBatch, mark_loss_weight: float = 1.0) -> EvalMetrics:
    model.eval()
    with torch.no_grad():
        output = model(
            batch.node_features, batch.node_type_ids, batch.node_mask,
            batch.edge_index, batch.edge_physics
        )
        loss_out = compute_loss(output, batch, mark_loss_weight=mark_loss_weight)

        risk = _valid_risk(batch)
        event = batch.edge_link_down_target[risk]
        event_time = batch.edge_event_time_delta[risk]
        lam = output.total_intensity()[risk].clamp_min(1e-8)

        horizons = sorted(set([
            min(1.0, batch.lookahead_s),
            min(5.0, batch.lookahead_s),
            min(10.0, batch.lookahead_s),
            min(30.0, batch.lookahead_s),
            float(batch.lookahead_s),
        ]))
        horizon_metrics: dict[str, dict[str, float]] = {}
        briers = []
        for h in horizons:
            labels_h = (event > 0.5) & (event_time <= h + 1e-6)
            prob_h = 1.0 - torch.exp(-lam * h)
            horizon_metrics[f"{h:g}s"] = {
                "auroc": compute_auc(prob_h, labels_h.float()),
                "auprc": compute_auprc(prob_h, labels_h.float()),
                "brier": brier_score(prob_h, labels_h.float()),
                "ece": compute_calibration_error(prob_h, labels_h.float()),
            }
            briers.append(horizon_metrics[f"{h:g}s"]["brier"])

        # Headline horizon is the configured forecast horizon.
        final_h = float(batch.lookahead_s)
        final_prob = 1.0 - torch.exp(-lam * final_h)
        final_auc = compute_auc(final_prob, event)
        final_auprc = compute_auprc(final_prob, event)
        final_brier = brier_score(final_prob, event)
        final_ece = compute_calibration_error(final_prob, event)
        survival_ece = compute_calibration_error(1.0 - final_prob, 1.0 - event)
        ibs = float(sum(briers) / len(briers)) if briers else float("nan")

        c_index = harrell_c_index(event_time, event, lam)
        observed = event > 0.5
        time_mae = float(event_time[observed].mean().item()) if False else float("nan")
        if observed.any():
            # Under a constant hazard, the model-implied conditional mean
            # event time is E[T | T<=tau] = 1/lambda - tau/(exp(lambda*tau)-1).
            lt = lam[observed]
            tau = final_h
            mean_cond = 1.0 / lt - tau / torch.expm1((lt * tau).clamp(max=50.0))
            time_mae = float((mean_cond - event_time[observed]).abs().mean().item())

        mark_mask = risk & (batch.edge_has_event_marks > 0.5)
        n_events = int(mark_mask.sum().item())
        if n_events:
            pdr_err = output.marks.pdr_mean()[mark_mask] - batch.edge_event_pdr[mark_mask]
            th_err = output.marks.throughput_mean()[mark_mask] - batch.edge_event_throughput[mark_mask]
            d_err = output.marks.delay_mean()[mark_mask] - batch.edge_event_delay[mark_mask]
            pdr_mae, pdr_rmse = float(pdr_err.abs().mean()), float(torch.sqrt((pdr_err**2).mean()))
            th_mae, th_rmse = float(th_err.abs().mean()), float(torch.sqrt((th_err**2).mean()))
            d_mae, d_rmse = float(d_err.abs().mean()), float(torch.sqrt((d_err**2).mean()))
        else:
            pdr_mae = pdr_rmse = th_mae = th_rmse = d_mae = d_rmse = float("nan")

        macro_f1, cause_brier = _cause_metrics(output, batch, risk)

    return EvalMetrics(
        val_loss=float(loss_out.total_loss),
        val_tpp_loss=float(loss_out.tpp_loss),
        val_mark_loss=float(loss_out.mark_loss),
        event_auc=final_auc,
        event_auprc=final_auprc,
        brier_score=final_brier,
        integrated_brier_score=ibs,
        calibration_error=final_ece,
        survival_calibration_error=survival_ece,
        c_index=c_index,
        failure_time_mae_s=time_mae,
        pdr_mae=pdr_mae, pdr_rmse=pdr_rmse,
        throughput_mae=th_mae, throughput_rmse=th_rmse,
        delay_mae=d_mae, delay_rmse=d_rmse,
        cause_macro_f1=macro_f1, cause_brier=cause_brier,
        n_edges=int(output.total_intensity().numel()),
        n_at_risk=int(risk.sum()),
        n_events=n_events,
        horizon_metrics=horizon_metrics,
    )
