"""Non-neural physics-preserving residual hazard learner.

The learner fits a cause-specific Poisson intensity with the physics hazard as
an *offset*.  For cause c, exposure tau and event indicator y_c,

    lambda_c = lambda_phys,c * exp(f_c(x)).

The negative log likelihood is

    tau * lambda_phys,c * exp(f_c) - y_c * f_c + const.

This is equivalent to a Poisson regression on pseudo-rate

    z = y_c / (tau * lambda_phys,c)

with sample weight

    w = tau * lambda_phys,c.

A histogram gradient-boosting regressor with Poisson loss therefore learns
only the residual multiplicative factor, while the physical hazard remains an
explicit offset.  No neural-network parameters are involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..dataset.window_builder import WindowBatch

CAUSE_NAMES = ("geo", "wave", "doppler", "current")
_EPS = 1e-8


@dataclass
class GBTConfig:
    max_iter: int = 250
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 30
    l2_regularization: float = 1.0
    max_bins: int = 255
    random_state: int = 0
    residual_clip: float = 2.0
    residual_scale: float = 1.0


class PhysicsOffsetGBT:
    """Four cause-specific gradient-boosted residual hazard models."""

    def __init__(self, config: GBTConfig | None = None) -> None:
        self.config = config or GBTConfig()
        self.models: dict[str, HistGradientBoostingRegressor] = {}
        self.feature_names_: list[str] = []
        self.fitted_ = False

    def fit(self, X: np.ndarray, physics: np.ndarray, event_cause: np.ndarray,
            at_risk: np.ndarray, horizon_s: float, event_time: np.ndarray | None = None) -> "PhysicsOffsetGBT":
        X = np.asarray(X, dtype=np.float64)
        physics = np.asarray(physics, dtype=np.float64)
        causes = np.asarray(event_cause, dtype=np.int64)
        risk = np.asarray(at_risk, dtype=bool)
        if X.ndim != 2 or physics.shape != (len(X), 4):
            raise ValueError("X must be [N,F] and physics must be [N,4]")
        if len(causes) != len(X) or len(risk) != len(X):
            raise ValueError("event_cause and at_risk must match X length")
        if horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        if event_time is None:
            # Backward-compatible fallback. New benchmarks should always pass
            # observed event time for failures and horizon for censoring.
            exposure_all = np.full(len(X), float(horizon_s), dtype=np.float64)
        else:
            exposure_all = np.asarray(event_time, dtype=np.float64)
            if len(exposure_all) != len(X):
                raise ValueError("event_time must match X length")
            exposure_all = np.where(np.isfinite(exposure_all), exposure_all, float(horizon_s))
            exposure_all = np.maximum(exposure_all, 0.0)

        base_mask = risk & np.isfinite(X).all(axis=1) & np.isfinite(physics).all(axis=1)
        for ci, name in enumerate(CAUSE_NAMES):
            base = np.clip(physics[:, ci], _EPS, None)
            mask = base_mask & (exposure_all > 0)
            # First-event likelihood exposure: observed failures contribute
            # their actual event time; censored edges contribute the full
            # lookahead. This is the compensator exposure for the continuous-
            # time first-event competing-risk likelihood.
            w = exposure_all[mask] * base[mask]
            y = ((causes[mask] == ci)).astype(np.float64)
            z = y / np.clip(w, _EPS, None)
            # Avoid pathological targets from near-zero physical hazards;
            # their contribution is already down-weighted by w.
            z = np.clip(z, 0.0, 1e6)
            model = HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=self.config.max_iter,
                learning_rate=self.config.learning_rate,
                max_leaf_nodes=self.config.max_leaf_nodes,
                min_samples_leaf=self.config.min_samples_leaf,
                l2_regularization=self.config.l2_regularization,
                max_bins=self.config.max_bins,
                random_state=self.config.random_state + ci,
                early_stopping=True,
            )
            model.fit(X[mask], z, sample_weight=w)
            self.models[name] = model
        self.fitted_ = True
        return self

    def predict_ratio(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("PhysicsOffsetGBT has not been fitted")
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros((len(X), 4), dtype=np.float64)
        for ci, name in enumerate(CAUSE_NAMES):
            ratio = np.clip(self.models[name].predict(X), _EPS, None)
            log_ratio = np.clip(np.log(ratio), -self.config.residual_clip, self.config.residual_clip)
            out[:, ci] = np.exp(self.config.residual_scale * log_ratio)
        return out

    def predict_hazard(self, X: np.ndarray, physics: np.ndarray) -> np.ndarray:
        physics = np.asarray(physics, dtype=np.float64)
        if physics.shape != (len(X), 4):
            raise ValueError("physics must be [N,4]")
        return np.clip(physics, _EPS, None) * self.predict_ratio(X)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PhysicsOffsetGBT":
        return joblib.load(path)


def continuous_time_competing_risk_nll(
    physics_hazard: np.ndarray,
    residual_log_ratio: np.ndarray,
    residual_scale: float,
    event_time: np.ndarray,
    event_cause: np.ndarray,
    at_risk: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Mean continuous-time competing-risk NLL for piecewise-constant hazards.

    For a censored edge, the contribution is::

        tau * sum_c lambda_c

    For an observed first event at time t from cause c*, it is::

        t * sum_c lambda_c - log(lambda_c*)

    where ``lambda_c = physics_hazard_c * exp(scale * residual_log_ratio_c)``.
    ``event_time`` must therefore contain the observed event time for failures
    and the full lookahead/exposure time for censored observations.

    This is the validation objective used to select the residual shrinkage
    scale; it is the continuous-time likelihood corresponding to the
    competing-risk hazard model, rather than a binary Brier surrogate.
    """
    physics_hazard = np.asarray(physics_hazard, dtype=np.float64)
    residual_log_ratio = np.asarray(residual_log_ratio, dtype=np.float64)
    event_time = np.asarray(event_time, dtype=np.float64)
    event_cause = np.asarray(event_cause, dtype=np.int64)
    at_risk = np.asarray(at_risk, dtype=bool)

    if physics_hazard.ndim != 2 or physics_hazard.shape[1] != 4:
        raise ValueError("physics_hazard must be [N,4]")
    if residual_log_ratio.shape != physics_hazard.shape:
        raise ValueError("residual_log_ratio must have shape [N,4]")
    n = len(physics_hazard)
    if len(event_time) != n or len(event_cause) != n or len(at_risk) != n:
        raise ValueError("event_time, event_cause and at_risk must match N")

    mask = at_risk & np.isfinite(physics_hazard).all(axis=1) & np.isfinite(residual_log_ratio).all(axis=1) & np.isfinite(event_time)
    if not np.any(mask):
        return float("nan")

    ph = np.clip(physics_hazard[mask], eps, None)
    lr = residual_log_ratio[mask]
    t = np.maximum(event_time[mask], 0.0)
    causes = event_cause[mask]

    log_ratio = np.clip(residual_scale * lr, -20.0, 20.0)
    hazards = ph * np.exp(log_ratio)
    total = np.sum(hazards, axis=1)
    loss = t * total

    observed = causes >= 0
    if np.any(observed):
        idx = np.flatnonzero(observed)
        c = causes[observed]
        valid_cause = (c >= 0) & (c < hazards.shape[1])
        if not np.all(valid_cause):
            raise ValueError("event_cause contains invalid cause IDs")
        loss[idx] -= np.log(np.maximum(hazards[idx, c], eps))

    return float(np.mean(loss))


def _edge_node_features(batch: WindowBatch) -> np.ndarray:
    """Construct leakage-safe edge features from the observed node window."""
    nf = batch.node_features.detach().cpu().numpy().astype(np.float64)  # [B,K,V,F]
    b, k, v, f = nf.shape
    edge = batch.edge_index.detach().cpu().numpy().astype(np.int64)
    eb = batch.edge_batch.detach().cpu().numpy().astype(np.int64)
    src, dst = edge
    sb, sl = src // v, src % v
    db, dl = dst // v, dst % v
    # [E,K,F]
    si, sj = nf[sb, :, sl, :], nf[db, :, dl, :]
    def stats(x: np.ndarray) -> np.ndarray:
        last = x[:, -1, :]
        mean = x.mean(axis=1)
        std = x.std(axis=1)
        delta = x[:, -1, :] - x[:, 0, :]
        return np.concatenate([last, mean, std, delta], axis=1)
    return np.concatenate([stats(si), stats(sj)], axis=1)


def _neighbor_physics_features(batch: WindowBatch) -> np.ndarray:
    """Per-edge mean physics hazard of the source/destination neighborhoods."""
    physics = batch.edge_physics.detach().cpu().numpy().astype(np.float64)
    edge = batch.edge_index.detach().cpu().numpy().astype(np.int64)
    eb = batch.edge_batch.detach().cpu().numpy().astype(np.int64)
    nf = batch.node_features.shape
    v = int(nf[2])
    src, dst = edge
    # Global node ids are batch-offset. Aggregate incident candidate-edge
    # physics for each endpoint; this uses only reference-time graph data.
    n_nodes = int(nf[0] * v)
    sums = np.zeros((n_nodes, 4), dtype=np.float64)
    counts = np.zeros(n_nodes, dtype=np.float64)
    for u, vv, ph in zip(src, dst, physics):
        sums[u] += ph; sums[vv] += ph
        counts[u] += 1.0; counts[vv] += 1.0
    means = sums / np.maximum(counts[:, None], 1.0)
    return np.concatenate([means[src], means[dst]], axis=1)


def edge_feature_matrix(batch: WindowBatch) -> tuple[np.ndarray, list[str]]:
    """Return deterministic, leakage-safe features and names."""
    node_part = _edge_node_features(batch)
    physics = batch.edge_physics.detach().cpu().numpy().astype(np.float64)
    neighbor = _neighbor_physics_features(batch)
    kind = batch.edge_kind_ids.detach().cpu().numpy().astype(np.float64)
    # Physics enters both as raw log-scale features and through neighborhood
    # context.  No target/event tensors are used here.
    phys_log = np.log1p(np.clip(physics, 0.0, None))
    X = np.concatenate([node_part, phys_log, neighbor, kind[:, None]], axis=1)
    f = node_part.shape[1]
    names = [f"node_pair_{i}" for i in range(f)]
    names += [f"log_physics_{n}" for n in CAUSE_NAMES]
    names += [f"neighbor_src_{n}" for n in CAUSE_NAMES]
    names += [f"neighbor_dst_{n}" for n in CAUSE_NAMES]
    names += ["edge_kind"]
    return X, names


def extract_training_arrays(batches: Iterable[WindowBatch]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, list[str]]:
    xs, ps, cs, rs = [], [], [], []
    horizons = []
    names: list[str] = []
    for batch in batches:
        X, names = edge_feature_matrix(batch)
        xs.append(X)
        ps.append(batch.edge_physics.detach().cpu().numpy())
        cs.append(batch.edge_event_cause_id.detach().cpu().numpy())
        rs.append((batch.edge_at_risk.detach().cpu().numpy() > 0.5))
        horizons.append(float(batch.lookahead_s))
    if not xs:
        raise ValueError("No batches supplied")
    if len(set(round(h, 8) for h in horizons)) != 1:
        raise ValueError("All batches must use the same lookahead_s")
    return np.concatenate(xs), np.concatenate(ps), np.concatenate(cs), np.concatenate(rs), horizons[0], names
