"""Check the exact event-time likelihood of the nominal physics generator.

At strength=0 the simulator generates failures from the time-varying nominal
physics intensity using adaptive Ogata thinning.  The standard benchmark
instead evaluates a piecewise-constant forecast hazard fixed at the reference
time.  This diagnostic recomputes the nominal time-varying compensator over
the actual observed event/censoring interval and compares it with that
forecast approximation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.training.train import MissionCache
from simulator.dataset.window_builder import build_window


def make_config(args):
    base = default_config(master_seed=args.corpus_seed)
    fleet = FleetConfig(
        usv_count=1,
        n_auv_min=3,
        n_auv_max=8,
        m_iout_min=10,
        m_iout_max=30,
        max_auv_speed_mps=base.fleet.max_auv_speed_mps,
    )
    env = MissionEnvelopeConfig(
        duration_s=args.mission_duration_s,
        node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m,
        depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state,
    )
    return MissionConfig(
        master_seed=args.corpus_seed,
        split="train",
        wave=base.wave,
        current=base.current,
        acoustic=base.acoustic,
        fleet=fleet,
        envelope=env,
        hazard=base.hazard,
        graph=base.graph,
    )


def _interpolate_endpoint_values(array, timestamps, node_indices, t_query):
    """Fast per-row interpolation matching event_generator's bracket rule."""
    node_indices = np.asarray(node_indices, dtype=np.int64)
    t_query = np.asarray(t_query, dtype=np.float64)
    sample_rate = 1.0 / (timestamps[1] - timestamps[0])
    tidx = np.clip(np.floor(t_query * sample_rate).astype(np.int64), 0, len(timestamps) - 1)
    t0 = timestamps[tidx]
    dt = t_query - t0
    x0 = array[tidx, node_indices, 0]
    y0 = array[tidx, node_indices, 1]
    z0 = array[tidx, node_indices, 2]
    vx0 = array[tidx, node_indices, 3]
    vy0 = array[tidx, node_indices, 4]
    return x0 + vx0 * dt, y0 + vy0 * dt, z0, vx0, vy0


def integrate_edges(mission, sample, quadrature_points: int):
    """Vectorized trapezoidal compensator for all at-risk edges in one window."""
    risk = sample.edge_at_risk.numpy() > 0.5
    if not np.any(risk):
        return np.empty(0), np.empty(0, dtype=int), np.empty(0), np.empty(0)

    edge_index = sample.edge_index.numpy()[:, risk]
    event_dt = sample.edge_event_time_delta.numpy()[risk].astype(np.float64)
    causes = sample.edge_event_cause_id.numpy()[risk].astype(np.int64)
    t_ref = float(sample.reference_time_s)

    # Each edge gets its own integration interval [t_ref, t_ref + event_dt].
    u = np.linspace(0.0, 1.0, quadrature_points, dtype=np.float64)
    t_end = t_ref + event_dt
    t_query = t_ref + event_dt[:, None] * u[None, :]
    flat_t = t_query.reshape(-1)
    flat_i = np.repeat(edge_index[0], quadrature_points)
    flat_j = np.repeat(edge_index[1], quadrature_points)

    xi, yi, zi, vxi, vyi = _interpolate_endpoint_values(
        mission.array, mission.timestamps, flat_i, flat_t
    )
    xj, yj, zj, vxj, vyj = _interpolate_endpoint_values(
        mission.array, mission.timestamps, flat_j, flat_t
    )
    distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
    geo, wave, doppler, current = mission.evaluator.cause_hazards_batch(
        distance, xi, yi, zi, xj, yj, zj, vxi, vyi, vxj, vyj, flat_t
    )
    total = (geo + wave + doppler + current).reshape(len(edge_index[0]), quadrature_points)
    compensator = np.trapezoid(total, t_query, axis=1)

    event_mask = causes >= 0
    log_event_hazard = np.zeros(len(causes), dtype=np.float64)
    if np.any(event_mask):
        ev = np.flatnonzero(event_mask)
        ev_t = t_end[ev]
        ex, ey, ez, evxi, evyi = _interpolate_endpoint_values(
            mission.array, mission.timestamps, edge_index[0, ev], ev_t
        )
        jx, jy, jz, jvxi, jvyi = _interpolate_endpoint_values(
            mission.array, mission.timestamps, edge_index[1, ev], ev_t
        )
        ed = np.sqrt((ex - jx) ** 2 + (ey - jy) ** 2 + (ez - jz) ** 2)
        egeo, ewave, edop, ecur = mission.evaluator.cause_hazards_batch(
            ed, ex, ey, ez, jx, jy, jz, evxi, evyi, jvxi, jvyi, ev_t
        )
        ecauses = np.stack([egeo, ewave, edop, ecur], axis=1)
        selected = ecauses[np.arange(len(ev)), causes[ev]]
        log_event_hazard[ev] = np.log(np.maximum(selected, 1e-12))

    oracle_loss = compensator - log_event_hazard
    return oracle_loss, causes, compensator, event_dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-missions", type=int, default=50)
    p.add_argument("--corpus-seed", type=int, default=2026)
    p.add_argument("--mission-duration-s", type=float, default=900)
    p.add_argument("--window-length", type=int, default=10)
    p.add_argument("--lookahead-s", type=float, default=30)
    p.add_argument("--eval-stride", type=int, default=5)
    p.add_argument("--quadrature-points", type=int, default=65)
    p.add_argument("--output", default="results/physics_consistency/event_time_oracle.json")
    args = p.parse_args()
    if args.quadrature_points < 5 or args.quadrature_points % 2 == 0:
        raise ValueError("--quadrature-points must be odd and >= 5")

    config = make_config(args)
    cache = MissionCache.generate(
        config, args.corpus_seed, args.n_missions,
        unmodeled_state_strength=0.0,
        unmodeled_state_mode="cause_specific",
    )

    oracle_losses = []
    ref_losses = []
    event_losses = []
    n_windows = n_at_risk = n_events = 0
    max_abs_integral_diff = 0.0
    mission_summaries = []

    for mission in cache.missions:
        valid = cache._valid_reference_indices(mission, args.lookahead_s)
        m_oracle = m_ref = 0.0
        m_count = 0
        for t_idx in valid[::args.eval_stride]:
            sample = build_window(
                mission.mission_id, mission.array, mission.node_ids, mission.node_types,
                mission.timestamps, mission.events, mission.evaluator,
                mission.candidate_range_factor, reference_t_idx=t_idx,
                window_length=args.window_length, lookahead_s=args.lookahead_s,
            )
            risk = sample.edge_at_risk.numpy() > 0.5
            if not np.any(risk):
                continue
            physics = sample.edge_physics.numpy()[risk]
            et = sample.edge_event_time_delta.numpy()[risk].astype(np.float64)
            causes = sample.edge_event_cause_id.numpy()[risk].astype(np.int64)
            ref_total = np.clip(physics.sum(axis=1), 1e-12, None)
            ref_loss = et * ref_total
            observed = causes >= 0
            if np.any(observed):
                idx = np.flatnonzero(observed)
                ref_loss[idx] -= np.log(np.maximum(physics[observed, causes[observed]], 1e-12))

            oloss, _, compensator, _ = integrate_edges(mission, sample, args.quadrature_points)
            oracle_losses.extend(oloss.tolist())
            ref_losses.extend(ref_loss.tolist())
            n_at_risk += len(ref_loss)
            n_events += int(observed.sum())
            n_windows += 1
            event_losses.extend((ref_loss[observed] - (et[observed] * ref_total[observed]) + et[observed] * ref_total[observed]).tolist())
            m_oracle += float(np.sum(oloss)); m_ref += float(np.sum(ref_loss)); m_count += len(ref_loss)
            max_abs_integral_diff = max(max_abs_integral_diff, float(np.max(np.abs(compensator - et * ref_total))))
        if m_count:
            mission_summaries.append({"mission_id": mission.mission_id, "n_at_risk": m_count, "oracle_nll": m_oracle/m_count, "reference_constant_nll": m_ref/m_count})

    oracle = float(np.mean(oracle_losses))
    ref = float(np.mean(ref_losses))
    rel = 100.0 * (ref - oracle) / ref if ref else math.nan
    result = {
        "strength": 0.0,
        "n_missions": args.n_missions,
        "n_windows": n_windows,
        "n_at_risk": n_at_risk,
        "n_failures": n_events,
        "quadrature_points": args.quadrature_points,
        "oracle_time_varying_physics_nll": oracle,
        "reference_time_piecewise_constant_physics_nll": ref,
        "piecewise_constant_excess_nll_percent": rel,
        "max_abs_reference_integral_discrepancy": max_abs_integral_diff,
        "interpretation": "The oracle uses the same nominal time-varying physics intensity and trajectory interpolation convention as event generation; the reference-time value is the benchmark's piecewise-constant approximation.",
        "missions": mission_summaries,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "missions"}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
