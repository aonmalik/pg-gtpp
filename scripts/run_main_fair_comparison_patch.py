"""Main fair comparison for PG-GTPP.

Compares physics_only, neural_only, pg_gtpp, physics_soft_reg and the seven
architecture-family baselines on identical mission-level train/val/test
splits. Adds the physics-offset GBT residual model and reports a common
continuous-time first-event NLL diagnostic.

For physics-aware models (physics_only, pg_gtpp, gbt_residual), the forecast
compensator integrates the nominal physics cause hazards along the known
future trajectory, multiplied by a history-only residual ratio. For models
without a physics path, the reference-time predicted hazard is held constant
through the forecast. This convention is explicit in the output JSON.
"""
from __future__ import annotations

import argparse, json, math, os
from pathlib import Path
import numpy as np
import torch

from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.training.train import MissionCache, TrainConfig, train
from simulator.training.eval import evaluate
from simulator.dataset.window_builder import build_window, collate_windows
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.model.baselines import BASELINE_REGISTRY, physics_regularized_loss
from simulator.model.gbt_residual import GBTConfig, PhysicsOffsetGBT, extract_training_arrays, edge_feature_matrix
from simulator.model.baselines.common import BaselineOutput

# Reuse the independently validated trajectory-integrated physics implementation.
# This avoids maintaining a second copy of the most error-prone numerical code.

CAUSE_NAMES = ("geo", "wave", "doppler", "current")
F_NODE = 11
ABLATIONS = {
    "physics_only": dict(learned_correction=False, use_physics_intensity=True, use_fusion_gate=False),
    "neural_only": dict(learned_correction=True, use_physics_intensity=False, use_fusion_gate=False, use_physics_attention=False),
    "pg_gtpp": dict(learned_correction=True, use_physics_intensity=True, use_fusion_gate=True),
}


def finite(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def make_config(a):
    base = default_config(master_seed=a.corpus_seed)
    fleet = FleetConfig(usv_count=1, n_auv_min=3, n_auv_max=8, m_iout_min=10, m_iout_max=30,
                        max_auv_speed_mps=base.fleet.max_auv_speed_mps)
    env = MissionEnvelopeConfig(duration_s=a.mission_duration_s, node_sample_rate_hz=1.0,
        area_size_m=base.envelope.area_size_m, depth_range_m=base.envelope.depth_range_m,
        sea_state=base.envelope.sea_state)
    return MissionConfig(master_seed=a.corpus_seed, split="train", wave=base.wave, current=base.current,
        acoustic=base.acoustic, fleet=fleet, envelope=env, hazard=base.hazard, graph=base.graph)


def split_three(cache, seed):
    tr, hold = cache.split(.30, seed)
    va, te = hold.split(.50, seed + 1)
    return tr, va, te


def interp(mission, idx, t):
    ts, arr = mission.timestamps, mission.array
    idx, t = np.asarray(idx, dtype=np.int64), np.asarray(t, dtype=np.float64)
    sr = 1.0 / (ts[1] - ts[0])
    k = np.clip(np.floor(t * sr).astype(np.int64), 0, len(ts) - 1)
    dt = t - ts[k]
    x0, y0, z0 = arr[k, idx, 0], arr[k, idx, 1], arr[k, idx, 2]
    vx, vy = arr[k, idx, 3], arr[k, idx, 4]
    return x0 + vx * dt, y0 + vy * dt, z0, vx, vy


def physics_at(mission, src, dst, t):
    ex, ey, ez, evx, evy = interp(mission, src, t)
    jx, jy, jz, jvx, jvy = interp(mission, dst, t)
    d = np.sqrt((ex-jx)**2 + (ey-jy)**2 + (ez-jz)**2)
    hs = mission.evaluator.cause_hazards_batch(d, ex, ey, ez, jx, jy, jz, evx, evy, jvx, jvy, t)
    return np.stack(hs, axis=1)


def integrated_losses(mcache, model=None, gbt=None, a=None, q=33, return_records=False):
    """Common first-event NLL.

    Physics-only is delegated to the validated integrated benchmark verbatim.
    Physics-aware learned models use the same trajectory integration and an
    aggregate event-time term. Non-physics models use their reference-time
    aggregate intensity as a constant hazard over the forecast.
    """
    if model is None and gbt is None:
        # Physics-only must not route through the GBT helper: that helper
        # constructs GBT feature matrices, which require a WindowBatch.  More
        # importantly, the primary comparison uses the same aggregate event
        # term as every other model.  Evaluate the nominal physics likelihood
        # directly with a unit cause-wise residual ratio.
        losses = []
        records = []
        for mission in mcache.missions:
            valid = mcache._valid_reference_indices(mission, a.lookahead_s)
            for tidx in valid[::a.eval_stride]:
                b = build_window(
                    mission.mission_id, mission.array, mission.node_ids, mission.node_types,
                    mission.timestamps, mission.events, mission.evaluator,
                    mission.candidate_range_factor, reference_t_idx=tidx,
                    window_length=a.window_length, lookahead_s=a.lookahead_s,
                )
                risk = b.edge_at_risk.numpy() > 0.5
                if not np.any(risk):
                    continue
                ei = b.edge_index.numpy()[:, risk]
                et = b.edge_event_time_delta.numpy()[risk].astype(np.float64)
                causes = b.edge_event_cause_id.numpy()[risk].astype(np.int64)
                tref = float(b.reference_time_s)
                u = np.linspace(0.0, 1.0, q)
                tq = tref + et[:, None] * u[None, :]
                flat = tq.reshape(-1)
                ii = np.repeat(ei[0], q)
                jj = np.repeat(ei[1], q)
                xi, yi, zi, vxi, vyi = interp(mission, ii, flat)
                xj, yj, zj, vxj, vyj = interp(mission, jj, flat)
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                hs = mission.evaluator.cause_hazards_batch(
                    d, xi, yi, zi, xj, yj, zj, vxi, vyi, vxj, vyj, flat
                )
                ph = np.stack(hs, axis=1).reshape(len(ei[0]), q, 4)
                comp = np.trapezoid(ph, x=tq[:, :, None], axis=1).sum(axis=1)
                loss = comp.copy()
                obs = causes >= 0
                if np.any(obs):
                    ev = np.flatnonzero(obs)
                    tev = tref + et[ev]
                    ex, ey, ez, evx, evy = interp(mission, ei[0, ev], tev)
                    jx, jy, jz, jvx, jvy = interp(mission, ei[1, ev], tev)
                    dd = np.sqrt((ex-jx)**2 + (ey-jy)**2 + (ez-jz)**2)
                    eh = np.stack(mission.evaluator.cause_hazards_batch(
                        dd, ex, ey, ez, jx, jy, jz,
                        evx, evy, jvx, jvy, tev
                    ), axis=1)
                    # Primary fair metric: aggregate first-event intensity.
                    loss[ev] -= np.log(np.maximum(eh.sum(axis=1), 1e-12))
                if not np.all(np.isfinite(loss)):
                    raise FloatingPointError("Non-finite physics-only integrated NLL")
                losses.extend(loss.tolist())
                if return_records:
                    for k, edge_pos in enumerate(np.flatnonzero(risk)):
                        records.append({"mission_id": str(mission.mission_id), "edge_src": int(ei[0,k]), "edge_dst": int(ei[1,k]), "event_time_delta_s": float(et[k]), "event_cause_id": int(causes[k]), "nll": float(loss[k])})
        if return_records:
            return (float(np.mean(losses)) if losses else float("nan"), records)
        return float(np.mean(losses)) if losses else float("nan")

    losses = []
    records = []
    for mission in mcache.missions:
        valid = mcache._valid_reference_indices(mission, a.lookahead_s)
        for tidx in valid[::a.eval_stride]:
            b = build_window(
                mission.mission_id, mission.array, mission.node_ids, mission.node_types,
                mission.timestamps, mission.events, mission.evaluator,
                mission.candidate_range_factor, reference_t_idx=tidx,
                window_length=a.window_length, lookahead_s=a.lookahead_s,
            )
            risk = b.edge_at_risk.numpy() > 0.5
            if not np.any(risk):
                continue
            fb = collate_windows([b])

            if gbt is not None:
                X, _ = edge_feature_matrix(fb)
                all_ratios = np.asarray(gbt.predict_ratio(X), dtype=np.float64)
                if all_ratios.shape[0] != risk.shape[0]:
                    raise ValueError(f"GBT ratio/edge mismatch: ratios={all_ratios.shape}, edges={risk.shape}")
                ratios = all_ratios[risk]
                physics_mode = True
            else:
                with torch.no_grad():
                    out = model(fb.node_features, fb.node_type_ids, fb.node_mask,
                                fb.edge_index, fb.edge_physics)
                if isinstance(model, PGGTPPModel) and model.fusion.heads["geo"].use_physics_intensity:
                    raw_p = fb.edge_physics.detach().cpu().numpy()[risk]
                    c = np.stack(
                        [out.fusion[n]["fused"].detach().cpu().numpy() for n in CAUSE_NAMES],
                        axis=1,
                    )[risk]

                    positive_p = raw_p > 0.0

                    if np.any(positive_p):
                        positive_ratios = c[positive_p] / raw_p[positive_p]

                        print(
                            "[PG DEBUG]"
                            f" c_shape={c.shape}"
                            f" raw_physics_min={np.nanmin(raw_p):.6g}"
                            f" raw_physics_max={np.nanmax(raw_p):.6g}"
                            f" raw_physics_zeros={np.sum(raw_p == 0.0)}"
                            f" raw_physics_nonfinite={np.sum(~np.isfinite(raw_p))}"
                            f" positive_ratio_min={np.nanmin(positive_ratios):.6g}"
                            f" positive_ratio_max={np.nanmax(positive_ratios):.6g}"
                            f" positive_ratio_mean={np.nanmean(positive_ratios):.6g}"
                            f" positive_ratio_median={np.nanmedian(positive_ratios):.6g}"
                            f" positive_ratio_p01={np.nanpercentile(positive_ratios, 1):.6g}"
                            f" positive_ratio_p99={np.nanpercentile(positive_ratios, 99):.6g}"
                            f" positive_ratio_nonfinite={np.sum(~np.isfinite(positive_ratios))}"
                            f" fused_min={np.nanmin(c):.6g}"
                            f" fused_max={np.nanmax(c):.6g}"
                            f" fused_zeros={np.sum(c == 0.0)}"
                        )
                    else:
                        print(
                            "[PG DEBUG]"
                            f" c_shape={c.shape}"
                            f" raw_physics_zeros={np.sum(raw_p == 0.0)}"
                            f" raw_physics_nonfinite={np.sum(~np.isfinite(raw_p))}"
                            f" fused_zeros={np.sum(c == 0.0)}"
                            " NO_POSITIVE_PHYSICS_HAZARDS"
                        )

                    # Use the original physics hazard for the integrated
                    # likelihood; numerical flooring is handled downstream.
                    p = np.clip(raw_p, 1e-12, None)
                    ratios = c / p

                    physics_mode = True
                else:
                    ratios = None
                    ref_total = np.clip(out.total_intensity().detach().cpu().numpy()[risk], 1e-12, None)
                    physics_mode = False

            ei = b.edge_index.numpy()[:, risk]
            et = b.edge_event_time_delta.numpy()[risk].astype(np.float64)
            causes = b.edge_event_cause_id.numpy()[risk].astype(np.int64)
            tref = float(b.reference_time_s)

            if physics_mode:
                # EXACT same trajectory/interpolation convention as the validated GBT benchmark.
                u = np.linspace(0.0, 1.0, q)
                tq = tref + et[:, None] * u[None, :]
                flat = tq.reshape(-1)
                ii = np.repeat(ei[0], q)
                jj = np.repeat(ei[1], q)
                xi, yi, zi, vxi, vyi = interp(mission, ii, flat)
                xj, yj, zj, vxj, vyj = interp(mission, jj, flat)
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                hs = mission.evaluator.cause_hazards_batch(
                    d, xi, yi, zi, xj, yj, zj, vxi, vyi, vxj, vyj, flat
                )
                ph = np.stack(hs, axis=1).reshape(len(ei[0]), q, 4)
                hz = ph * ratios[:, None, :]
                comp = np.trapezoid(hz, x=tq[:, :, None], axis=1).sum(axis=1)

                # Aggregate event intensity at the observed event time.
                obs = causes >= 0
                loss = comp.copy()
                if np.any(obs):
                    ev = np.flatnonzero(obs)
                    tev = tref + et[ev]
                    ex, ey, ez, evx, evy = interp(mission, ei[0, ev], tev)
                    jx, jy, jz, jvx, jvy = interp(mission, ei[1, ev], tev)
                    dd = np.sqrt((ex-jx)**2 + (ey-jy)**2 + (ez-jz)**2)
                    eh = np.stack(mission.evaluator.cause_hazards_batch(
                        dd, ex, ey, ez, jx, jy, jz, evx, evy, jvx, jvy, tev
                    ), axis=1) * ratios[ev]
                    loss[ev] -= np.log(np.maximum(eh.sum(axis=1), 1e-12))
            else:
                # Generic baselines: constant reference-time aggregate intensity.
                comp = ref_total * et
                loss = comp.copy()
                obs = causes >= 0
                if np.any(obs):
                    ev = np.flatnonzero(obs)
                    loss[ev] -= np.log(np.maximum(ref_total[ev], 1e-12))

            if not np.all(np.isfinite(loss)):
                raise FloatingPointError("Non-finite integrated NLL encountered")
            losses.extend(loss.tolist())
            if return_records:
                for k, edge_pos in enumerate(np.flatnonzero(risk)):
                    records.append({"mission_id": str(mission.mission_id), "edge_src": int(ei[0,k]), "edge_dst": int(ei[1,k]), "event_time_delta_s": float(et[k]), "event_cause_id": int(causes[k]), "nll": float(loss[k])})

    if return_records:
        return (float(np.mean(losses)) if losses else float("nan"), records)
    return float(np.mean(losses)) if losses else float("nan")


def train_physics_soft(model, cache, cfg, reg_weight):
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    model.train()
    for step in range(1, cfg.n_steps + 1):
        b = cache.sample_batch(rng, cfg.batch_size, cfg.window_length, cfg.lookahead_s)
        if b.edge_index.shape[1] == 0: continue
        opt.zero_grad()
        out = model(b.node_features, b.node_type_ids, b.node_mask, b.edge_index, b.edge_physics)
        loss = physics_regularized_loss(out, b, b.edge_physics, mark_loss_weight=cfg.mark_loss_weight, physics_reg_weight=reg_weight)
        loss.total_loss.backward(); opt.step()
    return model


def build_model(name, a, seed):
    torch.manual_seed(seed)
    if name in ABLATIONS:
        return PGGTPPModel(f_node=F_NODE, hidden_dim=a.hidden_dim, **ABLATIONS[name])
    return BASELINE_REGISTRY[name](f_node=F_NODE, hidden_dim=a.hidden_dim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-missions', type=int, default=50); p.add_argument('--corpus-seed', type=int, default=2026)
    p.add_argument('--seeds', default='42,43,44'); p.add_argument('--strengths', default='0')
    p.add_argument('--mission-duration-s', type=float, default=900); p.add_argument('--window-length', type=int, default=10)
    p.add_argument('--lookahead-s', type=float, default=30); p.add_argument('--eval-stride', type=int, default=10)
    p.add_argument('--quadrature-points', type=int, default=33); p.add_argument('--n-steps', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=16); p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=3e-3); p.add_argument('--mark-loss-weight', type=float, default=.25)
    p.add_argument('--physics-reg-weight', type=float, default=.1); p.add_argument('--val-every', type=int, default=250)
    p.add_argument('--val-stride', type=int, default=10); p.add_argument('--output-dir', default='results/main_fair_comparison')
    p.add_argument('--only', default=None); p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--gbt-max-iter', type=int, default=250); p.add_argument('--gbt-learning-rate', type=float, default=.05)
    p.add_argument('--gbt-max-leaf-nodes', type=int, default=31); p.add_argument('--gbt-min-samples-leaf', type=int, default=30)
    p.add_argument('--gbt-l2', type=float, default=1.0); p.add_argument('--gbt-scales', default='0,.125,.25,.375,.5,.75,1,1.25,1.5,2')
    p.add_argument('--allow-nondeterministic', action='store_true')
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    if not a.allow_nondeterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8'); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True, warn_only=True)
    names = list(BASELINE_REGISTRY) + list(ABLATIONS) + ['gbt_residual']
    if a.only: names = [n for n in names if n in set(a.only.split(','))]
    seeds = [int(x) for x in a.seeds.split(',') if x.strip()]; strengths = [float(x) for x in a.strengths.split(',') if x.strip()]
    summary = {'protocol': vars(a), 'models': names, 'conditions': {}}
    for strength in strengths:
        print(f'\n===== mismatch strength {strength:g} =====')
        cfg = make_config(a)
        cache = MissionCache.generate(cfg, a.corpus_seed, a.n_missions, unmodeled_state_strength=strength, unmodeled_state_mode='cause_specific')
        tr, va, te = split_three(cache, 2026)
        cond = {'strength': strength, 'models': {}}
        phys_nll = integrated_losses(te, a=a, q=a.quadrature_points)
        _, physics_records = integrated_losses(te, a=a, q=a.quadrature_points, return_records=True)
        physics_by_key = {(r['mission_id'],r['edge_src'],r['edge_dst'],r['event_time_delta_s']): r['nll'] for r in physics_records}

        # Independent reference-time diagnostic retained for auditability.
        cond['physics_only_oracle_integrated_nll'] = phys_nll
        print(f'physics_only | validated integrated NLL={phys_nll:.6f}')
        for seed in seeds:
            for name in names:
                path = out / f'strength_{strength:g}' / f'{name}_seed{seed}.json'; path.parent.mkdir(parents=True, exist_ok=True)
                if a.skip_existing and path.exists():
                    cond['models'][f'{name}_seed{seed}'] = json.loads(path.read_text()); continue
                if name == 'physics_only':
                    test_nll = phys_nll
                    _, physics_records = integrated_losses(te, a=a, q=a.quadrature_points, return_records=True)
                    path.with_name(path.stem + '_test_losses.json').write_text(json.dumps(physics_records, indent=2))
                    rec = {
                        'model': name,
                        'seed': seed,
                        'integrated_test_nll': test_nll,
                        'physics_nll': phys_nll,
                        'relative_nll_improvement': 0.0,
                        'selection_objective': 'fixed_physics_oracle',
                        'training_steps': 0,
                    }

                elif name == 'gbt_residual':
                    tb = tr.all_windows(a.window_length, a.lookahead_s, stride=a.eval_stride)
                    vb = va.all_windows(a.window_length, a.lookahead_s, stride=a.eval_stride)
                    Xtr, Ptr, Ctr, Rtr, tau, _ = extract_training_arrays([tb]); train_et = tb.edge_event_time_delta.numpy()
                    g = PhysicsOffsetGBT(GBTConfig(max_iter=a.gbt_max_iter, learning_rate=a.gbt_learning_rate,
                        max_leaf_nodes=a.gbt_max_leaf_nodes, min_samples_leaf=a.gbt_min_samples_leaf,
                        l2_regularization=a.gbt_l2, random_state=seed, residual_scale=1.0))
                    g.fit(Xtr, Ptr, Ctr, Rtr, tau, event_time=train_et)
                    best = (float('inf'), 0.0)
                    for s in [float(x) for x in a.gbt_scales.split(',') if x.strip()]:
                        g.config.residual_scale=s; vn=integrated_losses(va, gbt=g, a=a, q=a.quadrature_points)
                        if math.isfinite(vn) and vn < best[0]: best=(vn,s)
                    g.config.residual_scale=best[1]
                    test_nll=integrated_losses(te, gbt=g, a=a, q=a.quadrature_points)
                    _, gbt_records = integrated_losses(te, gbt=g, a=a, q=a.quadrature_points, return_records=True)
                    for r in gbt_records:
                        r['physics_nll'] = physics_by_key.get((r['mission_id'],r['edge_src'],r['edge_dst'],r['event_time_delta_s']))
                    path.with_name(path.stem + '_test_losses.json').write_text(json.dumps(gbt_records, indent=2))
                    rec={'model':name,'seed':seed,'integrated_test_nll':test_nll,'selected_residual_scale':best[1],
                         'selection_objective':'integrated_time_varying_nll','physics_nll':phys_nll,
                         'relative_nll_improvement':(phys_nll-test_nll)/phys_nll if phys_nll else None}
                else:
                    model=build_model(name,a,seed)
                    tc=TrainConfig(n_steps=a.n_steps,batch_size=a.batch_size,window_length=a.window_length,lookahead_s=a.lookahead_s,
                        learning_rate=a.learning_rate,mark_loss_weight=a.mark_loss_weight,log_every=0,seed=seed,
                        lr_schedule='cosine',val_every=a.val_every,val_stride=a.val_stride,restore_best_val=False)
                    if name == 'physics_soft_reg': train_physics_soft(model,tr,tc,a.physics_reg_weight)
                    else: train(model,tr,tc,val_cache=va)
                    vb=va.all_windows(a.window_length,a.lookahead_s,stride=a.val_stride); teb=te.all_windows(a.window_length,a.lookahead_s,stride=a.eval_stride)
                    vm=evaluate(model,vb,mark_loss_weight=a.mark_loss_weight); tm=evaluate(model,teb,mark_loss_weight=a.mark_loss_weight)
                    test_nll=integrated_losses(te,model=model,a=a,q=a.quadrature_points)
                    _, model_records = integrated_losses(te,model=model,a=a,q=a.quadrature_points,return_records=True)
                    for r in model_records:
                        r['physics_nll'] = physics_by_key.get((r['mission_id'],r['edge_src'],r['edge_dst'],r['event_time_delta_s']))
                    path.with_name(path.stem + '_test_losses.json').write_text(json.dumps(model_records, indent=2))
                    rec={'model':name,'seed':seed,'parameters':model.parameter_count(),'test_auc':finite(tm.event_auc),'test_auprc':finite(tm.event_auprc),
                         'test_brier':finite(tm.brier_score),'test_ece':finite(tm.calibration_error),'test_cindex':finite(tm.c_index),
                         'reference_test_nll':finite(tm.val_tpp_loss),'integrated_test_nll':test_nll,'val_auc':finite(vm.event_auc),
                         'physics_nll':phys_nll,'relative_nll_improvement':(phys_nll-test_nll)/phys_nll if phys_nll else None,
                         'selection_objective':'fixed_training_budget','training_steps':a.n_steps}
                path.write_text(json.dumps(rec,indent=2)); cond['models'][f'{name}_seed{seed}']=rec
                print(f'{name:18s} seed={seed}: integrated NLL={rec["integrated_test_nll"]:.6f}')
        summary['conditions'][f'strength_{strength:g}']=cond
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(f'\nSaved {out}/summary.json')

if __name__=='__main__': main()
