"""
Diagnose whether PG-GTPP is genuinely using its learned correction/gate or
collapsing to the physics-only solution.

Runs a small, controlled experiment on the same MissionCache/split protocol,
then reports:
  * physics vs PG-GTPP intensity agreement
  * gate distribution
  * neural correction and weighted contributions
  * prediction correlation
  * gradient norms for correction/gate parameters

This is a diagnostic, not a paper benchmark.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch

from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.model.pg_gtpp_model import PGGTPPModel
from simulator.training.train import MissionCache, TrainConfig, train
from simulator.training.loss import compute_loss
from simulator.training.eval import evaluate

F_NODE=11

def seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_config(args):
    base=default_config(master_seed=args.corpus_seed)
    fleet=FleetConfig(usv_count=1,n_auv_min=3,n_auv_max=8,m_iout_min=10,m_iout_max=30,
                      max_auv_speed_mps=base.fleet.max_auv_speed_mps)
    envelope=MissionEnvelopeConfig(duration_s=args.mission_duration_s,node_sample_rate_hz=1.0,
                                   area_size_m=base.envelope.area_size_m,
                                   depth_range_m=base.envelope.depth_range_m,
                                   sea_state=base.envelope.sea_state)
    return MissionConfig(master_seed=args.corpus_seed,split="train",wave=base.wave,
                         current=base.current,acoustic=base.acoustic,fleet=fleet,
                         envelope=envelope,hazard=base.hazard,graph=base.graph)

def train_model(model, cache, args, seed):
    cfg=TrainConfig(n_steps=args.n_steps,batch_size=args.batch_size,
                    window_length=args.window_length,lookahead_s=args.lookahead_s,
                    learning_rate=args.learning_rate,mark_loss_weight=args.mark_loss_weight,
                    log_every=args.log_every,checkpoint_every=None,seed=seed,
                    lr_schedule=args.lr_schedule)
    return train(model, cache, cfg)

def collect(model, batch):
    model.eval()
    with torch.no_grad():
        out=model(batch.node_features,batch.node_type_ids,batch.node_mask,
                  batch.edge_index,batch.edge_physics)
        total=out.total_intensity().detach()
        phys=batch.edge_physics.sum(-1).detach()
        if hasattr(out,"fusion"):
            gates=torch.stack([out.fusion[k]["gate"] for k in ("geo","wave","doppler","current")],dim=1)
            corr=torch.stack([out.fusion[k]["correction"] for k in ("geo","wave","doppler","current")],dim=1)
            pg=torch.stack([out.fusion[k]["fused"] for k in ("geo","wave","doppler","current")],dim=1)
            # PG-GTPP is multiplicative, not a convex blend.  Therefore
            # ``gate * physics`` is NOT the physical contribution.  The exact
            # decomposition is the fixed physics baseline plus the signed
            # learned intensity correction.
            phys_contrib=batch.edge_physics.sum(-1)
            neural_contrib=pg.sum(-1) - phys_contrib
        else:
            gates=corr=pg=phys_contrib=neural_contrib=None
    return dict(total=total,phys=phys,gates=gates,corr=corr,pg_causes=pg,
                phys_contrib=phys_contrib,neural_contrib=neural_contrib)

def stats(x):
    x=x.float().cpu().numpy().ravel()
    return f"mean={x.mean():.6g} std={x.std():.6g} min={x.min():.6g} max={x.max():.6g}"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--n-missions",type=int,default=20)
    p.add_argument("--corpus-seed",type=int,default=2026)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--mission-duration-s",type=float,default=900.)
    p.add_argument("--n-steps",type=int,default=500)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--window-length",type=int,default=10)
    p.add_argument("--lookahead-s",type=float,default=30.)
    p.add_argument("--learning-rate",type=float,default=3e-3)
    p.add_argument("--mark-loss-weight",type=float,default=1.)
    p.add_argument("--hidden-dim",type=int,default=64)
    p.add_argument("--log-every",type=int,default=200)
    p.add_argument("--eval-stride",type=int,default=5)
    p.add_argument("--lr-schedule",choices=["none","cosine"],default="cosine")
    p.add_argument("--allow-nondeterministic",action="store_true")
    args=p.parse_args()
    if not args.allow_nondeterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True)
    seed_everything(args.seed)
    cache=MissionCache.generate(make_config(args),args.corpus_seed,args.n_missions)
    train_cache, holdout_cache = cache.split(0.30, 0)
    val_cache, test_cache = holdout_cache.split(0.50, 1)
    print(
        f"Corpus: {args.n_missions} missions; "
        f"train={len(train_cache.missions)} "
        f"val={len(val_cache.missions)} "
        f"test={len(test_cache.missions)}"
    )

    print(
        "Protocol: exact fair-benchmark mission split = 70/15/15; "
        "validation is not used for test diagnostics."
    )
    models={}
    for label,flags in [
        ("physics_only",dict(learned_correction=False,use_physics_intensity=True,use_fusion_gate=False)),
        ("pg_gtpp",dict(learned_correction=True,use_physics_intensity=True,use_fusion_gate=True)),
    ]:
        print(f"\nTraining {label}...")
        seed_everything(args.seed)
        m=PGGTPPModel(f_node=F_NODE,hidden_dim=args.hidden_dim,**flags)
        train_model(m,train_cache,args,args.seed)
        models[label]=m
        print(f"parameters={m.parameter_count():,}")

    val_batch = val_cache.all_windows(
        args.window_length, args.lookahead_s, stride=args.eval_stride
    )
    batch = test_cache.all_windows(
        args.window_length, args.lookahead_s, stride=args.eval_stride
    )

    print("\n=== VALIDATION METRICS ===")
    for label, model in models.items():
        metrics = evaluate(model, val_batch, mark_loss_weight=args.mark_loss_weight)
        print(
            f"{label}: AUROC={metrics.event_auc:.4f} "
            f"AUPRC={metrics.event_auprc:.4f} "
            f"Brier={metrics.brier_score:.5f} "
            f"IBS={metrics.integrated_brier_score:.5f} "
            f"C={metrics.c_index:.4f}"
        )

    p=collect(models["physics_only"],batch)
    g=collect(models["pg_gtpp"],batch)
    delta=g["total"]-p["total"]
    absd=delta.abs()
    corr=np.corrcoef(g["total"].cpu().numpy(),p["total"].cpu().numpy())[0,1]
    print("\n=== TEST INTENSITY COMPARISON ===")
    print("physics-only:",stats(p["total"]))
    print("PG-GTPP:     ",stats(g["total"]))
    print("difference:  ",stats(delta))
    print(f"MAE={absd.mean().item():.8g} RMSE={torch.sqrt((delta**2).mean()).item():.8g} maxAE={absd.max().item():.8g}")
    print(f"Pearson correlation={corr:.8f}")

    print("\n=== PG-GTPP FUSION ===")
    print("gate overall:",stats(g["gates"]))
    print("correction overall:",stats(g["corr"]))
    print("physics contribution:",stats(g["phys_contrib"]))
    print("neural contribution:",stats(g["neural_contrib"]))
    print(f"gate > 0.90: {(g['gates']>0.90).float().mean().item()*100:.2f}%")
    print(f"gate < 0.10: {(g['gates']<0.10).float().mean().item()*100:.2f}%")
    print(f"mean signed neural correction / total: {(g['neural_contrib'].mean()/g['total'].mean()).item():.6f}")
    print(f"mean absolute neural correction / total: {(g['neural_contrib'].abs().mean()/g['total'].mean()).item():.6f}")

    # Gradient diagnostic on one fresh train batch.
    m=models["pg_gtpp"]; m.train()
    rng=np.random.default_rng(args.seed+123)
    tb=train_cache.sample_batch(rng,args.batch_size,args.window_length,args.lookahead_s)
    m.zero_grad(set_to_none=True)
    out=m(tb.node_features,tb.node_type_ids,tb.node_mask,tb.edge_index,tb.edge_physics)
    loss=compute_loss(out,tb).total_loss
    loss.backward()
    gate_grads=[]; corr_grads=[]
    for n,p0 in m.named_parameters():
        if p0.grad is None: continue
        v=p0.grad.detach().norm().item()
        if "gate_mlp" in n: gate_grads.append(v)
        if "correction_mlp" in n: corr_grads.append(v)
    print("\n=== GRADIENTS (fresh batch) ===")
    print(f"loss={loss.item():.6g}")
    print(f"gate MLP grad norms:       mean={np.mean(gate_grads):.6g} max={np.max(gate_grads):.6g}")
    print(f"correction MLP grad norms: mean={np.mean(corr_grads):.6g} max={np.max(corr_grads):.6g}")

    print("\n=== REFERENCE ===")
    print("If PG-GTPP has non-trivial gate/correction gradients and a measurable")
    print("PG-vs-physics intensity difference, the fusion is active. If gate")
    print("is ~1 and neural contribution ~0, PG-GTPP has learned to trust physics.")
    print("If gradients are ~0 or the PG output is exactly physics despite")
    print("non-extreme gates, inspect the implementation/training objective.")

if __name__=="__main__":
    main()
