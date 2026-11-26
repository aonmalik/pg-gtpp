"""Benchmark a non-neural physics-offset gradient-boosted residual learner.

Primary model:
    lambda_c = lambda_phys,c * exp(scale * clip(log r_c, -clip, clip))
where r_c is learned by a cause-specific Poisson GBT with lambda_phys as an
offset. The physics-only model is recovered exactly at scale=0.
"""
from __future__ import annotations

import argparse, json, math, os
from pathlib import Path
import numpy as np
import torch

from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.training.train import MissionCache
from simulator.model.gbt_residual import (GBTConfig, PhysicsOffsetGBT, extract_training_arrays, edge_feature_matrix, continuous_time_competing_risk_nll)
from simulator.training.eval import compute_auc, compute_auprc, brier_score, compute_calibration_error, harrell_c_index

F_NODE = 11

def seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_config(args):
    base=default_config(master_seed=args.corpus_seed)
    fleet=FleetConfig(usv_count=1,n_auv_min=3,n_auv_max=8,m_iout_min=10,m_iout_max=30,max_auv_speed_mps=base.fleet.max_auv_speed_mps)
    env=MissionEnvelopeConfig(duration_s=args.mission_duration_s,node_sample_rate_hz=1.0,area_size_m=base.envelope.area_size_m,depth_range_m=base.envelope.depth_range_m,sea_state=base.envelope.sea_state)
    return MissionConfig(master_seed=args.corpus_seed,split='train',wave=base.wave,current=base.current,acoustic=base.acoustic,fleet=fleet,envelope=env,hazard=base.hazard,graph=base.graph)

def split_three(cache, seed):
    tr, hold=cache.split(.30,seed); va, te=hold.split(.50,seed+1); return tr,va,te

def metrics(haz, batch):
    risk=batch.edge_at_risk.numpy()>0.5
    event=batch.edge_link_down_target.numpy()[risk]>0.5
    et=batch.edge_event_time_delta.numpy()[risk]
    causes=batch.edge_event_cause_id.numpy()[risk]
    ph=batch.edge_physics.numpy()[risk]
    lam=np.clip(haz[risk].sum(1),1e-8,None)
    tau=float(batch.lookahead_s)
    p=1-np.exp(-lam*tau)
    # Direct continuous-time NLL of the supplied hazard.  This is used for
    # validation model selection and reported alongside classification metrics.
    nll_loss = et*lam
    observed = event & (causes >= 0)
    if np.any(observed):
      idx=np.flatnonzero(observed)
      nll_loss[idx] -= np.log(np.maximum(haz[risk][observed, causes[observed]], 1e-12))
    nll=float(np.mean(nll_loss)) if len(nll_loss) else None
    return {
      'auc': finite(compute_auc(torch.from_numpy(p),torch.from_numpy(event.astype(float)))),
      'auprc': finite(compute_auprc(torch.from_numpy(p),torch.from_numpy(event.astype(float)))),
      'brier': finite(brier_score(torch.from_numpy(p),torch.from_numpy(event.astype(float)))),
      'ece': finite(compute_calibration_error(torch.from_numpy(p),torch.from_numpy(event.astype(float)))),
      'cindex': finite(harrell_c_index(torch.from_numpy(et),torch.from_numpy(event.astype(float)),torch.from_numpy(lam))),
      'nll': nll,
      'n_at_risk': int(risk.sum()), 'n_events': int(event.sum())
    }

def finite(x):
    try:
      x=float(x); return x if math.isfinite(x) else None
    except: return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n-missions',type=int,default=50); p.add_argument('--corpus-seed',type=int,default=2026)
    p.add_argument('--seeds',default='7,21,42'); p.add_argument('--strengths',default='0,1,2,3')
    p.add_argument('--mission-duration-s',type=float,default=900); p.add_argument('--window-length',type=int,default=10); p.add_argument('--lookahead-s',type=float,default=30)
    p.add_argument('--eval-stride',type=int,default=5); p.add_argument('--output-dir',default='results/gbt_residual')
    p.add_argument('--max-iter',type=int,default=250); p.add_argument('--learning-rate',type=float,default=.05); p.add_argument('--max-leaf-nodes',type=int,default=31); p.add_argument('--min-samples-leaf',type=int,default=30); p.add_argument('--l2',type=float,default=1.0)
    p.add_argument('--residual-scales',default='0,0.25,0.5,0.75,1.0'); p.add_argument('--residual-clip',type=float,default=2.0)
    p.add_argument('--mismatch-mode',choices=['cause_specific','total'],default='cause_specific')
    args=p.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    config=make_config(args); seeds=[int(x) for x in args.seeds.split(',') if x.strip()]; strengths=[float(x) for x in args.strengths.split(',') if x.strip()]
    scales=[float(x) for x in args.residual_scales.split(',') if x.strip()]
    summary={'protocol':vars(args),'conditions':{}}
    for strength in strengths:
      print('\n'+'='*72+f'\nstrength_{strength:g}\n'+'='*72)
      cache=MissionCache.generate(config,args.corpus_seed,args.n_missions,unmodeled_state_strength=strength,unmodeled_state_mode=args.mismatch_mode)
      tr,va,te=split_three(cache,2026)
      train_batch=tr.all_windows(args.window_length,args.lookahead_s,stride=args.eval_stride)
      val_batch=va.all_windows(args.window_length,args.lookahead_s,stride=args.eval_stride)
      test_batch=te.all_windows(args.window_length,args.lookahead_s,stride=args.eval_stride)
      Xtr,Ptr,Ctr,Rtr,tau,names=extract_training_arrays([train_batch])
      Xva, Pva, Cva, Rva, _, _=extract_training_arrays([val_batch])
      Xte, Pte, Cte, Rte, _, _=extract_training_arrays([test_batch])
      summary['conditions'][f'strength_{strength:g}']={'models':{},'n_train_edges':len(Xtr),'n_train_at_risk':int(Rtr.sum()),'n_train_events':int((Ctr>=0).sum())}
      for seed in seeds:
        seed_everything(seed)
        gbt=PhysicsOffsetGBT(GBTConfig(max_iter=args.max_iter,learning_rate=args.learning_rate,max_leaf_nodes=args.max_leaf_nodes,min_samples_leaf=args.min_samples_leaf,l2_regularization=args.l2,random_state=seed,residual_clip=args.residual_clip,residual_scale=1.0))
        print(f'  fitting GBT seed={seed}: {len(Xtr)} rows, {Rtr.sum()} at-risk')
        gbt.fit(Xtr,Ptr,Ctr,Rtr,tau)
        # Tune only residual shrinkage on validation.  The primary selection
        # objective is the continuous-time competing-risk NLL, which is the
        # actual likelihood of the piecewise-constant hazard model.  Brier is
        # retained as a secondary diagnostic but never touches the test set.
        ratio_va=gbt.predict_ratio(Xva)
        log_ratio_va=np.log(np.clip(ratio_va,1e-12,None))
        # Cva stores cause IDs; event-time delta is taken directly from the
        # validation WindowBatch so observed failures use t_event and censored
        # samples use the full lookahead.
        val_event_time=val_batch.edge_event_time_delta.numpy()
        val_event_cause=Cva
        best=None
        scale_records=[]
        for scale in scales:
          val_nll=continuous_time_competing_risk_nll(
              physics_hazard=Pva,
              residual_log_ratio=log_ratio_va,
              residual_scale=scale,
              event_time=val_event_time,
              event_cause=val_event_cause,
              at_risk=Rva,
          )
          hz=Pva*np.exp(np.clip(log_ratio_va*scale,-args.residual_clip,args.residual_clip))
          vm=metrics(hz,val_batch)
          record={'scale':float(scale),'val_nll':finite(val_nll),'val_brier':vm['brier'],'val_auc':vm['auc'],'val_auprc':vm['auprc'],'val_ece':vm['ece']}
          scale_records.append(record)
          score=val_nll if math.isfinite(val_nll) else float('inf')
          if best is None or score<best[0]: best=(score,scale,vm,record)
          print(f"    scale={scale:g} | val NLL={record['val_nll']} AUC={record['val_auc']} Brier={record['val_brier']} ECE={record['val_ece']}")
        scale=best[1]; gbt.config.residual_scale=scale
        hz_te=gbt.predict_hazard(Xte,Pte)
        m=metrics(hz_te,test_batch)
        m['selected_residual_scale']=scale
        m['selection_objective']='continuous_time_nll'
        m['val_nll']=best[3]['val_nll']
        m['val_brier']=best[3]['val_brier']
        m['val_auc']=best[3]['val_auc']
        m['val_auprc']=best[3]['val_auprc']
        m['val_ece']=best[3]['val_ece']
        m['scale_search']=scale_records
        m['seed']=seed
        summary['conditions'][f'strength_{strength:g}']['models'].setdefault('pg_gbt',[]).append(m)
        print(f"    selected scale={scale:g} | val NLL={best[3]['val_nll']} | test AUC={m['auc']} AUPRC={m['auprc']} Brier={m['brier']} ECE={m['ece']} NLL={m['nll']}")
      # analytical physics control on the same test batch
      phys=metrics(Pte,test_batch)
      summary['conditions'][f'strength_{strength:g}']['models']['physics_only']=phys
      print(f"  physics_only | AUC={phys['auc']} AUPRC={phys['auprc']} Brier={phys['brier']} ECE={phys['ece']} NLL={phys['nll']}")
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(f'\nSaved {out/"summary.json"}')

if __name__=='__main__': main()
