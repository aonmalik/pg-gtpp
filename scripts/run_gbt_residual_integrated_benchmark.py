"""Fair GBT residual benchmark with time-varying integrated physics likelihood.

This benchmark keeps the learned residual ratio constant over the forecast
window (it is computed from observed history), while integrating the nominal
physics cause hazards along the known trajectory. Validation scale selection
and test NLL therefore use the same time-varying compensator convention as the
nominal event generator.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np, torch
from simulator.mission_config import FleetConfig, MissionConfig, MissionEnvelopeConfig, default_config
from simulator.training.train import MissionCache
from simulator.model.gbt_residual import GBTConfig, PhysicsOffsetGBT, extract_training_arrays, edge_feature_matrix
from simulator.training.eval import compute_auc, compute_auprc, brier_score, compute_calibration_error, harrell_c_index
from simulator.dataset.window_builder import build_window

CAUSES=4

def make_config(args):
    base=default_config(master_seed=args.corpus_seed)
    fleet=FleetConfig(usv_count=1,n_auv_min=3,n_auv_max=8,m_iout_min=10,m_iout_max=30,max_auv_speed_mps=base.fleet.max_auv_speed_mps)
    env=MissionEnvelopeConfig(duration_s=args.mission_duration_s,node_sample_rate_hz=1.0,area_size_m=base.envelope.area_size_m,depth_range_m=base.envelope.depth_range_m,sea_state=base.envelope.sea_state)
    return MissionConfig(master_seed=args.corpus_seed,split='train',wave=base.wave,current=base.current,acoustic=base.acoustic,fleet=fleet,envelope=env,hazard=base.hazard,graph=base.graph)

def split_three(cache, seed):
    tr,hold=cache.split(.30,seed); va,te=hold.split(.50,seed+1); return tr,va,te

def finite(x):
    try:
        x=float(x); return x if math.isfinite(x) else None
    except Exception: return None

def ref_metrics(haz,batch):
    risk=batch.edge_at_risk.numpy()>0.5
    event=batch.edge_link_down_target.numpy()[risk]>0.5
    et=batch.edge_event_time_delta.numpy()[risk]
    causes=batch.edge_event_cause_id.numpy()[risk]
    lam=np.clip(haz[risk].sum(1),1e-8,None)
    p=1-np.exp(-lam*float(batch.lookahead_s))
    loss=et*lam
    obs=event & (causes>=0)
    if np.any(obs):
        idx=np.flatnonzero(obs); loss[idx]-=np.log(np.maximum(haz[risk][obs,causes[obs]],1e-12))
    return {'auc':finite(compute_auc(torch.from_numpy(p),torch.from_numpy(event.astype(float)))), 'auprc':finite(compute_auprc(torch.from_numpy(p),torch.from_numpy(event.astype(float)))), 'brier':finite(brier_score(torch.from_numpy(p),torch.from_numpy(event.astype(float)))), 'ece':finite(compute_calibration_error(torch.from_numpy(p),torch.from_numpy(event.astype(float)))), 'cindex':finite(harrell_c_index(torch.from_numpy(et),torch.from_numpy(event.astype(float)),torch.from_numpy(lam))), 'nll':float(np.mean(loss)), 'n_at_risk':int(risk.sum()), 'n_events':int(event.sum())}

def interp(mission, idx, t):
    ts=mission.timestamps; arr=mission.array
    idx=np.asarray(idx,dtype=np.int64); t=np.asarray(t,dtype=np.float64)
    sr=1.0/(ts[1]-ts[0]); k=np.clip(np.floor(t*sr).astype(np.int64),0,len(ts)-1)
    t0=ts[k]; dt=t-t0
    x0=arr[k,idx,0]; y0=arr[k,idx,1]; z0=arr[k,idx,2]; vx=arr[k,idx,3]; vy=arr[k,idx,4]
    return x0+vx*dt,y0+vy*dt,z0,vx,vy

def integrated_window(mission,sample,ratios,q):
    risk=sample.edge_at_risk.numpy()>0.5
    if not np.any(risk): return np.empty(0)
    ei=sample.edge_index.numpy()[:,risk]; et=sample.edge_event_time_delta.numpy()[risk].astype(float); causes=sample.edge_event_cause_id.numpy()[risk].astype(int)
    tref=float(sample.reference_time_s); u=np.linspace(0,1,q); tq=tref+et[:,None]*u[None,:]
    flat=tq.reshape(-1); ii=np.repeat(ei[0],q); jj=np.repeat(ei[1],q)
    xi,yi,zi,vxi,vyi=interp(mission,ii,flat); xj,yj,zj,vxj,vyj=interp(mission,jj,flat)
    d=np.sqrt((xi-xj)**2+(yi-yj)**2+(zi-zj)**2)
    hs=mission.evaluator.cause_hazards_batch(d,xi,yi,zi,xj,yj,zj,vxi,vyi,vxj,vyj,flat)
    ph=np.stack(hs,axis=1).reshape(len(ei[0]),q,CAUSES)
    # ratios are constant across forecast, preserving the causal/no-future constraint.
    hz=ph*np.asarray(ratios)[:,None,:]
    comp=np.trapezoid(hz,tq,axis=1).sum(axis=1)
    loss=comp.copy(); obs=causes>=0
    if np.any(obs):
        ev=np.flatnonzero(obs); tev=tref+et[ev]; ex,ey,ez,evx,evy=interp(mission,ei[0,ev],tev); jx,jy,jz,jvx,jvy=interp(mission,ei[1,ev],tev); dd=np.sqrt((ex-jx)**2+(ey-jy)**2+(ez-jz)**2)
        eh=np.stack(mission.evaluator.cause_hazards_batch(dd,ex,ey,ez,jx,jy,jz,evx,evy,jvx,jvy,tev),axis=1)*np.asarray(ratios)[ev]
        loss[ev]-=np.log(np.maximum(eh[np.arange(len(ev)),causes[ev]],1e-12))
    return loss

def integrated_split_nll(mcache, gbt, args, q):
    losses=[]
    for mission in mcache.missions:
        valid=mcache._valid_reference_indices(mission,args.lookahead_s)
        for tidx in valid[::args.eval_stride]:
            b=build_window(mission.mission_id,mission.array,mission.node_ids,mission.node_types,mission.timestamps,mission.events,mission.evaluator,mission.candidate_range_factor,reference_t_idx=tidx,window_length=args.window_length,lookahead_s=args.lookahead_s)
            risk=b.edge_at_risk.numpy()>0.5
            if not np.any(risk): continue
            X,_=edge_feature_matrix(b); ratios=gbt.predict_ratio(X)
            losses.extend(integrated_window(mission,b,ratios,q).tolist())
    return float(np.mean(losses)) if losses else float('nan')

def integrated_physics_nll(mcache,args,q):
    class Zero:
        def predict_ratio(self,X): return np.ones((len(X),4))
    return integrated_split_nll(mcache,Zero(),args,q)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--n-missions',type=int,default=50); p.add_argument('--corpus-seed',type=int,default=2026); p.add_argument('--seeds',default='42'); p.add_argument('--strengths',default='0,1,2,3,4,5'); p.add_argument('--mission-duration-s',type=float,default=900); p.add_argument('--window-length',type=int,default=10); p.add_argument('--lookahead-s',type=float,default=30); p.add_argument('--eval-stride',type=int,default=5); p.add_argument('--quadrature-points',type=int,default=33); p.add_argument('--output-dir',default='results/gbt_residual_integrated'); p.add_argument('--max-iter',type=int,default=200); p.add_argument('--learning-rate',type=float,default=.05); p.add_argument('--max-leaf-nodes',type=int,default=31); p.add_argument('--min-samples-leaf',type=int,default=30); p.add_argument('--l2',type=float,default=1.0); p.add_argument('--residual-scales',default='0,0.125,0.25,0.375,0.5,0.75,1.0,1.25,1.5,2.0'); p.add_argument('--residual-clip',type=float,default=2.0); p.add_argument('--mismatch-mode',choices=['cause_specific','total'],default='cause_specific')
    a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); cfg=make_config(a); seeds=[int(x) for x in a.seeds.split(',') if x.strip()]; strengths=[float(x) for x in a.strengths.split(',') if x.strip()]; scales=[float(x) for x in a.residual_scales.split(',') if x.strip()]
    summary={'protocol':vars(a),'conditions':{}}
    for strength in strengths:
        print('\n'+'='*72+f'\nstrength_{strength:g}\n'+'='*72); cache=MissionCache.generate(cfg,a.corpus_seed,a.n_missions,unmodeled_state_strength=strength,unmodeled_state_mode=a.mismatch_mode); tr,va,te=split_three(cache,2026)
        tb=tr.all_windows(a.window_length,a.lookahead_s,stride=a.eval_stride); vb=va.all_windows(a.window_length,a.lookahead_s,stride=a.eval_stride); teb=te.all_windows(a.window_length,a.lookahead_s,stride=a.eval_stride)
        Xtr,Ptr,Ctr,Rtr,tau,_=extract_training_arrays([tb]); Xva,Pva,Cva,Rva,_,_=extract_training_arrays([vb]); Xte,Pte,Cte,Rte,_,_=extract_training_arrays([teb]); train_et=tb.edge_event_time_delta.numpy()
        rec={'models':{}}
        for seed in seeds:
            g=PhysicsOffsetGBT(GBTConfig(max_iter=a.max_iter,learning_rate=a.learning_rate,max_leaf_nodes=a.max_leaf_nodes,min_samples_leaf=a.min_samples_leaf,l2_regularization=a.l2,random_state=seed,residual_clip=a.residual_clip,residual_scale=1.0)); print(f'  fitting GBT seed={seed}: {len(Xtr)} rows, {Rtr.sum()} at-risk'); g.fit(Xtr,Ptr,Ctr,Rtr,tau,event_time=train_et)
            best=None; scale_records=[]
            for scale in scales:
                g.config.residual_scale=scale
                vn=integrated_split_nll(va,g,a,a.quadrature_points); vr=integrated_physics_nll(va,a,a.quadrature_points)
                # secondary reference-time metrics
                hz=Pva*g.predict_ratio(Xva); vm=ref_metrics(hz,vb); row={'scale':scale,'integrated_val_nll':finite(vn),'reference_val_nll':vm['nll'],'val_auc':vm['auc'],'val_brier':vm['brier'],'val_ece':vm['ece']}; scale_records.append(row); print(f'    scale={scale:g} | integrated val NLL={row["integrated_val_nll"]} | ref NLL={row["reference_val_nll"]} AUC={row["val_auc"]} Brier={row["val_brier"]} ECE={row["val_ece"]}'); score=vn if math.isfinite(vn) else float('inf');
                if best is None or score<best[0]: best=(score,scale)
            g.config.residual_scale=best[1]; hz=Pte*g.predict_ratio(Xte); m=ref_metrics(hz,teb); m['selected_residual_scale']=best[1]; m['selection_objective']='integrated_time_varying_nll'; m['integrated_test_nll']=integrated_split_nll(te,g,a,a.quadrature_points); m['seed']=seed; m['scale_search']=scale_records; rec['models'].setdefault('pg_gbt',[]).append(m); print(f'    selected scale={best[1]:g} | integrated test NLL={m["integrated_test_nll"]} | ref test NLL={m["nll"]}')
        phys_ref=ref_metrics(Pte,teb); phys_int=integrated_physics_nll(te,a,a.quadrature_points); phys_ref['integrated_nll']=phys_int; rec['models']['physics_only']=phys_ref; print(f'  physics_only | integrated NLL={phys_int} | ref NLL={phys_ref["nll"]}')
        summary['conditions'][f'strength_{strength:g}']=rec
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(f'\nSaved {out/"summary.json"}')
if __name__=='__main__': main()
