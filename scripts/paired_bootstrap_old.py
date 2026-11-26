#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

def load(p): return json.loads(Path(p).read_text())
def mission_means(records):
    d={}
    for r in records: d.setdefault(str(r['mission_id']),[]).append(float(r['nll']))
    return {k:float(np.mean(v)) for k,v in d.items()}
def delta(a,b):
    A,B=mission_means(a),mission_means(b); keys=sorted(set(A)&set(B))
    if not keys: raise ValueError('No common missions')
    return np.array([A[k]-B[k] for k in keys])
def boot(d,n,rng):
    idx=rng.integers(0,len(d),size=(n,len(d)))
    x=d[idx].mean(axis=1)
    return float(d.mean()),*[float(v) for v in np.quantile(x,[.025,.975])]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='results/main_fair_comparison'); ap.add_argument('--strengths',default='0,5'); ap.add_argument('--seeds',default='42,43,44'); ap.add_argument('--n-boot',type=int,default=10000); ap.add_argument('--seed',type=int,default=2026); a=ap.parse_args()
    root=Path(a.root); rng=np.random.default_rng(a.seed); out={'protocol':vars(a),'conditions':{}}
    for st in [x.strip() for x in a.strengths.split(',') if x.strip()]:
        sd={}; base=root/f'strength_{st}'
        pooled_ph=[]; pooled_pg=[]
        for seed in [x.strip() for x in a.seeds.split(',') if x.strip()]:
            pgp=base/f'pg_gtpp_seed{seed}_test_losses.json'; php=base/f'physics_only_seed{seed}_test_losses.json'; gbp=base/f'gbt_residual_seed{seed}_test_losses.json'
            if not (pgp.exists() and php.exists()): continue
            pg,ph=load(pgp),load(php); d=delta(pg,ph); pooled_ph.append(d); m,lo,hi=boot(d,a.n_boot,rng); sd[f'seed_{seed}']={'n_missions':len(d),'pg_minus_physics':m,'ci95':[lo,hi]}
            if gbp.exists():
                d2=delta(pg,load(gbp)); pooled_pg.append(d2); m2,lo2,hi2=boot(d2,a.n_boot,rng); sd[f'seed_{seed}']['pg_minus_gbt']=m2; sd[f'seed_{seed}']['pg_minus_gbt_ci95']=[lo2,hi2]
        if pooled_ph:
            d=np.concatenate(pooled_ph); m,lo,hi=boot(d,a.n_boot,rng); sd['pooled_pg_minus_physics']={'n_mission_seed_pairs':len(d),'delta_nll':m,'ci95':[lo,hi]}
        if pooled_pg:
            d=np.concatenate(pooled_pg); m,lo,hi=boot(d,a.n_boot,rng); sd['pooled_pg_minus_gbt']={'n_mission_seed_pairs':len(d),'delta_nll':m,'ci95':[lo,hi]}
        out['conditions'][f'strength_{st}']=sd
    p=root/'paired_bootstrap.json'; p.write_text(json.dumps(out,indent=2)); print(f'Saved {p}')
if __name__=='__main__': main()
