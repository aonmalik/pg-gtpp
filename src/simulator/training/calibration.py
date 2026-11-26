"""Calibration utilities for PG-GTPP evaluation."""
from __future__ import annotations
import torch

def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, bins: int=10) -> float:
    p=probabilities.detach().flatten().clamp(0,1); y=labels.detach().flatten().float()
    if p.numel()==0: return float("nan")
    edges=torch.linspace(0,1,bins+1); err=0.0; total=0
    for i in range(bins):
        m=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        n=int(m.sum())
        if n:
            err += n*abs(float(p[m].mean())-float(y[m].mean())); total+=n
    return err/total if total else float("nan")
