"""Utilities for competing-risk PG-GTPP reporting."""
from __future__ import annotations
from collections import Counter


def cause_report(true_causes, predicted_causes, ambiguity=None):
    total=Counter()
    correct=Counter()
    for t,p in zip(true_causes,predicted_causes):
        total[str(t)] += 1
        if t==p: correct[str(t)] += 1
    out={k:{"samples":v,"accuracy":correct[k]/v if v else 0.0} for k,v in total.items()}
    if ambiguity is not None:
        out["ambiguity"]={"mean":sum(ambiguity)/max(1,len(ambiguity)),"fraction_low_confidence":sum(a<2 for a in ambiguity)/max(1,len(ambiguity))}
    return out
