"""Cause attribution reporting utilities."""
from __future__ import annotations
from collections import Counter

def cause_histogram(causes):
    return dict(Counter(causes))

def attribution_accuracy(predicted, truth, ambiguity=None, threshold=0.5):
    pairs=[]
    for i,(p,t) in enumerate(zip(predicted,truth)):
        if ambiguity is None or ambiguity[i] <= threshold:
            pairs.append(p==t)
    return sum(pairs)/len(pairs) if pairs else float("nan")
