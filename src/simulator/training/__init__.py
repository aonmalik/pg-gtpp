"""
PG-GTPP training pipeline (Phase 6).

Implements the loss function (piecewise-constant TPP NLL + mark NLL) and
the training loop (dataset iteration across missions, optimizer step,
checkpointing) that trains a PGGTPPModel against a generated corpus.

No new architecture lives here -- model/ already defines the forward pass
this package trains.
"""

from __future__ import annotations
