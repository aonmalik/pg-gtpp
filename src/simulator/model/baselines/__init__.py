"""
model/baselines/__init__.py

Seven baseline models, one per representative prior-work family identified
in the Phase 2 literature-gap table (brutal-review document), plus the
classical statistical control. Each is a drop-in replacement for
``PGGTPPModel`` against the exact same training loop, loss, and evaluation
code (``training/train.py``, ``training/loss.py``, ``training/eval.py``) --
see ``common.py`` for why that is possible without touching any of those
modules.

    name              literature-gap row                          graph?  physics?  temporal mechanism
    ----------------  -------------------------------------------  ------  --------  ---------------------------
    cox_perlink       domain lit. (classical, non-neural)           no      yes*      none (static covariates)
    rnn_perlink       domain lit. (shallow RNN, single-link)         no      no        per-node GRU
    gnn_generic       generic GNN / TGAT-TGN's "no grounding" half   yes     no        GRU-pooled window
    tgat_style        TGAT / TGN                                    yes     no        Time2Vec + attention
    dysat_style       DySAT / Dynamic Graph Transformer              yes     no        per-snapshot + temporal attn
    neural_hawkes     Neural TPP / Neural Hawkes                     no      no        per-edge LSTM
    physics_soft_reg  Physics-informed GNNs (general, soft loss)     yes     yes**     GRU-pooled window

    *  linear in the physics covariates, not the closed-form physics function.
    ** physics enters via a loss-level regularizer, not the model's forward pass.

Every entry's ``model_cls`` accepts ``(f_node: int, hidden_dim: int = ...,
**kwargs)`` and returns a ``BaselineOutput`` (see ``common.py``) from
``forward(node_features, node_type_ids, node_mask, edge_index,
edge_physics)``.
"""
from __future__ import annotations

from .common import BaselineOutput
from .cox_perlink import CoxPerLinkModel
from .rnn_perlink import ShallowRNNPerLinkModel
from .gnn_generic import GenericGNNModel
from .tgat_style import TGATStyleModel
from .dysat_style import DySATStyleModel
from .neural_hawkes import NeuralHawkesPerEdgeModel
from .physics_soft_reg import PhysicsSoftRegGNNModel, physics_regularized_loss

BASELINE_REGISTRY = {
    "cox_perlink": CoxPerLinkModel,
    "rnn_perlink": ShallowRNNPerLinkModel,
    "gnn_generic": GenericGNNModel,
    "tgat_style": TGATStyleModel,
    "dysat_style": DySATStyleModel,
    "neural_hawkes": NeuralHawkesPerEdgeModel,
    "physics_soft_reg": PhysicsSoftRegGNNModel,
}

__all__ = [
    "BaselineOutput",
    "BASELINE_REGISTRY",
    "CoxPerLinkModel",
    "ShallowRNNPerLinkModel",
    "GenericGNNModel",
    "TGATStyleModel",
    "DySATStyleModel",
    "NeuralHawkesPerEdgeModel",
    "PhysicsSoftRegGNNModel",
    "physics_regularized_loss",
]
