import numpy as np
import torch

from simulator.unmodeled_state import (
    CAUSE_OPERATIONAL_SENSITIVITY,
    cause_specific_truth_multiplier,
)
from simulator.model.pg_gtpp_model import PGGTPPModel


def test_cause_specific_multiplier_changes_relative_causes():
    components = np.array([[1.0, 0.0, 0.0]], dtype=float)
    m = cause_specific_truth_multiplier(components, 2.0)
    assert m.shape == (1, 4)
    assert np.all(m > 1.0)
    assert not np.allclose(m[:, 0], m[:, 2])
    assert np.allclose(CAUSE_OPERATIONAL_SENSITIVITY.sum(axis=1), 1.0)


def test_temporal_pg_forward_shapes():
    torch.manual_seed(0)
    model = PGGTPPModel(f_node=11, hidden_dim=16, n_heads=4, use_operational_temporal=True)
    b, k, v, e = 2, 5, 4, 4
    node_features = torch.randn(b, k, v, 11)
    node_features[..., 6] = torch.rand(b, k, v)
    node_features[..., 7] = torch.rand(b, k, v)
    node_types = torch.zeros(b, v, dtype=torch.long)
    node_mask = torch.ones(b, v)
    edge_index = torch.tensor([[0, 1, 4, 5], [1, 2, 5, 6]], dtype=torch.long)
    edge_physics = torch.rand(e, 4) * 0.05
    out = model(node_features, node_types, node_mask, edge_index, edge_physics)
    assert out.total_intensity().shape == (e,)
    assert out.marks.pdr_alpha.shape == (e,)
