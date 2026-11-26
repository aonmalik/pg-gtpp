import numpy as np

from simulator.unmodeled_state import operational_stress_batch, truth_hazard_multiplier


def test_operational_stress_is_bounded_and_finite():
    x = np.array([0.2, 0.8])
    s = operational_stress_batch(x, x, x, x, x, x, x, x, x, x)
    assert np.all(np.isfinite(s))
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_truth_multiplier_is_identity_at_zero_strength():
    stress = np.array([0.0, 0.25, 1.0])
    assert np.allclose(truth_hazard_multiplier(stress, 0.0), 1.0)


def test_truth_multiplier_increases_with_strength_for_positive_stress():
    stress = np.array([0.2, 0.8])
    m1 = truth_hazard_multiplier(stress, 1.0)
    m2 = truth_hazard_multiplier(stress, 2.0)
    assert np.all(m2 > m1)


def test_nominal_event_generator_preserves_physics_hazard_at_zero_strength():
    """Regression test for the strength=0 simulator path.

    At zero mismatch the truth hazard must be exactly the nominal physics
    hazard, so a normal seeded mission must still produce communication
    events. This catches accidental nesting of the nominal hazard assignment
    inside the mismatch branch.
    """
    import numpy as np
    from dataclasses import replace

    from simulator.event_generator import EventGenerator
    from simulator.mission_generator import MissionGenerator
    from simulator.mission_config import default_config
    from simulator.physics import AcousticChannel, CurrentField, WaveField
    from simulator.random_manager import RandomManager
    from simulator.trajectory_generator import TrajectoryGenerator

    config = default_config(master_seed=2026)
    config = replace(config, envelope=replace(config.envelope, duration_s=120.0))
    rm = RandomManager(master_seed=2026)
    mission_id = "regression_nominal_strength_zero"

    state = MissionGenerator(config, rm).generate(mission_id)
    array, node_ids = TrajectoryGenerator(config, rm).propagate(state)
    wave = WaveField.sample(config.wave, rm.get_numpy_rng(mission_id, "wave"))
    current = CurrentField.sample(config.current, rm.get_numpy_rng(mission_id, "current"))
    channel = AcousticChannel(config.acoustic)
    generator = EventGenerator(config, rm, channel, wave, current)
    node_types = {node.node_id: node.node_type for node in state.nodes}
    timestamps = np.arange(array.shape[0], dtype=np.float64)

    events_a = generator.generate(
        mission_id, array, node_ids, node_types, timestamps,
        unmodeled_state_strength=0.0, unmodeled_state_mode="cause_specific",
    )
    events_b = generator.generate(
        mission_id, array, node_ids, node_types, timestamps,
        unmodeled_state_strength=0.0, unmodeled_state_mode="cause_specific",
    )

    assert events_a, "strength=0 must not zero the nominal physics hazard"
    assert [e.event_id for e in events_a] == [e.event_id for e in events_b]
