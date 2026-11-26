import numpy as np
from simulator.model.gbt_residual import PhysicsOffsetGBT

def test_gbt_zero_scale_is_exact_physics():
    rng=np.random.default_rng(3)
    X=rng.normal(size=(300,8)); p=np.abs(rng.normal(size=(300,4)))+.01
    c=rng.integers(-1,4,size=300); risk=np.ones(300,dtype=bool)
    m=PhysicsOffsetGBT().fit(X,p,c,risk,30.0)
    m.config.residual_scale=0.0
    got=m.predict_hazard(X,p)
    assert np.allclose(got,p,rtol=1e-12,atol=1e-12)

def test_gbt_prediction_positive_and_finite():
    rng=np.random.default_rng(4)
    X=rng.normal(size=(250,5)); p=np.abs(rng.normal(size=(250,4)))+.01
    c=rng.integers(-1,4,size=250); risk=np.ones(250,dtype=bool)
    m=PhysicsOffsetGBT().fit(X,p,c,risk,30.0)
    got=m.predict_hazard(X,p)
    assert np.isfinite(got).all()
    assert (got>0).all()
