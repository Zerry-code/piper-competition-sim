import numpy as np
from piper_sim.joint_flexibility import configuration_quality

def test_soft_score_prefers_joint_margin_and_manipulability():
    low=np.full(6,-2.);high=np.full(6,2.);seed=np.zeros(6)
    middle=configuration_quality(seed,seed,low,high,np.eye(6))
    limit=configuration_quality(np.full(6,1.99),seed,low,high,np.eye(6))
    singular=configuration_quality(seed,seed,low,high,np.diag([1,1,1,1,1,1e-7]))
    assert middle['cost']<limit['cost'] and middle['cost']<singular['cost']

def test_all_joint_motion_is_allowed_and_scores_remain_finite():
    result=configuration_quality(np.ones(6),np.zeros(6),np.full(6,-2.),np.full(6,2.),np.eye(6))
    assert np.isfinite(result['cost']) and len(result['joint_margins'])==6


def test_default_ik_keeps_original_wrist_limits(monkeypatch):
    from types import SimpleNamespace as NS
    import piper_sim.traditional as module
    monkeypatch.setattr(module.mujoco,'MjData',lambda model:None)
    backend=NS(model=NS(site=lambda n:NS(id=0)),interface={'joints':{'lower_rad':[-2.6,0.,-2.9,-1.7,-1.22,-2.0944],'upper_rad':[2.6,3.14,0.,1.7,1.22,2.0944]}})
    ik=module.IK(backend)
    assert ik.lower[5]<-np.pi/2 and ik.upper[5]>np.pi/2
