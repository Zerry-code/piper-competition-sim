from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_sim.route_search import ready_position_route
from piper_sim.demo_skill import PickPlaceSkill


def test_ready_waypoint_is_required_and_validated():
    ready=np.ones(6)*.4;checked=[]
    points=ready_position_route(np.zeros(6),np.ones(6),[ready],lambda p:checked.append(p))
    assert any(np.allclose(q,ready) for q in points)
    assert checked and np.allclose(points[-1],1.)


def test_no_direct_bypass_when_ready_routes_unsafe():
    def reject(points):raise RuntimeError('collision')
    with pytest.raises(RuntimeError,match='collision'):
        ready_position_route(np.zeros(6),np.ones(6),[np.ones(6)*.4],reject)


def test_only_two_direct_trials_then_short_joint_route(monkeypatch):
    import piper_sim.route_search as routes
    calls=[]
    skill=PickPlaceSkill.__new__(PickPlaceSkill)
    skill.cfg={'ready_transport':True};skill.table=0.;skill.direct_transfer_attempts=0
    skill.b=NS(model=NS(site=lambda name:NS(id=0)))
    import mujoco
    monkeypatch.setattr(mujoco,'MjData',lambda model:NS())
    skill.ik=NS(solve=lambda *args:np.zeros(6))
    def direct(*args,**kwargs):calls.append('direct');raise RuntimeError('blocked')
    skill.cartesian_path=direct
    monkeypatch.setattr(routes,'IK',None,raising=False)
    import piper_sim.demo_skill as module
    monkeypatch.setattr(module,'joint_space_route',lambda start,goal,mids,validate,progress:[start,goal])
    monkeypatch.setattr(module,'IK',lambda *a,**k:NS(solve=lambda *args:np.zeros(6)))
    monkeypatch.setattr(routes,'ready_position_route',lambda start,goal,candidates,validate,progress:[start,goal])
    for _ in range(2):skill.ready_transport_path(np.zeros(6),np.array([.3,0,.3]),np.array([.5,0,.3]),0.)
    assert calls==['direct','direct'] and skill.direct_transfer_attempts==2
