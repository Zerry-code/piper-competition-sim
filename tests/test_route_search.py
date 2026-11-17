from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_sim.route_search import transfer_route,ordered_candidates

def segment(start,goal,yaw,current):
    if current[2]<.23 and goal[2]<.23 and np.linalg.norm(goal[:2]-current[:2])>.1:raise RuntimeError('direct branch unavailable')
    return [start,np.r_[goal,np.zeros(3)]]

def test_direct_failure_uses_validated_raised_bridge():
    checked=[]
    points,via=transfer_route(segment,np.zeros(6),np.array([.3,0,.2]),np.array([.5,.1,.2]),0,lambda p:checked.append(p))
    assert via and max(p[2] for p in via)>=.24
    assert len(checked)==1
    np.testing.assert_allclose(points[-1][:3],[.5,.1,.2])

def test_direct_is_preferred():
    points,via=transfer_route(segment,np.zeros(6),np.array([.3,0,.24]),np.array([.5,.1,.24]),0)
    assert not via

def test_invalid_routes_never_return_a_motion():
    def reject(points):raise RuntimeError('collision')
    with pytest.raises(RuntimeError,match='collision'):
        transfer_route(segment,np.zeros(6),np.array([.3,0,.2]),np.array([.5,.1,.2]),0,reject)

def test_quality_first_and_seeded_random_ties():
    candidates=[NS(center=np.array([.4,0,.1]),candidate_score=s,key=i) for i,s in enumerate([1.,1.,.4])]
    orders=[]
    for seed in range(8):
        order=ordered_candidates(candidates,np.zeros(3),np.random.default_rng(seed))
        assert order[-1].key==2
        orders.append(tuple(c.key for c in order))
    assert len(set(orders))>1
    a=ordered_candidates(candidates,np.zeros(3),np.random.default_rng(2))
    b=ordered_candidates(candidates,np.zeros(3),np.random.default_rng(2))
    assert [c.key for c in a]==[c.key for c in b]


def test_failed_grasp_point_is_not_selected_again():
    from piper_sim.demo_skill import PickPlaceSkill
    skill=PickPlaceSkill.__new__(PickPlaceSkill)
    skill.cfg={'random_seed':3,'tilt_candidates':[.15]}
    skill.b=NS(data=NS(site_xpos=np.array([[.3,0,.2]])),model=NS(site=lambda n:NS(id=0)))
    old=np.array([.4,0,.09]);new=np.array([.43,0,.09])
    target=NS(center=old,yaw=0.,grasp_yaws=[0.],rejected_points=[old.tolist()],grasp_quality={'alternatives':[dict(center=new.tolist(),yaw=0.,yaws=[0.])]})
    def plan_one(option,landing,floor,height,tilts):return {'center':option.center}
    skill._plan_one=plan_one
    plan=skill.plan(target,np.array([.4,-.2]),.06)
    np.testing.assert_allclose(plan['center'],new)
    assert not skill.planning_active


def test_single_yaw_still_searches_alternate_heights():
    from piper_sim.demo_skill import PickPlaceSkill
    skill=PickPlaceSkill.__new__(PickPlaceSkill)
    skill.cfg={'random_seed':3,'tilt_candidates':[.15],'transport_height_m':.20,'transport_height_candidates_m':[.18]}
    skill.b=NS(data=NS(site_xpos=np.array([[.3,0,.2]])),model=NS(site=lambda n:NS(id=0)))
    target=NS(center=np.array([.4,0,.09]),yaw=0.,grasp_yaws=[0.],grasp_quality={})
    visited=[]
    def plan_one(option,landing,floor,height,tilts):
        visited.append(height)
        if height==.20:raise RuntimeError('height not reachable')
        return {'height':height}
    skill._plan_one=plan_one
    assert skill.plan(target,np.array([.4,-.2]),.06)['height']==.18
    assert visited==[.20,.18]

def test_round_fruit_gets_radial_and_other_grasp_directions():
    from piper_sim.demo_skill import PickPlaceSkill
    skill=PickPlaceSkill.__new__(PickPlaceSkill)
    skill.cfg={'random_seed':3,'tilt_candidates':[.15]}
    skill.b=NS(data=NS(site_xpos=np.array([[.3,0,.2]])),model=NS(site=lambda n:NS(id=0)))
    target=NS(center=np.array([.5,.1,.09]),yaw=0.,grasp_quality={})
    seen=[]
    def plan_one(option,landing,floor,height,tilts):
        seen.extend(option.grasp_yaws)
        if len(seen)==1:raise RuntimeError('first direction blocked')
        return {'yaw':option.grasp_yaws[0]}
    skill._plan_one=plan_one
    skill.plan(target,np.array([.4,-.2]),.06)
    assert len(set(seen))>1
