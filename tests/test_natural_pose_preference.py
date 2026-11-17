from piper_sim.pose_preference import natural_tilts
from piper_sim.demo_skill import PickPlaceSkill
from types import SimpleNamespace as NS
import numpy as np


def test_preference_keeps_reverse_candidates_for_fallback():
    values=[-.9,1.8,.6,.35,-.3,.15]
    result=natural_tilts(values)
    assert result[0]==.35 and result.index(.6)<result.index(-.9)
    assert set(result)==set(values)


def test_search_uses_reverse_pose_if_preferred_candidates_fail():
    skill=PickPlaceSkill.__new__(PickPlaceSkill)
    skill.cfg={'natural_pose_preference':True,'tilt_candidates':[-.9,.35],'transport_height_m':.2}
    skill.b=NS(data=NS(site_xpos=np.array([[.3,0,.2]])),model=NS(site=lambda n:NS(id=0)))
    target=NS(center=np.array([.4,0,.05]),yaw=0.,grasp_yaws=[0.],grasp_quality={})
    visited=[]
    def plan_one(option,destination,floor,height,tilts):
        visited.append(tilts[0])
        if tilts[0]>=0:raise RuntimeError('candidate blocked')
        return {'tilt':tilts[0]}
    skill._plan_one=plan_one
    assert skill.plan(target,np.array([.5,0]),0.)['tilt']==-.9
    assert visited==[.35,-.9]


def test_fallback_is_not_starved_by_long_preferred_list():
    from piper_sim.pose_preference import fair_tilts
    values=[.15,.35,.6,.9,1.2,1.5,-.3,-.6,-.9]
    trials=fair_tilts(values)
    assert any(not 0<=t<=1.25 for t in trials[:3])
    assert set(trials)==set(values)


def test_placement_prefers_preserving_carried_tilt_over_flipping_upright():
    from piper_sim.pose_preference import placement_orientations
    poses=placement_orientations([-.9,.35,.6],[0.],-.9,0.)
    assert poses[0]==(-.9,0.)
    assert (.35,0.) in poses
