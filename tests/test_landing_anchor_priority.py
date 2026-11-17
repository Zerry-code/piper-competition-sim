from types import SimpleNamespace as NS
import numpy as np
from piper_sim.demo_skill import PickPlaceSkill


def test_try_new_orientation_at_primary_before_changing_landing():
    skill=PickPlaceSkill.__new__(PickPlaceSkill)
    skill.cfg={'tilt_candidates':[.35,.6],'transport_height_m':.2,'planning_budget_s':20.}
    skill.b=NS(data=NS(site_xpos=np.array([[.3,0,.2]])),model=NS(site=lambda n:NS(id=0)))
    target=NS(center=np.array([.4,0,.05]),yaw=0.,grasp_yaws=[0.],grasp_quality={},destination_options=[np.array([.55,.02])])
    primary=np.array([.5,0]);visited=[]
    def check(option,landing,floor,height,tilts):
        visited.append(landing.copy())
        if np.allclose(landing,primary) and tilts[0]==.35:raise RuntimeError('first orientation blocked')
        return {'tilt':tilts[0]}
    skill._plan_one=check
    result=skill.plan(target,primary,0.)
    assert np.allclose(result['destination_position'],primary)
    assert all(np.allclose(p,primary) for p in visited)
