import numpy as np
from piper_sim.fruit_placement import Footprint
from piper_sim.fresh_landing import visible_free_landings


def test_no_fresh_floor_observation_never_authorizes_a_landing():
    shape=Footprint(np.zeros((1,2)),np.array([.02]))
    tray={'center_xy_m':[.6,0.],'half_size_xy_m':[.1,.1],'placement_policy':'center_max_coverage'}
    assert visible_free_landings(shape,tray,np.empty((0,2)),set())==[]


def test_fresh_visible_landings_avoid_obstacles_and_failed_locations():
    shape=Footprint(np.zeros((1,2)),np.array([.02]))
    tray={'center_xy_m':[.6,0.],'half_size_xy_m':[.1,.1],'placement_policy':'center_max_coverage'}
    floor={(x,y) for x in range(48,72) for y in range(-12,13)}
    failed=[np.array([.6,0.])];obstacles=np.array([[.65,.04],[.65,.05]])
    points=visible_free_landings(shape,tray,obstacles,floor,failed=failed)
    assert points
    assert all(np.linalg.norm(p-failed[0])>=.014 for p in points)
    assert all(not shape.hits_points(p,obstacles) for p in points)


def test_repeated_release_failure_refreshes_vision_instead_of_same_descent():
    from types import SimpleNamespace as NS
    from run_harvest import FruitSkills
    skills=FruitSkills.__new__(FruitSkills)
    skills.skill=NS(release_verified=False,release_motion_locked=False)
    skills.plan={}
    calls=[]
    skills.refresh_held_landing=lambda target:calls.append('fresh_view') or True
    assert skills.resume_release(NS(release_retries=2))
    assert calls==['fresh_view']
