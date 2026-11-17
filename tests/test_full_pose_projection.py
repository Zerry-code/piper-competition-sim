from types import SimpleNamespace as NS
import numpy as np
from scipy.spatial.transform import Rotation
from piper_sim.fruit_placement import projected_footprint

def test_tilted_long_object_uses_full_rotation_not_yaw_only():
    x=np.linspace(-.10,.10,50)
    points=np.array([[u,y,z] for u in x for y in [-.012,.012] for z in [-.012,.012]])
    estimate=NS(points=points,center=np.zeros(3),object_center=np.zeros(3))
    flat=projected_footprint(estimate,'cucumber',np.eye(3),np.eye(3),np.zeros(3))
    upright=projected_footprint(estimate,'cucumber',np.eye(3),Rotation.from_euler('y',np.pi/2).as_matrix(),np.zeros(3))
    assert np.ptp(flat.vertices[:,0])>.20
    assert np.ptp(upright.vertices[:,0])<.05


def test_deferred_final_failure_routes_to_fresh_observation():
    from run_harvest import FruitSkills
    s=FruitSkills.__new__(FruitSkills);s.plan={'deferred_final':True};s.skill=NS(release_verified=False)
    calls=[];s.refresh_held_landing=lambda target:calls.append('fresh') or True
    assert s.resume_release(NS()) and calls==['fresh']
