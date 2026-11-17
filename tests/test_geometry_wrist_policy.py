from types import SimpleNamespace as NS
import numpy as np
from piper_sim.fruit_placement import visual_footprint,alternative_landings
from piper_sim.pair_placement import allocate_pairs


def test_grasp_wrist_limit_is_an_ik_bound_and_does_not_change_robot(monkeypatch):
    import piper_sim.traditional as module
    monkeypatch.setattr(module.mujoco,'MjData',lambda model:None)
    lower=[-2.6,0.,-2.96,-1.74,-1.22,-2.0944];upper=[2.6,3.14,0.,1.74,1.22,2.0944]
    backend=NS(model=NS(site=lambda name:NS(id=0)),interface={'joints':{'lower_rad':lower,'upper_rad':upper}})
    grasp=module.IK(backend,wrist_limit=np.pi/2);place=module.IK(backend)
    assert grasp.lower[5]==-np.pi/2 and grasp.upper[5]==np.pi/2
    assert place.upper[5]==2.0944 and upper[5]==2.0944 and lower[5]==-2.0944


def test_ellipsoid_footprint_retains_long_and_short_axes():
    angle=np.linspace(0,2*np.pi,200,endpoint=False)
    points=np.c_[.061*np.cos(angle),.043*np.sin(angle),np.full(200,.04)]
    shape=visual_footprint(NS(points=points,center=np.array([0,0,.04])),'pear')
    assert shape.kind=='ellipse' and np.all(shape._contains(points[:,:2]))
    extent=np.ptp(shape.vertices,axis=0)
    assert extent[0]>.12 and .085<extent[1]<.10


def test_pair_and_all_backups_remain_near_assigned_half_centroid():
    angle=np.linspace(0,2*np.pi,100)
    tracks=[NS(key=c,category=c,destination='red',estimate=NS(points=np.c_[.03*np.cos(angle),y+.03*np.sin(angle),np.full(100,.04)],center=np.array([0,y,.04]))) for c,y in [('apple',-.2),('persimmon',.2)]]
    tray={'center_xy_m':[.5865,0.],'half_size_xy_m':[.1,.1],'placement_policy':'center_max_coverage','allocation_mode':'pair_partition'}
    positions=allocate_pairs({'red':tray},tracks,{}, {'red':[.5865,0.]})
    options=alternative_landings({'red':tray},tracks,{},positions)
    for t in tracks:
        center=np.asarray(t.placement_partition['center'])
        assert np.linalg.norm(positions[t.key]-center)<.012
        assert all(np.linalg.norm(p-center)<=.025+1e-9 for p in options[t.key])
        assert all(np.min(.1-abs(p-tray['center_xy_m']))>.02 for p in options[t.key])
