import numpy as np
from types import SimpleNamespace as NS
from piper_sim.measured_fruit import ellipsoid_axes,measured_parts
from piper_sim.fruit_placement import allocate_fruit_positions,validate_landing,alternative_landings
from piper_sim.batch_audit import boundary_of_estimate

def test_measured_dimensions():
    assert np.isclose(ellipsoid_axes(.33,.27)[1],.27/np.pi)
    assert ellipsoid_axes(.26,.215)[0]>ellipsoid_axes(.26,.215)[1]
    parts=measured_parts('cucumber',[(np.zeros(3),np.array([.05,.015,.015]))])
    assert np.isclose(parts[0][1][0]*2,.26)

def test_oversized_cucumber_and_pear_share_card():
    tray=dict(center_xy_m=[.44,0.],half_size_xy_m=[.1,.1],placement_policy='center_max_coverage')
    x,y=np.meshgrid(np.linspace(-.13,.13,45),np.linspace(-.015,.015,8))
    long=np.c_[x.ravel()+.12,y.ravel()-.25,np.ones(x.size)*.03]
    a=np.linspace(0,2*np.pi,180)
    pear=np.c_[.12+.043*np.cos(a),.25+.043*np.sin(a),np.ones(len(a))*.03]
    tracks=[NS(key=c,category=c,destination='green',estimate=NS(points=p,center=np.mean(p,axis=0))) for c,p in [('cucumber',long),('pear',pear)]]
    positions=allocate_fruit_positions({'green':tray},tracks,{}, {'green':np.array([.44,0])})
    shapes={t.key:t.placement_footprint for t in tracks}
    for t in tracks:
        assert validate_landing(shapes[t.key],positions[t.key],tray,positions,shapes,t.key)
    assert not shapes['pear'].overlaps(positions['pear'],shapes['cucumber'],positions['cucumber'])
    assert alternative_landings({'green':tray},tracks,{},positions)['cucumber']
    p=long-long.mean(0)+np.array([.44,0,.03])
    assert boundary_of_estimate(NS(points=p),tray)['status']=='inside'
    assert boundary_of_estimate(NS(points=p),dict(tray,placement_policy='full_containment'))['status']=='outside'

def test_two_round_props_share_middle_instead_of_one_center_one_edge():
    tray=dict(center_xy_m=[.44,0],half_size_xy_m=[.1,.1],placement_policy='center_max_coverage')
    a=np.linspace(0,2*np.pi,120)
    tracks=[NS(key=c,category=c,destination='red',estimate=NS(points=np.c_[.3+.04*np.cos(a),y+.04*np.sin(a),np.full(len(a),.04)],center=np.array([.3,y,.04]))) for c,y in [('bell_pepper',-.25),('persimmon',.25)]]
    positions=allocate_fruit_positions({'red':tray},tracks,{}, {'red':np.array([.44,0])})
    assert all(np.min(np.array([.1,.1])-abs(p-[.44,0]))>.03 for p in positions.values())

def test_large_ellipsoid_pear_can_use_body_with_100mm_gripper():
    from piper_sim.grasp_geometry import pear_grasp_regions
    u,v=np.meshgrid(np.linspace(0,2*np.pi,160),np.linspace(.08,np.pi/2,70))
    points=np.c_[.06*np.sin(v).ravel()*np.cos(u).ravel(),.043*np.sin(v).ravel()*np.sin(u).ravel(),.043+.043*np.cos(v).ravel()]
    options=pear_grasp_regions(points,0.)
    assert any(.066<o['opening_m']<.10 for o in options)
