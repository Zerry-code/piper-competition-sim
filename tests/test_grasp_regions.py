import numpy as np
import mujoco
from piper_sim.grasp_geometry import banana_segment
from piper_sim.implicit_fruit import components,field
from piper_sim.control import load_interface
from piper_sim.trajectory import trajectory


def test_banana_region_is_on_straight_interior_body():
    parts=components('banana',np.random.default_rng(9))
    x,y=np.meshgrid(np.linspace(-.075,.075,160),np.linspace(-.045,.045,100))
    xyz=np.stack([x,y,np.zeros_like(x)],-1)
    points=xyz[field(xyz,parts)<0]
    center,yaw,quality=banana_segment(points)
    assert field(np.r_[center,0][None],parts)[0]<0
    assert .19<quality['position_fraction']<.81
    assert quality['radius_m']>.005
    assert np.isfinite(yaw)


def test_empty_motion_is_faster_under_its_own_bounded_profile():
    cfg=load_interface();points=[np.zeros(6),np.full(6,.6)]
    cfg['control']['max_joint_speed_rad_s']=[1.1]*6
    fast=trajectory(points,0,0,cfg,1.8)
    cfg['control']['max_joint_speed_rad_s']=[.34]*6
    slow=trajectory(points,0,0,cfg,.5)
    assert len(fast)<len(slow)*.7


def corner_fixture_height(condim,torsion=.01):
    xml=f'''<mujoco><option timestep=".002" integrator="implicitfast"/>
    <default><geom friction=".8 {torsion} .001" condim="{condim}" solref=".01 1"/></default>
    <worldbody><geom type="sphere" pos=".013 -.0272 .15" size=".01"/>
    <geom type="sphere" pos=".013 .0272 .15" size=".01"/>
    <body pos="0 0 .15"><freejoint/><geom type="box" size=".0175 .0175 .0175" mass=".05"/></body>
    </worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    for _ in range(500):mujoco.mj_step(model,data)
    return data.qpos[2]


def test_torsional_friction_can_artificially_hold_an_offset_point_grasp():
    # Synthetic regression only, not a claim about every real edge grasp.
    assert corner_fixture_height(3)<.10
    assert corner_fixture_height(6)>.12


def test_modest_spin_friction_does_not_restore_the_old_corner_lock():
    assert corner_fixture_height(4,.0005)<.10


def test_elongated_grasp_candidates_prefer_middle_with_safe_outer_backups():
    from piper_sim.grasp_geometry import banana_segment
    points=[]
    for x in np.linspace(0,.16,160):
        radius=.018 if x<.04 or x>.12 else .009
        for y in np.linspace(-radius,radius,20):points.append([.35+x,.10+y,.08])
    _,_,quality=banana_segment(np.array(points))
    assert .35<=quality['position_fraction']<=.65
    assert quality['alternatives']
    assert all(.20<=o['position_fraction']<=.80 for o in quality['alternatives'])
    flags=[1/3<=o['position_fraction']<=2/3 for o in quality['alternatives']]
    assert flags[0] and not all(flags)
    assert flags==sorted(flags,reverse=True)
