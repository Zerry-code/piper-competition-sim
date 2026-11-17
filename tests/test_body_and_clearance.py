import copy
import numpy as np
import yaml
from piper_sim.calibration import ROOT
from piper_sim.control import load_interface
from piper_sim.trajectory import trajectory
from piper_sim.grasp_geometry import main_body_region
from piper_sim.implicit_fruit import components,field
from piper_sim.competition_world import clear_of_trays


def test_body_section_avoids_pear_neck_from_partial_surface():
    parts=components('pear',np.random.default_rng(4))
    x,y,z=np.meshgrid(np.linspace(-.035,.035,55),np.linspace(-.035,.035,55),np.linspace(-.04,.05,70),indexing='ij')
    xyz=np.stack([x,y,z],-1)
    cloud=xyz[(abs(field(xyz,parts))<.065)&(y<.012)]
    cloud[:,2]+=.10
    center,quality=main_body_region(cloud,.06)
    assert center[2]<.110
    assert quality['radius_m']>.020
    assert len(quality['alternatives'])>1


def test_tray_exclusion_accounts_for_object_radius_and_wall():
    trays={'red':dict(center_xy_m=[.3,-.2],half_size_xy_m=[.07,.09])}
    assert not clear_of_trays([.3,-.05],.03,trays,.003,.06)
    assert clear_of_trays([.3,.01],.03,trays,.003,.06)


def test_transit_faster_without_changing_precision_or_gripper_speed():
    profiles=yaml.safe_load((ROOT/'config/motion_profiles.yaml').read_text(encoding='utf-8'))
    assert profiles['precision']==dict(speed=.34,acceleration=.5,gripper_speed=.040)
    cfg=load_interface();path=[np.zeros(6),np.full(6,.8)]
    for name,old_speed,old_acc in [('empty',1.10,1.8),('loaded',.60,.8)]:
        old=copy.deepcopy(cfg);old['control']['max_joint_speed_rad_s']=[old_speed]*6
        new=copy.deepcopy(cfg);new['control']['max_joint_speed_rad_s']=[profiles[name]['speed']]*6
        assert len(trajectory(path,0,0,new,profiles[name]['acceleration']))<len(trajectory(path,0,0,old,old_acc))


def test_physical_bounds_exclude_decorative_leaf_below_support():
    import mujoco
    from types import SimpleNamespace
    from piper_sim.competition_world import CompetitionWorld
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="fruit" pos="0 0 .1">
      <freejoint/><geom type="ellipsoid" size=".02 .025 .03"/>
      <geom type="box" pos="0 0 -.06" size=".01 .01 .01" contype="0" conaffinity="0" mass="0"/>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    world=object.__new__(CompetitionWorld);world.backend=SimpleNamespace(model=model,data=data)
    low,high=world.bounds('fruit')
    assert np.allclose(low,[-.02,-.025,.07])
    assert np.allclose(high,[.02,.025,.13])
