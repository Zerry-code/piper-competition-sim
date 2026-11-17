from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_sim.demo_skill import PickPlaceSkill
from piper_sim.release_safety import ReleaseBlocked
from piper_sim.trajectory import trajectory


def test_no_opening_before_actual_low_place_pose():
    skill=PickPlaceSkill.__new__(PickPlaceSkill);skill.cfg={'ready_transport':True};skill.table=0.
    skill.b=NS(model=NS(site=lambda n:NS(id=0)),data=NS(site_xpos=np.array([[.5,0.,.20]])))
    with pytest.raises(ReleaseBlocked):skill.require_low_release({'place':np.array([.5,0.,.05])})


def test_high_bottom_blocks_release_even_if_tcp_matches_plan():
    skill=PickPlaceSkill.__new__(PickPlaceSkill);skill.cfg={'ready_transport':True};skill.table=0.
    skill.carried_visual_cloud=np.array([[0,0,-.03]])
    skill.b=NS(model=NS(site=lambda n:NS(id=0)),data=NS(site_xpos=np.array([[.5,0.,.15]]),site_xmat=np.eye(3).reshape(1,9)))
    with pytest.raises(ReleaseBlocked):skill.require_low_release({'place':np.array([.5,0.,.15])})


def test_path_sampling_density_does_not_make_identical_motion_slow():
    interface={'control':{'policy_hz':20,'max_joint_speed_rad_s':[1.]*6,'max_gripper_speed_m_s':.05}}
    a=np.zeros(6);b=np.ones(6)
    simple=trajectory([a,b],0,0,interface)
    dense=np.vstack([np.linspace(a,b*.1,100),b])
    uneven=trajectory(dense,0,0,interface)
    assert abs(len(simple)-len(uneven))<=1


def test_curved_outline_keeps_inner_free_space():
    from piper_sim.geometric_footprint import estimate_geometry
    u=np.linspace(-.09,.09,100);points=np.array([[x,.7*x*x/.09+w,0.] for x in u for w in np.linspace(-.006,.006,5)])
    shape=estimate_geometry(NS(points=points,object_center=np.zeros(3)),'banana')
    assert shape.kind=='curved_sections'
    assert not shape._contains(np.array([[0.,.04]]))[0]
    assert shape._contains(np.array([[0.,0.]]))[0]


def test_reallocation_searches_half_region_before_truncating_options():
    from piper_sim.fruit_placement import Footprint,reallocate_final_landings
    shape=Footprint(np.zeros((1,2)),np.array([.02]))
    tray={'center_xy_m':[.5865,0.],'half_size_xy_m':[.1,.1],'placement_policy':'center_max_coverage'}
    partition={'center':[.5365,0.],'max_center_offset_m':.025}
    result=reallocate_final_landings(shape,tray,{}, {},'pear',np.empty((0,2)),np.array([.5365,0.]),partition=partition)
    assert result and all(np.linalg.norm(p-partition['center'])<=.025 for p in result['points'])


def test_small_xy_tracking_offset_does_not_block_low_release():
    skill=PickPlaceSkill.__new__(PickPlaceSkill);skill.cfg={'ready_transport':True};skill.table=0.
    skill.carried_visual_cloud=np.array([[0,0,-.03]])
    skill.b=NS(model=NS(site=lambda n:NS(id=0)),data=NS(site_xpos=np.array([[.507,0.,.035]]),site_xmat=np.eye(3).reshape(1,9)))
    skill.require_low_release({'place':np.array([.5,0.,.035])})


def test_release_height_correction_changes_target_instead_of_repeating(monkeypatch):
    import piper_sim.demo_skill as module
    skill=PickPlaceSkill.__new__(PickPlaceSkill);skill.cfg={};skill.table=0.
    skill.carried_visual_cloud=np.array([[0,0,-.03]])
    skill.b=NS(model=NS(site=lambda n:NS(id=0)),data=NS(site_xpos=np.array([[.5,0.,.043]]),site_xmat=np.eye(3).reshape(1,9)),measured=lambda:np.zeros(8))
    skill.cartesian_path=lambda *a:[np.zeros(6)];skill.validate_transfer=lambda p:None
    monkeypatch.setattr(module,'opening_clearance',lambda *a:True)
    plan={'place':np.array([.5,0.,.043]),'yaw':0.}
    skill.correct_release_height(plan)
    assert abs(plan['place'][2]-.039)<1e-9
