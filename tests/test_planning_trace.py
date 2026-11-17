from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_sim.demo_skill import PickPlaceSkill
from piper_sim.planning_trace import PlanningTrace
from piper_sim.route_search import transfer_route

def stub_skill():
    skill=PickPlaceSkill.__new__(PickPlaceSkill)
    skill.cfg={'random_seed':1,'tilt_candidates':[.15],'planning_target':'cucumber'}
    skill.b=NS(data=NS(site_xpos=np.array([[.3,0,.2]])),model=NS(site=lambda n:NS(id=0)))
    target=NS(center=np.array([.4,.1,.08]),yaw=0.,grasp_yaws=[0.],grasp_quality={})
    return skill,target

def test_timeout_is_unknown_and_does_not_claim_all_candidates_checked():
    skill,target=stub_skill();skill.cfg['planning_budget_s']=0.
    skill._plan_one=lambda *a:pytest.fail('No pose evaluation after zero budget')
    with pytest.raises(RuntimeError,match='规划预算耗尽'):skill.plan(target,np.array([.4,-.2]),.06)
    r=skill.last_planning_report
    assert r['status']=='timeout' and '未知' in r['conclusion']
    assert r['attempts']==0 and r['scope']['combination_upper_bound']>=1
    assert not skill.planning_active

def test_failed_attempt_records_stage_residual_and_scope():
    skill,target=stub_skill()
    def fail(*args):
        skill.planning_phase('hover')
        error=RuntimeError('IK unreachable: target=[.4 .1 .2]')
        error.position_error_m=.012;error.orientation_error_rad=.04;error.target_position=[.4,.1,.2]
        raise error
    skill._plan_one=fail
    with pytest.raises(RuntimeError):skill.plan(target,np.array([.4,-.2]),.06)
    r=skill.last_planning_report
    assert r['status']=='scheduled_finished' and '未证明' in r['conclusion']
    assert r['unique_combinations']==1
    event=next(e for e in r['events'] if e['event']=='attempt_failure')
    assert event['position_error_m']==.012 and '12.0毫米' in event['message']
    assert r['current']['phase']=='hover'
    assert r['scope_details']['grasps'][0]['center']==[.4,.1,.08]

def test_found_report_and_observer_finish_are_consistent():
    skill,target=stub_skill();seen=[];skill.cfg['planning_observer']=seen.append
    skill._plan_one=lambda *a:{'transfer_waypoints':[]}
    skill.plan(target,np.array([.4,-.2]),.06)
    assert skill.last_planning_report['status']=='found'
    assert seen[-1]['status']=='found' and seen[-1]['recent'][-1]['event']=='finished'

def test_route_progress_distinguishes_direct_failure_and_detour():
    events=[]
    def segment(start,goal,yaw,current):
        if np.linalg.norm(goal-current)>.18:raise RuntimeError('IK branch discontinuity')
        return [start,np.r_[goal,np.zeros(3)]]
    transfer_route(segment,np.zeros(6),np.array([.3,0,.2]),np.array([.55,0,.2]),0,progress=lambda event,**data:events.append(dict(event=event,**data)))
    assert events[0]['route_label']=='直线'
    assert any(e['event']=='route_failure' and e['reason_code']=='joint_jump' for e in events)
    assert any(e['event']=='route' and '绕行' in e['route_label'] for e in events)

def test_stop_during_planning_retains_interrupted_report():
    skill,target=stub_skill()
    class Stop(Exception):pass
    def observer(snapshot):
        if snapshot['status']=='searching':raise Stop()
    skill.cfg['planning_observer']=observer
    with pytest.raises(Stop):skill.plan(target,np.array([.4,-.2]),.06)
    assert skill.last_planning_report['status']=='interrupted' and not skill.planning_active


def test_real_ik_reporting_does_not_command_robot_motion(tmp_path):
    from piper_sim.fruit_world import ImplicitWorld,configure
    from piper_sim.demo_skill import TargetEstimate
    world=ImplicitWorld('l2',configure(2329111945,['cube_red','cube_green','cube_blue'],tmp_path),tmp_path,render=False)
    try:
        world.reset(2329111945)
        target=TargetEstimate(np.array([.3723971,.03659924,.07761217]),.03522433,.28785664,np.empty((0,0)),np.empty((0,3)),[0,0],0)
        target.grasp_yaws=[target.yaw+k*np.pi/2 for k in range(4)]
        snapshots=[]
        skill=PickPlaceSkill(world.backend,.06,lambda stage:None,dict(transport_height_m=.2,tilt_candidates=[.15,.35,.6,.75,.9,1.05],planning_observer=snapshots.append))
        before=world.backend.data.qpos.copy();time_before=world.backend.data.time
        skill.plan(target,np.array([.525,-.245]),.064)
        np.testing.assert_equal(before,world.backend.data.qpos)
        assert time_before==world.backend.data.time
        assert snapshots[-1]['status']=='found'
        assert any(e['event']=='route' for e in skill.last_planning_report['events'])
        assert any(e['event']=='phase' and e['phase']=='place' for e in skill.last_planning_report['events'])
    finally:world.close()
