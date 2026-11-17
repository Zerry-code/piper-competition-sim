from types import SimpleNamespace
import numpy as np
import pytest
from piper_sim.demo_skill import PickPlaceSkill
from piper_sim.release_safety import ReleaseBlocked,release_feedback
from piper_sim.fruit_world import ImplicitWorld,configure


def test_open_command_or_total_width_cannot_replace_both_finger_feedback(tmp_path):
    world=ImplicitWorld('l2',configure(710,['cube_red'],tmp_path),tmp_path,render=False)
    try:
        world.reset(710);backend=world.backend
        backend.previous_target[6]=.10
        backend.data.qpos[backend.fingers]=[.02,-.02]
        assert not release_feedback(backend)['opened']
        backend.data.qpos[backend.fingers]=[.03,-.04]
        assert not release_feedback(backend)['opened']
        backend.data.qpos[backend.fingers]=[.05,-.05]
        assert release_feedback(backend)['opened']
    finally:world.close()


def mock_skill(monkeypatch,opened=True,object_contact=False):
    backend=SimpleNamespace(measured=lambda:np.zeros(7),previous_target=np.zeros(7),
                            step=lambda action:None,data=SimpleNamespace(time=0.))
    skill=PickPlaceSkill(backend,.06,lambda stage:None,{})
    skill.hold=lambda seconds:None
    calls=[]
    skill.move=lambda q,width:calls.append(('open',width))
    skill.vertical_release_move=lambda dz,width:calls.append(('vertical',dz))
    skill.cartesian=lambda xyz,yaw,width:calls.append(('lateral_allowed',width))
    monkeypatch.setattr('piper_sim.demo_skill.release_feedback',lambda backend:dict(
        opened=opened,object_contact=object_contact,tray_contact=False))
    return skill,calls


def test_jammed_finger_timeout_cannot_authorize_lateral_retreat(monkeypatch):
    skill,calls=mock_skill(monkeypatch,opened=False)
    def jam(q,width):raise TimeoutError('finger blocked at tray rim')
    skill.move=jam
    with pytest.raises(ReleaseBlocked,match='opening unconfirmed'):
        skill.release_and_retreat(dict(above=np.array([.4,-.2,.2]),yaw=0))
    assert calls==[('vertical',.005)]*3
    assert not skill.release_verified
    assert not skill.retreat_verified


def test_open_then_vertical_clearance_precedes_any_other_move(monkeypatch):
    skill,calls=mock_skill(monkeypatch)
    skill.release_and_retreat(dict(above=np.array([.4,-.2,.2]),yaw=0))
    assert calls==[('open',.10),('vertical',.06),('lateral_allowed',.10)]
    assert skill.release_verified and skill.retreat_verified


def test_contact_after_vertical_lift_blocks_lateral_motion(monkeypatch):
    skill,calls=mock_skill(monkeypatch,object_contact=True)
    with pytest.raises(ReleaseBlocked,match='contact remains'):
        skill.release_and_retreat(dict(above=np.array([.4,-.2,.2]),yaw=0))
    assert not any(name=='lateral_allowed' for name,value in calls)
    assert not skill.retreat_verified
    assert skill.release_motion_locked


def test_failed_release_interlock_rejects_later_observation_motion(monkeypatch):
    skill,calls=mock_skill(monkeypatch)
    skill.release_motion_locked=True
    skill.stage='local_observe'
    with pytest.raises(ReleaseBlocked,match='interlock'):
        skill.tick(np.ones(7)*.1)


def test_physical_joint_limits_recover_crossed_finger_fault(tmp_path):
    import mujoco
    world=ImplicitWorld('l2',configure(710,['cube_red'],tmp_path),tmp_path,render=False)
    try:
        world.reset(710);backend=world.backend
        # Initial robot-joint fault only; fruit/box poses are untouched.
        backend.data.qpos[backend.fingers]=[-.0057,.0057]
        backend._set_target(np.r_[backend.measured()[:6],.10])
        for _ in range(500):mujoco.mj_step(backend.model,backend.data)
        assert release_feedback(backend)['opened']
        assert backend.data.qpos[backend.fingers[0]]>0
        assert backend.data.qpos[backend.fingers[1]]<0
    finally:world.close()


def test_release_retry_is_local_before_any_other_target_or_regrasp():
    from piper_sim.harvest import Harvest,Track,RetryableFailure
    calls=[]
    class Skills:
        last_attempt_closed=True
        def overview(self):return [Track('orange','orange',[.4,-.1,.08],'blue'),Track('next','apple',[.4,.1,.08],'red')]
        def observe_target(self,target,retry):calls.append(('observe',target.key))
        def pick(self,target):calls.append(('pick',target.key));return True
        def place_and_verify(self,target):
            if target.key=='orange':
                target.release_pending=True;raise RetryableFailure('tray rim blocked release',phase='release')
            return True
        def resume_release(self,target):
            calls.append(('local_release',target.key));target.release_pending=False;return True
        def recover(self,target):raise AssertionError('Do not carry the retained fruit to the old pickup location')
    report=Harvest(Skills()).run()
    assert report['status']=='complete'
    assert report['attempts']==2
    assert calls.index(('local_release','orange'))<calls.index(('observe','next'))
    assert calls.count(('pick','orange'))==1


def test_local_verification_can_replace_redundant_retreat_after_clearance(monkeypatch):
    skill,calls=mock_skill(monkeypatch)
    skill.cfg['verify_after_vertical_clearance']=True
    skill.b.model=SimpleNamespace(site=lambda n:SimpleNamespace(id=0))
    skill.b.data.site_xpos=np.array([[.4,-.2,.14]])
    skill.release_and_retreat(dict(above=np.array([.4,-.2,.2]),yaw=0))
    assert calls[0]==('open',.10) and calls[1][0]=='vertical'
    assert calls[1][1]>=.06 and len(calls)==2
    assert skill.release_verified and skill.retreat_verified
    assert not skill.release_motion_locked


def test_batch_release_lifts_only_safe_clearance_without_observation_height(monkeypatch):
    skill,calls=mock_skill(monkeypatch)
    skill.cfg.update(verify_after_vertical_clearance=True,release_to_observation_height=False)
    skill.release_and_retreat(dict(above=np.array([.4,-.2,.28]),yaw=0))
    assert calls==[('open',.10),('vertical',.06)]
    assert skill.release_verified and skill.retreat_verified


def test_repeated_clearance_failure_is_bounded_and_never_moves_to_next_target():
    from piper_sim.harvest import Harvest,Track,RetryableFailure
    calls=[]
    class Skills:
        def overview(self):return [Track('blocked','banana',[.4,-.1,.08],'red'),Track('next','pear',[.4,.1,.08],'green')]
        def observe_target(self,t,retry):calls.append(('observe',t.key))
        def pick(self,t):return True
        def place_and_verify(self,t):t.release_pending=True;raise RetryableFailure('release blocked',phase='release')
        def resume_release(self,t):calls.append(('retry',t.key));raise ReleaseBlocked('vertical clearance blocked')
    result=Harvest(Skills()).run()
    assert result['status']=='incomplete' and result['stop_reason']=='release_clearance_blocked'
    assert calls.count(('retry','blocked'))==3
    assert ('observe','next') not in calls and not result['completed']

def test_pepper_contact_clears_after_additional_vertical_steps(monkeypatch):
    skill,calls=mock_skill(monkeypatch)
    def feedback(backend):
        lifted=sum(c[1] for c in calls if c[0]=='vertical')
        return dict(opened=True,object_contact=lifted<.099,tray_contact=False)
    monkeypatch.setattr('piper_sim.demo_skill.release_feedback',feedback)
    skill.release_and_retreat(dict(above=np.array([.4,-.2,.2]),yaw=0))
    assert [c for c in calls if c[0]=='vertical']==[('vertical',.06),('vertical',.02),('vertical',.02)]
    assert calls[-1][0]=='lateral_allowed'
    assert skill.retreat_verified and not skill.release_motion_locked


def test_release_retry_wait_preserves_interlock_and_measured_pose():
    from run_harvest import FruitSkills
    skills=FruitSkills.__new__(FruitSkills)
    skills.b=SimpleNamespace(previous_target=np.array([.7]*6+[.10]),measured=lambda:np.array([.2]*6+[.068]))
    seen=[]
    skills.skill=SimpleNamespace(release_motion_locked=True,stage='release_lift_vertical')
    skills.skill.hold=lambda seconds:seen.append((skills.skill.stage,skills.b.previous_target.copy()))
    skills.wait_before_retry(SimpleNamespace(category='bell_pepper'),'release_clearance')
    assert seen[0][0]=='release_open'
    assert np.allclose(seen[0][1][:6],.2)
    assert skills.skill.release_motion_locked

def test_visible_pepper_surface_must_clear_fully_open_pads(tmp_path):
    import mujoco
    world=ImplicitWorld('l2',configure(710,['cube_red'],tmp_path),tmp_path,render=False)
    try:
        world.reset(710);b=world.backend;q=b.measured()[:6]
        d=mujoco.MjData(b.model);d.qpos[:6]=q;d.qpos[b.fingers]=[.05,-.05];mujoco.mj_forward(b.model,d)
        skill=PickPlaceSkill(b,.06,lambda stage:None,{})
        point=d.geom_xpos[b.model.geom('link7_pad').id].copy()
        target=SimpleNamespace(category='bell_pepper',points=np.array([point]))
        with pytest.raises(RuntimeError,match='指垫相交'):
            skill.check_visual_release(target,np.zeros(3),np.eye(3),q,np.zeros(3),0.)
        target.points=np.array([[10.,10.,10.]])
        skill.check_visual_release(target,np.zeros(3),np.eye(3),q,np.zeros(3),0.)
    finally:world.close()

def test_short_vertical_disengagement_when_long_lift_has_no_ik(monkeypatch):
    skill,calls=mock_skill(monkeypatch)
    def vertical(distance,width):
        calls.append(('vertical',distance))
        if distance>.021:raise ReleaseBlocked('No continuous vertical release clearance path')
    skill.vertical_release_move=vertical
    skill.release_and_retreat(dict(above=np.array([.4,-.2,.2]),yaw=0))
    assert [('vertical',.06),('vertical',.04),('vertical',.02)]==[c for c in calls if c[0]=='vertical']
    assert skill.retreat_verified and not skill.release_motion_locked
