import numpy as np
from piper_sim.harvest import Harvest, Track
from piper_sim.implicit_fruit import components, field
from piper_sim.fruit_perception import interior_pixel


class FakeSkills:
    def __init__(self,failures):self.failures=failures;self.calls=[]
    def overview(self):return [Track('left','apple',[.4,.1,.08],'red'),Track('right','banana',[.4,-.1,.08],'blue')]
    def observe_target(self,t,retry):self.calls.append(('observe',t.key,retry))
    def pick(self,t):
        self.calls.append(('pick',t.key))
        return t.attempts>self.failures.get(t.key,0)
    def place_and_verify(self,t):self.calls.append(('place',t.key));return True
    def recover(self,t):self.calls.append(('recover',t.key))


def test_three_misses_defer_without_empty_transport():
    skills=FakeSkills({'right':3})
    report=Harvest(skills).run()
    assert report['initial_order']==['right','left']
    assert report['completed']==['left','right']
    assert [x for x in skills.calls if x[0]=='pick']==[
        ('pick','right')]*3+[('pick','left'),('pick','right')]
    assert [x for x in skills.calls if x[0]=='place']==[('place','left'),('place','right')]
    assert ('observe','right',True) in skills.calls


def test_permanent_failure_is_explicit_incomplete():
    report=Harvest(FakeSkills({'right':100}),max_deferrals=2).run()
    assert report['status']=='incomplete'
    assert report['unresolved']==['right']
    assert report['attempts']==7


def test_live_scheduler_keeps_last_target_after_old_retry_limits():
    skills=FakeSkills({'right':63})
    report=Harvest(skills,max_attempts=None,max_deferrals=None).run()
    assert report['status']=='complete'
    assert report['completed']==['left','right']
    assert report['attempts']==65
    assert report['stop_reason']=='all_targets_confirmed'


def test_repeated_planning_failure_remains_pending_until_deadline():
    from piper_sim.harvest import RetryableFailure
    skills=FakeSkills({});count=0
    def unavailable(target,retry):
        nonlocal count
        count+=1
        if count>=12:raise TimeoutError('competition clock expired')
        raise RetryableFailure('No continuous grasp path',phase='planning')
    skills.observe_target=unavailable
    import pytest
    with pytest.raises(TimeoutError,match='clock expired'):
        Harvest(skills,max_attempts=None,max_deferrals=None).run()
    assert count==12
    assert not any(call[0]=='pick' for call in skills.calls)


def test_transient_observation_timeout_does_not_end_round_or_count_a_grasp():
    from piper_sim.harvest import RetryableFailure
    skills=FakeSkills({});original=skills.observe_target;failed=False
    def observe(target,retry):
        nonlocal failed
        if not failed:
            failed=True;raise RetryableFailure('local observation timed out',phase='observation')
        original(target,retry)
    skills.observe_target=observe
    report=Harvest(skills).run()
    assert report['status']=='complete'
    assert report['attempts']==2
    assert any(e.get('phase')=='observation' for e in report['events'])


def test_curved_banana_grasp_is_inside_not_hollow_centroid():
    parts=components('banana',np.random.default_rng(3))
    x,y=np.meshgrid(np.linspace(-.08,.08,160),np.linspace(-.06,.06,120))
    mask=field(np.stack([x,y,np.zeros_like(x)],-1),parts)<0
    u,v=interior_pixel(mask)
    assert mask[v,u]
    from scipy.ndimage import distance_transform_edt
    assert distance_transform_edt(mask)[v,u]>3


def test_shape_seed_reproducible_and_varied():
    a=components('pear',np.random.default_rng(1))
    b=components('pear',np.random.default_rng(1))
    c=components('pear',np.random.default_rng(2))
    np.testing.assert_allclose(a[0][1],b[0][1])
    assert not np.allclose(a[0][1],c[0][1])


def test_depth_noise_seed_and_near_range_mask():
    from piper_sim.calibration import simulated_depth_outputs
    camera={'depth':dict(min_valid_m=.3,max_valid_m=3.,meters_per_unit=.001,
                        noise_std_m=.001,dropout_probability=.01)}
    image=np.full((40,40),.5,dtype=np.float32)
    image[0,:]=.1
    a=simulated_depth_outputs(image,camera,np.random.default_rng(12))[0]
    b=simulated_depth_outputs(image,camera,np.random.default_rng(12))[0]
    np.testing.assert_equal(a,b)
    assert np.isnan(a[0]).all()
    assert np.nanstd(a)>.0005


def test_failed_placement_does_not_complete_target():
    skills=FakeSkills({})
    skills.place_and_verify=lambda target:False
    report=Harvest(skills,max_deferrals=1).run()
    assert report['completed']==[]
    assert report['status']=='incomplete'


def test_planning_failure_is_not_counted_as_empty_grasp():
    from piper_sim.harvest import RetryableFailure
    skills=FakeSkills({})
    def unavailable(target,retry):raise RetryableFailure('No reachable pose',phase='planning')
    skills.observe_target=unavailable
    report=Harvest(skills,max_deferrals=1).run()
    assert report['attempts']==0
    assert not any(call[0]=='pick' for call in skills.calls)
    assert all(event.get('gripper_closed') is False for event in report['events'])


def test_approach_without_closure_does_not_increment_grasp_attempts():
    skills=FakeSkills({})
    skills.last_attempt_closed=False
    skills.pick=lambda target:False
    report=Harvest(skills,max_deferrals=1).run()
    assert report['attempts']==0
    assert all(event.get('event')=='approach_failed' for event in report['events'])


def test_training_interface_no_object_pose_leak(tmp_path):
    from piper_sim.calibration import ROOT
    if not (ROOT/'assets/piper/scene.xml').exists():
        import pytest
        pytest.skip('Official robot assets not installed')
    from piper_sim.training_env import FruitEnv
    env=FruitEnv(categories=('apple',),output=tmp_path,render=False,horizon_s=.1)
    try:
        obs,_=env.reset(seed=51)
        assert 'objects' not in obs and 'positions' not in obs
        assert obs['joint_position_rad'].shape==(6,)
        env.step(np.zeros(7))
        _,_,terminated,truncated,_=env.step(np.zeros(7))
        assert truncated and not terminated
    finally:env.close()


def test_observation_limit_keeps_failed_target_unresolved():
    from piper_sim.harvest import RetryableFailure
    class Blocked(FakeSkills):
        def observe_target(self,t,retry):raise RetryableFailure('no new depth evidence')
    harvest=Harvest(Blocked({}),max_preparations_per_target=3)
    result=harvest.run()
    assert result['status']=='incomplete'
    assert set(result['unresolved'])=={'left','right'}
    assert result['attempts']==0
    assert any(e['event']=='preparation_limit_reached' for e in result['events'])


def test_unseen_boundary_does_not_trigger_full_search_and_next_target_runs():
    class BoundarySkills(FakeSkills):
        last_release_completed=True
        def pick(self,t):return True
        def place_and_verify(self,t):
            self.calls.append(('place',t.key))
            if t.key=='right':
                t.boundary_check_status='unknown';t.last_preparation_error='boundary unseen';return False
            return True
        def recover_dropped(self,t):raise AssertionError('No full-table search after release')
        def recover(self,t):raise AssertionError('No recovery for unknown boundary')
    skills=BoundarySkills({});r=Harvest(skills).run()
    assert r['status']=='incomplete' and r['completed']==['left'] and r['unresolved']==['right']
    assert ('place','left') in skills.calls


def test_unknown_boundary_after_release_retry_also_skips_global_search():
    from piper_sim.harvest import RetryableFailure
    class Skills(FakeSkills):
        last_release_completed=True
        def pick(self,t):return True
        def place_and_verify(self,t):
            if t.key=='right':t.release_pending=True;raise RetryableFailure('release blocked',phase='release')
            return True
        def resume_release(self,t):
            t.release_pending=False;t.boundary_check_status='unknown';return False
        def recover_dropped(self,t):raise AssertionError('Do not search the table after boundary-only inspection')
        def recover(self,t):raise AssertionError('No retained recovery')
    r=Harvest(Skills({})).run()
    assert r['completed']==['left'] and r['unresolved']==['right']
