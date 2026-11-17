from types import SimpleNamespace
import numpy as np
import pytest
from piper_sim.drop_recovery import GripMonitor,ObjectDropped,LandingTracker,landing_region
from piper_sim.harvest import Harvest,Track


def test_contact_bounce_is_filtered_but_persistent_loss_interrupts():
    monitor=GripMonitor(.1);monitor.arm()
    monitor.update(0,False,'transport');monitor.update(.05,True,'transport')
    monitor.update(.1,False,'transport');monitor.update(.15,False,'transport')
    with pytest.raises(ObjectDropped):monitor.update(.2,False,'transport')
    assert not monitor.armed
    monitor.update(.3,False,'release')


def test_region_requires_visible_extent_inside_tray_not_just_center():
    trays={'green':dict(center_xy_m=[.5,-.2],half_size_xy_m=[.07,.09])}
    geometry=dict(floor_thickness_m=.004)
    points=np.array([[.48,-.22,.07],[.52,-.18,.10]])
    assert landing_region(points,trays,.06,geometry)=='green'
    points[1,0]=.59
    assert landing_region(points,trays,.06,geometry)=='table'


def test_stationary_landing_requires_repeated_frames_and_resets_on_motion():
    tracker=LandingTracker(.4);points=np.array([[0,0,.07],[.03,.03,.10]])
    assert not tracker.update(0,points,'green')
    assert not tracker.update(.2,points+.01,'green')
    assert not tracker.update(.4,points+.01,'green')
    assert tracker.update(.6,points+.01,'green')


def test_landing_against_inner_wall_requests_another_view():
    trays={'green':dict(center_xy_m=[.5,-.2],half_size_xy_m=[.07,.09])}
    points=np.array([[.432,-.2905,.071],[.46,-.25,.10]])
    assert landing_region(points,trays,.06,dict(floor_thickness_m=.004))=='edge:green'


def test_slow_continuous_drift_does_not_look_stationary():
    tracker=LandingTracker(.4);points=np.array([[0,0,.07],[.03,.03,.10]])
    assert not tracker.update(0,points,'green')
    assert not tracker.update(.15,points+[.002,0,0],'green')
    assert not tracker.update(.3,points+[.004,0,0],'green')
    assert not tracker.update(.6,points+[.004,0,0],'green')
    assert tracker.update(.75,points+[.004,0,0],'green')


@pytest.mark.parametrize('phase',['lift','transport'])
@pytest.mark.parametrize('outcome',['placed','retry'])
def test_drop_recovery_takes_priority_over_next_target(phase,outcome):
    calls=[]
    class Skills:
        last_attempt_closed=True
        dropped=False
        def overview(self):return [Track('first','pear',[.4,-.1,.08],'green'),Track('next','apple',[.4,.1,.08],'red')]
        def observe_target(self,target,retry):calls.append(('observe',target.key))
        def pick(self,target):
            calls.append(('pick',target.key))
            if phase=='lift' and not self.dropped:
                self.dropped=True;raise ObjectDropped('lift')
            return True
        def place_and_verify(self,target):
            if phase=='transport' and not self.dropped:
                self.dropped=True;raise ObjectDropped('transport')
            calls.append(('place',target.key));return True
        def recover_dropped(self,target):calls.append(('drop_search',target.key));return outcome
        def recover(self,target):raise AssertionError('A drop needs visual search, not blind release/recovery')
    report=Harvest(Skills()).run()
    assert report['status']=='complete'
    assert report['completed']==['first','next']
    assert calls.index(('drop_search','first'))<calls.index(('pick','next'))
    assert calls.count(('pick','first'))==(1 if outcome=='placed' else 2)
    assert report['attempts']==(2 if outcome=='placed' else 3)
    assert not any(e.get('reason')=='three_failed_closures' for e in report['events'])


def test_unseen_drop_is_never_assumed_placed_or_grasped_at_old_position():
    class Skills:
        last_attempt_closed=True
        def overview(self):return [Track('fruit','pear',[.4,0,.08],'green')]
        def observe_target(self,*args,**kwargs):pass
        def pick(self,target):raise ObjectDropped('lift')
        def recover_dropped(self,target):return 'unseen'
    report=Harvest(Skills(),max_deferrals=2).run()
    assert report['status']=='incomplete'
    assert report['attempts']==1
    assert report['unresolved']==['fruit']


def test_unconfirmed_release_searches_destination_instead_of_old_pickup_location():
    class Skills:
        last_attempt_closed=True
        last_release_completed=True
        def overview(self):return [Track('fruit','pear',[.4,0,.08],'green')]
        def observe_target(self,*args,**kwargs):pass
        def pick(self,target):return True
        def place_and_verify(self,target):return False
        def recover_dropped(self,target):return 'placed'
        def recover(self,target):raise AssertionError('Already released: do not return to old pickup point')
    report=Harvest(Skills()).run()
    assert report['status']=='complete'
    assert report['attempts']==1
    assert any(e.get('event')=='placement_unconfirmed_search' for e in report['events'])
