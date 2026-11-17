from types import SimpleNamespace as NS
import numpy as np
from piper_sim.batch_audit import BatchBoundaryAudit,identify_audit_target,boundary_of_estimate
from piper_sim.harvest import Harvest,Track


def estimate(x,y=0.,size=.025):
    a,b=np.meshgrid(np.linspace(x-size,x+size,11),np.linspace(y-size,y+size,11))
    return NS(points=np.c_[a.ravel(),b.ravel(),np.ones(a.size)*.09])


def target(key='a'):
    t=Track(key,key,[.4,0,.09],'red',estimate=estimate(.4))
    t.destination_position=np.array([.4,0]);t.audit_pending=True
    return t


TRAY=dict(center_xy_m=[.4,0],half_size_xy_m=[.15,.15])


def test_boundary_inspection_identifies_category_before_adjusting():
    t=target()
    assert identify_audit_target(t,[('wrong',estimate(.55))],TRAY)['status']=='unknown'
    r=identify_audit_target(t,[('a',estimate(.55))],TRAY)
    assert r['status']=='outside' and r['category']=='a' and 'estimate' in r
    assert boundary_of_estimate(estimate(.4),TRAY)['status']=='inside'


def test_ambiguous_same_class_locations_are_not_guessed():
    r=identify_audit_target(target(),[('a',estimate(.35)),('a',estimate(.45))],TRAY)
    assert r['status']=='unknown'


def test_normal_release_completes_without_boundary_camera_or_wait():
    from run_harvest import FruitSkills
    skills=FruitSkills.__new__(FruitSkills);skills.navigation_events=[]
    skills.skill=NS(release_verified=True,retreat_verified=True)
    skills.b=NS(data=NS(time=12.3))
    skills.verify_placement=lambda t:(_ for _ in ()).throw(AssertionError('no per-item check'))
    t=target();assert skills.after_release(t)
    assert not t.audit_pending and t.boundary_check_status=='disabled_by_user'
    assert any(e['event']=='release_completed_without_boundary_audit' for e in skills.navigation_events)
    assert t.task_state=='complete'


def test_release_requires_real_opening_even_without_boundary_audit():
    import pytest
    from run_harvest import FruitSkills
    skills=FruitSkills.__new__(FruitSkills)
    skills.skill=NS(release_verified=False,retreat_verified=False)
    with pytest.raises(RuntimeError,match='张爪'):
        skills.after_release(target())


def test_audit_starts_after_all_releases_and_not_between_targets():
    calls=[]
    class Skills:
        def overview(self):return [target('a'),target('b')]
        def observe_target(self,t,retry):calls.append(('observe',t.key))
        def pick(self,t):return True
        def place_and_verify(self,t):calls.append(('release',t.key));return True
        def finalize_batch(self,targets,result):calls.append(('audit',len(targets)));return result
    result=Harvest(Skills()).run()
    assert result['status']=='complete'
    assert calls==[('observe','a'),('release','a'),('observe','b'),('release','b'),('audit',2)]
    calls.clear();Harvest(Skills(),defer_batch_audit=True).run()
    assert all(c[0]!='audit' for c in calls)


def test_batch_repairs_only_identified_outside_items_and_rechecks_all():
    ts=[target('a'),target('b')];calls=[]
    class Skills:
        def scan_batch_boundaries(self,targets):
            calls.append(('scan',len(targets)))
            return {t.key:dict(status='outside' if t.key=='b' and len(calls)==1 else 'inside',center=[.55,0,.09],estimate=t.estimate) for t in targets}
        def correct_boundary(self,t,state):calls.append(('correct',t.key));return dict(status='complete',attempts=1)
    result=BatchBoundaryAudit(Skills()).run(ts,dict(status='complete',completed=['a','b'],unresolved=[],attempts=2))
    assert result['status']=='complete' and result['attempts']==3
    assert calls==[('scan',2),('correct','b'),('scan',2)]


def test_missing_or_unresolved_boundary_never_reports_complete():
    class Skills:
        def scan_batch_boundaries(self,targets):return {'a':dict(status='unknown',reason='not seen')}
        def correct_boundary(self,*a):raise AssertionError('do not adjust unknown identity')
    r=BatchBoundaryAudit(Skills()).run([target()],dict(status='complete',completed=['a'],unresolved=[]))
    assert r['status']=='incomplete' and not r['completed'] and r['unresolved']==['a']


def test_no_safe_repair_point_does_not_repeat_unchanged_scans():
    calls=[]
    class Skills:
        def scan_batch_boundaries(self,targets):calls.append('scan');return {'a':dict(status='outside')}
        def correct_boundary(self,*args):return dict(status='incomplete',motion_attempted=False,attempts=0)
    r=BatchBoundaryAudit(Skills()).run([target()],dict(status='complete',completed=['a'],unresolved=[]))
    assert calls==['scan'] and r['status']=='incomplete'


def test_repair_is_bounded_when_object_remains_outside():
    calls=[]
    class Skills:
        def scan_batch_boundaries(self,targets):return {'a':dict(status='outside')}
        def correct_boundary(self,*args):calls.append('correct');return dict(status='incomplete',attempts=1)
    r=BatchBoundaryAudit(Skills(),max_correction_rounds=2).run([target()],dict(status='complete',completed=['a'],unresolved=[]))
    assert len(calls)==2 and r['status']=='incomplete'
