"""Regression: an intermediate release is never final success; plan one observed leg at a time."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_sim.harvest import Harvest,Track


def source_estimate():
    points=np.array([[x,y,.08] for x in np.linspace(.44,.56,40) for y in np.linspace(-.008,.008,5)])
    return NS(points=points,center=np.array([.5,0,.08]),object_center=np.array([.5,0,.08]),height=.03,yaw=0.,grasp_yaws=[0.,np.pi],grasp_quality={'alternatives':[dict(center=[.48,0],yaw=0.,yaws=[0.])]})


def first_plan(e,point,delta=0.):
    return dict(center=e.center.copy(),place=np.r_[point,.08],destination_position=np.asarray(point).tolist(),
        yaw=0.,place_yaw=delta,tilt=.6,retreat_q=np.zeros(6))


def setup_bridge(monkeypatch):
    import piper_sim.traditional as traditional
    from run_harvest import FruitSkills
    monkeypatch.setattr(traditional,'IK',lambda *a,**k:NS(tilt=k['tilt']))
    skills=FruitSkills.__new__(FruitSkills);skills.navigation_events=[];skills.observed_fruits={};skills.b=object()
    skills.w=NS(cfg=dict(trays={'yellow':dict(center_xy_m=[.4,-.2],half_size_xy_m=[.15,.15])},tray_geometry={'floor_thickness_m':.0005}))
    skills.skill=NS(cfg={'planning_budget_s':20.,'planning_target':'banana'},table=.06,last_planning_report=None)
    e=source_estimate();e.destination_options=[np.array([.39,0])]
    t=Track('banana','banana',[.5,0,.08],'yellow',estimate=e);t.staging_leg=1;t.destination_position=np.array([.47,0]);t.staging_final={'position':np.array([.4,-.2])}
    return skills,t,e


def test_first_leg_is_authorized_without_computing_the_second_leg(monkeypatch):
    skills,t,e=setup_bridge(monkeypatch);calls=[]
    def plan(estimate,point,floor):
        assert not getattr(estimate,'staging_prediction',False)
        calls.append(np.asarray(point).tolist())
        return first_plan(estimate,point)
    skills.skill.plan=plan
    result=skills.plan_staging_leg(t,e)
    assert result['destination_position']==[.47,0.]
    assert calls==[[.47,0.]]
    assert any(n.get('event')=='single_leg_ready' for n in skills.navigation_events)


def test_failure_of_sweet_leg_switches_to_goalward_segments(monkeypatch):
    skills,t,e=setup_bridge(monkeypatch)
    skills.skill.plan=lambda *args:(_ for _ in ()).throw(RuntimeError('current leg not found'))
    with pytest.raises(RuntimeError):skills.plan_staging_leg(t,e)
    assert t.staging_mode=='progressive' and t.staging_refresh_needed
    assert not any(n.get('event')=='staging_bridge_rejected' for n in skills.navigation_events)


def test_pending_batch_result_and_log_never_claim_final_placement():
    class Skills:
        def overview(self):return [Track('a','banana',[.4,0,.08],'yellow')]
        def observe_target(self,t,retry):pass
        def pick(self,t):return True
        def place_and_verify(self,t):t.audit_pending=True;return True
    r=Harvest(Skills(),defer_batch_audit=True).run()
    assert r['status']=='released_pending_audit' and r['completed']==[] and r['released']==['a']
    placed=[e for e in r['events'] if 'placed' in e]
    assert placed[0]['placed'] is False and placed[0]['placement_state']=='released_pending_audit'


def test_batch_audit_confirms_released_targets_even_when_completed_is_empty():
    from piper_sim.batch_audit import BatchBoundaryAudit
    t=Track('a','banana',[.4,0,.08],'yellow')
    r=BatchBoundaryAudit(NS(scan_batch_boundaries=lambda ts:{'a':dict(status='inside')})).run([t],dict(status='released_pending_audit',completed=[],released=['a'],unresolved=[]))
    assert r['status']=='complete' and r['completed']==['a']


def test_actual_second_leg_failure_requests_another_bridge_instead_of_repeating_forever():
    from test_staging import TwoLegSkills
    class Skills(TwoLegSkills):
        failed=False
        def observe_target(self,t,retry):
            if getattr(t,'staging_leg',0)==2 and not self.failed:
                self.failed=True;raise __import__('piper_sim.harvest',fromlist=['RetryableFailure']).RetryableFailure('actual second pose changed',phase='planning')
            return super().observe_target(t,retry)
        def finish_staging(self,t):
            super().finish_staging(t);t.staging_hops=getattr(t,'staging_hops',0)+1
        def request_restaging(self,t,reason):t.staging_leg=0;t.staging_pending=True;return True
    skills=Skills();r=Harvest(skills,max_preparations_per_target=4).run()
    assert r['status']=='complete'
    assert skills.calls.count(('begin','hard'))==2
    assert any(e.get('event')=='staging_bridge_replanned' for e in r['events'])
    intermediate=[e for e in r['events'] if e.get('placement_state')=='intermediate_released']
    assert len(intermediate)==2 and all(e['placed'] is False for e in intermediate)


def test_old_source_frame_is_not_reused_just_because_ik_did_not_advance_sim_time(monkeypatch):
    import run_harvest as runtime
    skills=runtime.FruitSkills.__new__(runtime.FruitSkills)
    monkeypatch.setattr(runtime,'time',NS(monotonic=lambda:100.))
    skills.prepare_target=lambda *args:pytest.fail('must not reuse the old frame')
    skills.capture_candidates=lambda **kwargs:(_ for _ in ()).throw(RuntimeError('fresh sensor requested'))
    t=Track('banana','banana',[.5,0,.08],'yellow',estimate=source_estimate())
    t.planning_retry_ready=True;t.planning_observed_wall=10.;t.planning_observed_at=0.
    with pytest.raises(RuntimeError,match='fresh sensor'):skills.observe_target(t,retry=False)
