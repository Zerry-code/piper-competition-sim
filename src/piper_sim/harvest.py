"""Sensor-only orchestration; evaluator truth must never be supplied as targets."""
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol
from piper_sim.drop_recovery import ObjectDropped


class RetryableFailure(RuntimeError):
    """A recoverable preparation/transfer failure, distinct from an empty closure."""
    def __init__(self,message,phase='observation'):
        super().__init__(message)
        self.phase=phase


@dataclass
class Track:
    key: str
    category: str
    position: list
    destination: str
    attempts: int = 0
    deferrals: int = 0
    estimate: object = None


class Skills(Protocol):
    def overview(self) -> list[Track]: ...
    def observe_target(self, target: Track, retry: bool): ...
    def pick(self, target: Track) -> bool: ...
    def place_and_verify(self, target: Track) -> bool: ...
    def recover(self, target: Track): ...


@dataclass
class Harvest:
    skills: Skills
    # Robot base +X forward, +Y left: rightmost is smallest base Y.
    right_axis: tuple = (0., -1., 0.)
    max_attempts: int | None = 60
    max_deferrals: int | None = None
    max_preparations_per_target: int | None = None
    events: list = field(default_factory=list)
    defer_staging: bool = False
    defer_batch_audit: bool = False

    def run(self):
        targets = self.skills.overview()
        self.targets=targets
        register=getattr(self.skills,'register_targets',None)
        if callable(register):register(targets)
        if len({t.key for t in targets}) != len(targets):
            raise ValueError("Overview track IDs must be unique")
        queue = deque(sorted(targets, key=lambda t: -sum(
            a*b for a, b in zip(self.right_axis, t.position))))
        initial_order = [t.key for t in queue]
        completed,unresolved,total,grasp_total=[],[],0,0
        staging_queue=deque()
        self.completed=completed
        def within_budget():return getattr(self.skills,'keep_retrying',False) or self.max_attempts is None or total<self.max_attempts
        def recover_safely(target):
            target.recovery_pending=True
            try:self.skills.recover(target)
            except (RuntimeError,TimeoutError) as exc:
                target.recovery_errors=getattr(target,'recovery_errors',0)+1
                target.last_recovery_error=str(exc)
                self.events.append(dict(target=target.key,event='retained_recovery_retry',error=str(exc)))
                reporter=getattr(self.skills,'report_recovery_failure',None)
                if reporter:reporter(target,str(exc))
                wait=getattr(self.skills,'wait_before_retry',None)
                if wait:wait(target,'retained_recovery')
                return False
            target.recovery_pending=False;target.recovery_errors=0
            return True
        def accept_placement(target):
            if getattr(target,'staging_leg',0)==1:
                final_landed=getattr(target,'staging_final_landed',False)
                self.skills.finish_staging(target)
                self.events.append(dict(target=target.key,event='staging_first_leg_complete',final_landed=final_landed))
                if not final_landed:return False
            completed.append(target.key)
            return True
        while (queue or staging_queue) and within_budget():
            if not queue and self.defer_staging:break
            from_staging=not queue
            target=(staging_queue if from_staging else queue).popleft()
            if not getattr(self.skills,'persistent_preparation',False) and self.max_preparations_per_target is not None and getattr(target,'preparation_total',0)>=self.max_preparations_per_target:
                unresolved.append(target.key)
                self.events.append(dict(target=target.key,event='preparation_limit_reached',error=getattr(target,'last_preparation_error','')))
                continue
            streak=0
            prep_failures=0
            placed=False
            defer_reason='budget'
            while streak<3 and within_budget():
                total+=1
                held=False
                if getattr(target,'recovery_pending',False):
                    if getattr(target,'recovery_errors',0)>=4 and not getattr(self.skills,'keep_retrying',False):
                        pending=[target.key]+[t.key for t in queue]+[t.key for t in staging_queue]+unresolved
                        return dict(status='incomplete',stop_reason='retained_object_recovery_blocked',released=list(completed),completed=[k for k in completed if not any(t.key==k and getattr(t,'audit_pending',False) for t in targets)],boundary_verification='pending_batch',unresolved=pending,attempts=grasp_total,events=self.events,blockers={target.key:target.last_recovery_error})
                    if not recover_safely(target):continue
                if getattr(target,'release_pending',False):
                    try:placed=bool(self.skills.resume_release(target))
                    except (RuntimeError,TimeoutError) as exc:
                        target.release_retries=getattr(target,'release_retries',0)+1
                        self.events.append(dict(target=target.key,event='release_clearance_retry',error=str(exc),retry=target.release_retries))
                        if target.release_retries>=3 and not getattr(self.skills,'keep_retrying',False):
                            pending=list(dict.fromkeys([target.key]+[t.key for t in queue]+[t.key for t in staging_queue]+unresolved))
                            return dict(status='incomplete',stop_reason='release_clearance_blocked',
                                completed=[k for k in completed if not any(t.key==k and getattr(t,'audit_pending',False) for t in targets)],
                                released=list(completed),unresolved=pending,boundary_verification='pending_batch',
                                attempts=grasp_total,events=self.events,blockers={target.key:str(exc)})
                        wait=getattr(self.skills,'wait_before_retry',None)
                        if wait:wait(target,'release_clearance')
                        continue
                    target.release_retries=0
                    if placed:
                        placed=accept_placement(target)
                        if placed:break
                        continue
                    boundary=getattr(target,'boundary_check_status',None)
                    if boundary=='unknown':
                        if getattr(self.skills,'keep_retrying',False):
                            target.release_pending=True
                            continue
                        unresolved.append(target.key)
                        self.events.append(dict(target=target.key,event='boundary_unchecked_continue'))
                        break
                    target.drop_search_pending=boundary!='outside'
                if getattr(target,'drop_search_pending',False):
                    try:outcome=self.skills.recover_dropped(target)
                    except (RuntimeError,TimeoutError) as exc:
                        outcome='unseen'
                        self.events.append(dict(target=target.key,event='drop_recovery_blocked',error=str(exc)))
                    self.events.append(dict(target=target.key,event='drop_recovery',outcome=outcome))
                    if outcome=='placed':
                        target.drop_search_pending=False
                        placed=accept_placement(target)
                        if placed:break
                        continue
                    if outcome=='unseen':
                        defer_reason='drop_target_not_found';break
                    target.drop_search_pending=False
                    streak=0;prep_failures=0
                try:
                    if getattr(target,'staging_pending',False):self.skills.begin_staging(target)
                    self.skills.observe_target(target,retry=target.attempts>0 or prep_failures>0)
                except RetryableFailure as exc:
                    prep_failures+=1
                    target.preparation_total=getattr(target,'preparation_total',0)+1
                    target.last_preparation_error=str(exc)
                    reporter=getattr(self.skills,'report_preparation_failure',None)
                    if reporter:reporter(target,str(exc),getattr(exc,'phase','observation'))
                    phase=getattr(exc,'phase','observation')
                    self.events.append(dict(target=target.key,event='preparation_failed',phase=phase,
                                            error=str(exc),gripper_closed=False))
                    defer_reason='preparation_unavailable'
                    if phase=='planning' and getattr(self.skills,'staging_enabled',False):
                        if not getattr(target,'staging_used',False):
                            target.staging_pending=True
                            self.events.append(dict(target=target.key,event='staging_deferred',reason='直接抓放搜索未找到方案，留到普通目标之后；整体可行性未知'))
                        elif getattr(target,'staging_leg',0)==2:
                            restage=getattr(self.skills,'request_restaging',None)
                            if callable(restage) and restage(target,str(exc)):
                                self.events.append(dict(target=target.key,event='staging_bridge_replanned',hops=getattr(target,'staging_hops',0),reason='本段未通过，继续单段向目标推进'))
                    if getattr(self.skills,'keep_retrying',False) and target.preparation_total%5==0:
                        target.rejected_points=[];target.rejected_regions=[];target.rejected_grasps=[]
                        target.local_view_failures=getattr(target,'local_view_failures',0)+1
                        self.events.append(dict(target=target.key,event='fresh_search_cycle',reason='新观察周期重新搜索，历史失败记录保留'))
                    if phase in ('planning','staging') or prep_failures>=2:break
                    continue
                target.attempts+=1
                try:held=bool(self.skills.pick(target))
                except ObjectDropped as exc:
                    if getattr(self.skills,'last_attempt_closed',True):grasp_total+=1
                    target.drop_search_pending=True
                    self.events.append(dict(target=target.key,event='object_dropped',phase=exc.stage,gripper_closed=True))
                    continue
                closed=bool(getattr(self.skills,'last_attempt_closed',True))
                if not closed:
                    target.attempts-=1
                    prep_failures+=1
                    self.events.append(dict(target=target.key,event='approach_failed',gripper_closed=False,
                        error=getattr(self.skills,'last_pick_error','Approach did not reach closure')))
                    if not recover_safely(target):continue
                    defer_reason='approach_unavailable'
                    if prep_failures>=2:break
                    continue
                grasp_total+=1
                try:placed=bool(held and self.skills.place_and_verify(target))
                except ObjectDropped as exc:
                    target.drop_search_pending=True
                    self.events.append(dict(target=target.key,event='object_dropped',phase=exc.stage,gripper_closed=True))
                    continue
                except RetryableFailure as exc:
                    placed=False
                    self.events.append(dict(target=target.key,event='transfer_failed',phase='placement',error=str(exc)))
                placement_state=('released_without_boundary_audit' if getattr(self.skills,'skip_boundary_checks',False) and placed else
                    'intermediate_released' if getattr(target,'staging_leg',0)==1 else
                    'released_pending_audit' if getattr(target,'audit_pending',False) else 'final_verified') if placed else 'not_placed'
                self.events.append(dict(target=target.key,attempt=target.attempts,gripper_closed=True,
                                        held=held,released=placed,placed=bool(placed and placement_state=='final_verified'),
                                        placement_state=placement_state))
                if placed:
                    placed=accept_placement(target)
                    if placed:break
                    streak=0;prep_failures=0
                    continue
                if getattr(target,'release_pending',False):
                    self.events.append(dict(target=target.key,event='release_retry_in_place'))
                    continue
                if held and getattr(self.skills,'last_release_completed',False) and getattr(target,'boundary_check_status',None)=='unknown':
                    if getattr(self.skills,'keep_retrying',False):
                        target.release_pending=True
                        continue
                    unresolved.append(target.key)
                    self.events.append(dict(target=target.key,event='boundary_unchecked_continue'))
                    break
                if held and getattr(self.skills,'last_release_completed',False) and getattr(target,'boundary_check_status',None)=='outside':
                    self.events.append(dict(target=target.key,event='boundary_outside_local_retry'))
                    continue
                if held and getattr(self.skills,'last_release_completed',False):
                    target.drop_search_pending=True
                    self.events.append(dict(target=target.key,event='placement_unconfirmed_search'))
                    continue
                if not recover_safely(target):continue
                if held:
                    defer_reason='placement_failed'
                    break
                streak+=1
                defer_reason='three_failed_closures' if streak>=3 else 'grasp_failed'
            if not placed:
                if target.key in unresolved:continue
                if streak>=3 and getattr(self.skills,'keep_retrying',False):
                    target.rejected_points=[];target.rejected_regions=[];target.rejected_grasps=[]
                    target.local_view_failures=getattr(target,'local_view_failures',0)+1
                target.deferrals+=1
                if not getattr(self.skills,'persistent_preparation',False) and self.max_deferrals is not None and target.deferrals>=self.max_deferrals:unresolved.append(target.key)
                else:
                    (staging_queue if getattr(target,'staging_pending',False) or getattr(target,'staging_used',False) else queue).append(target)
                    self.events.append(dict(target=target.key,event='deferred_to_tail',reason=defer_reason,
                                            failed_closures=streak,preparation_failures=prep_failures))
                    # Poll the clock/controls even if a failed plan made no motion.
                    wait=getattr(self.skills,'wait_before_retry',None)
                    if wait is not None:wait(target,defer_reason)
        deferred=[t.key for t in staging_queue] if self.defer_staging else []
        unresolved += [t.key for t in queue]+[t.key for t in staging_queue]
        result=dict(status='staging_deferred' if deferred and set(unresolved)==set(deferred) else 'complete' if not unresolved else 'incomplete',staging_deferred=deferred,initial_order=initial_order,
                    stop_reason=('all_targets_released_without_boundary_audit' if getattr(self.skills,'skip_boundary_checks',False) else 'all_targets_confirmed') if not unresolved else 'explicit_retry_budget',
                    released=list(completed),completed=[k for k in completed if not any(t.key==k and getattr(t,'audit_pending',False) for t in targets)],unresolved=unresolved,attempts=grasp_total,
                    preparation_cycles=total,events=self.events,
                    blockers={t.key:getattr(t,'last_preparation_error','抓取或落区尚未确认') for t in targets if t.key in unresolved})

        finalize=getattr(self.skills,'finalize_batch',None)
        if callable(finalize) and not self.defer_batch_audit and not deferred:
            result=finalize(targets,result)
            self.completed=result.get('completed',[])
        elif any(getattr(t,'audit_pending',False) for t in targets):
            result['boundary_verification']='pending_batch'
            if result['status']=='complete':
                result['status']='released_pending_audit'
                result['stop_reason']='all_targets_released_pending_batch_audit'
        return result
