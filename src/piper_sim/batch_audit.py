"""End-of-batch visual boundary audit and bounded corrective pick/place rounds."""
import numpy as np


def boundary_of_estimate(estimate,tray):
    points=np.asarray(estimate.points)
    if len(points)<20:return dict(status='unknown',reason='可见轮廓点不足')
    low,high=np.percentile(points[:,:2],[1,99],axis=0)
    center=np.asarray(tray['center_xy_m']);half=np.asarray(tray['half_size_xy_m'])
    overflow=float(max(np.max(center-half-low),np.max(high-center-half)))
    if tray.get('placement_policy')=='center_max_coverage':
        geometric=(low+high)/2
        distance=float(np.min(half-abs(geometric-center)))
        return dict(status='inside' if distance>.003 else 'outside' if distance<-.004 else 'unknown',center=np.r_[geometric,np.median(points[:,2])].tolist(),low=low.tolist(),high=high.tolist(),overflow_m=max(0.,overflow),edge_clearance_m=max(0.,distance),policy='center_max_coverage',reason='' if distance>.003 else '几何中心接近或超出卡纸边界')
    return dict(status='outside' if overflow>.004 else 'inside' if overflow<-.003 else 'unknown',
        center=np.median(points,axis=0).tolist(),low=low.tolist(),high=high.tolist(),
        overflow_m=max(0.,overflow),edge_clearance_m=max(0.,-overflow),
        reason='边界附近误差未能排除' if -.003<=overflow<=.004 else '')


def identify_audit_target(target,observations,tray):
    candidates=[]
    source=np.asarray(getattr(getattr(target,'estimate',None),'points',np.empty((0,3))))
    reference=np.asarray(target.destination_position)
    extent=np.linalg.norm(np.ptp(source[:,:2],axis=0)) if len(source) else None
    for category,estimate in observations:
        if category!=target.category or len(estimate.points)<20:continue
        measured=np.linalg.norm(np.ptp(np.asarray(estimate.points)[:,:2],axis=0))
        if extent is not None and not .5*extent<=measured<=1.8*extent:continue
        center=np.median(estimate.points,axis=0)[:2]
        candidates.append((float(np.linalg.norm(center-reference)),len(estimate.points),estimate))
    if not candidates:return dict(status='unknown',reason='未从新图像识别到该目标的完整轮廓')
    candidates.sort(key=lambda o:(o[0],-o[1]))
    anchor=candidates[0][2]
    cluster=[o for o in candidates if np.linalg.norm(np.median(o[2].points,axis=0)[:2]-np.median(anchor.points,axis=0)[:2])<.07]
    clustered={id(o[2]) for o in cluster}
    others=[o for o in candidates if id(o[2]) not in clustered]
    if others and abs(others[0][0]-candidates[0][0])<.06:
        return dict(status='unknown',reason='同类识别出现空间歧义，禁止根据猜测调整')
    estimate=max(cluster,key=lambda o:o[1])[2]
    return dict(boundary_of_estimate(estimate,tray),estimate=estimate,category=target.category)


def public_status(status):
    return {k:v for k,v in status.items() if k!='estimate'}


class BatchBoundaryAudit:
    def __init__(self,skills,max_correction_rounds=2):
        self.skills=skills;self.max_correction_rounds=max_correction_rounds

    def run(self,targets,result):
        # Repeated instructions for one object are checked against the LAST goal.
        released_order=list(dict.fromkeys(result.get('released',result.get('completed',[]))))
        released=set(released_order);unique={t.category:t for t in targets if t.key in released}
        targets=list(unique.values())
        if not targets:return result
        history=[];corrections=[];states={};unsafe=False
        import itertools
        persistent=getattr(self.skills,'keep_retrying',False)
        for round_index in (itertools.count() if persistent else range(self.max_correction_rounds+1)):
            states=self.skills.scan_batch_boundaries(targets)
            history.append(dict(round=round_index,objects={k:public_status(v) for k,v in states.items()}))
            outside=[t for t in targets if states.get(t.key,{}).get('status')=='outside']
            unknown=[t for t in targets if states.get(t.key,{}).get('status')=='unknown']
            if not outside:
                if persistent and unknown:
                    self.skills.wait_before_retry(unknown[0],'boundary_observation')
                    continue
                break
            if not persistent and round_index==self.max_correction_rounds:break
            moved=False
            for target in outside:
                repair=self.skills.correct_boundary(target,states[target.key])
                corrections.append(dict(target=target.key,**repair))
                moved=moved or repair.get('motion_attempted',True)
                if not repair.get('safe_to_scan',True):
                    states[target.key]=dict(status='unknown',reason='纠偏后尚未安全释放，保持未完成')
                    unsafe=True;break
            if unsafe:break
            if not moved:
                if persistent:
                    self.skills.wait_before_retry(outside[0],'boundary_correction')
                    continue
                break
        confirmed={t.key for t in targets if states.get(t.key,{}).get('status')=='inside'}
        uncertain={t.key for t in targets}-confirmed
        result=dict(result)
        result['released']=sorted(released)
        result['completed']=[key for key in released_order if key in confirmed]
        result['unresolved']=list(dict.fromkeys(result.get('unresolved',[])+sorted(uncertain)))
        result['status']='complete' if not result['unresolved'] else 'incomplete'
        result['stop_reason']='batch_boundaries_confirmed' if result['status']=='complete' else 'batch_boundary_or_execution_unresolved'
        result['attempts']=result.get('attempts',0)+sum(c.get('attempts',0) for c in corrections)
        blockers=dict(result.get('blockers',{}))
        for key in uncertain:blockers[key]=states.get(key,{}).get('reason') or '最终检查仍有越界物品'
        result['blockers']=blockers
        result['boundary_verification']='confirmed' if not uncertain else 'unresolved'
        result['boundary_audit']=dict(rounds=history,corrections=corrections,confirmed=sorted(confirmed),unresolved=sorted(uncertain))
        for t in targets:
            t.audit_pending=t.key not in confirmed
            t.boundary_check_status=states.get(t.key,{}).get('status','unknown')
        return result
