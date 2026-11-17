"""Single-frame local boundary inspection; no identity reclassification or stability timer."""
import numpy as np

def inspect_boundary(candidates,landing,tray,source_points=None,reservations=None,key=None):
    landing=np.asarray(landing);reservations=reservations or {}
    source=np.asarray(source_points) if source_points is not None else np.empty((0,3))
    extent=float(np.linalg.norm(np.ptp(source[:,:2],axis=0))) if len(source) else .08
    choices=[]
    for estimate in candidates:
        points=np.asarray(estimate.points)
        if len(points)<20:continue
        center=np.median(points[:,:2],axis=0);distance=float(np.linalg.norm(center-landing))
        if distance>max(.10,extent/2+.04):continue
        if any(k!=key and np.linalg.norm(center-np.asarray(p))+.02<distance for k,p in reservations.items()):continue
        measured=float(np.linalg.norm(np.ptp(points[:,:2],axis=0)))
        if len(source) and not .5*extent<=measured<=1.8*extent:continue
        choices.append((distance,points))
    choices.sort(key=lambda c:c[0])
    if not choices or len(choices)>1 and choices[1][0]-choices[0][0]<.015:
        return dict(status='unknown',reason='局部轮廓不完整或存在关联歧义')
    points=choices[0][1];low,high=np.percentile(points[:,:2],[1,99],axis=0)
    if tray.get('placement_policy')=='center_max_coverage':
        from piper_sim.batch_audit import boundary_of_estimate
        from types import SimpleNamespace
        return boundary_of_estimate(SimpleNamespace(points=points),tray)
    center=np.array(tray['center_xy_m']);half=np.array(tray['half_size_xy_m'])
    overflow=float(max(np.max(center-half-low),np.max(high-center-half)))
    status='outside' if overflow>.004 else 'inside' if overflow<-.003 else 'unknown'
    return dict(status=status,low=low.tolist(),high=high.tolist(),center=np.median(points,axis=0).tolist(),overflow_m=max(0.,overflow),edge_clearance_m=max(0.,-overflow),reason='边界附近误差未能排除' if status=='unknown' else '')
