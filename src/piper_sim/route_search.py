"""Bounded Cartesian detours and seeded, quality-first grasp ordering."""
import numpy as np
from piper_sim.planning_trace import explain

def ordered_candidates(candidates,current,rng):
    # Scores are geometry heuristics, not learned success probabilities.
    for c in candidates:
        c.selection_score=float(getattr(c,'candidate_score',1.))-.15*float(np.linalg.norm(c.center-current))-3*max(0.,float(np.linalg.norm(c.center[:2]))-.50)
    pending=sorted(candidates,key=lambda c:(getattr(c,'priority_tier',0),-c.selection_score));result=[]
    while pending:
        best=pending[0].selection_score
        tied=[c for c in pending if getattr(c,'priority_tier',0)==getattr(pending[0],'priority_tier',0) and best-c.selection_score<=.03*max(1,abs(best))]
        for i in rng.permutation(len(tied)):result.append(tied[i])
        ids={id(c) for c in tied};pending=[c for c in pending if id(c) not in ids]
    return result

def transfer_route(segment,start_q,current,destination,yaw,validate=lambda points:None,progress=None):
    """Prefer straight, then bounded raised/side bridges. No commands until validated."""
    current=np.array(current);destination=np.array(destination)
    high=max(.24,float(current[2]),float(destination[2]))
    midpoint=(current+destination)/2
    recipes=[[]]
    delta=destination[:2]-current[:2];length=np.linalg.norm(delta)
    normal=np.array([-delta[1],delta[0]])/max(length,1e-6)
    for offset in [.045,-.045,.08,-.08]:
        side=midpoint.copy();side[:2]+=normal*offset
        recipes.append([side])
    for height in [high,min(.32,high+.04)]:
        for offset in [0.,.045,-.045]:
            delta=destination[:2]-current[:2];length=np.linalg.norm(delta)
            side=np.array([-delta[1],delta[0]])/max(length,1e-6)
            middle=midpoint.copy();middle[:2]+=side*offset;middle[2]=height
            a=current.copy();a[2]=height;b=destination.copy();b[2]=height
            recipes.append([a,middle,b])
    errors=[]
    for route_index,waypoints in enumerate(recipes):
        label='直线' if not waypoints else ('同高度侧向绕行' if len(waypoints)==1 else '抬高后过渡绕行')+' '+str(route_index)
        if progress:progress('route',route_index=route_index,route_label=label,waypoints=[p.tolist() for p in waypoints])
        points=[np.array(start_q,copy=True)];position=current
        try:
            for goal in waypoints+[destination]:
                if np.linalg.norm(goal-position)<1e-6:continue
                path=segment(points[-1],goal,yaw,current=position)
                points.extend(path[1:]);position=goal
            validate(points)
            return points,[p.tolist() for p in waypoints]
        except RuntimeError as exc:
            if progress:progress('route_failure',route_index=route_index,route_label=label,**explain(exc))
            errors.append(str(exc))
    raise RuntimeError('直线及有限绕行均未通过：'+errors[-1])


def joint_space_route(start,goal,midpoints,validate,progress=None):
    """Bounded joint-space bridges when a fixed-orientation Cartesian line fails."""
    recipes=[[]]+[[np.asarray(mid)] for mid in midpoints]
    errors=[]
    for index,mids in enumerate(recipes):
        if progress:progress('route',route_index=100+index,route_label='关节空间连续绕行 '+str(index),waypoints=[])
        points=[np.asarray(start).copy()]
        for q in mids+[np.asarray(goal)]:
            steps=max(2,int(np.ceil(np.max(abs(q-points[-1]))/.065)))
            points.extend(np.linspace(points[-1],q,steps+1)[1:])
        try:validate(points);return points
        except RuntimeError as exc:
            errors.append(str(exc))
            if progress:progress('route_failure',route_index=100+index,route_label='关节空间连续绕行 '+str(index),**explain(exc))
    raise RuntimeError('关节空间候选暂未通过：'+(errors[-1] if errors else '无候选'))


def ready_position_route(start,goal,ready_candidates,validate,progress=None):
    """Require a ready waypoint: no hidden direct route and no intermediate release."""
    errors=[]
    for index,ready in enumerate(ready_candidates):
        if progress:progress('route',route_index=200+index,route_label='持物回准备位置再前往目标',waypoints=[])
        points=[np.asarray(start).copy()]
        for q in [np.asarray(ready),np.asarray(goal)]:
            steps=max(2,int(np.ceil(np.max(abs(q-points[-1]))/.025)))
            points.extend(np.linspace(points[-1],q,steps+1)[1:])
        try:validate(points);return points
        except RuntimeError as exc:errors.append(str(exc))
    raise RuntimeError('准备姿态路线本轮未通过，保留目标继续重试：'+(errors[-1] if errors else '准备位置IK未找到'))
