"""Two items per paper: compare centre-line and diagonal pair partitions."""
import itertools
import numpy as np

def select_round_tracks(tracks,task,seed):
    explicit={s['source'] for s in task.get('steps',[]) if s['source'] not in ('all','remaining')}
    chosen=[];omitted=[]
    for index,color in enumerate(['red','green','yellow']):
        group=[t for t in tracks if t.destination==color]
        named=[t for t in group if t.category in explicit]
        if len(named)>2:raise ValueError('同一区域明确指定超过两件，请调整指令：'+color)
        rest=[t for t in group if t.category not in explicit]
        rng=np.random.default_rng(np.random.SeedSequence([seed,index,932]))
        rng.shuffle(rest)
        selected=named+rest[:max(0,2-len(named))]
        chosen.extend(selected);omitted.extend(t.category for t in group if t not in selected)
    if len(chosen)>6:raise ValueError('每轮最多执行六件')
    if len(tracks)>=6 and len(chosen)!=6:raise ValueError('整轮需要三色区域各两件，请调整分组；不会擅自改写目标颜色')
    return chosen,omitted


def allocate_pairs(trays,tracks,observed,anchors):
    from piper_sim.fruit_placement import visual_footprint,_cloud
    result={}
    for color in sorted({t.destination for t in tracks}):
        group=[t for t in tracks if t.destination==color];tray=trays[color]
        if len(group)>2:raise ValueError('每张卡纸最多两个放置位：'+color)
        center=np.asarray(tray['center_xy_m']);half=np.asarray(tray['half_size_xy_m'])
        cloud=_cloud(observed,{t.category for t in group});best=None
        axes=[('中线-前后',np.array([1.,0.])),('中线-左右',np.array([0.,1.])),('对角线-正',np.array([1.,1.])),('对角线-反',np.array([1.,-1.]))]
        shapes={t.key:visual_footprint(t.estimate,t.category) for t in group}
        angle_options=[[0.,np.pi/2,np.pi/4,-np.pi/4] if shapes[t.key].kind in ('three_circles','ellipse','capsule','curved_outline','curved_sections','rounded_rectangle','oriented_rectangle') else [0.] for t in group]
        for name,axis in axes:
            for spread in ([0.] if len(group)==1 else ([.5,.46,.54,.42,.58] if np.count_nonzero(axis)==1 else [1/3,.29,.37,.25,.41])):
                for swap in ([1] if len(group)==1 else [1,-1]):
                    points=[center+axis*half*spread*swap*(1 if i==0 else -1) for i in range(len(group))]
                    for angles in itertools.product(*angle_options):
                        rotated=[shapes[t.key].rotated(angle) for t,angle in zip(group,angles)]
                        if len(group)==2 and rotated[0].overlaps(points[0],rotated[1],points[1]):continue
                        if any(sh.hits_points(p,cloud) for sh,p in zip(rotated,points)):continue
                        cover=[sh.coverage(tray,p) for sh,p in zip(rotated,points)]
                        # Prefer coverage and easy reach; both slots share the paper symmetrically.
                        ideal=0. if len(group)==1 else (.5 if np.count_nonzero(axis)==1 else 1/3)
                        center_penalty=20*sum(np.linalg.norm(p-(center+axis*half*ideal*swap*(1 if i==0 else -1))) for i,p in enumerate(points))
                        reach=sum((1+8*float(np.min(np.ptp(getattr(sh,'vertices',sh.offsets),axis=0))))*max(0.,np.linalg.norm(p)-.60) for sh,p in zip(rotated,points))
                        score=center_penalty+sum(1-c for c in cover)+3*reach
                        if best is None or score<best[0]:best=(score,points,rotated,angles,name,axis)
        if best is None:raise RuntimeError('两件物品的中线/对角线分区暂未找到不重叠方案：'+color)
        _,points,rotated,angles,name,axis=best
        for t,p,sh,angle in zip(group,points,rotated,angles):
            result[t.key]=p;t.placement_footprint=sh;t.placement_rotation=angle
            t.placement_margin_m=sh.clearance(tray,p)
            if len(group)==2:t.placement_partition=dict(scheme=name,axis=axis.tolist(),sign=1 if np.dot(p-center,axis)>=0 else -1,center=(center+axis*half*(.5 if np.count_nonzero(axis)==1 else 1/3)*(1 if np.dot(p-center,axis)>=0 else -1)).tolist(),max_center_offset_m=.025)
    return result
