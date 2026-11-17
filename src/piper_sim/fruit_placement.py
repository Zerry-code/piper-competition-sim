"""Visual footprint unions for collision-free tabletop reservations (metres)."""
from dataclasses import dataclass
import numpy as np

EDGE_MARGINS=(.05,.045,.04,.035,.03,.02,.01,.006)


def footprint_radius(estimate):
    points=np.asarray(estimate.points)[:,:2]
    fallback=np.median(points,axis=0)
    center=np.asarray(getattr(estimate,'object_center',getattr(estimate,'center',fallback)))[:2]
    return max(.025,float(np.max(np.linalg.norm(points-center,axis=1))))+.006


@dataclass
class Footprint:
    offsets: np.ndarray
    radii: np.ndarray
    kind: str='circle'

    def rotated(self,angle):
        c,s=np.cos(angle),np.sin(angle)
        return Footprint(self.offsets@np.array([[c,s],[-s,c]]),self.radii.copy(),self.kind)

    def coverage(self,tray,point):
        low=np.min(self.offsets-self.radii[:,None],axis=0);high=np.max(self.offsets+self.radii[:,None],axis=0)
        x,y=np.meshgrid(np.linspace(low[0],high[0],23),np.linspace(low[1],high[1],23))
        samples=np.c_[x.ravel(),y.ravel()]
        body=np.any(np.linalg.norm(samples[:,None,:]-self.offsets,axis=2)<=self.radii,axis=1)
        inside=np.all(abs(samples+point-tray['center_xy_m'])<=tray['half_size_xy_m'],axis=1)
        return float(np.sum(body&inside)/max(1,body.sum()))

    def clearance(self,tray,point):
        if tray.get('placement_policy')=='center_max_coverage':return float(np.min(np.asarray(tray['half_size_xy_m'])-abs(np.asarray(point)-tray['center_xy_m'])))
        centers=np.asarray(point)+self.offsets
        return float(np.min(np.asarray(tray['half_size_xy_m'])-abs(centers-tray['center_xy_m'])-self.radii[:,None]))

    def bounds(self,tray,margin):
        if tray.get('placement_policy')=='center_max_coverage':
            c=np.asarray(tray['center_xy_m']);h=np.asarray(tray['half_size_xy_m'])
            return np.maximum(c-h+.002,[.20,-np.inf]),np.minimum(c+h-.002,[.95,np.inf])
        low=np.asarray(tray['center_xy_m'])-tray['half_size_xy_m']+margin
        high=np.asarray(tray['center_xy_m'])+tray['half_size_xy_m']-margin
        return (np.maximum(np.max(low-self.offsets+self.radii[:,None],axis=0),[.20,-np.inf]),
                np.minimum(np.min(high-self.offsets-self.radii[:,None],axis=0),[.95,np.inf]))

    def overlaps(self,point,other,other_point):
        if hasattr(other,'vertices'):return other.overlaps(other_point,self,point)
        delta=(np.asarray(point)+self.offsets)[:,None,:]-(np.asarray(other_point)+other.offsets)[None,:,:]
        return bool(np.any(np.linalg.norm(delta,axis=2)<self.radii[:,None]+other.radii[None,:]-1e-10))

    def hits_points(self,point,cloud):
        if not len(cloud):return False
        return bool(np.any(np.linalg.norm(cloud[None,:,:]-(np.asarray(point)+self.offsets)[:,None,:],axis=2)<self.radii[:,None]+.006))

    def record(self):
        return dict(kind=self.kind,offsets_m=self.offsets.tolist(),radii_m=self.radii.tolist())


def visual_footprint(estimate,category=None):
    """Fit a category-appropriate planar geometry from visible RGB-D points."""
    known=('apple','pear','banana','cucumber','potato','bell_pepper','persimmon','cube_red','cube_green','cube_yellow','cube_blue')
    if category in known and len(estimate.points)>=3:
        from piper_sim.geometric_footprint import estimate_geometry
        return estimate_geometry(estimate,category)
    return Footprint(np.zeros((1,2)),np.array([footprint_radius(estimate)]))


def edge_clearance(tray,point,radius):
    if isinstance(radius,Footprint):return radius.clearance(tray,point)
    return float(np.min(np.array(tray['half_size_xy_m'])-abs(np.asarray(point)-tray['center_xy_m']))-radius)


def _cloud(observed,moving):
    clouds=[np.asarray(e.points)[:,:2] for category,e in observed.items() if category not in moving]
    return np.vstack(clouds) if clouds else np.empty((0,2))


def allocate_fruit_positions(trays,tracks,observed,anchors):
    if tracks and all(trays[t.destination].get('allocation_mode')=='pair_partition' for t in tracks):
        from piper_sim.pair_placement import allocate_pairs
        return allocate_pairs(trays,tracks,observed,anchors)
    result={};shapes={t.key:visual_footprint(t.estimate,t.category) for t in tracks}
    for t in tracks:
        if np.max(np.ptp(np.asarray(t.estimate.points)[:,:2],axis=0))>.30:
            raise RuntimeError('果蔬 '+t.category+' 的视觉轮廓异常，可能并入卡纸或邻近物体；需要重新观察，不能据此判断卡纸放不下')
    for color in sorted({t.destination for t in tracks}):
        group=sorted([t for t in tracks if t.destination==color],key=lambda t:-float(np.sum(shapes[t.key].radii**2)))
        tray=trays[color];obstacles=_cloud(observed,{t.category for t in group});found=False
        for margin in ((0.,) if tray.get('placement_policy')=='center_max_coverage' else EDGE_MARGINS):
            choices={}
            for t in group:
                base=shapes[t.key];options=[]
                angles=[0.,np.pi/2,-np.pi/2,np.pi/4,-np.pi/4] if base.kind in ('three_circles','ellipse','capsule','curved_outline','curved_sections','rounded_rectangle','oriented_rectangle') else [0.]
                for angle in angles:
                    shape=base.rotated(angle);low,high=shape.bounds(tray,margin)
                    if np.any(low>high):continue
                    candidates=[np.array([x,y]) for x in np.linspace(low[0],high[0],15) for y in np.linspace(low[1],high[1],15) if np.hypot(x,y)<=.95]
                    candidates.sort(key=lambda p:(-shape.coverage(tray,p),-shape.clearance(tray,p),float(np.linalg.norm(p-np.asarray(anchors[color])))))
                    if len(group)>1 and candidates and tray.get('placement_policy')!='center_max_coverage':
                        landmarks=[candidates[0],low,np.array([low[0],high[1]]),high,np.array([high[0],low[1]])]
                        seeds=[min(candidates,key=lambda p:np.linalg.norm(p-goal)) for goal in landmarks]
                        candidates=seeds+candidates
                    options.extend((p,shape,angle) for p in candidates if not shape.hits_points(p,obstacles))
                if tray.get('placement_policy')=='center_max_coverage':options.sort(key=lambda o:(-o[1].coverage(tray,o[0]),np.linalg.norm(o[0]-tray['center_xy_m'])))
                choices[t.key]=options
            if any(not choices[t.key] for t in group):continue
            reserved=[];budget=[25000];selected={};best=[float('inf'),None,None]
            centered=tray.get('placement_policy')=='center_max_coverage'
            costs={}
            for t in group:
                for point,shape,angle in choices[t.key]:
                    costs[(t.key,tuple(point),angle)]=(2*(1-shape.coverage(tray,point))+np.linalg.norm(point-np.asarray(tray['center_xy_m']))+3*max(0.,.035-shape.clearance(tray,point))+2*max(0.,np.linalg.norm(point)-.60)) if centered else 0.
                if centered:choices[t.key].sort(key=lambda o:costs[(t.key,tuple(o[0]),o[2])])
            def search(index,cost=0.):
                if index==len(group):
                    if cost<best[0]:best[:]=[cost,dict(result),dict(selected)]
                    return not centered
                t=group[index]
                for point,shape,angle in choices[t.key]:
                    budget[0]-=1
                    if budget[0]<0:return False
                    candidate_cost=cost+costs[(t.key,tuple(point),angle)]
                    if candidate_cost>=best[0]:continue
                    if any(shape.overlaps(point,old_shape,old) for old,old_shape in reserved):continue
                    reserved.append((point,shape));result[t.key]=point;selected[t.key]=(shape,angle)
                    if search(index+1,candidate_cost):return True
                    reserved.pop();result.pop(t.key,None)
                return False
            search(0)
            if best[1] is not None:
                result.update(best[1]);selected=best[2]
                found=True
                for t in group:
                    t.placement_footprint,t.placement_rotation=selected[t.key]
                    t.placement_margin_m=t.placement_footprint.clearance(tray,result[t.key])
                break
        if not found:raise RuntimeError('卡纸 '+color+' 当前视觉估计下未找到不重叠落点（没有数量上限，不支持叠放）')
    return result


def alternative_landings(trays,tracks,observed,positions):
    shapes={t.key:getattr(t,'placement_footprint',visual_footprint(t.estimate,t.category)) for t in tracks};result={}
    cloud=_cloud(observed,{o.category for o in tracks})
    for t in tracks:
        tray=trays[t.destination];shape=shapes[t.key]
        margin=max(0.,min(.05,shape.clearance(tray,positions[t.key])))
        low,high=shape.bounds(tray,margin);candidates=[]
        for x in np.linspace(low[0],high[0],9):
            for y in np.linspace(low[1],high[1],9):
                p=np.array([x,y])
                if not (.20<=x<=.95 and np.linalg.norm(p)<=.95):continue
                if tray.get('placement_policy')=='center_max_coverage' and shape.clearance(tray,p)<.002-1e-8:continue
                partition=getattr(t,'placement_partition',None)
                if partition and np.dot(p-np.asarray(tray['center_xy_m']),np.asarray(partition['axis']))*partition['sign']<.01:continue
                if partition and 'center' in partition and np.linalg.norm(p-np.asarray(partition['center']))>partition.get('max_center_offset_m',.025):continue
                if any(other.key!=t.key and shape.overlaps(p,shapes[other.key],positions[other.key]) for other in tracks):continue
                if shape.hits_points(p,cloud):continue
                candidates.append(p)
        anchor=np.asarray(getattr(t,'placement_partition',{}).get('center',positions[t.key]))
        candidates.sort(key=lambda p:(np.linalg.norm(p-anchor),-shape.coverage(tray,p),-shape.clearance(tray,p)))
        chosen=[positions[t.key]]

        for p in candidates:
            spacing=.008 if getattr(t,'placement_partition',None) else .025
            if all(np.linalg.norm(p-old)>spacing for old in chosen):chosen.append(p)
            if len(chosen)>=6:break
        result[t.key]=chosen
    return result


def validate_landing(shape,point,tray,reservations,shapes,key,margin=0.,cloud=None):
    if shape.clearance(tray,point)<margin-1e-8:return False
    if any(k!=key and shape.overlaps(point,shapes[k],p) for k,p in reservations.items() if k in shapes):return False
    return cloud is None or not shape.hits_points(point,cloud)


def reallocate_final_landings(shape,tray,reservations,shapes,key,cloud,anchor,limit=6,partition=None):
    """A regrasp changes orientation: search the final card again from new vision."""
    for margin in ((0.,) if tray.get('placement_policy')=='center_max_coverage' else EDGE_MARGINS):
        for angle in ([0.,np.pi/2,-np.pi/2,np.pi/4,-np.pi/4,np.pi] if shape.kind in ('three_circles','ellipse','capsule','curved_outline','curved_sections','rounded_rectangle','oriented_rectangle') else [0.]):
            rotated=shape.rotated(angle);low,high=rotated.bounds(tray,margin)
            if np.any(low>high):continue
            points=[np.array([x,y]) for x in np.linspace(low[0],high[0],9) for y in np.linspace(low[1],high[1],9)]
            points=[p for p in points if np.linalg.norm(p)<=.95 and validate_landing(rotated,p,tray,reservations,shapes,key,margin,cloud)]
            if partition and 'center' in partition:
                points=[p for p in points if np.linalg.norm(p-np.asarray(partition['center']))<=partition.get('max_center_offset_m',.025)]
                points.sort(key=lambda p:(np.linalg.norm(p-np.asarray(partition['center'])),-rotated.coverage(tray,p)))
            else:
                points.sort(key=lambda p:(-rotated.coverage(tray,p),-rotated.clearance(tray,p),np.linalg.norm(p-np.asarray(anchor))))
            if not points:continue
            chosen=[points[0]]
            for p in sorted(points,key=lambda p:np.linalg.norm(p-np.asarray(anchor))):
                if all(np.linalg.norm(p-old)>.025 for old in chosen):chosen.append(p)
                if len(chosen)>=limit:break
            return dict(points=chosen,rotation=angle,margin=margin)
    return None


def projected_footprint(estimate,category,grasp_rotation,place_rotation,grasp_center):
    """Project observed geometry after the full held-object rotation, not planar yaw."""
    from types import SimpleNamespace
    origin=np.asarray(getattr(estimate,'object_center',estimate.center))
    cloud=(np.asarray(estimate.points)-grasp_center)@grasp_rotation@place_rotation.T
    rotated_center=(origin-grasp_center)@grasp_rotation@place_rotation.T
    return visual_footprint(SimpleNamespace(points=cloud,center=rotated_center,object_center=rotated_center),category)


def plan_footprint(estimate,category,plan):
    from scipy.spatial.transform import Rotation
    if 'grasp_rotation' not in plan:return visual_footprint(estimate,category).rotated(plan.get('place_yaw',plan['yaw'])-plan['yaw'])
    rotation=Rotation.from_euler('z',plan.get('place_yaw',plan['yaw'])).as_matrix()@Rotation.from_euler('y',plan.get('place_tilt',plan['tilt'])).as_matrix()@np.diag([1.,-1.,-1.])
    return projected_footprint(estimate,category,plan['grasp_rotation'],rotation,plan['center'])
