"""RGB-D synthetic-color baseline and interior grasp-point estimation.

Color classifier is explicitly a synthetic baseline, not a trained real-fruit recognizer.
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label as label_volume
from skimage.morphology import skeletonize
from piper_sim.demo_skill import TargetEstimate
from piper_sim.calibration import camera_to_base
from piper_sim.implicit_fruit import COLORS
from piper_sim.table_plane import tabletop_residual
from piper_sim.grasp_geometry import banana_segment,main_body_region,pear_grasp_regions


def interior_pixel(mask):
    skeleton = skeletonize(mask)
    y,x = np.where(skeleton)
    if not len(x): raise ValueError('Empty object skeleton')
    thickness = distance_transform_edt(mask)[y,x]
    # Select a thick medial section near the visible body's median, never a hollow centroid.
    distance = np.hypot(x-np.median(x), y-np.median(y))
    score = thickness / (1+distance/max(float(thickness.max())*3,1))
    i = int(np.argmax(score))
    return int(x[i]), int(y[i])



def split_touching_colors(mask,base,rgb):
    """Separate substantial green/warm bodies in one touching RGB-D component."""
    points=base[mask]
    if len(points)<200 or np.max(np.ptp(points[:,:2],axis=0))<.10:return [mask]
    colors=rgb[mask].astype(float)/255.;r,g,b=colors.T
    green=(g>1.25*r)&(g>1.1*b)&(g-r>.035)
    warm=(r>=.9*g)&(r>1.2*b)&(r-b>.04)
    minimum=max(80,int(len(points)*.16))
    if green.sum()<minimum or warm.sum()<minimum:
        # Touching red pepper and orange persimmon are both warm bodies.
        green=(r>2*g)&(g/np.maximum(r,1e-6)<.18)
        warm=(g/np.maximum(r,1e-6)>.24)&(g/np.maximum(r,1e-6)<.65)&(b/np.maximum(r,1e-6)<.25)
        if green.sum()<minimum or warm.sum()<minimum:return [mask]
    centers=np.array([np.median(points[green],axis=0),np.median(points[warm],axis=0)])
    if np.linalg.norm(centers[0,:2]-centers[1,:2])<.035:return [mask]
    labels=np.argmin(np.linalg.norm(points[:,None,:]-centers[None,:,:],axis=2),axis=1)
    labels[green]=0;labels[warm]=1
    result=[]
    for identity in range(2):
        part=np.zeros(mask.shape,dtype=bool);part[mask]=labels==identity
        if part.sum()<80:return [mask]
        result.append(part)
    return result

def candidates_3d(obs,calibration,table,workspace=None,trays=None,tray_geometry=None):
    """Separate touching image silhouettes with connected components in metric XYZ."""
    v,u=np.indices(obs.depth_m.shape)
    z=obs.depth_m
    k=obs.intrinsics
    valid=np.isfinite(z)&(z>0)
    camera=np.stack([(u-k['cx'])*z/k['fx'],(v-k['cy'])*z/k['fy'],z],-1)
    base=np.zeros_like(camera)
    base[valid]=camera_to_base(camera[valid],obs.T_base_end,calibration)
    foreground_height,sigma=tabletop_residual(base[valid],table)
    heights=np.zeros(z.shape);heights[valid]=foreground_height
    foreground_height=heights
    threshold=max(.007,3.5*sigma)
    valid&=(foreground_height>threshold)&(foreground_height<.14)
    xmin,xmax,ymin,ymax=workspace or (.20,.63,-.14,.25)
    valid&=(base[...,0]>xmin)&(base[...,0]<xmax)&(base[...,1]>ymin)&(base[...,1]<ymax)
    if trays and tray_geometry:
        # Remove the configured container walls, which otherwise connect a fruit
        # to a tray in the component graph. This is fixture geometry, not fruit truth.
        wall=tray_geometry['wall_thickness_m']+.002
        for tray in trays.values():
            delta=abs(base[...,:2]-np.array(tray['center_xy_m']))
            half=np.array(tray['half_size_xy_m'])
            boundary=((abs(delta[...,0]-half[0])<wall)&(delta[...,1]<half[1]+wall))|((abs(delta[...,1]-half[1])<wall)&(delta[...,0]<half[0]+wall))
            valid&=~(boundary&(base[...,2]<table+tray_geometry['floor_thickness_m']+tray_geometry['wall_height_m']+.003))
    cloud=base[valid]
    if not len(cloud):return []
    voxels=np.floor((cloud-np.array([xmin,ymin,table]))/.004).astype(int)+1
    grid=np.zeros(tuple(voxels.max(0)+2),dtype=bool)
    grid[tuple(voxels.T)]=True
    volume,_=label_volume(grid,structure=np.ones((3,3,3),dtype=bool))
    labels=volume[tuple(voxels.T)]
    pixel_labels=np.zeros(z.shape,dtype=int)
    pixel_labels[valid]=labels
    result=[]
    masks=[]
    for identity in np.unique(labels):
        mask=pixel_labels==identity
        if mask.sum()<80:continue
        masks.extend(split_touching_colors(mask,base,obs.rgb) if hasattr(obs,'rgb') else [mask])
    for mask in masks:
        points=base[mask]
        extent=np.ptp(points,axis=0)
        if max(extent[:2])<.015 or extent[2]<.005:continue
        top=float(np.percentile(points[:,2],95))
        center=np.r_[np.median(points[:,:2],axis=0),(table+top)/2]
        ys,xs=np.where(mask)
        result.append(TargetEstimate(center,top-table,0.,mask,points,
            [float(xs.mean()),float(ys.mean())],0.))
    return result


def classify_visible_geometry(points,rgb,support):
    """Appearance and metric geometry only; no scene categories or object IDs."""
    points=np.asarray(points);color=np.median(rgb,axis=0).astype(float)/255
    normal=color/max(np.linalg.norm(color),1e-6)
    palette=dict(COLORS,cube_red=[.72,.025,.025,1],cube_green=[.12,.38,.025,1],cube_yellow=[.9,.75,.03,1])
    xy=points[:,:2]-np.mean(points[:,:2],axis=0)
    values,vectors=np.linalg.eigh(np.cov(xy.T))
    local=xy@vectors
    spans=np.percentile(local,98,axis=0)-np.percentile(local,2,axis=0)
    length=float(max(spans));width=float(min(spans));ratio=length/max(width,1e-6)
    top=float(np.percentile(points[:,2],95));height=top-support
    flat=float(np.mean(points[:,2]>top-.003))
    # A yellow surface alone never establishes a cube: require an elevated broad flat face.
    face=points[points[:,2]>top-.003]
    face_area=0.;face_short=0.;face_long=0.;face_fill=0.
    if len(face)>=20:
        rectangle=cv2.minAreaRect(face[:,:2].astype(np.float32))
        face_short,face_long=sorted(rectangle[1]);face_area=face_short*face_long
        if face_area>1e-8:face_fill=float(cv2.contourArea(cv2.convexHull(face[:,:2].astype(np.float32)))/face_area)
    # Side pixels change with viewpoint; judge the observed metric top face itself.
    cube_like=.040<height<.065 and .032<face_short<.070 and face_long<.075 and face_area>.0013 and face_fill>.85
    allowed=['apple','pear','banana','persimmon','bell_pepper','cucumber','potato']
    if cube_like:allowed=['cube_red','cube_green','cube_yellow']
    prototypes={c:[palette[c][:3]] for c in allowed}
    if 'pear' in prototypes:prototypes['pear'].append([.85,.95,.10])
    if 'apple' in prototypes:prototypes['apple'].append([.80,.04,.025])
    def score(category):
        return min(np.linalg.norm(normal-np.asarray(p)/max(np.linalg.norm(p),1e-6)) for p in prototypes[category])
    category=min(allowed,key=score)
    r,g,b=color
    yellowish=g>.035 and r>.035 and .55<g/max(r,1e-6)<1.45 and b<.30*max(r,g)
    if not cube_like and yellowish:
        if length>.085 and ratio>1.25 and height<.040:category='banana'
        elif .035<length<.095 and height>.028 and g>r*.8:category='pear'
    if category in ('banana','cucumber') and (length<.080 or ratio<1.25):
        category=min([c for c in allowed if c not in ('banana','cucumber')],key=score)
    return category,dict(method='rgb_metric_geometry',height_m=height,length_m=length,width_m=width,flat_fraction=flat,cube_like=bool(cube_like),top_face_area_m2=float(face_area),top_face_short_m=float(face_short),top_face_fill=float(face_fill))


def estimate_candidates(obs, calibration, table, allowed_categories=None,workspace=None,trays=None,tray_geometry=None):
    candidates = candidates_3d(obs, calibration, table,workspace,trays,tray_geometry)
    result = []
    for candidate in candidates:
        support=table
        if trays and tray_geometry:
            xy=np.median(candidate.points[:,:2],axis=0)
            if any(np.all(abs(xy-np.array(t['center_xy_m']))<np.array(t['half_size_xy_m'])) for t in trays.values()):
                support+=tray_geometry['floor_thickness_m']
        candidate.support_height_m=support
        top=float(np.percentile(candidate.points[:,2],95))
        candidate.height=top-support
        candidate.center[2]=(support+top)/2
        u,v = interior_pixel(candidate.mask)
        z = obs.depth_m[v,u]
        if not np.isfinite(z): continue
        k = obs.intrinsics
        point = camera_to_base(np.array([[(u-k['cx'])*z/k['fx'],
            (v-k['cy'])*z/k['fy'],z]]),obs.T_base_end,calibration)[0]
        candidate.center[:2] = point[:2]
        candidate.pixel = [u,v]
        near = candidate.points[np.linalg.norm(candidate.points[:,:2]-point[:2],axis=1)<.018]
        if len(near)>8:
            _,vectors = np.linalg.eigh(np.cov(near[:,:2].T))
            tangent = vectors[:,-1]
            candidate.yaw = float(np.arctan2(tangent[1],tangent[0]))
        category,candidate.recognition=classify_visible_geometry(candidate.points,obs.rgb[candidate.mask],support)
        if category.startswith('cube_'):
            top=float(np.percentile(candidate.points[:,2],90))
            face=candidate.points[candidate.points[:,2]>top-.003]
            candidate.center=np.r_[np.median(face[:,:2],axis=0),(support+top)/2]
            candidate.height=top-support
            corners=cv2.boxPoints(cv2.minAreaRect(face[:,:2].astype(np.float32)))
            edge=corners[1]-corners[0]
            candidate.yaw=float((np.arctan2(edge[1],edge[0])+np.pi/4)%(np.pi/2)-np.pi/4)
            candidate.grasp_yaws=[candidate.yaw+k*np.pi/2 for k in range(4)]
        elif category in ('banana','cucumber'):
            try:
                center,yaw,quality=banana_segment(candidate.points)
            except ValueError:
                candidate.grasp_quality=dict(method='segment_detail_insufficient',requires_refinement=True)
                result.append((category,candidate))
                continue
            candidate.center[:2]=center
            candidate.yaw=yaw
            candidate.grasp_yaws=[yaw,yaw+np.pi]
            candidate.grasp_quality=quality
            # Small angle changes can make a cylindrical section reachable while
            # retaining overlap with the pads. Bound them by visible thickness.
            tolerance=min(.45,float(np.arctan2(quality['radius_m'],.022)))
            offsets=[0.,tolerance/2,-tolerance/2,tolerance,-tolerance]
            candidate.grasp_yaws=[yaw+turn+offset for offset in offsets for turn in [0.,np.pi]]
            for option in quality['alternatives']:
                option['yaws']=[option['yaw']+turn+offset for offset in offsets for turn in [0.,np.pi]]
            candidate.object_center=(np.percentile(candidate.points,2,axis=0)+np.percentile(candidate.points,98,axis=0))/2
            candidate.preferred_place_yaws=[np.pi/2,-np.pi/2]
            # A curved fruit can contact the tray before a flat object's nominal
            # center height; release with a small clearance rather than pushing it.
            candidate.place_clearance_m=.010
        elif category=='pear':
            try:options=pear_grasp_regions(candidate.points,support)
            except ValueError:
                candidate.grasp_quality=dict(method='body_detail_insufficient',requires_refinement=True)
            else:
                best=options[0]
                candidate.center=np.array(best['center']);candidate.yaw=best['yaw']
                candidate.grasp_yaws=best['yaws']
                candidate.grasp_quality=dict(method='opposing_body_sections',alternatives=options[1:],opening_m=best['opening_m'])
                candidate.object_center=(np.percentile(candidate.points,2,axis=0)+np.percentile(candidate.points,98,axis=0))/2
        elif category in ('apple','orange','lemon','peach','plum','strawberry','persimmon','bell_pepper','potato'):
            points=candidate.points
            if category in ('pear','strawberry'):
                points=points[points[:,2]<np.percentile(points[:,2],85)]
            candidate.object_center=(np.percentile(points,2,axis=0)+np.percentile(points,98,axis=0))/2
            candidate.object_center[2]=(support+top)/2
            # Algebraic sphere fit to visible RGB-D surface, with explicit plausibility bounds.
            solution=np.linalg.lstsq(np.c_[2*points,np.ones(len(points))],
                                    np.sum(points*points,axis=1),rcond=None)[0]
            center=solution[:3]
            radius=np.sqrt(max(0,solution[3]+center@center))
            median=np.median(points,axis=0)
            if .012<radius<.045 and np.linalg.norm(center[:2]-median[:2])<.035 and table<center[2]<table+.06:
                candidate.center[:2]=center[:2]
                candidate.object_center[:2]=center[:2]
        result.append((category,candidate))
    return result
