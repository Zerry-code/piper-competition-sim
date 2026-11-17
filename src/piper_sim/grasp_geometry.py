"""Visible-geometry grasp regions; no simulator instance poses or labels."""
import numpy as np
from scipy.ndimage import binary_closing,binary_fill_holes,distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize


def main_body_region(points,table_height):
    """Rank visible horizontal cross-sections by robust body width, excluding a neck."""
    points=np.asarray(points,dtype=float)
    low,high=np.percentile(points[:,2],[5,95])
    options=[]
    for z in np.linspace(max(table_height+.014,low+.003),high-.003,14):
        section=points[abs(points[:,2]-z)<.0045]
        if len(section)<25:continue
        xy=section[:,:2]
        width=np.percentile(xy,95,axis=0)-np.percentile(xy,5,axis=0)
        if min(width)<.008:continue
        solution=np.linalg.lstsq(np.c_[2*xy,np.ones(len(xy))],np.sum(xy*xy,axis=1),rcond=None)[0]
        center=solution[:2]
        radius=np.sqrt(max(0,solution[2]+center@center))
        if not .012<radius<.055:continue
        if np.linalg.norm(center-np.median(xy,axis=0))>.035:continue
        residual=np.median(abs(np.linalg.norm(xy-center,axis=1)-radius))
        score=np.sqrt(np.prod(width))/(1+residual/.003)
        options.append(dict(score=float(score),center=[float(center[0]),float(center[1]),float(z)],
                            radius_m=float(radius)))
    if not options:raise ValueError('No sufficiently broad visible body section')
    options.sort(key=lambda item:item['score'],reverse=True)
    best=options[0]
    alternatives=[]
    for option in options:
        if option['score']<.85*best['score']:continue
        if any(abs(option['center'][2]-other['center'][2])<.003 for other in alternatives):continue
        alternatives.append(option)
        if len(alternatives)>=4:break
    return np.array(best['center']),dict(method='visible_broad_body_cross_section',
        radius_m=best['radius_m'],alternatives=alternatives)


def banana_segment(points,resolution=.0015):
    points=np.asarray(points)
    origin=points[:,:2].min(0)-3*resolution
    indices=np.rint((points[:,:2]-origin)/resolution).astype(int)
    mask=np.zeros(tuple(indices.max(0)+4),dtype=bool)
    mask[tuple(indices.T)]=True
    mask=binary_fill_holes(binary_closing(mask,iterations=2))
    skeleton=skeletonize(mask)
    nodes=np.argwhere(skeleton)
    if len(nodes)<12:raise ValueError('Insufficient visible banana centerline')
    lookup={tuple(p):i for i,p in enumerate(nodes)}
    rows=[];cols=[];weights=[]
    for i,p in enumerate(nodes):
        for delta in [(a,b) for a in [-1,0,1] for b in [-1,0,1] if a or b]:
            j=lookup.get(tuple(p+delta))
            if j is not None:rows.append(i);cols.append(j);weights.append(np.linalg.norm(delta))
    graph=csr_matrix((weights,(rows,cols)),shape=(len(nodes),len(nodes)))
    distance=dijkstra(graph,indices=0)
    start=int(np.argmax(np.where(np.isfinite(distance),distance,-1)))
    distance,previous=dijkstra(graph,indices=start,return_predecessors=True)
    end=int(np.argmax(np.where(np.isfinite(distance),distance,-1)))
    path=[end]
    while path[-1]!=start:
        parent=int(previous[path[-1]])
        if parent<0:raise ValueError('Disconnected banana centerline')
        path.append(parent)
    line=nodes[path]
    if len(line)<12:raise ValueError('Banana segment too short')
    thickness=distance_transform_edt(mask)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(line,axis=0),axis=1))]
    fractions=arc/max(arc[-1],1e-9)
    candidates=[]
    for i in range(max(4,len(line)//5),min(len(line)-4,4*len(line)//5)):
        if not .20<=fractions[i]<=.80:continue
        a=line[i]-line[i-4];b=line[i+4]-line[i]
        cosine=np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9)
        if cosine<.75:continue
        radius=thickness[tuple(line[i])]*resolution
        if not .005<radius<.025:continue
        score=radius*(.5+.5*cosine)/(1+8*abs(fractions[i]-.5))
        candidates.append((score,i))
    if not candidates:raise ValueError('No graspable body section; request a clearer view rather than grip an end')
    middle=[c for c in candidates if 1/3<=fractions[c[1]]<=2/3]
    _,i=max(middle or candidates)
    tangent=(line[i+4]-line[i-4]).astype(float)
    tangent/=np.linalg.norm(tangent)
    center=origin+line[i]*resolution
    alternatives=[]
    middle_options=sorted(middle,reverse=True)
    outer_options=sorted([c for c in candidates if c not in middle],reverse=True)
    for score,j in middle_options+outer_options:
        xy=origin+line[j]*resolution
        if any(np.linalg.norm(xy-np.array(a['center']))<.007 for a in alternatives):continue
        if sum(1/3<=a['position_fraction']<=2/3 for a in alternatives)>=6 and 1/3<=fractions[j]<=2/3:continue
        direction=line[j+4]-line[j-4]
        alternatives.append(dict(center=xy.tolist(),yaw=float(np.arctan2(direction[1],direction[0])),score=float(score),position_fraction=float(fractions[j])))
        if len(alternatives)>=12:break
    return center,float(np.arctan2(tangent[1],tangent[0])),dict(
        method='visible_metric_centerline_straight_section',radius_m=float(thickness[tuple(line[i])]*resolution),
        centerline_points=len(line),position_fraction=float(fractions[i]),alternatives=alternatives)


def pear_grasp_regions(points,table_height):
    """Rank body sections and closing directions using visible contact geometry."""
    points=np.asarray(points)
    _,body=main_body_region(points,table_height)
    candidates=[]
    for section in body['alternatives']:
        center=np.array(section['center'])
        local=points[abs(points[:,2]-center[2])<.008]
        if len(local)<25:continue
        for yaw in np.arange(0,2*np.pi,np.pi/12):
            normal=np.array([-np.sin(yaw),np.cos(yaw)])
            tangent=np.array([np.cos(yaw),np.sin(yaw)])
            projected=(local[:,:2]-center[:2])@normal
            lo,hi=np.percentile(projected,[2,98]);width=hi-lo
            if not .018<width<.094:continue
            # Require observed patches on both opposing sides, not only a top cap.
            left=local[projected<lo+.005];right=local[projected>hi-.005]
            if min(len(left),len(right))<5:continue
            span=min(np.ptp(left[:,:2]@tangent),np.ptp(right[:,:2]@tangent))
            if span<.006:continue
            balance=abs(hi+lo)/max(width,.001)
            score=section['score']*(1+min(span,.025)/.025)/(1+balance)
            candidates.append(dict(center=center.tolist(),yaw=float(yaw),yaws=[float(yaw)],score=float(score),opening_m=float(width+.006)))
    if not candidates:raise ValueError('No opposing visible body patches within gripper aperture')
    candidates.sort(key=lambda c:-c['score'])
    return candidates

def ellipsoid_pear_grasp(points,table_height):
    """Fit a measured-size ellipsoid prior to RGB-D; never read simulator pose."""
    from scipy.optimize import least_squares
    from piper_sim.measured_fruit import ellipsoid_axes
    points=np.asarray(points)
    if len(points)<200:raise ValueError('Insufficient pear surface')
    lo,hi=np.percentile(points,[2,98],axis=0)
    if max(hi[:2]-lo[:2])<.095 or not .045<hi[2]-table_height<.115:raise ValueError('Not a horizontal broad ellipsoid observation')
    cloud=points[::max(1,len(points)//1800)]
    covariance=np.cov(cloud[:,:2].T);_,axis=np.linalg.eigh(covariance);yaw=float(np.arctan2(axis[1,-1],axis[0,-1]))
    radii=ellipsoid_axes(.33,.27)/2
    xy=(lo[:2]+hi[:2])/2
    initial=np.r_[xy,table_height+radii[2],yaw,1.]
    def residual(p):
        angle=p[3];rotation=np.array([[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1.]])
        local=(cloud-p[:3])@rotation.T
        return np.linalg.norm(local/(radii*p[4]),axis=1)-1
    fit=least_squares(residual,initial,bounds=(np.r_[xy-.045,table_height+.028,yaw-np.pi/2,.85],np.r_[xy+.045,table_height+.065,yaw+np.pi/2,1.15]),loss='soft_l1',f_scale=.04,max_nfev=160)
    error=float(np.median(abs(residual(fit.x))))
    if error>.10:raise ValueError('Ellipsoid fit uncertain')
    center=fit.x[:3];angle=float(fit.x[3]);width=float(2*radii[1]*fit.x[4])
    if width>.094:raise ValueError('Body wider than available opening margin')
    quality=dict(method='measured_ellipsoid_surface_fit',fit_error=error,opening_m=width+.006,prior='user_measured_33cm_27cm_circumferences',alternatives=[])
    for dz in [.005,-.005]:quality['alternatives'].append(dict(center=(center+[0,0,dz]).tolist(),yaw=angle,yaws=[angle,angle+np.pi],score=1.-abs(dz)))
    for shift in [-.012,.012]:
        point=center+np.array([shift*np.cos(angle),shift*np.sin(angle),0.])
        quality['alternatives'].append(dict(center=point.tolist(),yaw=angle,yaws=[angle,angle+np.pi],score=.98))
    return center,angle,quality
