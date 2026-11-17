"""Select free positions from a freshly observed plane and raised RGB-D obstacles."""
import numpy as np

def visible_free_landings(shape,tray,obstacles,seen_plane,partition=None,failed=()):
    center=np.asarray(tray['center_xy_m']);half=np.asarray(tray['half_size_xy_m'])
    anchor=np.asarray(partition.get('center',center)) if partition else center
    vertices=getattr(shape,'vertices',None)
    low=vertices.min(0) if vertices is not None else np.min(shape.offsets-shape.radii[:,None],axis=0)
    high=vertices.max(0) if vertices is not None else np.max(shape.offsets+shape.radii[:,None],axis=0)
    x,y=np.meshgrid(np.linspace(low[0],high[0],9),np.linspace(low[1],high[1],9));samples=np.c_[x.ravel(),y.ravel()]
    if hasattr(shape,'_contains'):samples=samples[shape._contains(samples)]
    obstacles=np.asarray(obstacles).reshape(-1,2)
    if len(obstacles):obstacles=np.unique(np.round(obstacles/.004).astype(int),axis=0)*.004
    points=[]
    for dx in np.linspace(-half[0]+.012,half[0]-.012,15):
        for dy in np.linspace(-half[1]+.012,half[1]-.012,15):
            point=center+[dx,dy]
            if partition and np.dot(point-center,partition['axis'])*partition['sign']<.005:continue
            if any(np.linalg.norm(point-old)<.014 for old in failed):continue
            cells=np.floor((samples+point)/.01).astype(int)
            if sum(tuple(cell) in seen_plane for cell in cells)/max(1,len(cells))<.70:continue
            if shape.hits_points(point,obstacles):continue
            points.append(point)
    points.sort(key=lambda p:(np.linalg.norm(p-anchor),-shape.coverage(tray,p)))
    chosen=[]
    for point in points:
        if all(np.linalg.norm(point-old)>.015 for old in chosen):chosen.append(point)
        if len(chosen)>=6:break
    return chosen
