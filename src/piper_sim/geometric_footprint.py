"""Observed geometry polygons: ellipses, rounded rectangles and curved outlines."""
import cv2
import numpy as np
from piper_sim.fruit_placement import Footprint


def hull(points):
    return cv2.convexHull(np.asarray(points,dtype=np.float32)).reshape(-1,2).astype(float)


class GeometryFootprint(Footprint):
    def __init__(self,vertices,kind):
        self.vertices=hull(vertices)
        # Compatibility metadata only; collision/coverage never use these discs.
        super().__init__(self.vertices.copy(),np.zeros(len(self.vertices)),kind)

    def rotated(self,angle):
        c,s=np.cos(angle),np.sin(angle)
        return GeometryFootprint(self.vertices@np.array([[c,s],[-s,c]]),self.kind)

    def _contains(self,points,margin=0.):
        points=np.asarray(points);edges=np.roll(self.vertices,-1,axis=0)-self.vertices
        normals=np.c_[edges[:,1],-edges[:,0]]
        normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-10)
        limits=np.sum(normals*self.vertices,axis=1)+margin
        return np.all(points@normals.T<=limits+1e-9,axis=1)

    def coverage(self,tray,point):
        low=self.vertices.min(axis=0);high=self.vertices.max(axis=0)
        x,y=np.meshgrid(np.linspace(low[0],high[0],27),np.linspace(low[1],high[1],27))
        samples=np.c_[x.ravel(),y.ravel()];body=self._contains(samples)
        inside=np.all(abs(samples+point-tray['center_xy_m'])<=tray['half_size_xy_m'],axis=1)
        return float(np.sum(body&inside)/max(1,body.sum()))

    def clearance(self,tray,point):
        if tray.get('placement_policy')=='center_max_coverage':
            return float(np.min(np.asarray(tray['half_size_xy_m'])-abs(np.asarray(point)-tray['center_xy_m'])))
        return float(np.min(np.asarray(tray['half_size_xy_m'])-abs(self.vertices+point-tray['center_xy_m'])))

    def bounds(self,tray,margin):
        c=np.asarray(tray['center_xy_m']);half=np.asarray(tray['half_size_xy_m'])
        if tray.get('placement_policy')=='center_max_coverage':return np.maximum(c-half+.002,[.20,-np.inf]),np.minimum(c+half-.002,[.95,np.inf])
        return np.maximum(c-half+margin-self.vertices.min(axis=0),[.20,-np.inf]),np.minimum(c+half-margin-self.vertices.max(axis=0),[.95,np.inf])

    def overlaps(self,point,other,other_point):
        if hasattr(other,'parts'):return other.overlaps(other_point,self,point)
        if not isinstance(other,GeometryFootprint):
            for offset,radius in zip(other.offsets,other.radii):
                if cv2.pointPolygonTest(self.vertices.astype(np.float32),tuple((np.asarray(other_point)+offset-point).astype(float)),True)>-float(radius):return True
            return False
        a=self.vertices+point;b=other.vertices+other_point
        for polygon in (a,b):
            edges=np.roll(polygon,-1,axis=0)-polygon
            for edge in edges:
                normal=np.array([-edge[1],edge[0]])
                aa=a@normal;bb=b@normal
                if aa.max()<=bb.min()+1e-10 or bb.max()<=aa.min()+1e-10:return False
        return True

    def hits_points(self,point,cloud):
        for start in range(0,len(cloud),4096):
            if np.any(self._contains(np.asarray(cloud)[start:start+4096]-point,.006)):return True
        return False

    def record(self):
        return dict(kind=self.kind,vertices_m=self.vertices.tolist(),estimate_source='observed_rgbd_geometry',padding_m=.004)


class CompoundFootprint(GeometryFootprint):
    """Union of small convex observed sections retains banana's concave space."""
    def __init__(self,parts):
        self.parts=parts
        super().__init__(np.vstack([p.vertices for p in parts]),'curved_sections')
    def rotated(self,angle):return CompoundFootprint([p.rotated(angle) for p in self.parts])
    def _contains(self,points,margin=0.):return np.any([p._contains(points,margin) for p in self.parts],axis=0)
    def overlaps(self,point,other,other_point):
        return any(a.overlaps(point,b,other_point) for a in self.parts for b in getattr(other,'parts',[other]))
    def record(self):return dict(kind=self.kind,parts=[p.record() for p in self.parts],padding_m=.004)


def estimate_geometry(estimate,category):
    points=np.asarray(estimate.points)[:,:2];points=points[np.isfinite(points).all(axis=1)]
    center=np.asarray(getattr(estimate,'object_center',getattr(estimate,'center',np.median(points,axis=0))))[:2]
    origin=np.mean(points,axis=0);basis=np.linalg.svd(points-origin,full_matrices=False)[2]
    local=(points-origin)@basis.T;low=local.min(axis=0);high=local.max(axis=0)
    mid=(low+high)/2;axes=np.maximum((high-low)/2,.003)
    theta=np.linspace(0,2*np.pi,64,endpoint=False)
    if category=='banana':
        sections=[]
        pads=.004*np.c_[np.cos(theta[::4]),np.sin(theta[::4])]
        edges=np.linspace(low[0],high[0],10)
        for left,right in zip(edges[:-1],edges[1:]):
            band=points[(local[:,0]>=left-.001)&(local[:,0]<=right+.001)]
            if len(band)>=3:
                sections.append(GeometryFootprint((band[:,None,:]+pads[None,:,:]).reshape(-1,2)-center,'curved_section'))
        if sections:return CompoundFootprint(sections)
        # The convex envelope retains the observed bend without a length-sized ball.
        vertices=hull(points);kind='curved_outline'
    elif category=='cucumber':
        # Oriented capsule, matching measured length and transverse thickness.
        radius=axes[1];straight=max(0.,axes[0]-radius)
        local_shape=np.c_[radius*np.cos(theta)+straight*np.sign(np.cos(theta)),radius*np.sin(theta)]
        vertices=(local_shape+mid)@basis+origin;kind='capsule'
        # Include all measured surface points (e.g. an irregular cucumber end).
        vertices=hull(np.vstack([vertices,points]))
    elif category.startswith('cube_'):
        rectangle=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*axes
        vertices=(rectangle+mid)@basis+origin;kind='oriented_rectangle'
    else:
        # Ellipsoid cross-section for pear/apple/potato/persimmon; pepper has lobed,
        # nearly rectangular cross-section. A superellipse approximates that body.
        power=.55 if category=='bell_pepper' else 1.
        curve=np.c_[np.sign(np.cos(theta))*abs(np.cos(theta))**power,np.sign(np.sin(theta))*abs(np.sin(theta))**power]
        # Enclose measured departures from an ellipse without inflating to a sphere.
        normalized=(local-mid)/axes
        exponent=2/power
        scale=max(1.,float(np.percentile(np.sum(abs(normalized)**exponent,axis=1)**(1/exponent),98)))
        vertices=(curve*axes*scale+mid)@basis+origin
        kind='rounded_rectangle' if category=='bell_pepper' else 'ellipse'
    # Small sensor/contact allowance follows the outline, not an enclosing sphere.
    offsets=.004*np.c_[np.cos(theta[::4]),np.sin(theta[::4])]
    vertices=hull((vertices[:,None,:]+offsets[None,:,:]).reshape(-1,2))
    return GeometryFootprint(vertices-center,kind)
