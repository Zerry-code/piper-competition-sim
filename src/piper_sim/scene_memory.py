"""Sensor-only tabletop memory; stale hints never authorize a grasp or free slot."""
import numpy as np
from piper_sim.calibration import camera_to_base
from piper_sim.table_plane import tabletop_residual


class SceneMemory:
    def __init__(self,trays):
        self.trays=trays;self.objects={};self.hints={};self.cards={};self.seen_plane=set()
        self.obstacles=np.empty((0,2));self.frames=0;self.revision=0

    def observe(self,obs,calibration,table,candidates,cards):
        self.frames+=1;self.cards.update(cards)
        for color,center in cards.items():
            if color in self.trays:self.trays[color]['center_xy_m']=np.asarray(center).tolist()
        for category,estimate in candidates:
            if not category.startswith('cube_') or len(estimate.points)<100:continue
            self.objects[category]=estimate;self.hints[category]=estimate.center[:2].copy()
        z=obs.depth_m;k=obs.intrinsics;v,u=np.indices(z.shape);valid=np.isfinite(z)&(z>0)
        pixels=np.stack([(u-k['cx'])*z/k['fx'],(v-k['cy'])*z/k['fy'],z],-1)
        points=camera_to_base(pixels[valid],obs.T_base_end,calibration)
        height,sigma=tabletop_residual(points,table)
        plane=points[abs(height)<min(.012,max(.004,2.5*sigma)),:2]
        self.seen_plane.update(map(tuple,np.floor(plane/.01).astype(int)))
        raised=points[(height>max(.012,3.5*sigma))&(height<.10),:2]
        # Retain raised evidence until a new floor observation clears that cell.
        cleared=set(map(tuple,np.floor(plane/.01).astype(int)))
        old=[p for p in self.obstacles if tuple(np.floor(p/.01).astype(int)) not in cleared]
        self.obstacles=np.vstack([np.array(old).reshape(-1,2),raised[::5]])
        if len(self.obstacles)>50000:self.obstacles=self.obstacles[-50000:]

    def moved(self,category,commanded_xy):
        self.objects.pop(category,None)
        self.hints[category]=np.array(commanded_xy).copy() # navigation hint only
        self.seen_plane.clear() # recheck the actual destination plane after a move
        self.revision+=1

    def occupied(self,color,exclude=None):
        tray=self.trays[color];center=np.array(tray['center_xy_m']);half=np.array(tray['half_size_xy_m'])
        for category,estimate in self.objects.items():
            if category==exclude:continue
            radius=.025*(abs(np.cos(estimate.yaw))+abs(np.sin(estimate.yaw)))+.008
            if np.all(abs(estimate.center[:2]-center)<half+radius):return True
        return False

    def free_position(self,color,exclude=None,reserved=()):
        if color not in self.cards:return None
        tray=self.trays[color];center=np.array(tray['center_xy_m']);half=np.array(tray['half_size_xy_m'])
        anchor=np.array(self.cards[color])
        offsets=[(x,y) for x in [0,.025,-.025,.06,-.06,.09,-.09] for y in [0,.06,-.06,.09,-.09]]
        offsets+=list(map(tuple,(center-anchor+np.array([x,y]) for x in [-.05,0,.05] for y in [-.05,0,.05])))
        for margin in [.05,.04,.03,.02,.01]:
            for delta in sorted(offsets,key=lambda d:d[0]**2+d[1]**2):
                point=anchor+delta
                if np.any(abs(point-center)>half-(.041+margin)):continue
                if any(np.linalg.norm(point-np.asarray(p))<.095 for p in reserved):continue
                if any(category!=exclude and np.linalg.norm(point-e.center[:2])<.095 for category,e in self.objects.items()):continue
                if len(self.obstacles) and np.min(np.linalg.norm(self.obstacles-point,axis=1))<.045:continue
                footprint=[tuple(np.floor((point+[x,y])/.01).astype(int)) for x in np.linspace(-.035,.035,8) for y in np.linspace(-.035,.035,8)]
                if sum(p in self.seen_plane for p in footprint)/len(footprint)<.90:continue
                return point
        return None
