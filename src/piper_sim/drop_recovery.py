"""Grip-loss detection and RGB-D landing checks, without simulator object poses."""
import numpy as np


class ObjectDropped(RuntimeError):
    def __init__(self,stage):
        super().__init__('Loaded bilateral contact lost during '+stage)
        self.stage=stage


class GripMonitor:
    def __init__(self,debounce_s=.10):
        self.debounce_s=debounce_s
        self.disarm()

    def arm(self):
        self.armed=True
        self.missing_since=None

    def disarm(self):
        self.armed=False
        self.missing_since=None

    def update(self,now,retained,stage):
        if not self.armed:return
        if retained:self.missing_since=None;return
        if self.missing_since is None:self.missing_since=now
        if now-self.missing_since>=self.debounce_s-1e-9:
            self.disarm()
            raise ObjectDropped(stage)


def landing_region(points,trays,table,geometry):
    """Conservative visible extent, including tray floor height; no pose truth."""
    points=np.asarray(points)
    low,high=np.percentile(points,[1,99],axis=0)
    floor=table+geometry['floor_thickness_m']
    for color,tray in trays.items():
        center=np.array(tray['center_xy_m']);half=np.array(tray['half_size_xy_m'])
        if tray.get('placement_policy')=='center_max_coverage' and low[2]>=floor-.003:
            clearance=float(np.min(half-abs((low[:2]+high[:2])/2-center)))
            if clearance>=0:return color if clearance>.003 else 'edge:'+color
        # At a wall, the visible side alone cannot prove that the hidden side
        # is contained. Request another view instead of rearranging or completing.
        if (np.all(low[:2]>=center-half-.001) and np.all(high[:2]<=center+half+.001)
                and low[2]>=floor-.003):
            if min(np.min(low[:2]-(center-half)),np.min(center+half-high[:2]))<.003:
                return 'edge:'+color
            return color
    return 'table'


class LandingTracker:
    """Require a sustained stationary RGB-D extent before accepting the right tray."""
    def __init__(self,stable_s=.4):
        self.stable_s=stable_s
        self.previous=None
        self.region=None
        self.since=None

    def update(self,now,points,region):
        # Median alone can hide a rotating elongated fruit. Check both extent ends.
        bounds=np.percentile(points,[5,95],axis=0)
        if (self.previous is None or region!=self.region or
                np.max(np.abs(bounds-self.previous))>.003):
            self.since=now
            self.previous=bounds
        self.region=region
        return now-self.since>=self.stable_s-1e-9
