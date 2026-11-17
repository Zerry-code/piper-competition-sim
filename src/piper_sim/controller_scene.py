"""Control sees nominal fixture priors and image estimates, never scene inventory."""
import copy

class ControllerWorld:
    def __init__(self,world):
        self._world=world;self.cfg=copy.deepcopy(world.cfg)
        self.cfg.pop('object_catalog',None)
        for task in self.cfg.get('tasks',{}).values():
            task.pop('objects',None);task.pop('object_pool',None)
        for tray in self.cfg['trays'].values():
            tray['center_xy_m']=list(tray.get('reference_center_xy_m',tray['center_xy_m']))
    def __getattr__(self,name):
        if name in ('objects','bounds','object_specs'):raise AttributeError('Scene truth is not available to control')
        return getattr(self._world,name)
