from types import SimpleNamespace as NS
import numpy as np
from piper_sim.controller_scene import ControllerWorld

def test_controller_cannot_read_inventory_or_actual_card_jitter():
    world=NS(objects=[{'category':'pear'}],cfg={'trays':{'yellow':{'center_xy_m':[.58,.315],'reference_center_xy_m':[.58,.30]}}})
    control=ControllerWorld(world)
    assert control.cfg['trays']['yellow']['center_xy_m']==[.58,.30]
    try:control.objects
    except AttributeError:pass
    else:raise AssertionError('hidden inventory leaked')
    control.cfg['trays']['yellow']['center_xy_m']=[.58,.312]
    assert world.cfg['trays']['yellow']['center_xy_m']==[.58,.315]
