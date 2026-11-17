import xml.etree.ElementTree as ET
import numpy as np
import pytest
from piper_sim.session_controls import SessionControls,RestartRound
from piper_sim.fruit_world import configure,ImplicitWorld
from piper_sim.competition_world import has_support_path


def test_window_replay_random_task_and_pause_commands():
    controls=SessionControls()
    controls.key(ord('R'))
    assert controls.request==('replay','fruits')
    with pytest.raises(RestartRound):controls.check()
    controls.key(ord('1'))
    assert controls.request==('switch','blocks')
    controls.key(ord('N'))
    assert controls.request==('random','fruits')
    controls.key(ord('P'));assert controls.paused
    controls.key(ord('P'));assert not controls.paused


def test_visible_wrist_camera_and_random_block_positions(tmp_path):
    worlds=[]
    try:
        positions=[]
        for seed in [301,302]:
            output=tmp_path/str(seed);output.mkdir()
            cfg=configure(seed,['cube_red','cube_green','cube_blue'],output)
            world=ImplicitWorld('l2',cfg,output,render=False);worlds.append(world)
            world.reset(seed)
            root=ET.parse(output/'scene.xml').getroot()
            assert root.find(".//body[@name='wrist_sensor_housing']") is not None
            assert root.find(".//geom[@name='wrist_sensor_shell']").get('material')=='dabai_label'
            assert world.backend.model.camera('competition_camera').id>=0
            positions.append(world.initial_centers['object_1'])
        assert not np.allclose(*positions)
    finally:
        for world in worlds:world.close()


def test_stacked_fruit_support_cannot_travel_through_table_or_robot():
    assert has_support_path(10,20,{10:{11},11:{10,20},20:{11}},{10,11})
    assert not has_support_path(10,20,{10:{0},0:{10,20},20:{0}},{10,11})
    assert not has_support_path(10,20,{10:{7},7:{10,20},20:{7}},{10,11})
