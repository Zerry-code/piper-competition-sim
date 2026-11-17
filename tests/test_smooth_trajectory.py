import numpy as np
from piper_sim.trajectory import trajectory
from piper_sim.control import load_interface
from piper_sim.session_controls import SessionControls


def test_speed_acceleration_and_continuous_interior_waypoints():
    cfg=load_interface();dt=1/cfg['control']['policy_hz']
    points=np.array([np.full(6,x) for x in [0,.1,.2,.3,.4]])
    samples=trajectory(points,0,.07,cfg)
    path=np.vstack([np.zeros(7),samples])
    speed=np.diff(path[:,:6],axis=0)/dt
    acceleration=np.diff(speed,axis=0)/dt
    assert np.all(abs(speed)<=np.array(cfg['control']['max_joint_speed_rad_s'])+1e-5)
    assert np.max(abs(acceleration))<=.45+1e-3
    assert np.max(abs(np.diff(path[:,6])/dt))<=cfg['control']['max_gripper_speed_m_s']+1e-5
    interior=(path[1:,0]>.08)&(path[1:,0]<.32)
    assert np.all(speed[interior,0]>.005)
    np.testing.assert_allclose(samples[-1,:6],points[-1],atol=1e-12)
    assert abs(speed[0,0])<.001 and abs(speed[-1,0])<.001


def test_function_keys_do_not_need_letter_input():
    controls=SessionControls()
    controls.key(294);assert controls.request==('replay','fruits')
    controls.key(295);assert controls.request==('random','fruits')
    controls.key(296);assert controls.paused
