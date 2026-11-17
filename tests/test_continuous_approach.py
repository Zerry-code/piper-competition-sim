"""Regression for the last green cube's endpoint-only approach plan."""
import numpy as np
from piper_sim.fruit_world import ImplicitWorld,configure
from piper_sim.demo_skill import PickPlaceSkill,TargetEstimate


def test_last_green_cube_has_a_continuous_descent_and_lift(tmp_path):
    world=ImplicitWorld('l2',configure(2329111945,['cube_red','cube_green','cube_blue'],tmp_path),tmp_path,render=False)
    try:
        world.reset(2329111945)
        # Recorded RGB-D estimate, not the simulator object's hidden position.
        target=TargetEstimate(np.array([.3723971,.03659924,.07761217]),.03522433,.28785664,
                              np.empty((0,0)),np.empty((0,3)),[0,0],0)
        target.grasp_yaws=[target.yaw+k*np.pi/2 for k in range(4)]
        skill=PickPlaceSkill(world.backend,.06,lambda stage:None,
            dict(transport_height_m=.2,tilt_candidates=[.15,.35,.6,.75,.9,1.05]))
        plan=skill.plan(target,np.array([.525,-.245]),.064)
        near=plan['center'].copy();near[2]+=.05
        points=skill.cartesian_path(plan['hover_q'],near,plan['yaw'])
        points+=skill.cartesian_path(points[-1],plan['center'],plan['yaw'])[1:]
        points+=skill.cartesian_path(points[-1],plan['hover'],plan['yaw'])[1:]
        assert np.max(np.abs(np.diff(points,axis=0)))<=.5
        assert skill.closed_attempts==0
        assert np.allclose(world.backend.measured()[:6],[0,1,-1,0,1.15,0],atol=.01)
    finally:
        world.close()
