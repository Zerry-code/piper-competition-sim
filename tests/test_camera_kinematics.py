"""Observation IK needs camera FK, not collision/dynamics on every residual."""
import mujoco
import numpy as np
from piper_sim.fruit_world import ImplicitWorld,configure

def test_camera_and_tcp_kinematics_match_full_forward(tmp_path):
    world=ImplicitWorld('l2',configure(710,['cube_red'],tmp_path),tmp_path,render=False)
    try:
        world.reset(710);model=world.backend.model
        full=mujoco.MjData(model);fast=mujoco.MjData(model)
        for q in ([0,.4,-.8,0,1.2,0],[.4,1,-1.4,.1,1.8,.2],[-.3,.7,-1.1,0,1.4,-.2]):
            full.qpos[:]=world.backend.data.qpos;full.qpos[:6]=q
            fast.qpos[:]=full.qpos
            mujoco.mj_forward(model,full)
            mujoco.mj_kinematics(model,fast);mujoco.mj_camlight(model,fast)
            np.testing.assert_allclose(fast.cam_xpos,full.cam_xpos,atol=1e-12)
            np.testing.assert_allclose(fast.cam_xmat,full.cam_xmat,atol=1e-12)
            np.testing.assert_allclose(fast.site_xpos,full.site_xpos,atol=1e-12)
    finally:world.close()
