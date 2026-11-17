import numpy as np
import mujoco
from piper_sim.fruit_world import configure,ImplicitWorld


def test_fruit_stays_quiet_and_table_damping_disengages_after_lift(tmp_path):
    world=ImplicitWorld('l2',configure(140,['apple','pear','banana','lemon','orange','strawberry'],tmp_path),tmp_path,render=False)
    try:
        world.reset(140)
        b=world.backend
        ids=[b.model.body(o['id']).id for o in world.objects]
        initial=b.data.xpos[ids].copy()
        peak=0.
        for _ in range(2500):
            world._before_physics_step(b)
            mujoco.mj_step(b.model,b.data)
            peak=max(peak,float(np.linalg.norm(b.data.xpos[ids]-initial,axis=1).max()))
        assert peak<.0001
        bid=ids[0];joint=b.model.joint(world.objects[0]['id']+'_free')
        b.data.qpos[joint.qposadr[0]+2]+=.2
        mujoco.mj_forward(b.model,b.data)
        world._before_physics_step(b)
        dof=joint.dofadr[0]
        np.testing.assert_allclose(b.model.dof_damping[dof:dof+6],world._original_damping[bid])
        before=b.data.qvel[dof+3:dof+6].copy()
        b.data.xfrc_applied[bid,4]=.0005
        for _ in range(10):
            world._before_physics_step(b);mujoco.mj_step(b.model,b.data)
        assert np.linalg.norm(b.data.qvel[dof+3:dof+6]-before)>.01
    finally:world.close()
