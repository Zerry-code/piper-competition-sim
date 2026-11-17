"""Measured finger release and robot/known-tray clearance, never object pose truth."""
import mujoco
import numpy as np


class ReleaseBlocked(RuntimeError):
    """No lateral recovery is permitted until opening and vertical clearance succeed."""


def release_feedback(backend,width=.10):
    model,data=backend.model,backend.data
    fingers=[float(data.qpos[model.joint(f'joint{i}').qposadr[0]]) for i in [7,8]]
    opened=abs(fingers[0]-width/2)<.0025 and abs(fingers[1]+width/2)<.0025
    finger_bodies={model.body('link7').id,model.body('link8').id}
    object_contact=False;tray_contact=False
    for i,c in enumerate(data.contact[:data.ncon]):
        bodies=[int(model.geom_bodyid[g]) for g in [c.geom1,c.geom2]]
        if not finger_bodies.intersection(bodies):continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
        if force[0]<.10:continue
        other=1 if bodies[0] in finger_bodies else 0
        object_contact|=model.body_dofnum[bodies[other]]==6
        tray_contact|=model.geom([c.geom1,c.geom2][other]).name.startswith('tray_')
    return dict(opened=bool(opened),fingers_m=fingers,object_contact=bool(object_contact),tray_contact=bool(tray_contact))


def opening_clearance(backend,q,width=.10,margin=.001):
    """Check the finger sweep against known static trays using robot kinematics."""
    model=backend.model;data=mujoco.MjData(model)
    data.qpos[:6]=q
    finger_bodies={model.body('link7').id,model.body('link8').id}
    geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g] in finger_bodies
           and (model.geom_contype[g] or model.geom_conaffinity[g])]
    trays=[g for g in range(model.ngeom) if model.geom(g).name.startswith('tray_')]
    for gap in np.linspace(0,width,6):
        data.qpos[backend.fingers]=[gap/2,-gap/2]
        mujoco.mj_forward(model,data)
        for finger in geoms:
            for tray in trays:
                if mujoco.mj_geomDistance(model,data,finger,tray,.02,None)<margin:
                    return False
    return True
