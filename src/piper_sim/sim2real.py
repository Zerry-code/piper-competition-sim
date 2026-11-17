"""Seeded deployment-gap experiments. Parameters are hypotheses, never hardware specs."""
from collections import deque
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

class DeploymentGap:
    def __init__(self,profile,seed):
        self.profile=dict(profile);self.rng=np.random.default_rng(np.random.SeedSequence([seed,9371]))
        r=self.rng;p=self.profile
        self.translation=r.uniform(-p['camera_translation_m'],p['camera_translation_m'],3)
        self.rotation=r.uniform(-p['camera_rotation_deg'],p['camera_rotation_deg'],3)
        self.scale=1+float(r.uniform(-p['depth_scale_error'],p['depth_scale_error']))
        self.queue=deque();self.applied=None;self.samples=[]

    def install(self,backend,object_names):
        m=backend.model;p=self.profile
        cam=m.camera('competition_camera').id
        m.cam_pos[cam]+=self.translation
        quat=m.cam_quat[cam].copy()
        rotation=Rotation.from_quat(quat[[1,2,3,0]])*Rotation.from_euler('xyz',self.rotation,degrees=True)
        m.cam_quat[cam]=rotation.as_quat()[[3,0,1,2]]
        for name in object_names:
            bid=m.body(name).id;factor=float(self.rng.uniform(1-p['mass_fraction'],1+p['mass_fraction']))
            m.body_mass[bid]*=factor;m.body_inertia[bid]*=factor
            gids=np.where(m.geom_bodyid==bid)[0]
            friction=float(self.rng.uniform(1-p['friction_fraction'],1+p['friction_fraction']))
            m.geom_friction[gids,0]*=friction
            # Explicit pairs override geom friction: perturb their sliding terms too.
            for pair in range(m.npair):
                if m.pair_geom1[pair] in gids or m.pair_geom2[pair] in gids:m.pair_friction[pair,:2]*=friction
            self.samples.append(dict(object=name,mass_factor=factor,friction_factor=friction))
        mujoco.mj_setConst(m,backend.data)
        backend.deployment_gap=self
        self.applied=backend.measured().copy()
        self.sensor(backend)

    def sensor(self,backend):
        backend.camera['camera']['depth'].update(noise_std_m=self.profile['depth_std_m'],dropout_probability=self.profile['dropout'])

    def command(self,target):
        self.queue.append(np.array(target,copy=True))
        if len(self.queue)>self.profile['delay_steps']:self.applied=self.queue.popleft()
        return self.applied.copy()

    def report(self):
        return dict(profile=self.profile,camera_translation_m=self.translation.tolist(),camera_rotation_deg=self.rotation.tolist(),depth_scale=self.scale,objects=self.samples)
