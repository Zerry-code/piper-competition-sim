"""YCB textured scans where available; procedural fallback for unscanned vegetables."""
import yaml
from piper_sim.calibration import ROOT
from piper_sim.fruit_world import ImplicitWorld,configure
from piper_sim.competition_world import CompetitionWorld
from piper_sim.l1_world import configure_card_regions,remove_card_walls

CATEGORIES=['apple','pear','banana','persimmon','bell_pepper','cucumber','potato']


def configure_fruits(seed,output):
    path=configure(seed,CATEGORIES,output,measured=False)
    catalog_path=output/'catalog.yaml';catalog=yaml.safe_load(catalog_path.read_text(encoding='utf-8'))
    source=yaml.safe_load((ROOT/'config/competition/objects_scanned.yaml').read_text(encoding='utf-8'))['objects']
    for category in CATEGORIES:
        original=source.get(category,{})
        if original.get('visual_mesh',{}).get('texture') and original.get('collision_meshes'):
            catalog['objects'][category]=original
    props=yaml.safe_load((ROOT/'config/fruit_props.yaml').read_text(encoding='utf-8-sig'))
    for category,asset in catalog['objects'].items():
        if category.startswith('cube_'):continue
        asset['mass_kg_range']=props['mass_kg_range']
        asset['hollow_shell_fraction']=props['shell_relative_thickness']
        asset['measurement_status']=props['status']
    catalog_path.write_text(yaml.safe_dump(catalog),encoding='utf-8')
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg['tasks']['l2']['objects']=CATEGORIES
    configure_card_regions(cfg,seed)
    for tray in cfg['trays'].values():
        tray['placement_policy']='center_max_coverage';tray['allocation_mode']='pair_partition'
    cfg['spawn'].update(x_m=[.36,.59],y_m=[-.48,.48],tray_clearance_m=.01,radial_limit_m=.60)
    cfg['tasks']['l2']['mapping']={ 'apple':'red','pear':'green','banana':'yellow','persimmon':'red','bell_pepper':'red','cucumber':'green','potato':'yellow'}
    path.write_text(yaml.safe_dump(cfg),encoding='utf-8');return path


class DocumentFruitWorld(ImplicitWorld):
    def _build(self,rng):
        super()._build(rng)
        remove_card_walls(self.output/'scene.xml')
    def _settle(self,backend):
        import mujoco,numpy as np
        from collections import deque
        model,data=backend.model,backend.data
        joints=[model.joint(o['id']+'_free') for o in self.objects]
        history=deque(maxlen=21)
        for sample in range(600):
            for _ in range(round(.05/model.opt.timestep)):
                self._before_physics_step(backend);mujoco.mj_step(model,data)
            positions=np.array([data.qpos[j.qposadr[0]:j.qposadr[0]+7].copy() for j in joints])
            speeds=np.array([[np.linalg.norm(data.qvel[j.dofadr[0]:j.dofadr[0]+3]),np.linalg.norm(data.qvel[j.dofadr[0]+3:j.dofadr[0]+6])] for j in joints])
            history.append((positions,speeds))
            if sample<40 or len(history)<21:continue
            first=history[0][0];drift=np.max(np.linalg.norm(positions[:,:3]-first[:,:3],axis=1))
            angle=2*np.arccos(np.clip(abs(np.sum(positions[:,3:]*first[:,3:],axis=1)),0,1))
            means=np.mean([h[1] for h in history],axis=0)
            if drift<.0003 and max(angle)<.008 and max(means[:,0])<.001 and max(means[:,1])<.04:
                self.settle_report=dict(preparation_sim_s=(sample+1)*.05,window_s=1.,max_position_drift_m=float(drift));return
        raise RuntimeError('Scanned fruit did not physically settle within 30 seconds')

    def _add_scanned_object(self,root,body,name,category,asset,scale,mass):
        if 'implicit_parts' not in asset:
            return CompetitionWorld._add_scanned_object(self,root,body,name,category,asset,scale,mass)
        return super()._add_scanned_object(root,body,name,category,asset,scale,mass)
