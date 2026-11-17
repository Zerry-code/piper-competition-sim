"""Procedural PiPER fruit world built on the official robot and compound contacts."""
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
import yaml
from piper_sim.competition_world import CompetitionWorld, numbers
from piper_sim.calibration import ROOT, load_config
from piper_sim.implicit_fruit import FRUITS, COLORS, components, mesh, texture, field
from PIL import Image, ImageDraw, ImageFont

class ImplicitWorld(CompetitionWorld):
    def _valid_reset(self):
        if not super()._valid_reset():return False
        clearance=self.cfg['spawn'].get('tray_clearance_m',0)
        for obj in self.objects:
            low,high=self.bounds(obj['id'])
            for tray in self.cfg['trays'].values():
                half=np.array(tray['half_size_xy_m'])+self.cfg['tray_geometry']['wall_thickness_m']
                center=np.array(tray['center_xy_m'])
                gap=np.maximum(np.maximum(center-half-high[:2],low[:2]-(center+half)),0)
                if np.linalg.norm(gap)<clearance-.001:return False
        return True

    def _before_physics_step(self,backend):
        model,data=backend.model,backend.data
        if getattr(self,'_damping_model',None) is not model:
            self._damping_model=model
            self._damping_objects={model.body(o['id']).id:model.joint(o['id']+'_free').dofadr[0] for o in self.objects}
            self._original_damping={bid:model.dof_damping[d:d+6].copy() for bid,d in self._damping_objects.items()}
            self._support_geoms={model.geom('competition_table').id}|{
                model.geom('tray_'+color+'_floor').id for color in self.cfg['trays']}
        installation=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'installation_table')
        if installation>=0:self._support_geoms.add(installation)
        supported=set();touched=set()
        for contact in data.contact[:data.ncon]:
            geoms=(contact.geom1,contact.geom2)
            bodies=[int(model.geom_bodyid[g]) for g in geoms]
            for bid in bodies:
                if bid in self._damping_objects:
                    (supported if self._support_geoms.intersection(geoms) else touched).add(bid)
        for bid,dof in self._damping_objects.items():
            model.dof_damping[dof:dof+6]=self._original_damping[bid]
            if bid in supported and bid not in touched:
                model.dof_damping[dof:dof+3]+=.01
                model.dof_damping[dof+3:dof+6]+=.0002

    def _settle(self,backend):
        """Preparation ends only after all free objects remain physically quiet."""
        model,data=backend.model,backend.data
        dofs=[model.joint(o['id']+'_free').dofadr[0] for o in self.objects]
        quiet=0.;dt=model.opt.timestep
        maximum=20.
        for step in range(round(maximum/dt)):
            self._before_physics_step(backend)
            mujoco.mj_step(model,data)
            stable=all(np.linalg.norm(data.qvel[d:d+3])<.0005 and
                       np.linalg.norm(data.qvel[d+3:d+6])<.015 for d in dofs)
            quiet=quiet+dt if stable else 0.
            if (step+1)*dt>=1.5 and quiet>=.8:
                self.settle_report=dict(preparation_sim_s=(step+1)*dt,
                    quiet_window_s=quiet,max_linear_speed_m_s=max(np.linalg.norm(data.qvel[d:d+3]) for d in dofs),
                    max_angular_speed_rad_s=max(np.linalg.norm(data.qvel[d+3:d+6]) for d in dofs))
                return
        raise RuntimeError('Objects did not reach a stable resting state: '+str([(o['category'],float(np.linalg.norm(data.qvel[d:d+3])),float(np.linalg.norm(data.qvel[d+3:d+6]))) for o,d in zip(self.objects,dofs)]))

    def reset(self,seed=0):
        super().reset(seed)
        self.backend.sensor_rng=np.random.default_rng(seed)
        self.backend.camera['camera']['depth'].update(
            noise_std_m=.001,dropout_probability=.005)
        if hasattr(self.backend,"deployment_gap"):self.backend.deployment_gap.sensor(self.backend)
        return self.observe()

    def _build(self,rng):
        super()._build(rng)
        path=self.output/'scene.xml'
        tree=ET.parse(path)
        root=tree.getroot()
        # The light wrist-roll joint needs implicit damping under force clipping.
        # Preserve the existing servos on the other five joints and the gripper.
        root.find(".//joint[@name='joint6']").set('damping','6')
        root.find(".//position[@name='pos_joint6']").set('kv','0')
        # Prevent fast unloaded closing from crossing the physical zero stop and
        # trapping the pads on the wrong side of each other. Constraints remain
        # physical; no joint/object state is snapped or reset during operation.
        for name in ['joint7','joint8']:
            joint=root.find(f".//joint[@name='{name}']")
            joint.set('solreflimit','.004 1')
            joint.set('solimplimit','.999 .999 .0001 .5 2')
            joint.set('margin','.001')
        # Visible sensor shell is separate from the physical optical frame.
        assets=root.find('asset')
        label=self.output/'dabai_label.png'
        image=Image.new('RGB',(512,128),'#d2e4ec')
        draw=ImageDraw.Draw(image)
        draw.text((24,20),'DaBai  RGB-D',font=ImageFont.truetype('arial.ttf',52),fill='#163b50')
        image.save(label)
        ET.SubElement(assets,'texture',name='dabai_label',type='2d',file=str(label))
        ET.SubElement(assets,'material',name='dabai_label',texture='dabai_label',rgba='1 1 1 1',emission='.12')
        shell=root.find(".//geom[@name='wrist_sensor_shell']")
        if shell is not None:
            shell.set('material','dabai_label')
            shell.set('rgba','1 1 1 1')
        marker=root.find(".//body[@name='wrist_sensor_housing']")
        if marker is not None:
            ET.SubElement(marker,'geom',name='camera_status_led',type='sphere',
                pos='-.025 .007 -.002',size='.0025',rgba='0 .8 1 1',
                mass='0',contype='0',conaffinity='0',group='2')
        for light in tree.getroot().iter('light'):
            light.set('diffuse',numbers(rng.uniform(.65,.95,3)))
        # Finite-area pads retain a modest bounded spin friction, not an unlimited lock.
        for geom in root.iter('geom'):
            if geom.get('name','').startswith('object_') or geom.get('name','').endswith('_pad'):
                geom.set('condim','4')
                geom.set('friction',f"{self.cfg['physics']['friction']} 0.0005 0.00002")
                geom.set('solref','.01 1')
        contact=root.find('contact')
        if contact is None:contact=ET.SubElement(root,'contact')
        # Mild support rolling resistance is separate from finger/object contacts.
        # It damps unattended rolling without giving the gripper a torsional lock.
        mu=self.cfg['physics']['friction']
        for geom in root.iter('geom'):
            name=geom.get('name','')
            if name.startswith('object_') and geom.get('contype','1')!='0':
                for support in ['competition_table']+(['installation_table'] if self.cfg.get('venue') else [])+['tray_'+color+'_floor' for color in self.cfg['trays']]:
                    ET.SubElement(contact,'pair',geom1=name,geom2=support,condim='6',
                        friction=f'{mu} {mu} .00005 .00008 .00008',solref='.015 1')
        root.find('option').set('iterations','80')
        root.find('option').set('tolerance','1e-9')
        tree.write(path)

    def _add_scanned_object(self, root, body, name, category, asset, scale, mass):
        parts=[(np.array(p['pos'])*scale,np.array(p['size'])*scale) for p in asset['implicit_parts']]
        low=np.min([p-r for p,r in parts],axis=0)
        high=np.max([p+r for p,r in parts],axis=0)
        axes=[np.linspace(a,b,36) for a,b in zip(low,high)]
        xyz=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
        values=field(xyz,parts)
        occupied=xyz[(values<0)&(values>-asset.get('hollow_shell_fraction',float('inf')))]
        if len(occupied)<20:raise ValueError('Insufficient shell samples for inertia')
        center=occupied.mean(0)
        relative=occupied-center
        covariance=relative.T@relative/len(relative)
        inertia=mass*(np.trace(covariance)*np.eye(3)-covariance)
        ET.SubElement(body,'inertial',pos=numbers(center),mass=str(mass),
            fullinertia=numbers([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))
        assets = root.find('asset')
        ET.SubElement(assets,'mesh',name=name+'_implicit',
            file=str(ROOT/asset['visual_mesh']['file']),scale=numbers([scale]*3))
        ET.SubElement(assets,'texture',name=name+'_skin',type='2d',file=asset['skin'])
        ET.SubElement(assets,'material',name=name+'_skin',texture=name+'_skin',
            specular='.18',shininess='.18',rgba='1 1 1 1')
        ET.SubElement(body,'geom',type='mesh',mesh=name+'_implicit',mass='0',
            contype='0',conaffinity='0',material=name+'_skin',group='1')
        hi=np.array(asset['visual_mesh']['bounds_m'][1])*scale
        if category not in ('banana','orange','lemon','potato','cucumber'):
            ET.SubElement(body,'geom',type='capsule',size='.0018 .005',
                pos=numbers([0,0,hi[2]+.003]),rgba='.22 .12 .04 1',mass='0',
                contype='0',conaffinity='0',group='1')
        if category in ('apple','pear','strawberry','persimmon','bell_pepper'):
            ET.SubElement(body,'geom',type='ellipsoid',size='.008 .003 .001',
                pos=numbers([.005,0,hi[2]+.004]),rgba='.12 .34 .04 1',mass='0',
                contype='0',conaffinity='0',group='1')
        for i,part in enumerate(asset['implicit_parts']):
            ET.SubElement(body,'geom',name=f'{name}_implicit_{i}',type='ellipsoid',
                pos=numbers(np.array(part['pos'])*scale),
                size=numbers(np.array(part['size'])*scale),mass=str(mass/len(asset['implicit_parts'])),
                friction=f"{self.cfg['physics']['friction']} .00005 .00002",condim='3',
                rgba='0 0 0 0',group='3',solref='.005 1')


def configure(seed, categories, output, measured=False):
    cubes=('cube_red','cube_green','cube_blue')
    if not categories or len(set(categories))!=len(categories) or any(c not in FRUITS+cubes for c in categories):
        raise ValueError('Choose unique known fruit categories')
    rng = np.random.default_rng(seed)
    catalog = yaml.safe_load((ROOT/'config/competition/objects_scanned.yaml').read_text(encoding='utf-8'))['objects']
    catalog={key:catalog[key] for key in cubes}
    for category in [c for c in categories if c in FRUITS]:
        parts = components(category,rng)
        if measured:
            from piper_sim.measured_fruit import measured_parts
            parts=measured_parts(category,parts)
        path = output/f'{category}.obj'
        lo,hi = mesh(parts,path)
        skin=output/f'{category}_skin.png'
        texture(category,skin,rng)
        catalog[category] = dict(mass_kg_range=[.035,.10],
            visual_mesh=dict(file=str(path),bounds_m=[lo.tolist(),hi.tolist()]),
            skin=str(skin),
            implicit_parts=[dict(pos=p.tolist(),size=r.tolist()) for p,r in parts])
    catalog_path = output/'catalog.yaml'
    catalog_path.write_text(yaml.safe_dump({'objects':catalog}),encoding='utf-8')
    cfg = yaml.safe_load((ROOT/'config/competition/scene.yaml').read_text(encoding='utf-8'))
    cam = load_config()['camera']
    cfg['object_catalog'] = str(catalog_path)
    cfg['object_variation']['enabled'] = False
    cfg['robot_initial_qpos']=[0.,1.,-1.,0.,1.15,0.,.03,-.03]
    # Keep the fixture inside the wrist camera's observable/reachable workspace.
    # The original DaBai cannot supply depth below 30 cm at the near table edge.
    cfg['spawn'].update(x_m=[.33,.52],y_m=[-.10,.18],scale_range=[1,1],tray_clearance_m=.06)
    cfg['physics']['friction'] = float(rng.uniform(.7,1.2))
    # Procedural banana reaches 14 cm; upstream 11 cm-wide demo trays can reject it.
    for tray in cfg['trays'].values():
        tray['half_size_xy_m']=[.070,.095]
        tray['center_xy_m'][1]=-.245
    cfg['camera'].update(mount='wrist',width=cam['width'],height=cam['height'],
        intrinsics=cam['intrinsics'],depth_range_m=[.3,3])
    cfg['camera']['visual']=load_config()['simulation']['camera_visual']
    cfg['camera']['visual']['bracket_size_m']=[.045,.070,.004]
    cfg['tasks']['l2']['mapping'].update(cfg['tasks']['l1']['mapping'])
    cfg['tasks']['l2'].pop('object_pool',None)
    cfg['tasks']['l2']['objects'] = categories
    cfg['tasks']['l2']['time_limit_s'] = 1800
    path = output/'scene_config.yaml'
    path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    return path


