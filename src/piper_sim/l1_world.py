"""L1 document fixture: 50 mm RGB-yellow cubes and 200 mm flat cardstock."""
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import cv2
from piper_sim.fruit_world import ImplicitWorld,configure
from piper_sim.calibration import ROOT,camera_to_base
from piper_sim.table_plane import tabletop_residual

COLORS={'red':[.72,.025,.025,1],'green':[.12,.38,.025,1],'yellow':[.9,.75,.03,1]}


def configure_card_regions(cfg,seed=0):
    """Shared physical target definition for both document task routes."""
    venue=yaml.safe_load((ROOT/'config/venue.yaml').read_text(encoding='utf-8-sig'))
    cfg['venue']=venue
    cfg['enforce_time_limit']=venue.get('enforce_time_limit',False)
    cfg.setdefault('spawn',{})['venue_edge_row']=True
    seam=venue['installation_depth_m']-venue['base_from_rear_edge_m']
    depth=venue['total_depth_m']-venue['installation_depth_m']
    cfg['table'].update(center_xy_m=[seam+depth/2,0],half_size_xy_m=[depth/2,venue['object_table_width_m']/2],top_m=0.)
    cfg['support_surface']=dict(center_xy_m=[venue['total_depth_m']/2-venue['base_from_rear_edge_m'],0],half_size_xy_m=[venue['total_depth_m']/2,venue['installation_width_m']/2])
    slots=[[seam+venue['card_seam_gap_m']+venue['card_size_m']/2,p[1]] for p in venue['card_centers_xy_m']] if 'card_seam_gap_m' in venue else venue['card_centers_xy_m']
    colors=np.random.default_rng(np.random.SeedSequence([seed,4821])).permutation(list(COLORS))
    cfg['trays']={str(color):dict(center_xy_m=center,half_size_xy_m=[venue['card_size_m']/2]*2,rgba=COLORS[color]) for color,center in zip(colors,slots)}
    for color,tray in cfg['trays'].items():
        tray['reference_center_xy_m']=list(tray['center_xy_m'])
        tray['center_xy_m']=(np.asarray(tray['center_xy_m'])+np.array([0.,np.random.default_rng(np.random.SeedSequence([seed,913,list(COLORS).index(color)])).uniform(-.015,.015)])).tolist()
    cfg['tray_geometry'].update(floor_thickness_m=.0005,wall_height_m=0.,wall_thickness_m=0.)


def remove_card_walls(path):
    tree=ET.parse(path)
    for body in tree.getroot().iter('body'):
        if body.get('name','').startswith('tray_'):
            for geom in list(body.findall('geom')):
                if not geom.get('name','').endswith('_floor'):body.remove(geom)
    tree.write(path)


def configure_l1(seed,output):
    path=configure(seed,['cube_red','cube_green','cube_blue'],output)
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    catalog={'objects':{ 'cube_'+color:dict(mass_kg_range=[.06,.09],parts=[dict(type='box',size=[.025]*3,pos=[0,0,0],rgba=rgba)]) for color,rgba in COLORS.items()}}
    (output/'catalog.yaml').write_text(yaml.safe_dump(catalog),encoding='utf-8')
    configure_card_regions(cfg,seed)
    cfg['spawn'].update(x_m=[.40,.56],y_m=[-.48,.48],tray_clearance_m=.015,scale_range=[1,1])
    cfg['tasks']['l2'].update(objects=['cube_'+c for c in COLORS],mapping={'cube_'+c:c for c in COLORS})
    cfg['tasks']['l2']['time_limit_s']=300
    path.write_text(yaml.safe_dump(cfg),encoding='utf-8');return path


class L1World(ImplicitWorld):
    def _build(self,rng):
        super()._build(rng)
        remove_card_walls(self.output/'scene.xml')


def detect_card_targets(obs,calibration,table,expected_trays=None):
    """Visible colored plane only; cubes rejected by measured height above plane."""
    z=obs.depth_m;k=obs.intrinsics;v,u=np.indices(z.shape);valid=np.isfinite(z)&(z>0)
    camera=np.stack([(u-k['cx'])*z/k['fx'],(v-k['cy'])*z/k['fy'],z],-1)
    base=np.zeros_like(camera);base[valid]=camera_to_base(camera[valid],obs.T_base_end,calibration)
    residual,sigma=tabletop_residual(base[valid],table)
    height=np.full(z.shape,np.inf);height[valid]=residual
    valid&=(abs(height)<min(.012,max(.004,2.5*sigma)))&(base[...,0]>.05)&(base[...,0]<.95)&(abs(base[...,1])<.45)
    hsv=cv2.cvtColor(obs.rgb,cv2.COLOR_RGB2HSV)
    masks={'red':((hsv[...,0]<12)|(hsv[...,0]>170)), 'green':((hsv[...,0]>35)&(hsv[...,0]<90)), 'yellow':((hsv[...,0]>18)&(hsv[...,0]<35))}
    result={}
    for color,mask in masks.items():
        region=valid.copy()
        if expected_trays and color in expected_trays:
            prior=expected_trays[color]
            reference=np.asarray(prior.get('reference_center_xy_m',prior['center_xy_m']))
            region&=np.all(abs(base[...,:2]-reference)<np.asarray(prior['half_size_xy_m'])+.04,axis=2)
        pixels=(region&mask&(hsv[...,1]>90)&(hsv[...,2]>35)).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(pixels)
        if count<2:continue
        index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        points=base[labels==index]
        # Prefer a visible interior patch reachable by the arm, rather than the
        # center of a large partially visible card beyond the robot workspace.
        points=points[(points[:,0]>.22)&(points[:,0]<.95)]
        if len(points)<120:continue
        lo,hi=np.percentile(points[:,:2],[2,98],axis=0)
        if min(hi-lo)<.15:continue
        rectangle=cv2.minAreaRect(points[:,:2].astype(np.float32))
        result[color]=np.asarray(rectangle[0],dtype=float)
    return result


def card_observation_xy(cfg,color):
    point=np.array(cfg['trays'][color]['center_xy_m'],dtype=float)
    return point
