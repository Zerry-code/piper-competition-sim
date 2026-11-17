import numpy as np
import yaml
from types import SimpleNamespace
from piper_sim.l1_world import configure_l1,L1World,detect_card_targets


def test_document_dimensions_colors_and_flat_cards(tmp_path):
    config=configure_l1(721,tmp_path);cfg=yaml.safe_load(config.read_text(encoding='utf-8'))
    assert set(cfg['trays'])=={'red','green','yellow'}
    catalog=yaml.safe_load((tmp_path/'catalog.yaml').read_text(encoding='utf-8'))['objects']
    assert set(catalog)=={'cube_red','cube_green','cube_yellow'}
    for asset in catalog.values():assert asset['parts'][0]['size']==[.025]*3
    for card in cfg['trays'].values():assert card['half_size_xy_m']==[.1,.1]
    world=L1World('l2',config,tmp_path,render=False)
    try:
        world.reset(721)
        names=[world.backend.model.geom(g).name for g in range(world.backend.model.ngeom)]
        assert 'tray_yellow_floor' in names
        assert not any(name.startswith('tray_') and not name.endswith('_floor') for name in names)
    finally:world.close()


def test_card_detection_uses_color_and_measured_plane_not_colored_cube():
    rgb=np.zeros((30,30,3),dtype=np.uint8);rgb[:]=[230,195,8]
    obs=SimpleNamespace(rgb=rgb,depth_m=np.full((30,30),.0605),
        intrinsics=dict(fx=10,fy=10,cx=-80,cy=0),T_base_end=np.eye(4))
    calibration={'hand_eye':{'T_end_camera':np.eye(4).tolist()}}
    assert 'yellow' in detect_card_targets(obs,calibration,.06)
    obs.depth_m[:]=.11
    assert not detect_card_targets(obs,calibration,.06)
