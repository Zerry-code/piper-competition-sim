import yaml
import numpy as np
from piper_sim.l1_world import configure_l1,card_observation_xy


def test_layout_has_10cm_edge_gaps_and_color_shuffle(tmp_path):
    arrangements=[]
    for seed in [930,931,932,933,934]:
        output=tmp_path/str(seed);output.mkdir()
        cfg=yaml.safe_load(configure_l1(seed,output).read_text(encoding='utf-8'))
        positions=[t['reference_center_xy_m'] for t in cfg['trays'].values()]
        assert all(abs(t['center_xy_m'][1]-t['reference_center_xy_m'][1])<=.015 for t in cfg['trays'].values())
        assert sorted(p[1] for p in positions)==[-.30,0,.30]
        assert all(np.isclose(p[0],.5865) for p in positions)
        ordered=sorted(cfg['trays'].values(),key=lambda t:t['center_xy_m'][1])
        for a,b in zip(ordered,ordered[1:]):
            gap=b['reference_center_xy_m'][1]-b['half_size_xy_m'][1]-a['reference_center_xy_m'][1]-a['half_size_xy_m'][1]
            assert np.isclose(gap,.10)
        arrangements.append(tuple(tuple(cfg['trays'][c]['center_xy_m']) for c in ['red','green','yellow']))
        for color in cfg['trays']:
            view=card_observation_xy(cfg,color)
            assert np.sign(view[1])==np.sign(cfg['trays'][color]['center_xy_m'][1])
    assert len(set(arrangements))>1


def test_same_seed_replays_same_color_layout(tmp_path):
    values=[]
    for name in ['first','second']:
        output=tmp_path/name;output.mkdir()
        values.append(yaml.safe_load(configure_l1(931,output).read_text(encoding='utf-8'))['trays'])
    assert values[0]==values[1]


def test_preview_config_and_seam_gap_without_spawn_block():
    from piper_sim.l1_world import configure_card_regions
    cfg={'table':{},'tray_geometry':{}}
    configure_card_regions(cfg,7)
    seam=cfg['venue']['installation_depth_m']-cfg['venue']['base_from_rear_edge_m']
    for tray in cfg['trays'].values():
        assert np.isclose(tray['center_xy_m'][0]-tray['half_size_xy_m'][0]-seam,.05)
    assert cfg['enforce_time_limit'] is False
