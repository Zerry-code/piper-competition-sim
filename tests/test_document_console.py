import yaml
from piper_sim.document_fruits import configure_fruits,CATEGORIES
from piper_sim.implicit_fruit import components
import numpy as np


def test_document_fruit_catalog_has_textured_scans_and_six_required_types(tmp_path):
    config=configure_fruits(882,tmp_path)
    cfg=yaml.safe_load(config.read_text(encoding='utf-8'))
    assert cfg['tasks']['l2']['objects']==CATEGORIES
    catalog=yaml.safe_load((tmp_path/'catalog.yaml').read_text(encoding='utf-8'))['objects']
    for name in ['pear','banana']:
        assert catalog[name]['collision_meshes']
        assert catalog[name]['source_id']
        assert 'texture' in catalog[name]['visual_mesh']
    for name in ['persimmon','bell_pepper','cucumber','potato']:
        assert catalog[name]['implicit_parts']
        assert 'skin' in catalog[name]


def test_vegetable_shapes_are_not_all_spherical():
    rng=np.random.default_rng(12)
    pepper=components('bell_pepper',rng);cucumber=components('cucumber',rng)
    assert len(pepper)==4
    assert len(cucumber)>10
    low=np.min([p-r for p,r in cucumber],axis=0);high=np.max([p+r for p,r in cucumber],axis=0)
    assert high[0]-low[0]>3*(high[1]-low[1])


def test_fruit_destinations_are_the_same_flat_cards_as_l1(tmp_path):
    from piper_sim.l1_world import configure_l1
    from piper_sim.document_fruits import DocumentFruitWorld
    blocks=tmp_path/'blocks';blocks.mkdir()
    fruits=tmp_path/'fruits';fruits.mkdir()
    a=yaml.safe_load(configure_l1(882,blocks).read_text(encoding='utf-8'))
    b=yaml.safe_load(configure_fruits(882,fruits).read_text(encoding='utf-8'))
    assert {c:{k:v for k,v in t.items() if k not in ('placement_policy','allocation_mode')} for c,t in b['trays'].items()}==a['trays']
    assert all(t['placement_policy']=='center_max_coverage' for t in b['trays'].values())
    assert a['tray_geometry']==b['tray_geometry']
    assert set(b['tasks']['l2']['mapping'].values())=={'red','green','yellow'}
    world=DocumentFruitWorld('l2',fruits/'scene_config.yaml',fruits,render=False)
    try:
        world.reset(882)
        model=world.backend.model
        names=[model.geom(g).name for g in range(model.ngeom)]
        assert not any(name.startswith('tray_') and not name.endswith('_floor') for name in names)
        for color in ['red','green','yellow']:
            assert np.allclose(model.geom('tray_'+color+'_floor').size,[.1,.1,.00025])
    finally:world.close()
