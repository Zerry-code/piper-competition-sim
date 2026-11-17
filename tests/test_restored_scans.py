import yaml
from piper_sim.document_fruits import configure_fruits
from piper_sim.calibration import ROOT

def test_original_scanned_mesh_texture_collision_are_restored(tmp_path):
    configure_fruits(930,tmp_path)
    catalog=yaml.safe_load((tmp_path/'catalog.yaml').read_text(encoding='utf-8'))['objects']
    for category in ['apple','pear','banana']:
        item=catalog[category]
        assert item['source_id'] and 'implicit_parts' not in item
        assert (ROOT/item['visual_mesh']['file']).exists()
        assert (ROOT/item['visual_mesh']['texture']).exists()
        assert item['collision_meshes']
