import numpy as np
import yaml
from piper_sim.sim2real import DeploymentGap
from piper_sim.calibration import ROOT
from piper_sim.l1_world import L1World,configure_l1


def profile(name):return yaml.safe_load((ROOT/'config/sim2real.yaml').read_text(encoding='utf-8'))['profiles'][name]

def test_seeded_gap_repeatable_and_baseline_identity():
    a=DeploymentGap(profile('mild'),91);b=DeploymentGap(profile('mild'),91)
    assert a.report()==b.report()
    z=DeploymentGap(profile('baseline'),91)
    assert z.scale==1 and np.all(z.translation==0) and np.all(z.rotation==0)

def test_delay_fifo_preserves_command_order():
    gap=DeploymentGap(profile('stress'),1);gap.applied=np.zeros(7)
    np.testing.assert_equal(gap.command(np.ones(7)),np.zeros(7))
    np.testing.assert_equal(gap.command(np.ones(7)*2),np.zeros(7))
    np.testing.assert_equal(gap.command(np.ones(7)*3),np.ones(7))
    np.testing.assert_equal(gap.command(np.ones(7)*4),np.ones(7)*2)

def test_physical_camera_bias_does_not_correct_nominal_calibration(tmp_path):
    cfg=configure_l1(91,tmp_path);data=yaml.safe_load(cfg.read_text());data['deployment_gap']=profile('mild');cfg.write_text(yaml.safe_dump(data))
    world=L1World('l2',cfg,tmp_path,render=False)
    try:
        world.reset(91);backend=world.backend
        nominal=np.array(backend.camera['hand_eye']['T_end_camera'])
        np.testing.assert_equal(nominal,np.array(yaml.safe_load((ROOT/'config/eye_in_hand.yaml').read_text(encoding='utf-8'))['hand_eye']['T_end_camera']))
        assert backend.deployment_gap.profile['delay_steps']==1
        assert len(world.deployment_report['objects'])==3
        assert all(.85<=o['mass_factor']<=1.15 for o in world.deployment_report['objects'])
        assert backend.camera['camera']['depth']['noise_std_m']==.002
    finally:world.close()


def test_display_rgb_does_not_consume_sensor_rng_or_render_depth(tmp_path):
    from types import SimpleNamespace as NS
    world=L1World('l2',configure_l1(91,tmp_path),tmp_path,render=False)
    try:
        world.reset(91);backend=world.backend;calls=[]
        backend.renderer=NS(disable_depth_rendering=lambda:calls.append('rgb'),update_scene=lambda *a,**k:None,render=lambda:np.zeros((2,2,3),dtype=np.uint8),close=lambda:None)
        before=repr(backend.sensor_rng.bit_generator.state)
        assert backend.camera_rgb().shape==(2,2,3)
        assert repr(backend.sensor_rng.bit_generator.state)==before
        assert calls==['rgb']
    finally:world.close()
