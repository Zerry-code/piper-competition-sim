"""Regression for L1 cards disappearing under hand-eye/depth stress errors."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import piper_sim.l1_world as cards
import piper_sim.scene_memory as memory
from piper_sim.table_plane import tabletop_residual


def observed_scene(seed=12):
    y,x=np.mgrid[:150,:150]
    px=.25+x*.002;py=-.15+y*.002
    z=.067+.018*py+np.random.default_rng(seed).normal(0,.004,x.shape)
    rgb=np.empty((*x.shape,3),dtype=np.uint8);rgb[:]=[230,195,8]
    cube=(px>.48)&(py>.08)
    z[cube]+=.05
    rgb[cube]=[230,8,8]
    points=np.stack([px,py,z],axis=-1)
    obs=NS(rgb=rgb,depth_m=np.ones(x.shape),intrinsics=dict(fx=1,fy=1,cx=0,cy=0),T_base_end=np.eye(4))
    return obs,points,cube


@pytest.mark.parametrize('seed',[12,28,43])
def test_noisy_biased_card_survives_without_accepting_cube(monkeypatch,seed):
    obs,points,cube=observed_scene(seed)
    monkeypatch.setattr(cards,'camera_to_base',lambda *a:points.reshape(-1,3))
    detected=cards.detect_card_targets(obs,{},.06)
    assert set(detected)=={'yellow'}
    assert np.linalg.norm(detected['yellow']-[.399,-.001])<.025


def test_floor_support_and_obstacles_use_same_observed_plane(monkeypatch):
    obs,points,cube=observed_scene()
    monkeypatch.setattr(memory,'camera_to_base',lambda *a:points.reshape(-1,3))
    scene=memory.SceneMemory({'yellow':dict(center_xy_m=[.4,0],half_size_xy_m=[.15,.15])})
    scene.observe(obs,{},.06,[],{'yellow':np.array([.4,0])})
    assert scene.free_position('yellow') is not None
    assert len(scene.obstacles)>0
    assert (scene.obstacles[:,0]>.46).mean()>.95


def test_unreasonable_plane_fit_does_not_relabel_cube_as_table():
    obs,points,cube=observed_scene()
    points[...,2]+=.05
    residual,sigma=tabletop_residual(points,.06)
    assert np.median(residual)>.05

def test_scan_reports_missing_color_and_preserves_per_frame_evidence():
    from run_l1 import L1Skills
    skills=object.__new__(L1Skills)
    skills.scene_memory=NS(cards={},hints={},frames=0,free_position=lambda *a,**k:None)
    skills.card_targets=skills.scene_memory.cards
    skills.survey_views=0;skills.navigation_events=[]
    skills.w=NS(cfg={'trays':{'yellow':dict(center_xy_m=[.22,-.26])}})
    skills.capture_candidates=lambda:[]
    skills.view=lambda *a:[]
    with pytest.raises(RuntimeError,match='红色方块、黄色卡纸'):
        skills.scan(['cube_red'],['yellow'])
    assert skills.navigation_events[-1]['event']=='l1_survey_incomplete'
    assert skills.navigation_events[-1]['missing']==['cube_red','yellow']
    assert skills.survey_views>0
    assert any(e['event']=='l1_visual_confirmation' for e in skills.navigation_events)


def test_scan_retries_missing_paper_from_closer_view_only():
    from run_l1 import L1Skills
    skills=object.__new__(L1Skills)
    skills.scene_memory=NS(cards={},hints={},frames=0,free_position=lambda *a,**k:np.array([.3,-.2]))
    skills.card_targets=skills.scene_memory.cards
    skills.survey_views=0;skills.navigation_events=[]
    skills.w=NS(cfg={'trays':{'yellow':dict(center_xy_m=[.22,-.26])}})
    estimate=NS(points=np.ones((120,3)))
    skills.capture_candidates=lambda:[('cube_red',estimate)]
    heights=[]
    def view(xy,height):
        heights.append(height)
        if height==.40:skills.card_targets['yellow']=np.array([.3,-.2])
        return []
    skills.view=view
    skills.scan(['cube_red'],['yellow'])
    assert heights[-1]==.40
    assert skills.navigation_events[-1]['event']=='adaptive_survey'
