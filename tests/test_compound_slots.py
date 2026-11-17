from types import SimpleNamespace
import numpy as np
import pytest
from piper_sim.scene_memory import SceneMemory
from piper_sim.task_plan import resolve_plan


def scene():
    trays={c:dict(center_xy_m=xy,half_size_xy_m=[.15,.15]) for c,xy in [('red',[.2,-.25]),('green',[.5,-.25]),('yellow',[.65,.08])]}
    memory=SceneMemory(trays)
    memory.cards={'red':np.array([.27,-.25]),'green':np.array([.47,-.25]),'yellow':np.array([.565,.08])}
    memory.seen_plane={(x,y) for x in range(0,82) for y in range(-45,46)}
    memory.objects={c:SimpleNamespace(center=np.array([*xy,.085]),yaw=0) for c,xy in [('cube_red',[.565,.08]),('cube_green',[.4,.12]),('cube_yellow',[.36,.22])]}
    return memory


def test_corrected_user_sequence_uses_empty_red_card_and_excludes_completed_red():
    memory=scene()
    plan={'steps':[dict(source='cube_green',destination='green'),dict(source='remaining',destination='empty_card')]}
    actions=resolve_plan(plan,memory,['cube_red','cube_green','cube_yellow'],completed=['cube_red'])
    assert [(a['category'],a['destination']) for a in actions]==[('cube_green','green'),('cube_yellow','red')]


def test_all_same_color_expands_all_objects_even_previously_completed():
    actions=resolve_plan({'steps':[dict(source='all',destination='same_color')]},scene(),
                         ['cube_red','cube_green','cube_yellow'],completed=['cube_red'])
    assert [(a['category'],a['destination']) for a in actions]==[
        ('cube_red','red'),('cube_green','green'),('cube_yellow','yellow')]


def test_remaining_same_color_excludes_completed_objects():
    actions=resolve_plan({'steps':[dict(source='remaining',destination='same_color')]},scene(),
                         ['cube_red','cube_green','cube_yellow'],completed=['cube_red'])
    assert [(a['category'],a['destination']) for a in actions]==[
        ('cube_green','green'),('cube_yellow','yellow')]


def test_occupied_card_can_offer_separate_space_but_never_same_landing_point():
    memory=scene();point=memory.free_position('yellow',exclude='cube_green')
    assert point is not None
    assert np.linalg.norm(point-memory.objects['cube_red'].center[:2])>=.095


def test_unobserved_plane_cannot_be_assumed_free():
    memory=scene();memory.seen_plane.clear()
    assert memory.free_position('green') is None


def test_no_empty_card_rejects_whole_plan_before_execution():
    memory=scene()
    memory.objects['cube_green'].center[:2]=[.47,-.25]
    memory.objects['cube_yellow'].center[:2]=[.27,-.25]
    with pytest.raises(ValueError,match='没有足够'):
        resolve_plan({'steps':[dict(source='cube_yellow',destination='empty_card')]},memory,['cube_red','cube_green','cube_yellow'])


def test_command_hint_is_not_fresh_visual_evidence():
    memory=scene();memory.moved('cube_red',[.57,.08])
    assert 'cube_red' not in memory.objects
    assert 'cube_red' in memory.hints
    assert not memory.seen_plane


def test_adaptive_scan_stops_without_extra_arm_motion_when_fresh_evidence_is_ready():
    import sys
    from piper_sim.calibration import ROOT
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_l1 import L1Skills
    skills=object.__new__(L1Skills)
    skills.scene_memory=scene();skills.card_targets=skills.scene_memory.cards
    skills.survey_views=0;skills.navigation_events=[]
    estimate=SimpleNamespace(points=np.zeros((120,3)))
    skills.capture_candidates=lambda:[('cube_red',estimate)]
    def unexpected(*args):raise AssertionError('Ready sensor evidence must not cause a full-table rescan')
    skills.view=unexpected
    skills.scan(['cube_red'],['green'])
    assert skills.survey_views==0
