import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_harvest as runtime

def setup(monkeypatch,candidates,landing=(.4,0)):
    skill=runtime.FruitSkills.__new__(runtime.FruitSkills)
    skill.b=NS(data=NS(time=0.));skill.skill=NS(release_verified=True,retreat_verified=True,table=.06)
    skill.skill.hold=lambda seconds:setattr(skill.b.data,'time',skill.b.data.time+seconds)
    skill.w=NS(cfg={'trays':{'red':dict(center_xy_m=[.4,0],half_size_xy_m=[.15,.15])}})
    skill.boundary_candidates=lambda:candidates
    views=[];skill.view=lambda *args,**kwargs:views.append(args)
    skill.navigation_events=[];skill.bin_placements={'red':0}
    monkeypatch.setattr(runtime,'release_feedback',lambda b:{})
    return skill,NS(key='banana',category='banana',destination='red',destination_position=landing),views

def cloud(x0,x1):
    x,y=np.meshgrid(np.linspace(x0,x1,6),np.linspace(-.02,.02,5))
    return NS(points=np.c_[x.ravel(),y.ravel(),np.ones(x.size)*.09])

def test_inside_boundary_is_one_frame_without_identity_or_wait(monkeypatch):
    skill,target,views=setup(monkeypatch,[cloud(.38,.42)])
    assert skill.verify_placement(target)
    assert target.boundary_check_status=='inside'
    assert skill.b.data.time==0 and not views

def test_missing_image_is_unknown_after_only_one_local_view(monkeypatch):
    skill,target,views=setup(monkeypatch,[])
    assert not skill.verify_placement(target)
    assert target.boundary_check_status=='unknown'
    assert len(views)==1 and skill.b.data.time==0

def test_outside_boundary_returns_observed_local_retry_anchor(monkeypatch):
    skill,target,views=setup(monkeypatch,[cloud(.49,.56)],(.49,0))
    assert not skill.verify_placement(target)
    assert target.boundary_check_status=='outside'
    assert .52<target.position[0]<.54 and not views


def test_small_visible_fragment_is_not_enough_to_claim_inside(monkeypatch):
    skill,target,views=setup(monkeypatch,[cloud(.39,.41)])
    target.estimate=NS(points=cloud(.33,.47).points)
    assert not skill.verify_placement(target)
    assert target.boundary_check_status=='unknown' and len(views)==1

def test_other_reserved_item_is_not_mistaken_for_just_released_item(monkeypatch):
    skill,target,views=setup(monkeypatch,[cloud(.44,.46)])
    target.placement_reservations={'banana':np.array([.4,0]),'potato':np.array([.45,0])}
    assert not skill.verify_placement(target)
    assert target.boundary_check_status=='unknown'
