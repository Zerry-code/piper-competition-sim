from types import SimpleNamespace as NS
import numpy as np
from piper_sim.demo_skill import PickPlaceSkill
from piper_sim.timing import summarize

def test_stable_contact_ends_unused_closing_stroke(monkeypatch):
    import piper_sim.trajectory as trajectory
    monkeypatch.setattr(trajectory,'trajectory',lambda *a,**kw:[np.zeros(7) for _ in range(40)])
    skill=PickPlaceSkill.__new__(PickPlaceSkill);count=[0]
    skill.cfg={'grasp_check':lambda:count[0]>=3}
    skill.b=NS(previous_target=np.r_[np.zeros(6),.07],data=NS(time=0.),interface={},measured=lambda:np.r_[np.zeros(6),.04])
    skill.motion_profile=lambda:1.
    def tick(goal):count[0]+=1;skill.b.data.time+=.05
    skill.tick=tick;skill.hold=lambda seconds:None
    skill.close_until_stable()
    assert 5<=count[0]<40

def test_missing_contact_keeps_full_close_attempt(monkeypatch):
    import piper_sim.trajectory as trajectory
    monkeypatch.setattr(trajectory,'trajectory',lambda *a,**kw:[np.zeros(7) for _ in range(10)])
    skill=PickPlaceSkill.__new__(PickPlaceSkill);count=[0]
    skill.cfg={'grasp_check':lambda:False}
    skill.b=NS(previous_target=np.r_[np.zeros(6),.07],data=NS(time=0.),interface={},measured=lambda:np.zeros(7))
    skill.motion_profile=lambda:1.
    def tick(goal):count[0]+=1;skill.b.data.time+=.05
    skill.tick=tick;skill.hold=lambda seconds:None
    skill.close_until_stable();assert count[0]==10

def test_wall_and_sim_timings_are_separate():
    r=summarize([dict(stage='planning',sim=0,wall=0),dict(stage='move',sim=0,wall=3)],2,5)
    assert r['stages']['planning']==dict(sim_s=0,wall_s=3)
    assert r['stages']['move']==dict(sim_s=2,wall_s=2)
