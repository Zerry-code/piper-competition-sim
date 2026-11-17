from types import SimpleNamespace as NS
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_harvest as runtime
from piper_sim.harvest import Harvest,Track,RetryableFailure


def recovery_skill(monkeypatch,support_after_failure):
    skill=runtime.FruitSkills.__new__(runtime.FruitSkills);calls=[];state={'lowered':False}
    skill.b=NS(model=NS(site=lambda n:NS(id=0)),data=NS(site_xpos=np.array([[.4,.1,.20]])),previous_target=np.zeros(7),measured=lambda:np.zeros(7))
    skill.plan={'center':np.array([.4,.1,.09]),'yaw':0.};skill.navigation_events=[]
    def lower(dz,width):
        calls.append('lower');state['lowered']=True;raise TimeoutError('contact-limited lowering')
    skill.skill=NS(grip_monitor=NS(disarm=lambda:None),release_motion_locked=False,table=.06,hold=lambda seconds:None,vertical_release_move=lower,release_and_retreat=lambda plan:calls.append('release'))
    monkeypatch.setattr(runtime,'contact_quality',lambda backend:{'loaded':True})
    monkeypatch.setattr(runtime,'retained_support',lambda backend:state['lowered'] and support_after_failure)
    monkeypatch.setattr(runtime,'release_feedback',lambda backend:dict(opened=False,object_contact=True,tray_contact=False))
    return skill,calls


def test_supported_object_can_release_after_downward_tracking_failure(monkeypatch):
    skill,calls=recovery_skill(monkeypatch,True)
    skill.recover(Track('pear','pear',[.4,.1,.09],'green'))
    assert calls==['lower','release']
    assert any(e['event']=='retained_object_supported' for e in skill.navigation_events)


def test_unsupported_object_is_never_opened_after_failed_lowering(monkeypatch):
    skill,calls=recovery_skill(monkeypatch,False)
    with pytest.raises(RetryableFailure):skill.recover(Track('pear','pear',[.4,.1,.09],'green'))
    assert calls==['lower']


class Skills:
    last_attempt_closed=True
    def __init__(self,permanent=False):self.calls=[];self.picks=0;self.errors=0;self.permanent=permanent
    def overview(self):return [Track('pear','pear',[.4,-.1,.09],'green'),Track('next','apple',[.4,.1,.09],'red')]
    def observe_target(self,t,retry):self.calls.append(('observe',t.key))
    def pick(self,t):self.picks+=1;return self.picks>1
    def place_and_verify(self,t):return True
    def recover(self,t):
        self.calls.append(('recover',t.key));self.errors+=1
        if self.permanent or self.errors==1:raise RetryableFailure('still holding, lower blocked')
    def wait_before_retry(self,*args):pass


def test_transient_retained_recovery_retries_before_any_next_target():
    s=Skills();report=Harvest(s).run()
    assert report['status']=='complete'
    assert s.calls[:4]==[('observe','pear'),('recover','pear'),('recover','pear'),('observe','pear')]
    assert report['completed']==['pear','next']


def test_permanent_retained_block_keeps_all_pending_and_never_moves_to_next():
    s=Skills(permanent=True);report=Harvest(s).run()
    assert report['status']=='incomplete' and report['stop_reason']=='retained_object_recovery_blocked'
    assert set(report['unresolved'])=={'pear','next'}
    assert ('observe','next') not in s.calls
