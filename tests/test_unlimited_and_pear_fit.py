from types import SimpleNamespace as NS
import time
import numpy as np
from run_harvest import FruitSkills
from piper_sim.l1_world import L1World,configure_l1
from piper_sim.grasp_geometry import ellipsoid_pear_grasp

def test_unlimited_simulation_advances_past_reference_time(tmp_path):
    w=L1World('l2',configure_l1(721,tmp_path),tmp_path,render=False)
    try:
        w.reset(721);w.cfg['tasks']['l2']['time_limit_s']=.01
        w.step(np.zeros(7),observe=False);w.step(np.zeros(7),observe=False)
        assert w.elapsed_s>.01
        assert not w.evaluation()['timed_out']
        before=w.elapsed_s;w.cfg['enforce_time_limit']=True;w.step(np.zeros(7),observe=False)
        assert w.elapsed_s==before
    finally:w.close()

def test_wall_deadline_is_disabled_in_unlimited_mode():
    skill=FruitSkills.__new__(FruitSkills)
    skill.w=NS(cfg={'enforce_time_limit':False,'tasks':{'l2':{'time_limit_s':1}}})
    skill.wall_started=time.monotonic()-10000
    skill.check_deadline()

def test_pear_surface_fit_recovers_body_center_and_short_axis():
    from piper_sim.measured_fruit import ellipsoid_axes
    radii=ellipsoid_axes(.33,.27)/2
    u,v=np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(.1,np.pi/2,50))
    center=np.array([.35,.30,radii[2]])
    points=np.c_[radii[0]*np.sin(v).ravel()*np.cos(u).ravel(),radii[1]*np.sin(v).ravel()*np.sin(u).ravel(),radii[2]*np.cos(v).ravel()]+center
    fitted,yaw,quality=ellipsoid_pear_grasp(points,0.)
    assert np.linalg.norm(fitted-center)<.003
    assert abs(np.sin(yaw))<.05
    assert quality['opening_m']<.10

def test_unlimited_release_retries_do_not_stop_at_three():
    from piper_sim.harvest import Harvest,Track,RetryableFailure
    from piper_sim.release_safety import ReleaseBlocked
    class Skills:
        keep_retrying=True
        retries=0
        def overview(self):return [Track('pear','pear',[.3,.2,.04],'green')]
        def observe_target(self,*a,**k):pass
        def pick(self,t):return True
        def place_and_verify(self,t):t.release_pending=True;raise RetryableFailure('wait for release',phase='release')
        def resume_release(self,t):
            self.retries+=1
            if self.retries<6:raise ReleaseBlocked('still blocked')
            t.release_pending=False;return True
        def wait_before_retry(self,*a):pass
    skills=Skills();result=Harvest(skills,max_attempts=1).run()
    assert result['status']=='complete' and skills.retries==6


def test_unlimited_audit_reobserves_unknown_until_confirmed():
    from piper_sim.batch_audit import BatchBoundaryAudit
    class Skills:
        keep_retrying=True
        calls=0
        def scan_batch_boundaries(self,targets):
            self.calls+=1
            return {'pear':dict(status='inside' if self.calls>=4 else 'unknown',reason='need another image')}
        def wait_before_retry(self,*a):pass
    skills=Skills();target=NS(key='pear',category='pear')
    result=BatchBoundaryAudit(skills,max_correction_rounds=1).run([target],dict(released=['pear'],completed=[],unresolved=[]))
    assert result['status']=='complete' and skills.calls==4
