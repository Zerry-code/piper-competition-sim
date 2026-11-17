from types import SimpleNamespace as NS
import numpy as np
import piper_sim.fruit_perception as perception

def test_tilted_biased_table_is_not_merged_into_fruit(monkeypatch):
    y,x=np.mgrid[0:120,0:140]
    px=.25+x*.0025;py=-.10+y*.0025
    rng=np.random.default_rng(12)
    z=.071+.025*py+rng.normal(0,.001,x.shape)
    fruit=(px-.42)**2+(py-.05)**2<.025**2
    z[fruit]+=.03+np.sqrt(np.maximum(0,.025**2-(px[fruit]-.42)**2-(py[fruit]-.05)**2))
    base=np.stack([px,py,z],axis=-1)
    monkeypatch.setattr(perception,'camera_to_base',lambda p,t,c:base.reshape(-1,3))
    obs=NS(depth_m=np.ones(x.shape),intrinsics=dict(cx=0,cy=0,fx=1,fy=1),T_base_end=np.eye(4))
    candidates=perception.candidates_3d(obs,{},.06)
    assert len(candidates)==1
    assert np.max(np.ptp(candidates[0].points[:,:2],axis=0))<.06


def test_visible_cucumber_is_kept_for_verification_when_grasp_fit_fails(monkeypatch):
    x,y=np.mgrid[0:40,0:8]
    points=np.c_[.35+x.ravel()*.003,.10+y.ravel()*.002,np.full(x.size,.08)]
    candidate=NS(points=points,center=np.array([.4,.1,.07]),mask=np.ones((20,20),dtype=bool),yaw=0.)
    color=(np.array(perception.COLORS['cucumber'][:3])*255).astype(np.uint8)
    obs=NS(rgb=np.tile(color,(20,20,1)),depth_m=np.ones((20,20))*.5,intrinsics=dict(cx=10,cy=10,fx=20,fy=20),T_base_end=np.eye(4))
    monkeypatch.setattr(perception,'candidates_3d',lambda *a,**k:[candidate])
    monkeypatch.setattr(perception,'camera_to_base',lambda *a,**k:np.array([[.4,.1,.08]]))
    def fail(points):raise ValueError('partial visible segment')
    monkeypatch.setattr(perception,'banana_segment',fail)
    estimates=perception.estimate_candidates(obs,{},.06,['cucumber'])
    assert estimates[0][0]=='cucumber'
    assert estimates[0][1].grasp_quality['requires_refinement']
