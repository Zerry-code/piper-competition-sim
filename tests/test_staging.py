"""Staging is a real two-pick workflow, deferred behind ordinary targets."""
from types import SimpleNamespace as NS
import numpy as np
from piper_sim.harvest import Harvest,Track,RetryableFailure
from piper_sim.fruit_placement import Footprint
from piper_sim.staging import staging_options,floor_coverage


class TwoLegSkills:
    staging_enabled=True
    def __init__(self,blocked_second=False,unknown_release=False):
        self.calls=[];self.blocked_second=blocked_second;self.unknown_release=unknown_release
        self.tracks=[Track('hard','banana',[.5,-.1,.08],'yellow'),Track('easy','pear',[.4,.1,.08],'green')]
    def overview(self):return self.tracks
    def observe_target(self,t,retry):
        self.calls.append(('observe',t.key,getattr(t,'staging_leg',0)))
        if t.key=='hard' and not getattr(t,'staging_used',False):raise RetryableFailure('direct search exhausted',phase='planning')
        if self.blocked_second and getattr(t,'staging_leg',0)==2:raise RetryableFailure('second search exhausted',phase='planning')
    def begin_staging(self,t):
        self.calls.append(('begin',t.key));t.staging_leg=1;t.staging_pending=False;t.staging_used=True
    def finish_staging(self,t):self.calls.append(('finish',t.key));t.staging_leg=2
    def pick(self,t):self.calls.append(('pick',t.key,getattr(t,'staging_leg',0)));return True
    def place_and_verify(self,t):
        self.calls.append(('place',t.key,getattr(t,'staging_leg',0)))
        self.last_release_completed=True
        if self.unknown_release and getattr(t,'staging_leg',0)==1:t.boundary_check_status='unknown';return False
        return True
    def recover(self,t):raise AssertionError('no recovery expected')


def test_staging_runs_last_and_requires_two_distinct_picks():
    skills=TwoLegSkills();r=Harvest(skills,max_preparations_per_target=3).run()
    assert r['completed']==['easy','hard'] and r['status']=='complete'
    assert [c for c in skills.calls if c[0]=='pick']==[('pick','easy',0),('pick','hard',1),('pick','hard',2)]
    assert skills.calls.index(('place','easy',0))<skills.calls.index(('begin','hard'))
    assert skills.calls.index(('finish','hard'))<skills.calls.index(('observe','hard',2))


def test_failed_second_leg_never_counts_intermediate_landing_as_complete():
    skills=TwoLegSkills(blocked_second=True);r=Harvest(skills,max_preparations_per_target=3).run()
    assert r['status']=='incomplete' and r['completed']==['easy'] and r['unresolved']==['hard']
    assert skills.calls.count(('begin','hard'))==1


def test_unknown_intermediate_boundary_does_not_start_second_pick():
    skills=TwoLegSkills(unknown_release=True);r=Harvest(skills,max_preparations_per_target=3).run()
    assert r['status']=='incomplete' and ('finish','hard') not in skills.calls
    assert ('pick','hard',2) not in skills.calls


def test_external_l1_deferral_does_not_move_hard_target_early():
    skills=TwoLegSkills();r=Harvest(skills,defer_staging=True).run()
    assert r['status']=='staging_deferred' and r['staging_deferred']==['hard']
    assert ('begin','hard') not in skills.calls and r['completed']==['easy']
    skills.tracks=[skills.tracks[0]]
    r=Harvest(skills,max_preparations_per_target=3).run()
    assert r['status']=='complete' and r['completed']==['hard']


def plane_memory():
    return NS(seen_plane={(x,y) for x in range(15,65) for y in range(-40,41)},obstacles=np.empty((0,2)))


def test_staging_requires_visible_floor_and_avoids_cards_and_reservations():
    shape=Footprint(np.zeros((1,2)),np.array([.035]));memory=plane_memory()
    table=dict(center_xy_m=[.4,0],half_size_xy_m=[.35,.4])
    cards={'red':dict(center_xy_m=[.54,-.26],half_size_xy_m=[.15,.15])}
    opts=staging_options(shape,table,cards,memory,{'other':np.array([.39,0])},{'other':shape},'this',[.56,.2])
    assert opts
    assert all(not shape.overlaps(o['point'],shape,[.39,0]) for o in opts)
    assert all(floor_coverage(o['shape'],o['point'],memory.seen_plane)>=.92 for o in opts)
    memory.seen_plane=set()
    assert not staging_options(shape,table,cards,memory,{}, {},'this',[.56,.2])


def test_middle_circle_search_is_exhausted_before_outer_grasp_candidates():
    from piper_sim.demo_skill import PickPlaceSkill
    skill=PickPlaceSkill.__new__(PickPlaceSkill)
    skill.cfg={'random_seed':1,'tilt_candidates':[.15,.35],'transport_height_m':.2}
    skill.b=NS(data=NS(site_xpos=np.array([[.4,0,.2]])),model=NS(site=lambda n:NS(id=0)))
    points=np.array([[x,y,.08] for x in np.linspace(.31,.49,80) for y in np.linspace(-.01,.01,8)])
    target=NS(center=np.array([.4,0,.08]),object_center=np.array([.4,0,.08]),points=points,yaw=0.,grasp_yaws=[0.,np.pi],
        grasp_quality=dict(method='visible_metric_centerline_straight_section',alternatives=[dict(center=[.33,0,.08],yaw=0.,yaws=[0.],score=100.)]))
    seen=[]
    def attempt(option,landing,floor,height,tilts):
        seen.append(float(option.center[0]))
        if option.center[0]>.35:raise RuntimeError('middle fixture constrained')
        return dict(center=option.center)
    skill._plan_one=attempt
    plan=skill.plan(target,np.array([.4,-.2]),.064)
    assert plan['center'][0]==.33
    assert seen[-1]==.33 and all(x==.4 for x in seen[:-1])


def test_web_l1_batch_defers_hard_action_until_other_actions_finish(tmp_path,monkeypatch):
    import web_console as web
    from web_console import Console
    calls=[]
    actions=[dict(category='cube_red',destination='yellow'),dict(category='cube_green',destination='green')]
    monkeypatch.setattr(web,'parse_plan',lambda *a,**k:dict(status='ready',steps=[dict(source=a['category'],destination=a['destination']) for a in actions]))
    monkeypatch.setattr(web,'resolve_plan',lambda *a,**k:actions)
    class L1Fake:
        staging_enabled=True
        def __init__(self,*a,prepared_tracks=None,**kwargs):self.prepared=prepared_tracks;self.navigation_events=[];self.survey_views=0
        def scan(self,*a,**k):pass
        def overview(self):
            for t in self.prepared:t.destination_position=np.array([.4,-.2])
            return self.prepared
        def observe_target(self,t,retry):
            if t.category=='cube_red' and not getattr(t,'staging_used',False):raise RetryableFailure('no direct path',phase='planning')
        def begin_staging(self,t):t.staging_leg=1;t.staging_pending=False;t.staging_used=True
        def finish_staging(self,t):t.staging_leg=2
        def pick(self,t):calls.append((t.category,getattr(t,'staging_leg',0)));return True
        def place_and_verify(self,t):return True
    monkeypatch.setattr(web,'L1Skills',L1Fake)
    app=Console.__new__(Console);app.output=tmp_path;app.task_requests=0;app.completed_categories=set()
    app.scene_memory=NS(objects={a['category']:NS(center=np.array([.4,0,.08])) for a in actions},moved=lambda *a:None)
    app.world=NS(begin_attempt=lambda:None,elapsed_s=0.,evaluation=lambda:{})
    state={};app.update=lambda **kw:state.update(kw);app.stage=lambda *a:None
    app.bind_callback=lambda *a:None;app.render=lambda **k:None
    app.execute_l1_plan('test',list(app.scene_memory.objects))
    assert state['result']['status']=='complete'
    assert calls==[('cube_green',0),('cube_red',1),('cube_red',2)]


def test_local_footprint_refresh_reselects_staging_space_instead_of_reusing_invalid_point(tmp_path):
    from run_harvest import FruitSkills
    from piper_sim.fruit_placement import visual_footprint
    table=dict(center_xy_m=[.4,0],half_size_xy_m=[.35,.4],top_m=.06)
    trays={'yellow':dict(center_xy_m=[.22,-.26],half_size_xy_m=[.15,.15])}
    points=np.array([[x,y,.08] for x in np.linspace(.48,.61,50) for y in np.linspace(.16,.18,8)])
    estimate=NS(center=np.median(points,axis=0),object_center=np.median(points,axis=0),points=points,grasp_quality={})
    target=Track('banana','banana',[.54,.17,.08],'yellow',estimate=estimate)
    target.staging_leg=1;target.staging_surface_memory=plane_memory();target.staging_final={'position':np.array([.22,-.26])}
    target.destination_position=np.array([.22,-.26]);target.destination_options=[]
    target.placement_region=table;target.placement_floor=.06
    skills=FruitSkills.__new__(FruitSkills);skills.w=NS(cfg=dict(table=table,trays=trays,tray_geometry={'floor_thickness_m':.002}))
    skills.navigation_events=[];skills.output=tmp_path
    def plan(e,point,floor):return dict(destination_position=point.tolist(),yaw=0.,place_yaw=e.preferred_place_rotation,center=e.center)
    skills.skill=NS(cfg={},table=.06,callback=lambda *a:None,plan=plan,last_planning_report=None)
    skills.plan_staging_leg=lambda target,e:plan(e,target.destination_position,.06)
    skills.prepare_target(target,estimate)
    assert np.linalg.norm(target.destination_position-[.39,0])<.13
    assert target.staging_verification_region
    assert target.placement_footprint.kind=='curved_sections'
    shape=visual_footprint(estimate,'banana').rotated(target.placement_rotation)
    assert floor_coverage(shape,target.destination_position,target.staging_surface_memory.seen_plane)>=.92


def test_intermediate_release_retry_still_requires_second_pick():
    skills=TwoLegSkills();original=skills.place_and_verify;blocked=[False]
    def place(t):
        if getattr(t,'staging_leg',0)==1 and not blocked[0]:
            blocked[0]=True;t.release_pending=True
            raise RetryableFailure('finger opening blocked',phase='release')
        return original(t)
    def resume(t):t.release_pending=False;return True
    skills.place_and_verify=place;skills.resume_release=resume
    r=Harvest(skills,max_preparations_per_target=3).run()
    assert r['status']=='complete'
    assert ('pick','hard',2) in skills.calls and skills.calls.count(('finish','hard'))==1


def test_drop_that_actually_lands_in_final_region_can_skip_second_pick():
    from piper_sim.drop_recovery import ObjectDropped
    skills=TwoLegSkills();original=skills.pick;dropped=[False]
    def pick(t):
        if getattr(t,'staging_leg',0)==1 and not dropped[0]:dropped[0]=True;raise ObjectDropped('transport')
        return original(t)
    def recover(t):t.staging_final_landed=True;return 'placed'
    skills.pick=pick;skills.recover_dropped=recover
    r=Harvest(skills,max_preparations_per_target=3).run()
    assert r['status']=='complete' and ('pick','hard',2) not in skills.calls
    assert any(e.get('final_landed') for e in r['events'])


def test_staging_alternatives_keep_short_transfer_and_spatial_diversity():
    from piper_sim.staging import select_staging_landings
    points=[[.39,0],[.43,0],[.35,0],[.39,.06],[.43,.06],[.39,-.06],[.47,.12]]
    options=[dict(point=np.array(p),rotation=0.) for p in points]
    picked=select_staging_landings(options,[.54,.12])
    assert len(picked)==6
    assert np.allclose(picked[0]['point'],[.39,0])
    assert np.allclose(picked[1]['point'],[.47,.12])
    assert any(o['point'][1]<0 for o in picked)


def test_second_leg_reallocates_old_slot_for_changed_long_axis_without_overlap():
    from piper_sim.fruit_placement import reallocate_final_landings,visual_footprint
    tray=dict(center_xy_m=[.22,-.26],half_size_xy_m=[.15,.15])
    points=np.array([[x,y,.08] for x in np.linspace(-.07,.07,80) for y in np.linspace(-.012,.012,8)])
    e=NS(points=points,object_center=np.zeros(3));shape=visual_footprint(e,'banana')
    other=Footprint(np.zeros((1,2)),np.array([.0455]));reserved={'potato':np.array([.27448,-.20552])}
    fresh=reallocate_final_landings(shape,tray,reserved,{'potato':other},'banana',np.empty((0,2)),[.2,-.2924])
    assert fresh
    rotated=shape.rotated(fresh['rotation'])
    for p in fresh['points']:
        assert rotated.clearance(tray,p)>=fresh['margin']-1e-8
        assert not rotated.overlaps(p,other,reserved['potato'])


def test_second_leg_reallocation_respects_new_visual_occupancy():
    from piper_sim.fruit_placement import reallocate_final_landings
    tray=dict(center_xy_m=[.4,0],half_size_xy_m=[.15,.15])
    shape=Footprint(np.zeros((1,2)),np.array([.03]))
    cloud=np.array([[x,y] for x in np.linspace(.25,.55,70) for y in np.linspace(-.15,.15,70)])
    assert reallocate_final_landings(shape,tray,{}, {},'target',cloud,[.4,0]) is None
