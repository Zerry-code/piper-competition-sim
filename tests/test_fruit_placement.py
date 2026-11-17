from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_sim.fruit_placement import allocate_fruit_positions

def tracks(n):
    return [NS(key=str(i),category=str(i),destination='yellow',estimate=NS(points=np.array([[-.016,-.016,0],[.016,.016,0]]))) for i in range(n)]

def test_six_items_on_one_card_no_fixed_count_limit():
    ts=tracks(6);trays={'yellow':dict(center_xy_m=[.4,-.2],half_size_xy_m=[.15,.15])}
    positions=allocate_fruit_positions(trays,ts,{}, {'yellow':[.4,-.2]})
    assert len(positions)==6
    for i,p in positions.items():
        assert np.all(abs(p-[.4,-.2])<.12)
        for j,q in positions.items():
            if i!=j:assert np.linalg.norm(p-q)>.057

def test_physical_capacity_is_reported():
    ts=tracks(1);ts[0].estimate.points=np.array([[-.145,-.145,0],[.145,.145,0]])
    with pytest.raises(RuntimeError,match='没有数量上限'):
        allocate_fruit_positions({'yellow':dict(center_xy_m=[.4,0],half_size_xy_m=[.15,.15])},ts,{}, {'yellow':[.4,0]})


def test_contaminated_footprint_is_not_reported_as_capacity_failure():
    ts=tracks(1);ts[0].estimate.points=np.array([[-.2,0,0],[.2,0,0]])
    with pytest.raises(RuntimeError,match='视觉轮廓异常'):
        allocate_fruit_positions({'yellow':dict(center_xy_m=[.4,0],half_size_xy_m=[.15,.15])},ts,{}, {'yellow':[.4,0]})


def test_slots_avoid_far_card_edge():
    ts=tracks(2)
    slots=allocate_fruit_positions({'yellow':dict(center_xy_m=[.54,-.26],half_size_xy_m=[.15,.15])},ts,{}, {'yellow':[.65,-.30]})
    assert all(p[0]<=.56 and np.linalg.norm(p)<=.60 for p in slots.values())


def test_alternate_landings_preserve_other_reserved_footprints():
    from piper_sim.fruit_placement import alternative_landings
    ts=tracks(2);trays={'yellow':dict(center_xy_m=[.4,-.2],half_size_xy_m=[.15,.15])}
    positions=allocate_fruit_positions(trays,ts,{}, {'yellow':[.4,-.2]})
    options=alternative_landings(trays,ts,{},positions)
    assert all(len(p)>1 for p in options.values())
    for i,ps in options.items():
        for p in ps:
            for j,q in positions.items():
                if i!=j:assert np.linalg.norm(p-q)>.057


def test_entire_footprint_and_all_alternates_prefer_five_cm_clearance():
    from piper_sim.fruit_placement import footprint_radius,edge_clearance,alternative_landings
    ts=tracks(2);trays={'yellow':dict(center_xy_m=[.4,-.2],half_size_xy_m=[.15,.15])}
    positions=allocate_fruit_positions(trays,ts,{}, {'yellow':[.25,-.3]})
    options=alternative_landings(trays,ts,{},positions)
    for t in ts:
        assert edge_clearance(trays['yellow'],positions[t.key],footprint_radius(t.estimate))>=.05-1e-8
        assert all(edge_clearance(trays['yellow'],p,footprint_radius(t.estimate))>=.05-1e-8 for p in options[t.key])

def test_clearance_can_relax_for_capacity_without_overlap_or_count_limit():
    from piper_sim.fruit_placement import footprint_radius,edge_clearance
    ts=tracks(2)
    for t in ts:t.estimate.points=np.array([[-.04,-.04,0],[.04,.04,0]])
    trays={'yellow':dict(center_xy_m=[.4,0],half_size_xy_m=[.15,.15])}
    p=allocate_fruit_positions(trays,ts,{}, {'yellow':[.4,0]})
    assert np.linalg.norm(p['0']-p['1'])>=sum(footprint_radius(t.estimate) for t in ts)
    assert all(edge_clearance(trays['yellow'],p[t.key],footprint_radius(t.estimate))>0 for t in ts)
    assert any(t.placement_margin_m<.05 for t in ts)


def test_cube_landing_also_prefers_five_cm_from_entire_footprint():
    from piper_sim.scene_memory import SceneMemory
    tray=dict(center_xy_m=[.4,0],half_size_xy_m=[.15,.15])
    memory=SceneMemory({'red':tray});memory.cards['red']=np.array([.53,0])
    memory.seen_plane={(x,y) for x in range(20,61) for y in range(-20,21)}
    point=memory.free_position('red')
    assert point is not None
    assert np.min(np.array(tray['half_size_xy_m'])-abs(point-tray['center_xy_m']))-.036>=.05


def long_estimate(curved=False):
    u=np.linspace(-.09,.09,91)
    centerline=np.column_stack([u,.7*u*u/.09 if curved else np.zeros_like(u)])
    points=np.vstack([np.column_stack([centerline[:,0],centerline[:,1]+w,np.zeros(len(u))]) for w in np.linspace(-.012,.012,9)])
    return NS(points=points,object_center=np.median(points,axis=0))


@pytest.mark.parametrize('curved,category',[(False,'cucumber'),(True,'banana')])
def test_geometric_outline_covers_visible_points_without_enclosing_ball(curved,category):
    from piper_sim.fruit_placement import visual_footprint
    e=long_estimate(curved);shape=visual_footprint(e,category)
    assert shape.kind==('curved_sections' if curved else 'capsule')
    assert np.all(shape._contains(e.points[:,:2]-e.object_center[:2]))
    assert np.ptp(shape.vertices[:,0])<.20
    if not curved:assert np.ptp(shape.vertices[:,1])<.04


def test_three_circles_accept_parallel_items_rejected_by_enclosing_discs():
    from piper_sim.fruit_placement import visual_footprint,footprint_radius,validate_landing
    e=long_estimate();shape=visual_footprint(e,'cucumber')
    a=np.array([.4,-.04]);b=np.array([.4,.04]);tray=dict(center_xy_m=[.4,0],half_size_xy_m=[.15,.15])
    assert np.linalg.norm(a-b)<2*footprint_radius(e)
    assert not shape.overlaps(a,shape,b)
    assert validate_landing(shape,a,tray,{'other':b},{'other':shape},'this')
    assert not validate_landing(shape.rotated(np.pi/2),a,tray,{'other':b},{'other':shape},'this')


def test_rotated_long_footprint_checks_whole_shape_against_card_and_obstacles():
    from piper_sim.fruit_placement import visual_footprint,validate_landing
    shape=visual_footprint(long_estimate(),'cucumber');tray=dict(center_xy_m=[.4,0],half_size_xy_m=[.15,.15])
    p=np.array([.47,0])
    assert shape.clearance(tray,p)<0
    assert shape.rotated(np.pi/2).clearance(tray,p)>.04
    assert not validate_landing(shape,p,tray,{}, {},'a')
    assert validate_landing(shape.rotated(np.pi/2),p,tray,{}, {},'a')
    obstacle=p+shape.offsets[0]
    assert shape.hits_points(p,np.array([obstacle]))


def test_long_allocation_and_alternates_use_reserved_union_shapes():
    from piper_sim.fruit_placement import alternative_landings
    ts=[NS(key=c,category=c,destination='yellow',estimate=long_estimate(c=='banana')) for c in ('banana','cucumber')]
    trays={'yellow':dict(center_xy_m=[.4,0],half_size_xy_m=[.15,.15])}
    ps=allocate_fruit_positions(trays,ts,{}, {'yellow':[.4,0]})
    opts=alternative_landings(trays,ts,{},ps)
    for t in ts:
        assert t.placement_footprint.kind in ('capsule','curved_sections')
        assert t.placement_margin_m>0
        other=next(o for o in ts if o is not t)
        for p in opts[t.key]:
            assert not t.placement_footprint.overlaps(p,other.placement_footprint,ps[other.key])
            assert t.placement_footprint.clearance(trays['yellow'],p)>=min(.05,t.placement_margin_m)-1e-8


def test_actual_place_orientation_must_pass_visual_reservation_guard(tmp_path):
    from piper_sim.fruit_world import ImplicitWorld,configure
    from piper_sim.demo_skill import PickPlaceSkill,TargetEstimate
    world=ImplicitWorld('l2',configure(2329111945,['cube_red','cube_green','cube_blue'],tmp_path),tmp_path,render=False)
    try:
        world.reset(2329111945)
        target=TargetEstimate(np.array([.3723971,.03659924,.07761217]),.03522433,.28785664,np.empty((0,0)),np.empty((0,3)),[0,0],0)
        target.grasp_yaws=[target.yaw+k*np.pi/2 for k in range(4)]
        skill=PickPlaceSkill(world.backend,.06,lambda stage:None,dict(transport_height_m=.2,tilt_candidates=[.15,.35,.6,.75,.9,1.05]))
        plan=skill.plan(target,np.array([.525,-.245]),.064)
        checks=[]
        target.placement_validator=lambda point,delta:checks.append((point.copy(),delta)) or False
        target.grasp_yaws=[plan['yaw']];skill.planning_deadline=float('inf')
        before=world.backend.data.qpos.copy()
        with pytest.raises(RuntimeError):skill._plan_one(target,np.array([.525,-.245]),.064,.2,[plan['tilt']])
        assert checks
        assert all(np.allclose(point,[.525,-.245]) for point,delta in checks)
        np.testing.assert_equal(before,world.backend.data.qpos)
    finally:world.close()
