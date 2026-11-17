from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_sim.harvest import Harvest,Track,RetryableFailure
from piper_sim.staging import progressive_options
from piper_sim.fruit_placement import Footprint
from piper_sim.route_search import joint_space_route
from piper_sim.fruit_perception import split_touching_colors


def test_many_preparation_misses_do_not_discard_unattempted_target():
    class Skills:
        persistent_preparation=True
        def __init__(self):self.observations=0;self.picks=0
        def overview(self):return [Track('cucumber','cucumber',[.4,0,.08],'green')]
        def observe_target(self,t,retry):
            self.observations+=1
            if self.observations<11:raise RetryableFailure('awaiting fresh view',phase='observation')
        def pick(self,t):self.picks+=1;return True
        def place_and_verify(self,t):return True
    skills=Skills();r=Harvest(skills,max_attempts=None,max_preparations_per_target=2,max_deferrals=1).run()
    assert r['status']=='complete' and skills.observations==11 and skills.picks==1
    assert not any(e.get('event')=='preparation_limit_reached' for e in r['events'])


def test_progressive_landings_all_move_toward_goal_and_prefer_full_goal_step():
    shape=Footprint(np.zeros((1,2)),np.array([.025]));source=np.array([.39,0]);goal=np.array([.52,-.26])
    memory=NS(seen_plane={(x,y) for x in range(10,70) for y in range(-45,46)},obstacles=np.empty((0,2)))
    table=dict(center_xy_m=[.4,0],half_size_xy_m=[.35,.45]);trays={'red':dict(center_xy_m=goal,half_size_xy_m=[.15,.15])}
    options=progressive_options(shape,table,trays,memory,{}, {},'pear',source,goal,'red')
    assert options and np.linalg.norm(options[0]['point']-goal)<1e-8
    assert all(o['goal_distance']<np.linalg.norm(goal-source)-.019 for o in options)
    memory.seen_plane=set()
    assert not progressive_options(shape,table,trays,memory,{}, {},'pear',source,goal,'red')


def test_no_three_hop_cap_silently_abandons_a_pending_target():
    from run_harvest import FruitSkills
    skills=FruitSkills.__new__(FruitSkills);skills.skill=NS(cfg={});skills.navigation_events=[]
    t=Track('pear','pear',[.4,0,.08],'green');t.staging_hops=7;t.staging_leg=2
    assert skills.request_restaging(t,'current direct route search exhausted')
    assert t.staging_pending and t.staging_mode=='progressive'


def test_joint_bridge_uses_checked_bend_when_straight_joint_path_collides():
    def validate(points):
        if any(abs(p[0])<.15 and p[1]<.5 for p in points):raise RuntimeError('collision fixture')
    path=joint_space_route(np.array([-1.,0]),np.array([1.,0]),[np.array([0.,1.])],validate)
    assert max(p[1] for p in path)>.99
    assert max(np.max(abs(b-a)) for a,b in zip(path,path[1:]))<=.065+1e-9


def test_joint_bridge_does_not_bypass_collision_rejection():
    with pytest.raises(RuntimeError,match='暂未通过'):
        joint_space_route(np.zeros(2),np.ones(2),[],lambda p:(_ for _ in ()).throw(RuntimeError('collision')))


def touching_fixture():
    x,y=np.meshgrid(np.linspace(.3,.46,40),np.linspace(.1,.14,15));base=np.stack([x,y,np.ones_like(x)*.08],axis=-1)
    mask=np.ones(x.shape,dtype=bool);rgb=np.empty(x.shape+(3,),dtype=np.uint8)
    rgb[:,:22]=[20,120,30];rgb[:,22:]=[190,170,45]
    return mask,base,rgb


def test_touching_green_cucumber_and_warm_pear_are_not_one_large_pear():
    mask,base,rgb=touching_fixture();parts=split_touching_colors(mask,base,rgb)
    assert len(parts)==2
    assert np.array_equal(parts[0]|parts[1],mask) and not np.any(parts[0]&parts[1])
    assert np.median(base[parts[0]],axis=0)[0]<np.median(base[parts[1]],axis=0)[0]-.035


def test_small_green_stem_or_white_highlight_does_not_split_one_fruit():
    mask,base,rgb=touching_fixture();rgb[:]=[190,170,45];rgb[:,:4]=[20,120,30]
    assert len(split_touching_colors(mask,base,rgb))==1
    rgb[:]=[20,120,30];rgb[:,:10]=[210,210,210]
    assert len(split_touching_colors(mask,base,rgb))==1


def test_future_empty_goal_reservations_do_not_block_real_intermediate_step():
    from run_harvest import FruitSkills
    skills=FruitSkills.__new__(FruitSkills)
    current=Track('pear','pear',[.4,0,.08],'green');future=Track('cucumber','cucumber',[.45,.3,.08],'green')
    released=Track('potato','potato',[.3,.1,.08],'yellow')
    released.last_released_position=np.array([.25,-.2]);released.last_released_footprint=Footprint(np.zeros((1,2)),np.array([.04]))
    skills.round_targets={'pear':current,'cucumber':future,'potato':released}
    current.placement_reservations={'cucumber':np.array([.4,-.2]),'potato':np.array([.22,-.25])}
    current.placement_footprints={}
    positions,_=skills.physical_reservations(current)
    assert 'cucumber' not in positions
    np.testing.assert_equal(positions['potato'],released.last_released_position)


def test_progressive_choices_include_short_segments_when_long_move_is_blocked():
    from piper_sim.staging import select_progressive_landings
    source=np.array([.4,0]);goal=np.array([.4,-.3])
    options=[dict(point=source+f*(goal-source),rotation=0.,goal_distance=(1-f)*.3) for f in [1.,.95,.9,.85,.8,.65,.45,.25]]
    selected=select_progressive_landings(options,source,goal)
    assert np.linalg.norm(selected[0]['point']-goal)<1e-9
    assert any(np.linalg.norm(o['point']-source)<.1 for o in selected)
    assert len(selected)<=6
