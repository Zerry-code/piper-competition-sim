from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_sim.pair_placement import select_round_tracks,allocate_pairs

def test_seven_visible_six_selected_preserving_explicit_objects():
    names=['bell_pepper','persimmon','pear','cucumber','banana','potato','apple']
    tracks=[NS(category=c,destination='red' if i<2 else 'green' if i<4 else 'yellow') for i,c in enumerate(names)]
    task={'steps':[{'source':c} for c in names[:4]]}
    chosen,left=select_round_tracks(tracks,task,123)
    assert len(chosen)==6 and len(left)==1
    assert {t.category for t in chosen}.issuperset(names[:4])
    assert sorted(t.category for t in chosen)==sorted(t.category for t in select_round_tracks(tracks,task,123)[0])
    assert all(sum(t.destination==color for t in chosen)==2 for color in ['red','green','yellow'])

def test_three_explicit_items_cannot_be_silently_dropped():
    tracks=[NS(category=c,destination='red') for c in ['apple','pear','banana']]
    with pytest.raises(ValueError,match='明确指定超过两件'):
        select_round_tracks(tracks,{'steps':[{'source':t.category} for t in tracks]},1)

def test_pair_is_symmetric_and_nonoverlapping():
    angles=np.linspace(0,2*np.pi,100)
    tracks=[NS(key=c,category=c,destination='red',estimate=NS(points=np.c_[.04*np.cos(angles),y+.04*np.sin(angles),np.full(100,.04)],center=np.array([0,y,.04]))) for c,y in [('apple',-.2),('persimmon',.2)]]
    tray={'center_xy_m':[.60,0.],'half_size_xy_m':[.1,.1],'placement_policy':'center_max_coverage'}
    positions=allocate_pairs({'red':tray},tracks,{}, {'red':[.60,0.]})
    a,b=tracks
    assert np.allclose((positions[a.key]+positions[b.key])/2,[.6,0])
    assert not a.placement_footprint.overlaps(positions[a.key],b.placement_footprint,positions[b.key])
    assert a.placement_partition['scheme']==b.placement_partition['scheme']
    assert a.placement_partition['sign']==-b.placement_partition['sign']
