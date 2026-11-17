import pytest
from piper_sim.fruit_task import resolve_fruit_plan,validate_named_objects
from piper_sim.document_fruits import CATEGORIES

MAPPING=dict(apple='red',pear='green',banana='yellow',persimmon='red',bell_pepper='red',cucumber='green',potato='yellow')

def plan(steps,remainder='none',scope='selected'):
    return dict(steps=[dict(source=s,destination=d) for s,d in steps],remainder_destination=remainder,scope=scope,unknown_objects=[],question='')

def test_other_fruit_expands_to_pepper_instead_of_silently_skipping_it():
    result=resolve_fruit_plan('梨和黄瓜放绿，其他水果放黄',plan([('pear','green'),('cucumber','green')],'yellow','all'),CATEGORIES,MAPPING)
    assert set(result['assignments'])==set(CATEGORIES)
    assert result['assignments']['bell_pepper']=='yellow'
    assert set(result['expected_categories'])==set(CATEGORIES)

def test_old_five_item_expansion_cannot_complete_a_remaining_task():
    p=plan([(c,MAPPING[c]) for c in CATEGORIES if c!='bell_pepper'])
    with pytest.raises(ValueError,match='未保留'):resolve_fruit_plan('其他水果放黄色区域',p,CATEGORIES,MAPPING)

def test_unknown_orange_is_not_ignored_or_replaced():
    with pytest.raises(ValueError,match='没有“橙子”'):validate_named_objects('把橙子和柿子放到红色区域，其他水果放黄色区域',CATEGORIES)

def test_explicitly_named_pepper_cannot_be_hidden_in_remainder():
    with pytest.raises(ValueError,match='明确点名'):
        resolve_fruit_plan('彩椒放红色，其他水果放黄色',plan([('persimmon','red')],'yellow','all'),CATEGORIES,MAPPING)

def test_only_banana_is_still_a_valid_partial_task():
    r=resolve_fruit_plan('把香蕉放到黄色卡纸上',plan([('banana','yellow')]),CATEGORIES,MAPPING)
    assert r['expected_categories']==['banana'] and r['scope']=='selected'

def test_all_scope_requires_all_inventory_items():
    with pytest.raises(ValueError,match='完整分组计划缺少'):
        resolve_fruit_plan('把所有果蔬分类',plan([('banana','yellow')],scope='all'),CATEGORIES,MAPPING)

def test_default_mapping_all_remains_supported():
    r=resolve_fruit_plan('把所有果蔬按类别分类',plan([('all','mapping')],scope='all'),CATEGORIES,MAPPING)
    assert r['assignments']==MAPPING


def test_runtime_audit_rejects_five_completed_when_six_requested(monkeypatch,tmp_path):
    import sys
    from pathlib import Path
    from types import SimpleNamespace as NS
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import web_console as web
    task=dict(action='grouped',category='all',destination='mapping',assignments=MAPPING,expected_categories=CATEGORIES)
    monkeypatch.setattr(web,'parse_fruit_task',lambda *a,**k:task)
    monkeypatch.setattr(web,'WebFruitSkills',lambda *a,**k:NS(navigation_events=[]))
    fake=dict(status='complete',completed=[c for c in CATEGORIES if c!='bell_pepper'],unresolved=[])
    monkeypatch.setattr(web,'Harvest',lambda *a,**k:NS(events=[],run=lambda:fake))
    app=web.Console(output_root=tmp_path);app.mode='l2';app.output=tmp_path
    app.world=NS(cfg={'tasks':{'l2':{'mapping':MAPPING}}},begin_attempt=lambda:None,evaluation=lambda:{},elapsed_s=0.)
    app.bind_callback=lambda skills:None;app.stage=lambda name:None;app.render=lambda **kwargs:None
    app.execute('所有果蔬分类')
    assert app.data['result']['status']=='incomplete'
    assert app.data['result']['unresolved']==['bell_pepper']


def test_apple_is_in_scene_mapping_and_remainder_plan():
    assert 'apple' in CATEGORIES
    result=resolve_fruit_plan('彩椒和柿子放红色，其他水果放黄色',plan([('bell_pepper','red'),('persimmon','red')],'yellow','all'),CATEGORIES,MAPPING)
    assert result['assignments']['apple']=='yellow'
    assert len(result['expected_categories'])==7
