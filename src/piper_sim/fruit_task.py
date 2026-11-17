"""Grouped commands with explicit remainder and inventory coverage checks."""
import json,re,urllib.request
from piper_sim.local_language import MODEL

NAMES={'pear':'梨','banana':'香蕉','persimmon':'柿子','bell_pepper':'彩椒','cucumber':'黄瓜','potato':'土豆','apple':'苹果','orange':'橙子','strawberry':'草莓','lemon':'柠檬','grape':'葡萄'}
COLORS=('red','green','yellow','mapping')

def validate_named_objects(text,categories):
    for category,name in NAMES.items():
        pattern=r'(?:^|把|将|和|与|及|、|，|,)\s*'+re.escape(name)+r'(?:和|与|及|、|放|移|送|，|,|到|全|都|$)'
        if category not in categories and re.search(pattern,text):
            raise ValueError('当前场景没有“'+name+'”。可操作：'+'、'.join(NAMES.get(c,c) for c in categories)+'。请确认物品名称，未执行动作。')

def resolve_fruit_plan(text,plan,categories,mapping):
    validate_named_objects(text,categories)
    if plan.get('unknown_objects'):raise ValueError('指令含当前场景不存在的物品：'+'、'.join(plan['unknown_objects'])+'；未执行动作')
    if plan.get('question'):raise ValueError(plan['question'])
    assignments={};steps=plan.get('steps',[]);explicit=set()
    for step in steps:
        source=step['source'];dest=step['destination']
        if source not in list(categories)+['all','remaining'] or dest not in COLORS:raise ValueError('无效果蔬计划')
        selected=list(categories) if source=='all' else [c for c in categories if c not in assignments] if source=='remaining' else [source]
        if source!='remaining':explicit.update(selected)
        for category in selected:
            color=mapping[category] if dest=='mapping' else dest
            if category in assignments and assignments[category]!=color:raise ValueError('同一物品被指定到不同区域，请明确')
            assignments[category]=color
    remainder=plan.get('remainder_destination','none')
    if remainder not in COLORS+('none',):raise ValueError('无效剩余物品目的地')
    if remainder!='none':
        for category in categories:
            if category not in assignments:assignments[category]=mapping[category] if remainder=='mapping' else remainder
    mentions_remainder=bool(re.search(r'其他|其它|其余|剩下|剩余',text))
    whole=mentions_remainder or bool(re.search(r'(?:所有|全部)(?:的)?(?:水果|果蔬|物品)|(?:水果|果蔬|物品)(?:全部|全|都)',text)) or plan.get('scope')=='all'
    if mentions_remainder and remainder=='none' and not any(s['source']=='remaining' for s in steps):
        raise ValueError('指令包含“其他/剩余物品”，模型未保留其目的区域；整条未执行，请明确剩余物品放置颜色')
    named_missing=[NAMES[c] for c in categories if NAMES.get(c,c) in text and c not in explicit]
    if named_missing:raise ValueError('模型漏掉明确点名的物品：'+'、'.join(named_missing)+'；整条未执行')
    missing=set(categories)-set(assignments)
    if whole and missing:raise ValueError('完整分组计划缺少：'+'、'.join(NAMES.get(c,c) for c in categories if c in missing)+'；整条未执行')
    if not assignments:raise ValueError('请明确果蔬和目标区域')
    return dict(action='grouped',category='all',destination='mapping',assignments=assignments,steps=steps,remainder_destination=remainder,scope='all' if whole else 'selected',expected_categories=list(categories) if whole else list(assignments))

def parse_fruit_task(text,categories,mapping):
    validate_named_objects(text,categories)
    step={'type':'object','properties':{'source':{'type':'string','enum':list(categories)},'destination':{'type':'string','enum':list(COLORS)}},'required':['source','destination'],'additionalProperties':False}
    schema={'type':'object','properties':{'steps':{'type':'array','items':step},'remainder_destination':{'type':'string','enum':['none']+list(COLORS)},'scope':{'type':'string','enum':['selected','all']},'unknown_objects':{'type':'array','items':{'type':'string'}},'question':{'type':'string'}},'required':['steps','remainder_destination','scope','unknown_objects','question'],'additionalProperties':False}
    prompt=('你解析果蔬分组任务。当前真实可用类别只有：'+','.join(categories)+'。梨pear、香蕉banana、柿子persimmon、彩椒bell_pepper、黄瓜cucumber、土豆potato、苹果apple。'
        'steps逐项保留明确点名的物品与颜色；其他/其它/其余/剩下/剩余果蔬必须填remainder_destination，让控制器展开所有未点名物品，不要自行猜测或只列举部分剩余物品。'
        '包含剩余分组或所有果蔬时scope=all；只指定部分物品时scope=selected且remainder_destination=none。'
        '红red绿green黄yellow，按默认类别分类：steps为空且remainder_destination=mapping；全部放同一区域：steps为空且remainder_destination为该颜色。分组只在steps列出点名物品。默认类别mapping不能代替用户明确颜色。'
        '当前没有的名称写unknown_objects并填写question，steps置空，绝对不忽略或替换，例如橙子不等于柿子。明确任务unknown_objects=[]且question为空。'
        '例如香蕉和柿子放黄色、梨黄瓜放绿色、其他水果放红色：steps含banana/yellow,persimmon/yellow,pear/green,cucumber/green；remainder_destination=red,scope=all。')
    examples=[('把所有果蔬按类别分类',dict(steps=[],remainder_destination='mapping',scope='all',unknown_objects=[],question='')),
        ('把黄瓜和土豆放到红色区域，梨放到黄色区域，其余果蔬放到绿色区域',dict(steps=[dict(source='cucumber',destination='red'),dict(source='potato',destination='red'),dict(source='pear',destination='yellow')],remainder_destination='green',scope='all',unknown_objects=[],question='')),
        ('把香蕉放到黄色卡纸上',dict(steps=[dict(source='banana',destination='yellow')],remainder_destination='none',scope='selected',unknown_objects=[],question=''))]
    messages=[dict(role='system',content=prompt)]
    for example,answer in examples:messages.extend([dict(role='user',content=example),dict(role='assistant',content=json.dumps(answer,ensure_ascii=False))])
    messages.append(dict(role='user',content=text))
    payload=dict(model=MODEL,temperature=0,max_tokens=900,messages=messages,response_format={'type':'json_schema','json_schema':{'name':'fruit_plan','strict':True,'schema':schema}})
    req=urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=90) as response:raw=json.load(response)
    return resolve_fruit_plan(text,json.loads(raw['choices'][0]['message']['content']),categories,mapping)
