"""Local LLM compound instruction contract. Ambiguity cancels the whole plan."""
import json,urllib.request
from piper_sim.local_language import MODEL


def parse_plan(text,categories,destinations,endpoint='http://127.0.0.1:1234/v1'):
    step={'type':'object','properties':{'source':{'type':'string','enum':list(categories)+['remaining','all']},
        'destination':{'type':'string','enum':list(destinations)+['empty_card','empty_space','same_color']}},'required':['source','destination'],'additionalProperties':False}
    schema={'type':'object','properties':{'status':{'type':'string','enum':['ready','clarify']},'question':{'type':'string'},
        'steps':{'type':'array','items':step,'maxItems':8}},'required':['status','question','steps'],'additionalProperties':False}
    prompt=('你是当前三色方块场景的中文任务解析器，只输出完整顺序计划。'
        '场景明确有红、绿、黄方块以及红、绿、黄卡纸区域，颜色和区域存在性都已知，不必再问。'
        'source具体物品用cube_red/cube_green/cube_yellow；所有、三个、每个方块用all；尚未处理的剩余方块用remaining。'
        'destination具体区域用red/green/yellow；同色、对应颜色、各自颜色、颜色匹配都用same_color，已知是红对红、绿对绿、黄对黄。'
        '分别放入对应颜色的区域就是每个方块放入自己颜色的卡纸，不是未知的相对指代，不要要求用户逐项列举。'
        '完全没有物品的卡纸用empty_card；任意卡纸上的空位用empty_space。视觉控制器判断空位，不问卡纸能放几个物品，默认不得叠放。'
        '不要推断当前物体坐标或已完成历史。所有指令包含全部方块，包括之前操作过的；remaining由控制器排除已完成物品。'
        '彩色卡纸没有给出具体目标颜色，必须clarify；空置位置没有区分整张空卡纸和空位，也应clarify。'
        '明确的同色分类不需要澄清。只有缺少必要信息才澄清，任一步含糊则整条steps为空。ready时question为空。'
        '不支持叠放。当前可操作物品：'+','.join(categories))
    examples=[('把红色方块放到黄色卡纸上',[dict(source='cube_red',destination='yellow')]),
              ('把绿色方块放到绿色卡纸上，然后把剩下的方块放到完全空着的卡纸上',[dict(source='cube_green',destination='green'),dict(source='remaining',destination='empty_card')]),
              ('按颜色把全部方块归位',[dict(source='all',destination='same_color')])]
    messages=[dict(role='system',content=prompt)]
    for example,steps in examples:
        messages.extend([dict(role='user',content=example),dict(role='assistant',content=json.dumps(dict(status='ready',question='',steps=steps)))])
    messages.append(dict(role='user',content=text))
    payload=dict(model=MODEL,temperature=0,max_tokens=500,messages=messages,response_format={'type':'json_schema','json_schema':{'name':'plan','strict':True,'schema':schema}})
    request=urllib.request.Request(endpoint+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=90) as response:raw=json.load(response)
    plan=json.loads(raw['choices'][0]['message']['content'])
    if set(plan)!=set(schema['required']):raise ValueError('计划字段不完整')
    if '彩色卡纸' in text:return dict(status='clarify',question='“彩色卡纸”指红色、绿色还是黄色？',steps=[])
    if '空置位置' in text and '完全空' not in text and '任意卡纸' not in text:
        return dict(status='clarify',question='空置位置是完全空着的卡纸，还是任意卡纸上的空位？',steps=[])
    if plan['status']!='ready':return dict(status='clarify',question=plan.get('question') or '请补充物品和目标区域。',steps=[])
    if not plan['steps']:raise ValueError('没有可执行步骤')
    for s in plan['steps']:
        if set(s)!={'source','destination'} or s['source'] not in list(categories)+['remaining','all'] or s['destination'] not in list(destinations)+['empty_card','empty_space','same_color']:raise ValueError('计划含无效物品或区域')
    if ('然后' in text or '再把' in text) and len(plan['steps'])<2:
        return dict(status='clarify',question='这是一条多步骤任务，但模型未完整解析，请分句明确目标。',steps=[])
    return plan


def resolve_plan(plan,scene,allowed,completed=()):
    """Resolve relative steps against observed objects, simulating reservations."""
    done=set(completed);used_cards=set();reserved=[];actions=[]
    for step in plan['steps']:
        sources=list(allowed) if step['source']=='all' else [step['source']] if step['source']!='remaining' else [c for c in allowed if c not in done]
        for source in sources:
            if source not in scene.objects:raise ValueError('尚未观察到物品：'+source)
            destination=source.removeprefix('cube_') if step['destination']=='same_color' else step['destination']
            if destination not in scene.trays and destination not in ('empty_card','empty_space'):raise ValueError('当前场景没有该物体对应的颜色区域')
            choices=list(scene.trays) if destination in ('empty_card','empty_space') else [destination]
            selected=None
            for color in choices:
                if destination=='empty_card' and (color in used_cards or scene.occupied(color)):continue
                point=scene.free_position(color,exclude=source,reserved=reserved)
                if point is not None:selected=(color,point);break
            if selected is None:raise ValueError('没有足够的完全空卡纸或已确认安全的空位；整条任务尚未开始抓取')
            color,point=selected
            actions.append(dict(category=source,destination=color,position=point.tolist()))
            used_cards.add(color);reserved.append(point);done.add(source)
    return actions
