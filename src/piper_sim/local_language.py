"""Local text-only Qwen client; never silently substitutes a rule parser."""
import json
import urllib.request

MODEL = 'Qwen3-4B-Instruct-2507'


def parse_instruction(text, endpoint='http://127.0.0.1:1234/v1', model=MODEL,categories=None,destinations=None):
    schema = {'type':'object','properties':{
        'action': {'type':'string','enum':['sort_all','pick_place','clarify']},
        'category': {'type':'string','enum':['all','apple','pear','banana','lemon',
            'orange','strawberry','peach','plum','persimmon','bell_pepper','cucumber','potato','cube_red','cube_green','cube_blue','cube_yellow','all_blocks','unknown']},
        'destination': {'type':'string','enum':['mapping','red','green','blue','yellow','unknown']}},
        'required':['action','category','destination'],'additionalProperties':False}
    if categories is not None:schema['properties']['category']['enum']=list(dict.fromkeys(['all','unknown']+list(categories)))
    if destinations is not None:schema['properties']['destination']['enum']=list(dict.fromkeys(['mapping','unknown']+list(destinations)))
    payload = dict(model=model, temperature=0, max_tokens=180,
        messages=[{'role':'system','content':
            '把中文机械臂指令转换为JSON。支持全部水果或全部方块按颜色/类别分类，以及单个指定类别放入红绿黄卡纸或指定颜色区域。'
            '全部水果分类用sort_all/all/mapping；全部方块分类用sort_all/all_blocks/mapping。'
            '红绿黄方块分别是cube_red/cube_green/cube_yellow；蓝方块是cube_blue。柿子persimmon，彩椒bell_pepper，黄瓜cucumber，土豆potato。'
            '物体名必须依据用户原文，不能把目的区域的颜色当作目标物体。梨=pear，香蕉=banana，柿子=persimmon，彩椒=bell_pepper，黄瓜=cucumber，土豆=potato。'
            '例如把红色方块放到黄色卡纸上必须输出pick_place/cube_red/yellow。只有明确要求全部分类才用sort_all。单个目标用pick_place，并填写明确目的托盘。未指定单类目标的目的地或含糊指令必须clarify。不要编造坐标。'},
            {'role':'user','content':text}],
        response_format={'type':'json_schema','json_schema':{
            'name':'task','strict':True,'schema':schema}})
    request = urllib.request.Request(endpoint+'/chat/completions',
        data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=90) as response:
        result = json.load(response)
    task = json.loads(result['choices'][0]['message']['content'])
    if set(task) != set(schema['required']): raise ValueError('Invalid task fields')
    for k,v in task.items():
        if v not in schema['properties'][k]['enum']: raise ValueError('Invalid task value')
    if task['action']!='clarify' and task['category'] not in ('all','all_blocks','unknown') and task['destination'] not in ('mapping','unknown'):
        task['action']='pick_place'
    if task['action']=='pick_place' and (task['category'] in ('all','all_blocks','unknown') or
                                        task['destination'] in ('mapping','unknown')):
        raise ValueError('Ambiguous pick/place task')
    return task
