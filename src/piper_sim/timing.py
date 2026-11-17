"""Stage intervals use wall timestamps and simulator timestamps independently."""
def summarize(logs,end_sim,end_wall):
    stages={}
    for index,item in enumerate(logs):
        next_item=logs[index+1] if index+1<len(logs) else dict(sim=end_sim,wall=end_wall)
        row=stages.setdefault(item['stage'],dict(sim_s=0.,wall_s=0.))
        row['sim_s']+=max(0.,next_item['sim']-item['sim'])
        row['wall_s']+=max(0.,next_item.get('wall',0)-item.get('wall',0))
    return dict(sim_s=end_sim,wall_s=end_wall,stages=stages)
