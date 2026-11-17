"""Bounded planning evidence, kept separate from robot commands and rendering."""
import time,math
from collections import Counter,deque

PHASES={'hover':'到达抓取点上方','near':'接近目标','descend':'下降到抓取点','lift':'夹起后的提升','reorient':'调整搬运姿态','transfer':'搬运路径','preplace':'接近放置区','place':'下放物品','opening':'张爪空间','retreat':'放置后的退出路径'}
NAMES={'pear':'梨','banana':'香蕉','persimmon':'柿子','bell_pepper':'彩椒','cucumber':'黄瓜','potato':'土豆','cube_red':'红方块','cube_green':'绿方块','cube_yellow':'黄方块'}

def explain(error):
    text=str(error)
    if 'IK unreachable' in text:code,label='ik_unreachable','该位姿未通过逆运动学检查'
    elif 'discontinuity' in text:code,label='joint_jump','相邻路径点的关节变化过大'
    elif '碰撞' in text or 'collision' in text:code,label='collision','路径碰撞检查未通过'
    elif '预算' in text or 'budget' in text:code,label='candidate_budget','本候选的计算时间用完'
    elif 'opening' in text or '张爪' in text:code,label='opening','张爪空间不足'
    elif '视觉占地' in text:code,label='placement_footprint','旋转后的占地不满足边距或避让要求'
    elif 'transfer orientation' in text:code,label='transfer_orientation','抓取可达，搬运或放置姿态尚未通过'
    else:code,label='other','候选方案检查未通过'
    detail={}
    for field in ('target_position','position_error_m','orientation_error_rad','joint_index','joint_delta_rad'):
        if hasattr(error,field):detail[field]=getattr(error,field)
    if 'position_error_m' in detail:label+=f"（位置残差{detail['position_error_m']*1000:.1f}毫米，姿态残差{math.degrees(detail['orientation_error_rad']):.1f}°）"
    return dict(reason_code=code,reason=label,error=text,**detail)

class PlanningTrace:
    def __init__(self,target,budget,observer=None):
        self.target=NAMES.get(target,target or '当前目标');self.budget=budget;self.observer=observer
        self.started=time.monotonic();self.events=[];self.recent=deque(maxlen=10);self.counts=Counter();self.failures=Counter();self.unique=set();self.current={};self.status='searching';self.scope={};self.truncated=0
    def emit(self,event,**data):
        elapsed=round(time.monotonic()-self.started,3);self.counts[event]+=1
        if event=='scope':self.scope=data
        if event=='attempt':
            self.current=data.copy();self.unique.add((data['grasp_index'],data['landing_index'],data['yaw_deg'],data['tilt_deg'],data['height_m']))
        if event=='phase':self.current.update(phase=data['phase'],phase_label=PHASES.get(data['phase'],data['phase']))
        if event=='route':self.current.update(route_label=data['route_label'])
        if event.endswith('failure') and 'reason_code' in data:self.failures[self.current.get('phase_label','候选')+'：'+data['reason'].split('（')[0]]+=1
        if event=='attempt':message=f"尝试{self.counts[event]}：抓取点{data['grasp_index']}/{data['grasp_count']}，落点{data['landing_index']}/{data['landing_count']}，方向{data['yaw_deg']:.1f}°、倾角{data['tilt_deg']:.1f}°、高度{data['height_m']*100:.1f}厘米"
        elif event=='phase':message='检查：'+PHASES.get(data['phase'],data['phase'])
        elif event=='route':message='尝试搬运路线：'+data['route_label']
        elif event.endswith('failure'):message=self.current.get('phase_label','候选')+'：'+data.get('reason',data.get('error','未通过'))
        elif event=='waypoint':message=f"{self.current.get('phase_label','路径')}：采样点{data['index']}/{data['total']}"
        elif event=='found':message='已找到通过本次检查的抓放方案'
        elif event=='scope':message=f"候选范围：{data['grasp_count']}个抓取点、{data['landing_count']}个落点、{data['tilt_count']}种倾角、{data['height_count']}种高度"
        else:message=data.get('message',event)
        item=dict(data,event=event,elapsed_s=elapsed,message=message,phase=self.current.get('phase'))
        if event in ('pose_failure','orientation_failure','route_failure') or event=='attempt_failure' and not hasattr(self,'last_failure'):self.last_failure=item
        self.recent.append(item)
        if len(self.events)<20000:self.events.append(item)
        else:self.truncated+=1
        self.latest=item
        if self.observer:self.observer(self.snapshot())
    def snapshot(self):
        return dict(target=self.target,status=self.status,elapsed_s=round(time.monotonic()-self.started,2),budget_s=self.budget,scope={k:v for k,v in self.scope.items() if k.endswith('_count') or k=='combination_upper_bound'},current=dict(self.current),attempts=self.counts['attempt'],unique_combinations=len(self.unique),route_checks=self.counts['route'],failures=dict(self.failures),recent=list(self.recent),latest=getattr(self,'latest',None),last_failure=getattr(self,'last_failure',None))
    def finish(self,status,error=None):
        self.status=status
        labels={'found':'找到可执行候选','timeout':'达到时间预算，整体可行性未知','scheduled_finished':'本轮候选检查结束，未证明整体不可行','interrupted':'搜索被中断'}
        self.emit('finished',message=labels[status],error=str(error) if error else None)
        return dict(self.snapshot(),scope_details=self.scope,events=self.events,truncated_events=self.truncated,conclusion=labels[status],evidence_scope='本轮有限抓取点、角度、高度、落点及有限绕行；不是全局不可达证明')
