"""New task console. Only the owner thread touches MuJoCo/OpenGL."""
import argparse,base64,io,json,queue,secrets,threading,time,webbrowser
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import yaml
import mujoco
import numpy as np
from PIL import Image
from run_harvest import ROOT,FruitSkills,Harvest,RoundTimedOut
from run_l1 import L1Skills
from piper_sim.l1_world import L1World,configure_l1,detect_card_targets,card_observation_xy,configure_card_regions
from piper_sim.document_fruits import DocumentFruitWorld,configure_fruits,CATEGORIES
from piper_sim.local_language import parse_instruction,MODEL
from piper_sim.task_plan import parse_plan,resolve_plan
from piper_sim.scene_memory import SceneMemory
from piper_sim.harvest import Track
from piper_sim.timing import summarize
from piper_sim.fruit_task import parse_fruit_task
from piper_sim.fruit_placement import allocate_fruit_positions,alternative_landings,visual_footprint


class Cancelled(Exception):pass


class WebFruitSkills(FruitSkills):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.bin_placements={'red':0,'green':0,'yellow':0};self.card_targets={};self.observed_fruits={}

    def capture_candidates(self,recovery=False):
        candidates=super().capture_candidates(True)
        self.card_targets.update(detect_card_targets(self.last_observation,self.calibration,self.skill.table,self.w.cfg['trays']))
        for color,center in self.card_targets.items():self.w.cfg['trays'][color]['center_xy_m']=np.asarray(center).tolist()
        self.observed_fruits.update(dict(candidates))
        return candidates

    def overview(self):
        self.overview_calls+=1;seen={}
        assignments=self.task.get('assignments') if self.task else None
        required=set(assignments) if assignments else set(CATEGORIES) if not self.task or self.task['category']=='all' else {self.task['category']}
        destinations=set(assignments.values()) if assignments else {self.task['destination']} if self.task and self.task['destination']!='mapping' else {self.mapping[c] for c in required}
        inventory_task=bool(self.task and self.task.get('scope')=='all')
        if inventory_task:required={step['source'] for step in self.task.get('steps',[]) if step['source'] not in ('all','remaining')}
        def collect(candidates):
            for category,estimate in candidates:
                if category not in required and not inventory_task:continue
                if category in seen and len(estimate.points)<=len(seen[category].estimate.points):continue
                destination=assignments[category] if assignments else self.task['destination'] if self.task and self.task['destination']!='mapping' else self.mapping[category]
                seen[category]=Track(category,category,np.median(estimate.points,axis=0).tolist(),destination,estimate=estimate)
        collect(self.capture_candidates())
        views=[[.44,.20],[.44,-.20],[.38,0.]]
        survey_views=0
        for xy in views:
            if not inventory_task and required.issubset(seen) and destinations.issubset(self.card_targets):break
            try:
                self.view_stage=['survey_left','survey_right','survey_front'][survey_views]
                survey_views+=1
                collect(self.view(xy,.50 if survey_views==3 else .46))
            except (RuntimeError,TimeoutError):continue
        self.view_stage='observe'
        self.navigation_events.append(dict(event='three_view_survey',views=survey_views,confirmed=sorted(seen),cards=sorted(self.card_targets)))
        if not required.issubset(seen):raise RuntimeError('总览未确认全部目标：'+','.join(required-set(seen)))
        tracks=list(seen.values())
        if self.task and assignments:
            from piper_sim.pair_placement import select_round_tracks
            tracks,omitted=select_round_tracks(tracks,self.task,self.w.seed)
            self.task['assignments']={t.category:t.destination for t in tracks}
            self.task['expected_categories']=[t.category for t in tracks]
            self.task['unselected_categories']=omitted
            self.navigation_events.append(dict(event='round_selection',selected=self.task['expected_categories'],unselected=omitted,capacity_per_region=2))
            if omitted:
                message='本轮处理'+str(len(tracks))+'件；留置：'+'、'.join({'apple':'苹果','pear':'梨','banana':'香蕉','potato':'土豆','persimmon':'柿子','cucumber':'黄瓜','bell_pepper':'彩椒'}.get(c,c) for c in omitted)
                self.skill.stage=message;self.skill.callback(message)
        if not destinations.issubset(self.card_targets):raise RuntimeError('尚未确认全部目标卡纸')
        self.skill.stage='allocate_pair_regions';self.skill.callback(self.skill.stage)
        positions=allocate_fruit_positions(self.w.cfg['trays'],tracks,self.observed_fruits,self.card_targets)
        self.skill.stage='allocate_backup_landings';self.skill.callback(self.skill.stage)
        alternatives=alternative_landings(self.w.cfg['trays'],tracks,self.observed_fruits,positions)
        footprints={t.key:t.placement_footprint for t in tracks}
        for target in tracks:
            target.placement_footprints=footprints
            target.destination_position=positions[target.key]
            target.destination_options=alternatives[target.key]
            self.navigation_events.append(dict(target=target.key,event='placement_edge_margin',planned_clearance_m=target.placement_margin_m,preferred_m=.05,footprint=target.placement_footprint.record(),rotation_rad=target.placement_rotation))
            target.placement_reservations=positions
            if hasattr(target,'placement_partition'):self.navigation_events.append(dict(target=target.key,event='pair_partition',**target.placement_partition))
        return tracks



class Console:
    def __init__(self,initial_mode='l1',output_root=None):
        self.last_planning_report=None
        self.gap_profile='baseline'
        self.task_requests=0;self.current_record=None
        self.initial_mode=initial_mode
        self.output_root=output_root or ROOT/'artifacts/web-console'
        self.commands=queue.Queue();self.lock=threading.RLock();self.world=None;self.renderer=None
        self.paused=False;self.stop=False;self.index=0;self.last_render=0;self.started=None
        self.data=dict(model=MODEL,busy=False,mode='l1',stage='准备中',logs=[],rgb=None,overview=None,sim=0,wall=0,result=None)

    def update(self,**values):
        with self.lock:self.data.update(values)

    def stage(self,name):
        with self.lock:
            if name!=self.data['stage']:self.data['logs']=(self.data['logs']+[dict(stage=name,sim=self.world.elapsed_s if self.world and self.world.backend else 0,wall=time.monotonic()-self.started if self.started else 0)])[-1000:]
            self.data['stage']=name

    def reset(self,mode,seed=None):
        previous={c:tuple(t['center_xy_m']) for c,t in self.world.cfg['trays'].items()} if self.world else {}
        self.last_planning_report=None
        self.update(planning=None,object_progress={})
        self.started=None;self.paused=False;self.stop=False
        self.current_record=None
        self.update(parsed=None)
        if self.renderer:
            self.renderer.close();self.renderer=None
        if self.world:
            self.world.close();self.world=None
        self.index+=1;self.mode=mode;self.seed=seed if seed is not None else secrets.randbits(32)
        if seed is None and previous:
            for _ in range(100):
                preview={'table':{},'tray_geometry':{}};configure_card_regions(preview,self.seed)
                if any(tuple(preview['trays'][c]['center_xy_m'])!=previous[c] for c in previous):break
                self.seed=secrets.randbits(32)
        self.output=self.output_root/f'scene-{self.index}-{self.seed}';self.output.mkdir(parents=True,exist_ok=True)
        cfg=configure_l1(self.seed,self.output) if mode=='l1' else configure_fruits(self.seed,self.output)
        config=yaml.safe_load(cfg.read_text(encoding='utf-8'))
        profiles=yaml.safe_load((ROOT/'config/sim2real.yaml').read_text(encoding='utf-8'))
        config['deployment_gap']=profiles['profiles'][self.gap_profile]
        cfg.write_text(yaml.safe_dump(config),encoding='utf-8')
        self.world=(L1World if mode=='l1' else DocumentFruitWorld)('l2',cfg,self.output,render=True)
        self.world.cfg['tasks']['l2']['time_limit_s']=300;self.world.reset(self.seed)
        self.update(gap_profile=self.gap_profile,deployment_gap=self.world.deployment_report,time_limit_enforced=self.world.cfg.get('enforce_time_limit',True))
        self.scene_memory=SceneMemory(self.world.cfg['trays']);self.completed_categories=set()
        self.renderer=mujoco.Renderer(self.world.backend.model,height=400,width=640)
        self.update(mode=mode,seed=self.seed,result=None,logs=[],sim=0,wall=0,timing=None)
        self.stage('等待中文指令');self.render(force=True)

    def render(self,force=False):
        if not force and time.monotonic()-self.last_render<.20:return
        wrist_rgb=self.world.backend.camera_rgb()
        camera=mujoco.MjvCamera();camera.lookat[:]=[.40,0,0];camera.distance=1.60;camera.azimuth=135;camera.elevation=-45
        self.renderer.update_scene(self.world.backend.data,camera=camera)
        def encode(rgb):
            data=io.BytesIO();Image.fromarray(rgb).save(data,format='JPEG',quality=80)
            return 'data:image/jpeg;base64,'+base64.b64encode(data.getvalue()).decode()
        self.update(rgb=encode(wrist_rgb),overview=encode(self.renderer.render()),sim=round(self.world.elapsed_s,2),wall=round(time.monotonic()-self.started,1) if self.started else 0)
        self.last_render=time.monotonic()

    def callback(self,skills,original,stage):
        if self.stop:raise Cancelled('用户停止：场景已保留')
        while self.paused:
            if self.stop:raise Cancelled('用户停止')
            self.render();time.sleep(.05)
        self.update(current_issue=getattr(skills,'current_issue',None))
        self.stage(stage);original(stage);self.render()
        # Present continuous motion at approximately real simulation time.
        now=time.monotonic()
        remaining=.05-(now-getattr(self,'last_tick',now))
        if remaining>0:time.sleep(remaining)
        self.last_tick=time.monotonic()

    def planning_update(self,snapshot):
        self.update(planning=snapshot,wall=round(time.monotonic()-self.started,1) if self.started else 0)
        if self.stop and snapshot['status']=='searching':raise Cancelled('用户停止：路径搜索已中断')

    def planning_report(self,report):
        with self.lock:self.last_planning_report=report

    def object_progress(self,event):
        with self.lock:
            progress=dict(self.data.get('object_progress',{}));progress[event['target']]=event
            self.data['object_progress']=progress

    def execute(self,text):
        self.started=time.monotonic();self.stop=False;self.paused=False;self.update(result=None,logs=[],instruction=text,current_issue=None,timing=None,parsed=None,planning=None,object_progress={})
        self.stage('本地千问理解指令')
        allowed=['cube_red','cube_green','cube_yellow'] if self.mode=='l1' else CATEGORIES
        if self.mode=='l1':return self.execute_l1_plan(text,allowed)
        try:task=parse_fruit_task(text,allowed,self.world.cfg['tasks']['l2']['mapping'])
        except ValueError as exc:
            self.task_requests+=1
            self.current_record=dict(instruction=text,model=MODEL,result=dict(status='needs_clarification',error=str(exc)))
            self.current_record_path=self.output/f'task-{self.task_requests}.json'
            self.current_record_path.write_text(json.dumps(self.current_record,ensure_ascii=False,indent=2),encoding='utf-8')
            self.update(result=self.current_record['result']);self.stage('指令需澄清：未执行动作');self.render(force=True)
            return
        self.update(parsed=task)
        if task['action']=='clarify' or (task['action']=='pick_place' and task['category'] not in allowed):
            raise ValueError('请明确当前场景中的物品及目的区域，例如“把红色方块放到黄色卡纸上”。')
        if self.mode=='l1' and task['action']!='pick_place':raise ValueError('L1每次指定一个方块及卡纸颜色，请补充完整任务。')
        if self.mode=='l1' and task['destination'] not in ('red','green','yellow'):raise ValueError('L1只有红、绿、黄卡纸。')
        if self.mode=='l2' and task['destination'] not in ('mapping','red','green','yellow'):raise ValueError('水果场景使用红、绿、黄卡纸区域；请指定对应区域。')
        self.world.begin_attempt()
        if self.mode=='l1':
            skills=L1Skills(self.world,self.output,None,source=task['category'].removeprefix('cube_'),destination=task['destination'])
        else:skills=WebFruitSkills(self.world,self.output,task)
        self.bind_callback(skills)
        self.task_requests+=1
        self.current_record=dict(instruction=text,model=MODEL,parsed=task)
        self.current_record_path=self.output/f'task-{self.task_requests}.json'
        self.current_record_path.write_text(json.dumps(self.current_record,ensure_ascii=False,indent=2),encoding='utf-8')
        harvest=Harvest(skills,max_attempts=None,max_deferrals=None,max_preparations_per_target=8)
        try:result=harvest.run()
        finally:
            self.current_record.update(events=harvest.events,navigation=skills.navigation_events,evaluation=self.world.evaluation())
            self.current_record_path.write_text(json.dumps(self.current_record,ensure_ascii=False,indent=2),encoding='utf-8')
        expected=set(task.get('expected_categories',task['assignments']))
        missing=expected-set(result.get('completed',[]))
        result.update(requested=sorted(expected),not_requested=[c for c in CATEGORIES if c not in expected])
        if result['status']=='complete' and missing:
            result.update(status='incomplete',unresolved=sorted(missing),blockers={c:'任务清单中的目标尚未执行' for c in missing})
        self.update(result=result);self.stage('本条指令完成：'+str(len(result.get('completed',[])))+'件' if result['status']=='complete' else '任务未完成')
        self.current_record['result']=result
        self.current_record['timing']=summarize(self.data['logs'],self.world.elapsed_s,time.monotonic()-self.started)
        self.update(timing=self.current_record['timing'])
        self.current_record_path.write_text(json.dumps(self.current_record,ensure_ascii=False,indent=2),encoding='utf-8')
        (self.output/'last-task.json').write_text(json.dumps(dict(instruction=text,parsed=task,result=result,evaluation=self.world.evaluation()),ensure_ascii=False,indent=2),encoding='utf-8')
        self.render(force=True)

    def bind_callback(self,skills):
        skills.record_stage_images=False
        skills.record_sensor_frames=False
        skills.skill.cfg['progress_observer']=self.object_progress
        skills.skill.cfg['planning_observer']=self.planning_update
        skills.skill.cfg['planning_report_observer']=self.planning_report
        original=skills.skill.callback
        skills.skill.callback=lambda stage:self.callback(skills,original,stage)

    def execute_l1_plan(self,text,allowed):
        plan=parse_plan(text,allowed,['red','green','yellow']);self.update(parsed=plan)
        record=dict(instruction=text,model=MODEL,parsed=plan)
        self.task_requests+=1;self.current_record=record
        self.current_record_path=self.output/f'task-{self.task_requests}.json'
        self.current_record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        if plan['status']!='ready':
            self.stage('需要澄清：未执行动作');self.update(result=dict(status='needs_clarification',error=plan['question']))
            record['result']=self.data['result']
        else:
            self.world.begin_attempt()
            dynamic=any(s['destination'] in ('empty_card','empty_space') or s['source'] in ('remaining','all') for s in plan['steps'])
            sources=[s['source'] for s in plan['steps'] if s['source'] not in ('remaining','all')]
            if any(s['source']=='all' for s in plan['steps']):sources=list(allowed)
            destinations=['red','green','yellow'] if dynamic else list(dict.fromkeys(s['source'].removeprefix('cube_') if s['destination']=='same_color' else s['destination'] for s in plan['steps']))
            scanner=L1Skills(self.world,self.output,None,scene_memory=self.scene_memory);self.bind_callback(scanner)
            self.stage('按需复核物品和卡纸空位')
            survey_start=self.world.elapsed_s
            try:
                scanner.scan(sources,destinations,inventory=dynamic)
            finally:
                record['survey_navigation']=scanner.navigation_events
                record['survey_views']=scanner.survey_views
                record['survey_sim_s']=self.world.elapsed_s-survey_start
                self.current_record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            survey_sim_s=self.world.elapsed_s-survey_start
            actions=resolve_plan(plan,self.scene_memory,allowed,self.completed_categories)
            self.update(parsed=dict(plan,actions=actions))
            record['actions']=actions
            self.current_record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            results=[];deferred=[];skill_runs=[];audit_tracks=[];halted=False
            for index,action in enumerate(actions):
                category=action['category'];estimate=self.scene_memory.objects.get(category)
                if estimate is None:raise ValueError('目标位置已失效，需要重新观察')
                track=Track(category,category,estimate.center.tolist(),action['destination'],estimate=estimate)
                skills=L1Skills(self.world,self.output,None,scene_memory=self.scene_memory,prepared_tracks=[track]);self.bind_callback(skills)
                name={'cube_red':'红色方块','cube_green':'绿色方块','cube_yellow':'黄色方块'}[category]
                color={'red':'红色','green':'绿色','yellow':'黄色'}[action['destination']]
                self.stage(f'执行步骤 {index+1}/{len(actions)}：{name} → {color}卡纸')
                skill_runs.append(skills);audit_tracks.append(track)
                can_defer=sum(a['category']==category for a in actions)==1
                result=Harvest(skills,max_attempts=None,max_deferrals=None,max_preparations_per_target=8,defer_staging=can_defer,defer_batch_audit=True).run()
                results.append(result)
                if result['status']=='staging_deferred':
                    deferred.append((len(results)-1,action,track,skills))
                    self.stage(name+'：继续搜索持物经过准备位置的路线')
                    continue
                if result['status'] not in ('complete','released_pending_audit'):halted=True;break
                self.completed_categories.add(category)
                self.scene_memory.moved(category,track.destination_position)
                # Wrist image may be too close for valid depth; it is only a hint
                # until a following high view actually confirms the moved cube.
            if not halted:
                for slot,action,track,skills in deferred:
                    self.stage('重试持物经准备位置搬运 '+action['category'])
                    results[slot]=Harvest(skills,max_attempts=None,max_deferrals=None,max_preparations_per_target=8,defer_batch_audit=True).run()
                    if results[slot]['status'] not in ('complete','released_pending_audit'):break
                    self.completed_categories.add(action['category'])
                    self.scene_memory.moved(action['category'],track.destination_position)
            record['navigation']=[event for skills in skill_runs for event in skills.navigation_events]
            result=dict(status='complete' if len(results)==len(actions) and all(r['status'] in ('complete','released_pending_audit') for r in results) else 'incomplete',steps=results)
            finalizer=getattr(skill_runs[-1],'finalize_batch',None) if skill_runs else None
            if callable(finalizer):
                latest={t.key:t for t in audit_tracks};positions={k:np.asarray(t.destination_position) for k,t in latest.items()}
                footprints={k:visual_footprint(t.estimate,t.category).rotated(getattr(t,'placement_rotation',0.)) for k,t in latest.items()}
                for t in latest.values():t.placement_reservations=positions;t.placement_footprints=footprints
                delivered={key for step in results for key in step.get('released',step.get('completed',[]))}
                provisional=dict(result,completed=list(delivered),unresolved=[key for key in latest if key not in delivered],
                    attempts=sum(step.get('attempts',0) for step in results),blockers={})
                result=finalizer(list(latest.values()),provisional)
                self.completed_categories.difference_update(result.get('unresolved',[]))
                self.completed_categories.update(result.get('completed',[]))
                # Audit/corrections add navigation events after the initial steps.
                record['navigation']=[event for skills in skill_runs for event in skills.navigation_events]
            self.update(result=result);self.stage('任务完成' if result['status']=='complete' else '任务未完成')
            record.update(actions=actions,result=result,survey_views=scanner.survey_views,survey_sim_s=survey_sim_s,evaluation=self.world.evaluation())
        self.current_record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        self.render(force=True)

    def loop(self):
        self.reset(self.initial_mode)
        while True:
            action,payload=self.commands.get()
            try:
                if action=='reset':
                    self.gap_profile=payload.get('gap_profile',self.gap_profile)
                    self.reset(payload.get('mode',self.mode))
                elif action=='run':self.execute(payload['instruction'])
            except Exception as exc:
                self.stage('已停止' if isinstance(exc,Cancelled) else '比赛时间已到，保留当前进度' if isinstance(exc,RoundTimedOut) else '需要处理')
                self.update(result=dict(status='stopped' if isinstance(exc,Cancelled) else 'timed_out' if isinstance(exc,RoundTimedOut) else 'failed',error=str(exc)))
                if self.current_record is not None:
                    self.current_record['result']=self.data['result']
                    self.current_record_path.write_text(json.dumps(self.current_record,ensure_ascii=False,indent=2),encoding='utf-8')
            finally:
                if self.current_record is not None and self.started:
                    self.current_record['timing']=summarize(self.data['logs'],self.world.elapsed_s,time.monotonic()-self.started)
                    self.update(timing=self.current_record['timing'])
                    self.current_record_path.write_text(json.dumps(self.current_record,ensure_ascii=False,indent=2),encoding='utf-8')
                self.update(busy=False)


def server(console,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,value,status=200,kind='application/json; charset=utf-8'):
            data=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/':return self.send((ROOT/'assets/ui/task_console.html').read_bytes(),kind='text/html; charset=utf-8')
            if self.path=='/api/state':
                with console.lock:return self.send(dict(console.data))
            if self.path=='/api/planning-report':
                with console.lock:report=console.last_planning_report
                return self.send(report if report else {'error':'暂无完成的搜索记录'},200 if report else 404)
            self.send({'error':'Not found'},404)
        def do_POST(self):
            try:
                if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{port}',f'http://localhost:{port}'):raise ValueError('Origin rejected')
                size=int(self.headers.get('Content-Length','0'))
                if size>8192:raise ValueError('Request too large')
                body=json.loads(self.rfile.read(size) or b'{}');action=self.path.removeprefix('/api/')
                with console.lock:
                    if action=='pause':console.paused=not console.paused;console.data['paused']=console.paused
                    elif action=='stop':console.stop=True;console.paused=False
                    elif action in ('run','reset'):
                        if console.data['busy']:raise ValueError('当前任务尚未结束')
                        if action=='run' and (not isinstance(body.get('instruction'),str) or not 0<len(body['instruction'])<=2000):raise ValueError('请输入中文任务')
                        if action=='reset' and body.get('mode',console.data['mode']) not in ('l1','l2'):raise ValueError('Invalid mode')
                        if action=='reset' and body.get('gap_profile',console.gap_profile) not in ('baseline','mild','stress'):raise ValueError('未知误差档位')
                        console.data['busy']=True;console.commands.put((action,body))
                    else:raise ValueError('Unknown action')
                self.send({'accepted':True})
            except Exception as exc:self.send({'error':str(exc)},400)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['l1','l2'],default='l1');p.add_argument('--port',type=int,default=8772);p.add_argument('--no-browser',action='store_true');args=p.parse_args()
    console=Console(args.mode);http=server(console,args.port)
    threading.Thread(target=http.serve_forever,daemon=True).start()
    if not args.no_browser:webbrowser.open(f'http://127.0.0.1:{args.port}')
    console.loop()
