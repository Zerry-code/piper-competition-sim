"""Document L1: explicit target -> RGB mask + depth -> physical IK -> closed loop."""
import argparse,json,secrets,time
import numpy as np
from run_harvest import ROOT,FruitSkills,Harvest,RoundTimedOut,Track,SessionControls,RestartRound
from piper_sim.l1_world import L1World,configure_l1,detect_card_targets,card_observation_xy
from piper_sim.scene_memory import SceneMemory


class L1Skills(FruitSkills):
    def __init__(self,*args,source=None,destination=None,scene_memory=None,prepared_tracks=None,**kwargs):
        super().__init__(*args,**kwargs);self.source=source;self.destination=destination
        self.bin_placements={'red':0,'green':0,'yellow':0}
        self.scene_memory=scene_memory or SceneMemory(self.w.cfg['trays'])
        self.scene_memory.trays=self.w.cfg['trays']
        self.card_targets=self.scene_memory.cards
        self.prepared_tracks=prepared_tracks;self.survey_views=0

    def capture_candidates(self,recovery=False):
        candidates=super().capture_candidates(True)
        obs=self.last_observation
        cards=detect_card_targets(obs,self.calibration,self.skill.table,self.w.cfg['trays'])
        self.scene_memory.observe(obs,self.calibration,self.skill.table,candidates,cards)
        return candidates

    def scan(self,sources,destinations,inventory=False):
        fresh=set()
        required={'cube_red','cube_green','cube_yellow'} if inventory else set(sources)
        def capture(candidates):
            fresh.update(c for c,e in candidates if c.startswith('cube_') and len(e.points)>=100)
            self.navigation_events.append(dict(event='l1_visual_confirmation',frame=self.scene_memory.frames,
                fresh=sorted(fresh),cards=sorted(self.card_targets),
                missing_sources=sorted(required-fresh),missing_cards=sorted(set(destinations)-set(self.card_targets))))
        def ready():
            required={'cube_red','cube_green','cube_yellow'} if inventory else set(sources)
            return required.issubset(fresh) and all(c in self.card_targets and self.scene_memory.free_position(c,exclude=next(iter(sources),None)) is not None for c in destinations)
        capture(self.capture_candidates())
        if ready():
            self.navigation_events.append(dict(event='adaptive_survey',views=0,frames=self.scene_memory.frames,fresh=sorted(fresh)))
            return
        views=[]
        for category in sources:
            if category in self.scene_memory.hints:views.append(self.scene_memory.hints[category])
        if inventory:
            views.extend(self.scene_memory.hints.values())
        views.extend([[.44,.20],[.44,-.20],[.38,0.]])
        card_views={color:card_observation_xy(self.w.cfg,color) for color in self.w.cfg['trays']}
        views.extend(card_views[c] for c in destinations)
        views.extend([[.45,.10],[.52,.18],[.38,.04]]+list(card_views.values()))
        visited=[]
        for xy in views:
            if ready():break
            if any(np.linalg.norm(np.asarray(xy)-old)<.025 for old in visited):continue
            visited.append(np.asarray(xy).copy())
            try:candidates=self.view(np.asarray(xy),.46)
            except (RuntimeError,TimeoutError):continue
            self.survey_views+=1;capture(candidates)
        # A second, closer view is useful for a partially occluded color patch.
        # Only revisit missing evidence; never substitute configured object truth.
        if not ready():
            retry_views=[card_views[c] for c in destinations if c not in self.card_targets or self.scene_memory.free_position(c,exclude=next(iter(sources),None)) is None]
            if required-fresh:retry_views.extend([[.45,.10],[.38,.04]])
            for xy in retry_views:
                if ready():break
                try:candidates=self.view(np.asarray(xy),.40)
                except (RuntimeError,TimeoutError):continue
                self.survey_views+=1;capture(candidates)
        if not ready():
            names={'cube_red':'红色方块','cube_green':'绿色方块','cube_yellow':'黄色方块','red':'红色卡纸','green':'绿色卡纸','yellow':'黄色卡纸'}
            missing=sorted(required-fresh)+sorted(set(destinations)-set(self.card_targets))
            blocked=[c for c in destinations if c in self.card_targets and self.scene_memory.free_position(c,exclude=next(iter(sources),None)) is None]
            details=[]
            if missing:details.append('尚未从图像确认方块身份或三维位置：'+'、'.join(names[c] for c in missing))
            if blocked:details.append('尚未确认安全空位：'+'、'.join(names[c] for c in blocked))
            self.navigation_events.append(dict(event='l1_survey_incomplete',missing=missing,blocked_cards=blocked))
            raise RuntimeError('；'.join(details)+'。已尝试补充视角，保留场景等待复核，未执行盲抓。')
        self.navigation_events.append(dict(event='adaptive_survey',views=self.survey_views,frames=self.scene_memory.frames,fresh=sorted(fresh)))

    def overview(self):
        self.overview_calls+=1
        if self.prepared_tracks is not None:
            selected=self.prepared_tracks
        else:
            sources=['cube_'+self.source] if self.source else ['cube_red','cube_green','cube_yellow']
            destinations=[self.destination] if self.destination else ['red','green','yellow']
            self.scan(sources,destinations,inventory=self.source is None)
            selected=[Track(c,c,self.scene_memory.objects[c].center.tolist(),self.destination or c.removeprefix('cube_'),estimate=self.scene_memory.objects[c]) for c in sources]
        for target in selected:
            point=self.scene_memory.free_position(target.destination,exclude=target.category)
            if point is None:
                xy=card_observation_xy(self.w.cfg,target.destination)
                self.view(xy,.46);self.survey_views+=1
                point=self.scene_memory.free_position(target.destination,exclude=target.category)
            if point is None:raise RuntimeError('目标卡纸没有经过视觉确认的安全空位，禁止叠放')
            target.destination_position=np.asarray(point)
        (self.output/'l1_plan.json').write_text(json.dumps([dict(category=t.category,destination=t.destination,observed_card_target=t.destination_position.tolist()) for t in selected],indent=2),encoding='utf-8')
        return selected


def main(forced_seed=None):
    p=argparse.ArgumentParser();p.add_argument('--source',choices=['red','green','yellow']);p.add_argument('--destination',choices=['red','green','yellow']);p.add_argument('--classify',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--viewer',action='store_true');p.add_argument('--time-limit',type=float,default=300)
    args=p.parse_args()
    if (not args.classify and not (args.source and args.destination)) or (args.classify and (args.source or args.destination)):
        p.error('Specify --source and --destination; optional --classify is a separate batch experiment')
    seed=forced_seed if forced_seed is not None else args.seed if args.seed is not None else secrets.randbits(32)
    output=ROOT/'artifacts/l1'/str(seed);output.mkdir(parents=True,exist_ok=True)
    world=L1World('l2',configure_l1(seed,output),output,render=True);world.cfg['tasks']['l2']['time_limit_s']=args.time_limit
    if args.source:world.cfg['tasks']['l2']['mapping']['cube_'+args.source]=args.destination
    controls=SessionControls('l1')
    viewer=None;report={'seed':seed,'status':'failed','pipeline':'RGB color mask + registered depth + IK; no LLM'}
    try:
        world.reset(seed)
        skills=L1Skills(world,output,None,source=args.source,destination=args.destination)
        if args.viewer:
            import mujoco.viewer
            viewer=mujoco.viewer.launch_passive(world.backend.model,world.backend.data,key_callback=lambda key:controls.key(key) if key in (294,295,296) else None)
            viewer.cam.lookat[:]=[.42,0,.06];viewer.cam.distance=1.25;viewer.cam.azimuth=135;viewer.cam.elevation=-45
            skills.viewer=viewer;skills.controls=controls
            original_display=skills.update_display
            def l1_display(stage):
                original_display(stage)
                # Label the explicit task; no legacy fruit/block switching hint.
                viewer.set_texts((None,None,f"L1 {args.source or 'all'} -> {args.destination or 'same color'} | SIM {world.elapsed_s:.1f}s/{args.time_limit:.0f}s",stage+" | F5 replay | F6 random | F7 pause"))
            skills.update_display=l1_display
        harvest=Harvest(skills,max_attempts=None,max_deferrals=None)
        report.update(harvest.run())
    except RoundTimedOut:report['status']='timed_out'
    except RestartRound:report['status']='restarted_by_user'
    except Exception as exc:report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        try:report['evaluation']=world.evaluation()
        except Exception as exc:report['evaluation']={'error':str(exc)}
        requested=[o for o in world.objects if args.source is None or o['category']=='cube_'+args.source]
        success=[o['id'] for o in requested if (args.destination or o['category'].removeprefix('cube_')) in world.events.get(o['id'],{}).get('placed',[])]
        report['requested_evaluation']=dict(source=args.source,destination=args.destination,expected=len(requested),placed=success)
        if report['status']=='complete' and len(success)!=len(requested):report['status']='incomplete'
        if 'skills' in locals():report['sensor_events']=skills.navigation_events
        if 'harvest' in locals():report['events']=harvest.events
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('sensor_events','evaluation')},ensure_ascii=True),flush=True)
        if viewer:
            while viewer.is_running() and controls.request is None:
                skills.update_display(f"{report['status']} | placed {len(success)}/{len(requested)}")
                viewer.sync();time.sleep(.05)
            viewer.close()
        world.close()
    return controls.request,seed


if __name__=='__main__':
    forced=None
    while True:
        request,seed=main(forced)
        if request is None:break
        forced=seed if request[0]=='replay' else secrets.randbits(32)
