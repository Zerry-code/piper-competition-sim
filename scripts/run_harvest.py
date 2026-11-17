from piper_sim.traditional import IK
from piper_sim.fruit_placement import plan_footprint
"""Windows-compatible independent MuJoCo fruit environment and harvest pipeline."""
from uuid import uuid4
import argparse
import copy
import json
import sys
import time
import secrets
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image
from piper_sim.competition_world import CompetitionWorld, numbers, look_rotation
from piper_sim.calibration import load_config, ROOT
from piper_sim.demo_skill import PickPlaceSkill, GraspEmptyError
from piper_sim.harvest import Harvest, Track, RetryableFailure
from piper_sim.implicit_fruit import FRUITS, COLORS, components, mesh
from piper_sim.fruit_perception import estimate_candidates,candidates_3d
from piper_sim.local_language import parse_instruction
from piper_sim.session_controls import SessionControls, RestartRound
from piper_sim.grasp_contact import contact_quality,retained_support
from piper_sim.placement_boundary import inspect_boundary
from piper_sim.fruit_placement import visual_footprint,validate_landing,reallocate_final_landings
from piper_sim.batch_audit import BatchBoundaryAudit,identify_audit_target,public_status
from piper_sim.planning_trace import NAMES
from piper_sim.scene_memory import SceneMemory
from piper_sim.staging import staging_options,staging_surface_safe,select_staging_landings,progressive_options,select_progressive_landings
from piper_sim.drop_recovery import ObjectDropped,LandingTracker,landing_region
from piper_sim.release_safety import ReleaseBlocked,release_feedback


from piper_sim.fruit_world import ImplicitWorld, configure


class RoundTimedOut(Exception):
    pass


class FruitSkills:
    def __init__(self,world,output,task):
        from piper_sim.controller_scene import ControllerWorld
        self.w,self.b,self.output,self.task = ControllerWorld(world),world.backend,output,task
        self.calibration = load_config()
        self.controls = None
        self.viewer = None
        self.last_frame_time = time.monotonic()
        self.wall_started=time.monotonic()
        self.last_camera_refresh=0.
        self.finished_wall=None
        self.counter = 0
        self.last_stage=None
        self.frame=0
        self.bin_placements={'red':0,'green':0,'blue':0}
        self.skill = PickPlaceSkill(self.b,world.cfg['table']['top_m'],self.snapshot,
            dict(natural_pose_preference=True,ready_transport=True,minimum_carried_clearance_m=.105,ready_position_m=[.28,0.,.30],adaptive_transport_height=True,random_seed=int(world.seed),tray_colors=list(world.cfg['trays']),transport_height_m=.20,transport_height_candidates_m=[.18,.22],max_task_s=1800,settle_timeout_s=4,lift_verify_hold_s=.35,place_verify_hold_s=0.,verify_after_vertical_clearance=True,release_to_observation_height=False,grasp_check=lambda:contact_quality(self.b)['accepted'],grasp_ready=lambda:contact_quality(self.b).get('min_force_n',0.)>=2.,grasp_retained=lambda:contact_quality(self.b)['loaded'],tilt_candidates=[.15,.35,.6,.9,1.2,1.5,-.3,-.6,-.9,1.8,2.1,2.5,3.0],motion_profiles=yaml.safe_load((ROOT/'config/motion_profiles.yaml').read_text(encoding='utf-8'))),
            stepper=lambda a:world.step(a,observe=False))
        self.mapping = copy.deepcopy(world.cfg['tasks']['l2']['mapping'])
        self.plan = None
        self.overview_calls=0
        self.navigation_events=[]
        self.phase_times={}
        self.surface_memory=SceneMemory(self.w.cfg['trays'])
        self.staging_enabled=False
        self.skip_boundary_checks=True
        self.persistent_preparation=True
        self.keep_retrying=not world.cfg.get('enforce_time_limit',True)
        if self.keep_retrying:self.skill.cfg['max_task_s']=float('inf')
        self.round_targets={}
        self.defer_boundary_checks=True
        self.observed_fruits={}
        self.skill.cfg['deadline_check']=self.check_deadline
        self.last_sim_time=float(self.b.data.time)

    def update_display(self,stage):
        if self.viewer is None:return
        if self.finished_wall is not None:stage=getattr(self,'round_summary',stage)
        elapsed=self.w.elapsed_s
        limit=self.w.cfg['tasks']['l2']['time_limit_s']
        wall=(self.finished_wall or time.monotonic())-self.wall_started
        timer=f'SIM {elapsed:6.1f}s / {limit:.0f}s   LEFT {max(0,limit-elapsed):6.1f}s   WALL {wall:6.1f}s' if self.w.cfg.get('enforce_time_limit',True) else f'SIM {elapsed:6.1f}s   WALL {wall:6.1f}s   NO TIME LIMIT'
        self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
            timer,
            f'{stage}\nF5 replay | F6 random | F7 pause | 1 blocks | 2 fruits'))
        if time.monotonic()-self.last_camera_refresh>.10:
            wrist_rgb=self.b.camera_rgb()
            viewport=self.viewer.viewport
            width=max(128,min(280,int(viewport.width*.28)))
            height=int(width*wrist_rgb.shape[0]/wrist_rgb.shape[1])
            rgb=np.array(Image.fromarray(wrist_rgb).resize((width,height)))
            self.viewer.set_images((mujoco.MjrRect(viewport.left+max(0,viewport.width-width-12),viewport.bottom+12,width,height),rgb))
            self.last_camera_refresh=time.monotonic()

    def snapshot(self,stage):
        now=float(self.b.data.time)
        self.phase_times[stage]=self.phase_times.get(stage,0.)+max(0,now-self.last_sim_time)
        self.last_sim_time=now
        self.check_deadline()
        if self.w.cfg.get('enforce_time_limit',True) and self.w.elapsed_s>=self.w.cfg['tasks']['l2']['time_limit_s']-1e-8:
            raise RoundTimedOut('Simulation time budget exhausted')
        if self.controls:
            self.controls.check()
            while self.controls.paused and self.viewer and self.viewer.is_running():
                self.update_display('PAUSED')
                self.viewer.sync()
                time.sleep(.05)
                self.controls.check()
        if self.viewer is not None:
            if not self.viewer.is_running(): raise RuntimeError('Viewer closed by user')
            self.update_display(stage)
            self.viewer.sync()
            remaining=.05-(time.monotonic()-self.last_frame_time)
            if remaining>0:time.sleep(remaining)
            self.last_frame_time=time.monotonic()
        if stage==self.last_stage:return
        if stage.startswith('release_'):
            self.navigation_events.append(dict(event='release_state',stage=stage,
                time_s=float(self.b.data.time),tcp=self.b.data.site_xpos[self.b.model.site('tcp').id].tolist(),
                feedback=release_feedback(self.b)))
        self.last_stage=stage
        self.frame+=1
        if not getattr(self,'record_stage_images',True):return
        camera=mujoco.MjvCamera()
        camera.lookat[:]=[.36,-.035,.1]
        camera.distance=1.05
        camera.azimuth=135
        camera.elevation=-40
        self.b.renderer.update_scene(self.b.data,camera=camera)
        Image.fromarray(self.b.renderer.render().copy()).save(self.output/f'phase_{self.frame:03}_{stage}.png')

    def view(self, xy, height,recovery=False,holding=False):
        """Raise before lateral motion, then aim wrist optical camera at target region."""
        camid = self.b.model.camera('competition_camera').id
        data = mujoco.MjData(self.b.model)
        lower,upper = self.b.model.jnt_range[self.b.joints].T
        target=np.r_[xy,self.skill.table+.025]
        def residual(q):
            data.qpos[:6]=q
            mujoco.mj_kinematics(self.b.model,data)
            mujoco.mj_camlight(self.b.model,data)
            rotation=data.cam_xmat[camid].reshape(3,3)
            optical=np.diag([1.,-1.,-1.])@rotation.T@(target-data.cam_xpos[camid])
            tcp_z=data.site_xpos[self.b.model.site('tcp').id,2]
            return np.r_[optical[:2],.8*(optical[2]-height),
                         2*max(0,self.skill.table+.18-tcp_z),.012*q[3],.012*q[5]]
        best = None
        view_solutions=[]
        for seed in [self.b.measured()[:6],[0,.4,-.8,0,1.2,0],[0,1,-1.4,0,1.8,0]]:
            sol = least_squares(residual,np.clip(seed,lower+1e-5,upper-1e-5),
                                bounds=(lower+1e-6,upper-1e-6),max_nfev=180)
            view_solutions.append(sol)
            if best is None or np.linalg.norm(sol.fun)<np.linalg.norm(best.fun): best=sol
            if not holding and np.linalg.norm(sol.fun[:4])<.012:break
        residual(best.x)
        rotation=data.cam_xmat[camid].reshape(3,3)
        optical=np.diag([1.,-1.,-1.])@rotation.T@(target-data.cam_xpos[camid])
        if optical[2]<self.b.camera['camera']['depth']['min_valid_m']+.015 or np.linalg.norm(optical[:2])/optical[2]>.2:
            raise RuntimeError(f'Overview cannot keep the workspace within valid camera depth: {optical}')
        self.skill.stage=getattr(self,'view_stage','observe')
        if holding:
            for candidate in sorted(view_solutions,key=lambda candidate:np.linalg.norm(candidate.fun)):
                residual(candidate.x)
                rot=data.cam_xmat[camid].reshape(3,3)
                depth=np.diag([1.,-1.,-1.])@rot.T@(target-data.cam_xpos[camid])
                if depth[2]<self.b.camera['camera']['depth']['min_valid_m']+.015 or np.linalg.norm(depth[:2])/depth[2]>.2:continue
                try:self.skill.held_transition(np.linspace(self.b.measured()[:6],candidate.x,35));break
                except (RuntimeError,TimeoutError):continue
            else:raise ReleaseBlocked('持物观察姿态未通过安全路径检查，换视角而非松开夹爪')
        else:self.skill.move(best.x,.10)
        self.skill.hold(.05 if self.skill.stage in ('check_placement_boundary','check_staging_release') else .3)
        if self.skill.stage in ('check_placement_boundary','check_staging_release'):return []
        return self.capture_candidates(recovery=recovery)

    def capture_candidates(self,recovery=False):
        obs = self.b.observe()
        self.last_observation=obs
        self.counter+=1
        if getattr(self,'record_sensor_frames',True):
            Image.fromarray(obs.rgb).save(self.output/f'wrist_{self.counter:03}.png')
            np.savez_compressed(self.output/f'rgbd_{self.counter:03}.npz',
                depth=obs.depth_m,T_base_end=obs.T_base_end,intrinsics=json.dumps(obs.intrinsics))
        self.surface_memory.observe(obs,self.calibration,self.skill.table,[],{})
        kwargs={}
        if recovery:
            center=np.array(self.w.cfg['table']['center_xy_m']);half=np.array(self.w.cfg['table']['half_size_xy_m'])
            left=center[0]-half[0]
            if self.w.cfg.get('venue'):left=-self.w.cfg['venue']['base_from_rear_edge_m']
            kwargs=dict(workspace=(left,center[0]+half[0],center[1]-half[1],center[1]+half[1]),
                        trays=self.w.cfg['trays'],tray_geometry=self.w.cfg['tray_geometry'])
        candidates=estimate_candidates(obs,self.calibration,self.skill.table,None,**kwargs)
        for _,estimate in candidates:estimate.observed_wall=time.monotonic()
        self.navigation_events.append(dict(event='visual_candidates',candidates=[dict(category=c,center=np.median(e.points,axis=0).tolist(),recognition=getattr(e,'recognition',{})) for c,e in candidates]))
        self.observed_fruits.update(dict(candidates))
        return candidates

    def overview(self):
        self.overview_calls+=1
        seen=[]
        self.skill.stage='raise_for_overview'
        self.skill.move(np.array([0.,.7,-1.,0.,1.15,0.]),.10)
        self.skill.hold(.2)
        # Multiple high views are allowed when one reachable view cannot cover the table.
        for xy in [[.37,-.06],[.44,.02],[.37,.12],[.44,.18],[.34,.04]]:
            try: candidates=self.view(xy,.46)
            except (RuntimeError,TimeoutError) as exc:
                print('Skipped unreachable survey view:',str(exc),flush=True)
                continue
            for category,estimate in candidates:
                anchor=np.median(estimate.points,axis=0)
                existing=next((t for t in seen if t.category==category),None)
                if existing is not None:
                    if len(estimate.points)>len(existing.estimate.points):
                        existing.position=anchor.tolist()
                        existing.estimate=estimate
                    continue
                if self.task and self.task['category'] not in ('all',category): continue
                destination = self.task['destination'] if self.task and self.task['destination']!='mapping' else self.mapping[category]
                seen.append(Track(f'track_{len(seen)+1}',category,anchor.tolist(),destination,estimate=estimate))
        if not seen:raise RuntimeError('No sensor-visible fruit; overview cannot establish a queue')
        counts={color:0 for color in self.mapping.values()}
        for target in sorted(seen,key=lambda t:t.position[1]):
            destination=np.array(self.w.cfg['trays'][target.destination]['center_xy_m'],dtype=float)
            if not target.category.startswith('cube_'):
                first_offset=0.0 if target.category=='banana' else -.020 if target.category=='pear' else -.04
                second_offset=.020 if target.category=='banana' else .045
                destination[1]+=first_offset if counts[target.destination]==0 else second_offset
            target.destination_position=destination
            counts[target.destination]+=1
        (self.output/'task_plan.json').write_text(json.dumps([
            dict(target=t.key,category=t.category,estimated_position=t.position,
                 destination=t.destination,position=t.destination_position.tolist()) for t in seen],indent=2),encoding='utf-8')
        return seen

    def reacquire_target(self,target):
        self.progress(target,'reacquiring','当前未看清目标，扩大视角重新分离识别，继续保留抓取任务')
        anchor=np.asarray(target.position)[:2]
        views=[anchor,np.array([.44,.20]),np.array([.44,-.20]),np.array([.38,0.]),anchor+[-.05,0.]]
        for index,xy in enumerate(views):
            self.check_deadline()
            xy=np.asarray(xy,dtype=float).copy();xy[0]=np.clip(xy[0],.28,.54)
            self.view_stage='reacquire_target'
            try:candidates=self.view(xy,.48 if index else .44,recovery=True)
            except (RuntimeError,TimeoutError) as exc:
                self.navigation_events.append(dict(target=target.key,event='reacquire_view_blocked',error=str(exc)));continue
            finally:self.view_stage='observe'
            self.navigation_events.append(dict(target=target.key,event='reacquire_visible',view=xy.tolist(),
                candidates=[dict(category=c,center=np.median(e.points,axis=0).tolist(),extent=np.ptp(e.points,axis=0).tolist()) for c,e in candidates]))
            matches=[e for c,e in candidates if c==target.category]
            if matches:
                estimate=min(matches,key=lambda e:np.linalg.norm(np.median(e.points,axis=0)[:2]-anchor))
                target.position=np.median(estimate.points,axis=0).tolist();target.local_view_failures=0
                return self.prepare_target(target,estimate)
        target.local_view_failures=getattr(target,'local_view_failures',0)+1
        raise RetryableFailure('本组视角尚未确认目标；保持待抓取并继续搜索，不判不可达',phase='observation')

    def observe_target(self,target,retry):
        if getattr(target,'staging_leg',0)==1:self.surface_memory=target.staging_surface_memory
        if getattr(target,'staging_leg',0)==1 and getattr(target,'staging_refresh_needed',False):
            self.surface_memory=target.staging_surface_memory
            goal=np.asarray(target.staging_final['position'],dtype=float);source=np.asarray(target.position)[:2]
            view_options=[goal,(goal+source)/2,np.array([.39,0.])]
            index=getattr(target,'segment_view_index',0);target.segment_view_index=index+1
            xy=view_options[index%len(view_options)].copy()
            try:self.view(xy,.44,recovery=True)
            except (RuntimeError,TimeoutError):pass
            target.staging_refresh_needed=False;target.planning_retry_ready=False
        if getattr(target,'local_view_failures',0)>=2:return self.reacquire_target(target)
        if getattr(target,'planning_retry_ready',False):
            target.planning_retry_ready=False
            if time.monotonic()-getattr(target,'planning_observed_wall',0)<1.5:
                return self.prepare_target(target,target.estimate)
        if getattr(target,'drop_estimate_ready',False):
            target.drop_estimate_ready=False
            return self.prepare_target(target,target.estimate)
        # Reuse a fresh, sufficiently detailed current view before moving the camera.
        if not retry and not getattr(target,'recovered_from_drop',False):
            matches=[e for category,e in self.capture_candidates(recovery=True) if category==target.category and len(e.points)>=500 and not getattr(e,'grasp_quality',{}).get('requires_refinement')]
            if matches:
                estimate=min(matches,key=lambda e:np.linalg.norm(np.median(e.points,axis=0)[:2]-np.asarray(target.position)[:2]))
                if np.linalg.norm(np.median(estimate.points,axis=0)[:2]-np.asarray(target.position)[:2])<.08:
                    self.navigation_events.append(dict(target=target.key,event='current_view_sufficient'))
                    return self.prepare_target(target,estimate)
        # Use the stored target anchor and current joints; do not return to an overview/home pose.
        anchor=np.asarray(target.position,dtype=float)
        if getattr(target,'local_view_failures',0)>0:
            try:
                n=target.local_view_failures
                offsets=[[-.045,0],[.035,.045],[.035,-.045],[0,0]]
                xy=anchor[:2]+offsets[(n-1)%len(offsets)]
                self.navigation_events.append(dict(target=target.key,event='alternate_local_view',index=n,view_xy=xy.tolist()))
                candidates=self.view(xy,.46,recovery=getattr(target,'recovered_from_drop',False))
            except (RuntimeError,TimeoutError) as exc:
                self.b.previous_target[:6]=self.b.measured()[:6]
                target.local_view_failures+=1
                raise RetryableFailure('High retry observation unavailable: '+str(exc),phase='observation') from exc
            matches=[e for category,e in candidates if category==target.category]
            if not matches:
                target.local_view_failures+=1
                raise RetryableFailure('Target not visible from high retry view',phase='observation')
            result=self.prepare_target(target,min(matches,key=lambda e:np.linalg.norm(np.median(e.points,axis=0)[:2]-anchor[:2])))
            target.local_view_failures=0
            return result
        data=mujoco.MjData(self.b.model)
        camid=self.b.model.camera('competition_camera').id
        seed=self.b.measured()[:6].copy()
        lower,upper=self.b.model.jnt_range[self.b.joints].T
        desired_depth=.39 if retry else .36
        def residual(q):
            data.qpos[:6]=q
            mujoco.mj_kinematics(self.b.model,data)
            mujoco.mj_camlight(self.b.model,data)
            rotation=data.cam_xmat[camid].reshape(3,3)
            optical=np.diag([1.,-1.,-1.])@rotation.T@(anchor-data.cam_xpos[camid])
            tcp_z=data.site_xpos[self.b.model.site('tcp').id,2]
            return np.r_[optical[:2],.8*(optical[2]-desired_depth),.012*(q-seed),
                         2*max(0,self.skill.table+.10-tcp_z),.012*q[3],.012*q[5]]
        neutral=np.array([np.arctan2(anchor[1],anchor[0]),.7,-1.1,0,1.5,0])
        options=[least_squares(residual,np.clip(start,lower+1e-5,upper-1e-5),
            bounds=(lower+1e-6,upper-1e-6),max_nfev=130) for start in [neutral,seed]]
        solution=min(options,key=lambda result:np.linalg.norm(result.fun))
        residual(solution.x)
        rotation=data.cam_xmat[camid].reshape(3,3)
        optical=np.diag([1.,-1.,-1.])@rotation.T@(anchor-data.cam_xpos[camid])
        if optical[2]<.315 or np.linalg.norm(optical[:2])/optical[2]>.23:
            self.snapshot('local_view_unavailable')
            target.local_view_failures=getattr(target,'local_view_failures',0)+1
            raise RetryableFailure('No feasible local wrist view of the stored target',phase='observation')
        self.skill.stage='local_observe'
        try:
            self.skill.move(solution.x,.10)
            self.skill.hold(.10)
        except TimeoutError as exc:
            self.b.previous_target[:6]=self.b.measured()[:6]
            target.local_view_failures=getattr(target,'local_view_failures',0)+1
            self.snapshot('local_observe_retry')
            raise RetryableFailure('Local observation tracking timed out; retry with a high view',phase='observation') from exc
        self.navigation_events.append(dict(target=target.key,retry=retry,
            start_joints=seed.tolist(),view_joints=solution.x.tolist(),desired_depth=desired_depth))
        candidates=self.capture_candidates(recovery=getattr(target,'recovered_from_drop',False))
        nearby=[(np.linalg.norm(np.median(e.points,axis=0)[:2]-target.position[:2]),e)
                for category,e in candidates if category==target.category]
        # Categories are unique in this fixture; retain identity even after a failed
        # grasp pushes the object beyond its original 7 cm association gate.
        if not nearby:
            self.snapshot('target_not_visible')
            target.local_view_failures=getattr(target,'local_view_failures',0)+1
            raise RetryableFailure('Target not reacquired',phase='observation')
        estimate=min(nearby,key=lambda x:x[0])[1]
        return self.prepare_target(target,estimate)

    def progress(self,target,state,reason):
        target.task_state=state
        event=dict(target=target.key,category=target.category,name=NAMES.get(target.category,target.category),
            state=state,reason=reason,staging_leg=getattr(target,'staging_leg',0),staging_hops=getattr(target,'staging_hops',0))
        self.navigation_events.append(dict(event,event='object_progress'))
        observer=getattr(self.skill,'cfg',{}).get('progress_observer')
        if observer:observer(event)

    def save_planning_report(self,target,phase=None):
        report=getattr(self.skill,'last_planning_report',None)
        if report is None or report is getattr(self,'last_saved_planning_report',None):return
        self.last_saved_planning_report=report
        filename='planning-'+target.category+'-'+uuid4().hex[:10]+'.json'
        saved=dict(report,target_category=target.category,bridge_phase=phase)
        (self.output/filename).write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
        self.navigation_events.append(dict(target=target.key,event='planning_search_report',file=filename,
            status=report['status'],attempts=report['attempts'],elapsed_s=report['elapsed_s'],bridge_phase=phase))
        observer=self.skill.cfg.get('planning_report_observer')
        if observer:observer(saved)

    def check_deadline(self):
        if not self.w.cfg.get('enforce_time_limit',True):return
        limit=self.w.cfg['tasks']['l2']['time_limit_s']
        if time.monotonic()-self.wall_started>=limit:
            raise RoundTimedOut('实际比赛时间已到；保留进度，不将搜索失败当作物理不可达')

    def register_targets(self,targets):
        self.round_targets.update({t.key:t for t in targets})

    def physical_reservations(self,target):
        positions={};shapes={}
        for key,point in getattr(target,'placement_reservations',{}).items():
            if key==target.key:continue
            other=getattr(self,'round_targets',{}).get(key)
            if other is not None:
                if not hasattr(other,'last_released_position'):continue
                positions[key]=other.last_released_position
                shape=getattr(other,'last_released_footprint',None)
                if shape is not None:shapes[key]=shape
            else:
                positions[key]=point
                if key in getattr(target,'placement_footprints',{}):shapes[key]=target.placement_footprints[key]
        return positions,shapes

    def configure_staging_landing(self,target,estimate):
        shape=visual_footprint(estimate,target.category);source=np.median(estimate.points,axis=0)
        reservations,shapes=self.physical_reservations(target)
        memory=target.staging_surface_memory
        if hasattr(self,'last_observation'):memory.observe(self.last_observation,self.calibration,self.skill.table,[],{})
        mode=getattr(target,'staging_mode','sweet')
        if mode=='progressive':
            goal=target.staging_final['position']
            options=progressive_options(shape,self.w.cfg.get('support_surface',self.w.cfg['table']),self.w.cfg['trays'],memory,
                reservations,shapes,target.key,source,goal,target.destination)
        else:
            options=staging_options(shape,self.w.cfg.get('support_surface',self.w.cfg['table']),self.w.cfg['trays'],memory,
                reservations,shapes,target.key,source)
            # Prefer the central dexterous workspace, not another far-side short hop.
            options=[o for o in options if .33<=o['point'][0]<=.44 and abs(o['point'][1])<=.10]
        if not options and mode=='progressive':
            # A retreat to the dexterous region may initially move AWAY from
            # the final paper; do not forbid that safe recovery step.
            options=staging_options(shape,self.w.cfg.get('support_surface',self.w.cfg['table']),self.w.cfg['trays'],memory,reservations,shapes,target.key,source)
            options=[o for o in options if .33<=o['point'][0]<=.44 and abs(o['point'][1])<=.10]
            if options:mode='sweet';target.staging_mode='sweet'
        if not options:
            target.local_view_failures=getattr(target,'local_view_failures',0)+1
            if mode=='sweet':target.staging_mode='progressive'
            target.staging_refresh_needed=True
            raise RetryableFailure('当前视图尚未确认本段安全落点，继续寻找新视角和落点',phase='staging')
        best=options[0];target.placement_rotation=best['rotation']
        target.destination_position=best['point'].copy()
        same=[o for o in options if abs(o['rotation']-best['rotation'])<1e-6]
        if mode=='progressive':target.destination_options=[o['point'] for o in select_progressive_landings(options,source,target.staging_final['position'])]
        else:target.destination_options=[o['point'] for o in select_staging_landings(options,source)]
        cards=self.w.cfg['trays'] if mode=='sweet' else {c:t for c,t in self.w.cfg['trays'].items() if c!=target.destination}
        target.placement_extra_validator=lambda sh,point:staging_surface_safe(sh,point,cards,memory)
        target.placement_region=dict(self.w.cfg.get('support_surface',self.w.cfg['table']))
        target.placement_floor=self.skill.table
        self.navigation_events.append(dict(target=target.key,event='segment_landing_candidates',mode=mode,
            source=source.tolist(),goal=np.asarray(target.staging_final['position']).tolist(),
            candidates=[np.asarray(p).tolist() for p in target.destination_options]))

    def plan_staging_leg(self,target,estimate):
        mode=getattr(target,'staging_mode','sweet')
        self.progress(target,'planning_sweet_leg' if mode=='sweet' else 'planning_progressive_leg',
            '只规划并执行到前方灵活区的这一段，放下后再算下一段' if mode=='sweet' else '保底分段：选择当前可达且更接近目标的落点')
        self.skill.cfg['planning_target']=NAMES.get(target.category,target.category)+(' · 前方灵活区单段' if mode=='sweet' else ' · 向目标推进单段')
        try:
            plan=self.skill.plan(estimate,target.destination_position,self.skill.table)
            self.save_planning_report(target,'single_leg_'+mode)
            self.navigation_events.append(dict(target=target.key,event='single_leg_ready',mode=mode,
                destination=plan['destination_position'],second_leg_not_required=True))
            return plan
        except RuntimeError:
            self.save_planning_report(target,'single_leg_'+mode)
            if mode=='sweet':
                target.staging_mode='progressive';target.staging_refresh_needed=True
                self.progress(target,'segment_pending','到灵活区的本段尚未通过，改用向目标推进的保底分段')
            raise

    def request_restaging(self,target,reason):
        target.staging_leg=0;target.staging_pending=True;target.staging_mode='progressive'
        target.planning_retry_ready=False;target.local_view_failures=0
        self.progress(target,'staging_replan','本次直达搜索未通过，继续逐段向目标推进；没有判定目标失败')
        self.navigation_events.append(dict(target=target.key,event='progressive_replan_requested',reason=reason,
            completed_hops=getattr(target,'staging_hops',0)))
        return True

    def prepare_target(self,target,estimate):
        estimate.category=target.category
        if hasattr(target,'placement_partition'):estimate.placement_partition=target.placement_partition
        if getattr(estimate,'grasp_quality',{}).get('requires_refinement'):
            self.snapshot('body_detail_insufficient')
            target.local_view_failures=getattr(target,'local_view_failures',0)+1
            raise RetryableFailure('Target visible, but broad body section needs a clearer local view',phase='observation')
        target.estimate=estimate
        estimate.rejected_grasps=getattr(target,'rejected_grasps',[])
        estimate.rejected_regions=getattr(target,'rejected_regions',[])
        estimate.rejected_points=getattr(target,'rejected_points',[])
        if getattr(target,'staging_leg',0)==1:self.configure_staging_landing(target,estimate)
        if getattr(target,'staging_leg',0)==2:
            tray=self.w.cfg['trays'][target.destination]
            if not getattr(target,'staging_goal_observed',False):
                self.skill.stage='staging_final_region';self.skill.callback(self.skill.stage)
                xy=np.asarray(tray['center_xy_m'],dtype=float).copy()
                try:self.view(xy,.44,recovery=True)
                except (RuntimeError,TimeoutError) as exc:
                    raise RetryableFailure('中转后最终卡纸空位未能重新确认：'+str(exc),phase='observation') from exc
                target.staging_goal_observed=True
            obstacles=[np.asarray(e.points)[:,:2] for c,e in self.observed_fruits.items() if c!=target.category]
            cloud=np.vstack(obstacles) if obstacles else np.empty((0,2))
            actual_positions,actual_shapes=self.physical_reservations(target)
            fresh=reallocate_final_landings(visual_footprint(estimate,target.category),tray,
                actual_positions,actual_shapes,
                target.key,cloud,target.staging_final['position'])
            if fresh is None:raise RetryableFailure('中转后当前轮廓下未找到最终区域安全空位，保留未完成',phase='planning')
            target.destination_position=fresh['points'][0];target.destination_options=fresh['points']
            target.placement_rotation=fresh['rotation']
            self.navigation_events.append(dict(target=target.key,event='staging_final_reallocated',
                destination=target.destination,points=[p.tolist() for p in fresh['points']],rotation_rad=fresh['rotation'],margin_m=fresh['margin']))
        options=[np.asarray(target.destination_position)]+list(getattr(target,'destination_options',[]))
        options=list({tuple(np.asarray(p,dtype=float)):np.asarray(p,dtype=float) for p in options}.values())
        reservations,shapes=self.physical_reservations(target)
        base_shape=visual_footprint(estimate,target.category)
        rotation=getattr(target,'placement_rotation',0.)
        planned_shape=base_shape.rotated(rotation)
        tray=getattr(target,'placement_region',None) or self.w.cfg['trays'][target.destination]
        # Preserve reserved shapes; do not reintroduce the old enclosing circle.
        cloud_parts=[np.asarray(e.points)[:,:2] for category,e in getattr(self,'observed_fruits',{}).items()
                     if category!=target.category and category not in reservations]
        cloud=np.vstack(cloud_parts) if cloud_parts else np.empty((0,2))
        extra=getattr(target,'placement_extra_validator',lambda shape,point:True)
        partition=getattr(target,'placement_partition',None)
        def near_partition_center(point):
            return not partition or 'center' not in partition or np.linalg.norm(np.asarray(point)-partition['center'])<=partition.get('max_center_offset_m',.025)+1e-9
        options=[p for p in options if validate_landing(planned_shape,p,tray,reservations,shapes,target.key,cloud=cloud) and extra(planned_shape,p)]
        if not options and getattr(target,'staging_leg',0)!=1:
            all_cloud=[np.asarray(e.points)[:,:2] for c,e in self.observed_fruits.items() if c!=target.category]
            occupied=np.vstack(all_cloud) if all_cloud else np.empty((0,2))
            fresh=reallocate_final_landings(base_shape,tray,reservations,shapes,target.key,occupied,target.destination_position,partition=getattr(target,'placement_partition',None))
            if fresh is not None:
                target.placement_rotation=rotation=fresh['rotation'];planned_shape=base_shape.rotated(rotation)
                options=fresh['points']
        options=[p for p in options if near_partition_center(p)]
        if not options:raise RetryableFailure('半区中心附近需重新寻找不重叠落点，继续更新观察与规划',phase='planning')
        anchor=np.asarray(partition['center']) if partition and 'center' in partition else np.asarray(target.destination_position)
        options.sort(key=lambda p:(np.linalg.norm(p-anchor),-planned_shape.coverage(tray,p),-min(.05,planned_shape.clearance(tray,p))))
        target.destination_position=np.asarray(options[0])
        estimate.destination_options=options[1:]
        margin=.002 if tray.get('placement_policy')=='center_max_coverage' else max(0.,min(.05,planned_shape.clearance(tray,options[0])))
        estimate.preferred_place_rotation=rotation
        estimate.placement_validator=lambda point,delta:validate_landing(
            base_shape.rotated(delta),point,tray,reservations,shapes,target.key,margin,cloud) and extra(base_shape.rotated(delta),point) and near_partition_center(point)
        from piper_sim.fruit_placement import projected_footprint
        estimate.placement_validator_3d=lambda point,grasp_rot,place_rot,grasp_center:validate_landing(
            projected_footprint(estimate,target.category,grasp_rot,place_rot,grasp_center),point,tray,reservations,shapes,target.key,margin,cloud) and near_partition_center(point)
        self.navigation_events.append(dict(target=target.key,grasp_center=estimate.center.tolist(),grasp_region=getattr(estimate,'grasp_quality',{})))
        target.position=np.median(estimate.points,axis=0).tolist()
        destination=np.array(target.destination_position,dtype=float)
        self.skill.cfg['defer_final_until_ready']=getattr(target,'direct_search_rounds',0)>=2
        self.skill.cfg['direct_attempts_used']=getattr(target,'direct_transfer_attempts',0)
        self.skill.cfg['search_cycle']=getattr(target,'direct_search_rounds',0)
        previous_report=getattr(self.skill,'last_planning_report',None)
        self.skill.cfg['planning_target']=target.category if not getattr(target,'staging_leg',0) else NAMES.get(target.category,target.category)+' · 中转第'+str(target.staging_leg)+'段'
        try:
            self.skill.stage='plan_grasp_and_place'
            self.skill.callback(self.skill.stage)
            if getattr(target,'staging_leg',0)==1:
                self.plan=self.plan_staging_leg(target,estimate)
            else:
                if getattr(target,'staging_leg',0)==2:self.progress(target,'planning_final_leg','中转已释放，正在按实际图像规划最终放置')
                self.plan=self.skill.plan(estimate,destination,
                    getattr(target,'placement_floor',self.skill.table+self.w.cfg['tray_geometry']['floor_thickness_m']))
            target.destination_position=np.asarray(self.plan.get('destination_position',destination))
            if hasattr(target,'placement_reservations'):target.placement_reservations[target.key]=target.destination_position.copy()
            target.placement_rotation=float(self.plan['place_yaw']-self.plan['yaw'])
            target.placement_footprint=base_shape.rotated(target.placement_rotation)
            if hasattr(target,'placement_footprints'):target.placement_footprints[target.key]=target.placement_footprint
            if getattr(target,'staging_leg',0)==1:
                shape=target.placement_footprint
                low=np.min(shape.offsets-shape.radii[:,None],axis=0)
                high=np.max(shape.offsets+shape.radii[:,None],axis=0)
                target.staging_verification_region=dict(center_xy_m=(target.destination_position+(low+high)/2).tolist(),half_size_xy_m=((high-low)/2+.025).tolist())
            self.navigation_events.append(dict(target=target.key,event='placement_footprint_selected',
                footprint=target.placement_footprint.record(),rotation_rad=target.placement_rotation,
                clearance_m=target.placement_footprint.clearance(tray,target.destination_position)))
            self.navigation_events.append(dict(target=target.key,event='grasp_route_selected',grasp_wrist_limit_deg=None,center=self.plan['center'].tolist(),destination=target.destination_position.tolist(),detour=self.plan.get('transfer_waypoints',[]),score=self.plan.get('selection_score'),grasp_tilt=self.plan.get('tilt'),place_tilt=self.plan.get('place_tilt'),place_yaw=self.plan.get('place_yaw')))
        except RuntimeError as exc:
            target.direct_search_rounds=getattr(target,'direct_search_rounds',0)+1
            self.snapshot('planning_unavailable')
            target.planning_retry_ready=True
            target.planning_observed_at=float(self.b.data.time)
            target.planning_observed_wall=getattr(estimate,'observed_wall',0)
            raise RetryableFailure(str(exc),phase='planning') from exc
        finally:
            target.direct_transfer_attempts=getattr(self.skill,'direct_transfer_attempts',0)
            if getattr(self.skill,'last_planning_report',None) is not previous_report:self.save_planning_report(target)

    def begin_staging(self,target):
        if getattr(target,'staging_leg',0):return
        final=dict(position=np.asarray(target.destination_position).copy(),options=list(getattr(target,'destination_options',[])),
            rotation=getattr(target,'placement_rotation',0.),footprint=getattr(target,'placement_footprint',visual_footprint(target.estimate,target.category)))
        target.staging_final=final
        target.staging_mode=getattr(target,'staging_mode','sweet')
        anchor=np.asarray(target.position)[:2]
        if .33<=anchor[0]<=.44 and abs(anchor[1])<=.10:target.staging_mode='progressive'
        target.staging_leg=1;target.staging_pending=False;target.staging_used=True;target.staging_final_landed=False
        self.surface_memory=SceneMemory(self.w.cfg['trays']);target.staging_surface_memory=self.surface_memory
        self.skill.stage='staging_observe';self.skill.callback(self.skill.stage)
        xy=[.39,0.] if target.staging_mode=='sweet' else np.asarray(final['position'])
        xy=np.asarray(xy,dtype=float)
        try:self.view(xy,.44,recovery=True)
        except (RuntimeError,TimeoutError) as exc:
            target.local_view_failures=getattr(target,'local_view_failures',0)+1
            raise RetryableFailure('本段落点视图待更新：'+str(exc),phase='staging') from exc
        self.configure_staging_landing(target,target.estimate)
        target.planning_retry_ready=False
        self.navigation_events.append(dict(target=target.key,event='staging_selected',mode=target.staging_mode,
            point=target.destination_position.tolist(),second_leg_not_required=True))
        self.skill.stage='staging_plan_first';self.skill.callback(self.skill.stage)

    def finish_staging(self,target):
        """Called only after release and visual boundary confirmation."""
        final=target.staging_final
        target.destination_position=final['position'];target.destination_options=final['options']
        target.placement_rotation=final['rotation'];target.staging_leg=2;target.staging_goal_observed=False
        target.staging_hops=getattr(target,'staging_hops',0)+1
        for field in ('placement_region','placement_floor','placement_extra_validator','staging_verification_region','staging_surface_memory'):
            if hasattr(target,field):delattr(target,field)
        if hasattr(target,'placement_reservations'):target.placement_reservations[target.key]=final['position'].copy()
        if hasattr(target,'placement_footprints'):target.placement_footprints[target.key]=final['footprint']
        target.planning_retry_ready=False;target.drop_estimate_ready=False
        target.rejected_points=[];target.rejected_regions=[];target.rejected_grasps=[]
        target.local_view_failures=0;target.boundary_check_status=None
        self.navigation_events.append(dict(target=target.key,event='staging_release_confirmed',
            observed_position=list(target.position),next='重新观察与抓取，再规划到最终区域'))
        self.progress(target,'awaiting_final_regrasp','中转已安全释放，最终放置尚未完成，继续重新抓取')
        self.skill.stage='staging_regrasp';self.skill.callback(self.skill.stage)

    def pick(self,target):
        self.current_issue=None
        previous_closures=self.skill.closed_attempts
        self.last_attempt_closed=False
        self.last_pick_error=''
        try:
            self.skill.pick(self.plan)
            checks=[]
            for _ in range(3):
                check=contact_quality(self.b)
                checks.append(check)
                if not check['accepted']:
                    self.navigation_events.append(dict(target=target.key,grip_check=checks))
                    self.remember_failed_region(target)
                    return False
                self.skill.hold(.10)
            self.navigation_events.append(dict(target=target.key,grip_check=checks))
            return True
        except ObjectDropped:
            self.remember_failed_region(target)
            raise
        except (GraspEmptyError,TimeoutError,RuntimeError) as exc:
            self.last_pick_error=str(exc)
            self.remember_failed_region(target)
            if self.skill.closed_attempts==previous_closures and self.plan is not None:
                target.rejected_grasps=getattr(target,'rejected_grasps',[])+[(self.plan['tilt'],self.plan['yaw'])]
                self.navigation_events.append(dict(target=target.key,event='rejected_approach_pose',
                    tilt=self.plan['tilt'],yaw=self.plan['yaw'],reason=str(exc)))
            if self.viewer is not None and not self.viewer.is_running():raise
            self.snapshot('grasp_failed_after_close' if self.skill.closed_attempts>previous_closures else 'approach_failed_before_close')
            return False
        finally:
            self.last_attempt_closed=self.skill.closed_attempts>previous_closures

    def remember_failed_region(self,target):
        if self.plan is None:return
        target.rejected_points=getattr(target,'rejected_points',[])+[self.plan.get('grasp_candidate_center',self.plan['center'].tolist())]
        target.rejected_regions=getattr(target,'rejected_regions',[])+[(self.plan['center'].tolist(),float(self.plan['yaw']))]
        self.navigation_events.append(dict(target=target.key,event='rejected_body_grasp',center=self.plan['center'].tolist(),yaw=float(self.plan['yaw'])))

    def place_and_verify(self,target):
        self.last_release_completed=False
        try:self.skill.place(self.plan)
        except ObjectDropped as exc:
            if exc.stage!='place':raise
            # Floor support can unload the pads before the planned open command.
            # Release locally before any observation/transport recovery.
            target.release_pending=True
            self.navigation_events.append(dict(target=target.key,event='placement_contact_lost',reason=str(exc)))
            raise RetryableFailure('Contact lost at placement: verify opening and vertical clearance locally',phase='release') from exc
        except ReleaseBlocked as exc:
            target.release_pending=True
            self.navigation_events.append(dict(target=target.key,event='release_blocked',reason=str(exc)))
            raise RetryableFailure(str(exc),phase='release') from exc
        except (RuntimeError,TimeoutError) as exc:
            self.last_release_completed=self.skill.release_verified
            raise RetryableFailure(str(exc)) from exc
        self.last_release_completed=self.skill.release_verified
        target.destination_position=np.asarray(self.plan.get('destination_position',target.destination_position))
        target.placement_rotation=float(self.plan['place_yaw']-self.plan['yaw'])
        if hasattr(target,'placement_reservations'):target.placement_reservations[target.key]=target.destination_position.copy()
        if hasattr(target,'placement_footprints'):
            target.placement_footprint=plan_footprint(target.estimate,target.category,self.plan)
            target.placement_footprints[target.key]=target.placement_footprint
        return self.after_release(target)

    def refresh_held_landing(self,target):
        from piper_sim.fresh_landing import visible_free_landings
        self.skill.stage='release_reobserve';self.skill.callback(self.skill.stage)
        failed=getattr(target,'failed_release_landings',[])
        failed.append(np.asarray(self.plan.get('destination_position',target.destination_position)).copy())
        target.failed_release_landings=failed[-12:]
        if self.b.data.site_xpos[self.b.model.site('tcp').id,2]<self.skill.table+.18:
            self.skill.lift_for_fresh_view(self.plan)
        tray=self.w.cfg['trays'][target.destination]
        anchor=np.asarray(tray['center_xy_m']).copy()
        offsets=[[-.025,0.],[0.,.035],[0.,-.035],[.02,0.]]
        anchor+=offsets[int(getattr(target,'release_retries',0))%len(offsets)]
        self.view_stage='release_reobserve'
        try:
            candidates=self.view(anchor,.40,recovery=True,holding=True)
        finally:self.view_stage='observe'
        # Use the new frame's floor evidence, not the old overview memory.
        fresh=SceneMemory(self.w.cfg['trays'])
        fresh.observe(self.last_observation,self.calibration,self.skill.table,[],{})
        cloud_parts=[np.asarray(e.points)[:,:2] for category,e in candidates if category!=target.category]
        obstacles=np.vstack(cloud_parts) if cloud_parts else np.empty((0,2))
        shape=getattr(target,'placement_footprint',visual_footprint(target.estimate,target.category))
        points=visible_free_landings(shape,tray,obstacles,fresh.seen_plane,
            getattr(target,'placement_partition',None),target.failed_release_landings)
        self.navigation_events.append(dict(target=target.key,event='fresh_release_space',
            observed_cells=len(fresh.seen_plane),new_landings=[p.tolist() for p in points],failed=[p.tolist() for p in target.failed_release_landings]))
        if not points:raise ReleaseBlocked('新图像尚未确认安全空位，继续换观察角度；不在旧落点重算')
        estimate=self.plan['final_target']
        estimate.destination_options=points[1:]
        estimate.placement_partition=None
        estimate.placement_validator=lambda point,delta:validate_landing(
            visual_footprint(estimate,target.category).rotated(delta),point,tray,{}, {},target.key,cloud=obstacles)
        from piper_sim.fruit_placement import projected_footprint
        estimate.placement_validator_3d=lambda point,grasp_rot,place_rot,grasp_center:validate_landing(
            projected_footprint(estimate,target.category,grasp_rot,place_rot,grasp_center),point,tray,{}, {},target.key,cloud=obstacles)
        self.plan.update(final_destination=points[0],destination_position=points[0].tolist(),fresh_observed=True)
        self.plan.pop('original_release_z',None)
        target.destination_position=points[0]
        self.skill.final_plan_from_ready(self.plan)
        self.skill.place(self.plan)
        target.release_pending=False
        target.destination_position=np.asarray(self.plan['destination_position'])
        target.placement_rotation=float(self.plan['place_yaw']-self.plan['yaw'])
        target.placement_footprint=plan_footprint(target.estimate,target.category,self.plan)
        if hasattr(target,'placement_reservations'):target.placement_reservations[target.key]=target.destination_position.copy()
        if hasattr(target,'placement_footprints'):target.placement_footprints[target.key]=target.placement_footprint
        return self.after_release(target)

    def resume_release(self,target):
        if self.plan.get('deferred_final') and not self.skill.release_verified:
            return self.refresh_held_landing(target)
        if getattr(self.skill,'release_motion_locked',False):
            self.skill.release_and_retreat(self.plan)
            target.release_pending=False
            return self.after_release(target)
        if not self.skill.release_verified and getattr(target,'release_retries',0)>=2:
            return self.refresh_held_landing(target)
        if not self.skill.release_verified:
            # A blocked descent must be retried before any opening command.
            try:self.skill.correct_release_height(self.plan)
            except (RuntimeError,TimeoutError):
                return self.refresh_held_landing(target)
            self.skill.stage='place';self.skill.ik=IK(self.b,tilt=self.plan.get('place_tilt',self.plan['tilt']))
            self.skill.cartesian(self.plan['place'],self.plan.get('place_yaw',self.plan['yaw']),0)
        if not (self.skill.release_verified and self.skill.retreat_verified):self.skill.release_and_retreat(self.plan)
        target.release_pending=False
        self.last_release_completed=self.skill.release_verified
        return self.after_release(target)

    def after_release(self,target):
        if not self.skill.release_verified or not self.skill.retreat_verified:
            raise RetryableFailure('张爪或退出尚未确认，继续安全释放',phase='release')
        target.last_released_position=np.asarray(target.destination_position).copy()
        target.last_released_footprint=getattr(target,'placement_footprint',None)
        target.audit_pending=False;target.boundary_check_status='disabled_by_user'
        self.navigation_events.append(dict(target=target.key,event='release_completed_without_boundary_audit',
            destination=target.destination,planned_position=target.last_released_position.tolist()))
        self.progress(target,'complete','实际张爪并退出完成；按当前规则不进行边界巡检')
        return True

    def finalize_batch(self,targets,result):
        result['boundary_verification']='disabled_by_user'
        return result

    def scan_batch_boundaries(self,targets):
        self.current_issue=None
        self.skill.stage='batch_boundary_audit';self.skill.callback(self.skill.stage)
        self.navigation_events.append(dict(event='batch_audit_started',targets=[t.key for t in targets],time_s=float(self.b.data.time)))
        observations=[];remaining=set(t.destination for t in targets)
        sid=self.b.model.site('tcp').id;current=self.b.data.site_xpos[sid,:2].copy()
        visited=[]
        while remaining:
            color=min(remaining,key=lambda c:np.linalg.norm(np.asarray(self.w.cfg['trays'][c]['center_xy_m'])-current))
            remaining.remove(color);tray=self.w.cfg['trays'][color]
            xy=np.asarray(tray['center_xy_m'],dtype=float).copy()
            self.current_issue=dict(target={'red':'红色','green':'绿色','yellow':'黄色'}.get(color,color)+'卡纸',phase='末尾巡检',reason='识别物品并检查区域边界')
            self.view_stage='batch_boundary_audit'
            try:observations.extend(self.view(xy,.44,recovery=True));current=xy;visited.append(color)
            except (RuntimeError,TimeoutError) as exc:
                self.navigation_events.append(dict(event='batch_audit_view_unavailable',destination=color,error=str(exc)))
            finally:self.view_stage='observe'
        states={t.key:identify_audit_target(t,observations,self.w.cfg['trays'][t.destination]) for t in targets}
        # At most one extra local view per unclear region, not one per object.
        unclear={t.destination for t in targets if states[t.key]['status']=='unknown'}
        for color in unclear:
            tray=self.w.cfg['trays'][color];xy=np.asarray(tray['center_xy_m'],dtype=float).copy()
            xy[0]-=.035
            self.view_stage='batch_boundary_audit'
            try:observations.extend(self.view(xy,.46,recovery=True))
            except (RuntimeError,TimeoutError):pass
            finally:self.view_stage='observe'
        if unclear:states={t.key:identify_audit_target(t,observations,self.w.cfg['trays'][t.destination]) for t in targets}
        for t in targets:
            state=states[t.key]
            if 'estimate' in state:
                self.observed_fruits[t.category]=state['estimate']
                if hasattr(t,'placement_reservations'):t.placement_reservations[t.key]=np.asarray(state['center'])[:2]
                if hasattr(t,'placement_footprints'):t.placement_footprints[t.key]=visual_footprint(state['estimate'],t.category)
            self.navigation_events.append(dict(target=t.key,event='batch_boundary_result',destination=t.destination,time_s=float(self.b.data.time),**public_status(state)))
        self.current_issue=None
        return states

    def correct_boundary(self,target,state):
        self.current_issue=dict(target=NAMES.get(target.category,target.category),phase='末尾纠偏',reason='已确认越界，重新放入指定区域')
        self.skill.stage='batch_boundary_correct';self.skill.callback(self.skill.stage)
        self.navigation_events.append(dict(target=target.key,category=target.category,event='batch_boundary_correction_started',destination=target.destination,observed_center=state['center'],time_s=float(self.b.data.time)))
        target.position=state['center'];target.estimate=state['estimate']
        target.rejected_points=[];target.rejected_regions=[];target.rejected_grasps=[]
        target.local_view_failures=0;target.planning_retry_ready=False
        target.batch_previous_preparations=getattr(target,'preparation_total',0)
        target.preparation_total=0;target.deferrals=0
        target.drop_estimate_ready=False;target.release_pending=False;target.drop_search_pending=False
        tray=self.w.cfg['trays'][target.destination]
        cloud_parts=[np.asarray(e.points)[:,:2] for c,e in self.observed_fruits.items() if c!=target.category]
        cloud=np.vstack(cloud_parts) if cloud_parts else np.empty((0,2))
        fresh=reallocate_final_landings(visual_footprint(target.estimate,target.category),tray,
            getattr(target,'placement_reservations',{}),getattr(target,'placement_footprints',{}),
            target.key,cloud,target.destination_position)
        if fresh is None:return dict(status='incomplete',attempts=0,safe_to_scan=True,motion_attempted=False,error='越界物品已识别，但未找到安全调整落点')
        target.destination_position=fresh['points'][0];target.destination_options=fresh['points']
        target.placement_rotation=fresh['rotation'];target.batch_correction=True
        owner=self
        class CorrectionSkills:
            finalize_batch=None
            def overview(self):return [target]
            def __getattr__(self,name):return getattr(owner,name)
        try:
            result=Harvest(CorrectionSkills(),max_attempts=6,max_preparations_per_target=3,max_deferrals=2).run()
            safe=not contact_quality(self.b)['loaded'] and not target.release_pending
            return dict(status=result['status'],attempts=result.get('attempts',0),safe_to_scan=bool(safe),motion_attempted=True,events=result.get('events',[]),blockers=result.get('blockers',{}))
        finally:target.batch_correction=False

    def verify_placement(self,target):
        self.navigation_events.append(dict(target=target.key,event='release_verified',
            opened=self.skill.release_verified,vertical_clear=self.skill.retreat_verified,
            time_s=float(self.b.data.time),feedback=release_feedback(self.b)))
        self.skill.stage='check_staging_release' if getattr(target,'staging_leg',0)==1 else 'check_placement_boundary'
        if hasattr(self.skill,'callback'):self.skill.callback(self.skill.stage)
        destination=np.asarray(target.destination_position)
        target.boundary_check_status='unknown'
        for view_index in range(2):
            if view_index:
                xy=destination.copy()
                depth=min(.44,max(.38,.32+getattr(getattr(target,'estimate',None),'height',.02)))
                self.view_stage='check_staging_release' if getattr(target,'staging_leg',0)==1 else 'check_placement_boundary'
                try:self.view(xy,depth,recovery=True)
                except (RuntimeError,TimeoutError) as exc:
                    self.navigation_events.append(dict(target=target.key,event='boundary_view_unavailable',error=str(exc)))
                    break
                finally:self.view_stage='observe'
            visible=self.boundary_candidates()
            result=inspect_boundary(visible,destination,getattr(target,'staging_verification_region',None) or getattr(target,'placement_region',None) or self.w.cfg['trays'][target.destination],
                getattr(getattr(target,'estimate',None),'points',None),getattr(target,'placement_reservations',{}),target.key)
            self.navigation_events.append(dict(target=target.key,event='placement_boundary',view=view_index,**result))
            if result['status']=='inside':
                target.boundary_check_status='inside'
                if getattr(target,'staging_leg',0)==1:
                    final_check=inspect_boundary(visible,destination,self.w.cfg['trays'][target.destination],getattr(target.estimate,'points',None))
                    target.staging_final_landed=final_check['status']=='inside'
                    if target.staging_final_landed:target.audit_pending=True
                target.position=result['center']
                target.last_released_position=np.asarray(result['center'])[:2]
                target.last_released_footprint=getattr(target,'placement_footprint',None)
                if getattr(target,'staging_leg',0)!=1:self.bin_placements[target.destination]+=1
                return True
            if result['status']=='outside':
                target.boundary_check_status='outside';target.position=result['center']
                target.rejected_points=[];target.rejected_regions=[];target.local_view_failures=0
                return False
        target.last_preparation_error='已释放物品，局部边界未看清；保留待检查，不启动全桌寻找'
        return False

    def boundary_candidates(self):
        obs=self.b.observe();self.last_observation=obs
        surface=self.w.cfg.get('support_surface',self.w.cfg['table'])
        center=np.array(surface['center_xy_m']);half=np.array(surface['half_size_xy_m'])
        return candidates_3d(obs,self.calibration,self.skill.table,
            workspace=(center[0]-half[0],center[0]+half[0],center[1]-half[1],center[1]+half[1]),
            trays=self.w.cfg['trays'],tray_geometry=self.w.cfg['tray_geometry'])

    def recover(self,target):
        self.skill.grip_monitor.disarm()
        sid=self.b.model.site('tcp').id
        if self.plan is not None and contact_quality(self.b)['loaded'] and not self.skill.release_motion_locked:
            current=self.b.data.site_xpos[sid].copy()
            if np.linalg.norm(current[:2]-self.plan['center'][:2])>.025:
                raise RetryableFailure('物品仍夹持且已离开原抓取点，保持夹持，暂不允许横向回拖',phase='retained_recovery')
            self.skill.stage='recovery_lower'
            supported=False
            for step in range(35):
                if retained_support(self.b):
                    self.skill.hold(.15)
                    if retained_support(self.b):supported=True;break
                if not contact_quality(self.b)['loaded']:
                    target.drop_search_pending=True
                    self.navigation_events.append(dict(target=target.key,event='recovery_contact_lost'))
                    self.current_issue=None
                    return
                current=self.b.data.site_xpos[sid].copy()
                floor_limit=self.skill.table+.025
                if current[2]<=floor_limit:break
                distance=-min(.005,current[2]-floor_limit)
                try:self.skill.vertical_release_move(distance,0)
                except (RuntimeError,TimeoutError) as exc:
                    self.b.previous_target[:6]=self.b.measured()[:6]
                    self.skill.hold(.15)
                    self.navigation_events.append(dict(target=target.key,event='recovery_lower_blocked',step=step,error=str(exc),supported=retained_support(self.b)))
                    if retained_support(self.b):supported=True;break
                    raise RetryableFailure('下放暂时受阻且尚未确认桌面支撑，保持夹持并重试',phase='retained_recovery') from exc
            if not supported:
                raise RetryableFailure('未确认物品得到桌面支撑，保留夹持并重试下放',phase='retained_recovery')
            self.navigation_events.append(dict(target=target.key,event='retained_object_supported'))
        position=self.b.data.site_xpos[sid].copy()
        feedback=release_feedback(self.b)
        if position[2]>=.20 and feedback['opened'] and not feedback['object_contact'] and not feedback['tray_contact']:
            self.skill.release_motion_locked=False
            self.current_issue=None
            return
        retreat=dict(self.plan or {});retreat.setdefault('yaw',0.)
        retreat['above']=position.copy();retreat['above'][2]=max(.20,position[2]+.06)
        self.skill.release_and_retreat(retreat)
        self.navigation_events.append(dict(target=target.key,event='retained_recovery_released'))
        self.current_issue=None

    def report_recovery_failure(self,target,error):
        names={'pear':'梨','banana':'香蕉','cucumber':'黄瓜','persimmon':'柿子','potato':'土豆','bell_pepper':'彩椒'}
        self.current_issue=dict(target=names.get(target.category,target.category),phase='夹持恢复',reason=error)
        self.navigation_events.append(dict(target=target.key,event='retained_recovery_blocked',error=error))

    def report_preparation_failure(self,target,error,phase):
        names={'pear':'梨','banana':'香蕉','cucumber':'黄瓜','persimmon':'柿子','potato':'土豆','bell_pepper':'彩椒'}
        self.current_issue=dict(target=names.get(target.category,target.category),phase=phase,reason=error)
        self.navigation_events.append(dict(target=target.key,event='preparation_blocked',phase=phase,error=error))
        self.snapshot('定位受阻：'+self.current_issue['target']+'；'+error)

    def wait_before_retry(self,target,reason):
        # Poll without leaving the release-only safety state or resuming a stale goal.
        if self.skill.release_motion_locked:
            self.skill.stage='release_open'
            self.b.previous_target[:6]=self.b.measured()[:6]
        else:self.skill.stage='retry_pending_'+target.category
        self.skill.hold(.20)

    def raise_for_drop_search(self):
        """Plan a short vertical clearance move using only measured robot FK."""
        data=mujoco.MjData(self.b.model)
        q=self.b.measured()[:6].copy();data.qpos[:6]=q;mujoco.mj_kinematics(self.b.model,data)
        sid=self.b.model.site('tcp').id
        origin=data.site_xpos[sid].copy();rotation=data.site_xmat[sid].reshape(3,3).copy()
        if origin[2]>=self.skill.table+.20:
            self.skill.stage='drop_raise';self.skill.hold(.10)
            return origin
        lower,upper=self.b.model.jnt_range[self.b.joints].T
        for height in [.08,.06,.04,.02,.01]:
            points=[q.copy()]
            for dz in np.linspace(0,height,round(height/.01)+1)[1:]:
                destination=origin+np.array([0,0,dz]);seed=points[-1]
                def residual(candidate):
                    data.qpos[:6]=candidate;mujoco.mj_kinematics(self.b.model,data)
                    angular=Rotation.from_matrix(rotation@data.site_xmat[sid].reshape(3,3).T).as_rotvec()
                    return np.r_[data.site_xpos[sid]-destination,.12*angular,.001*(candidate-seed)]
                solution=least_squares(residual,np.clip(seed,lower+1e-5,upper-1e-5),
                    bounds=(lower+1e-6,upper-1e-6),max_nfev=100)
                error=residual(solution.x)
                if np.linalg.norm(error[:3])>.002 or np.linalg.norm(error[3:6])>.005 or np.max(abs(solution.x-seed))>.5:break
                points.append(solution.x)
            else:
                from piper_sim.trajectory import trajectory
                self.skill.stage='drop_raise'
                acceleration=self.skill.motion_profile()
                width=self.b.previous_target[6]
                for goal in trajectory(points,width,width,self.b.interface,acceleration):self.skill.tick(goal)
                self.skill.hold(.10)
                return origin
        raise RuntimeError('No continuous vertical clearance path for drop search')

    def recover_dropped(self,target):
        self.skill.grip_monitor.disarm()
        self.snapshot('drop_detected')
        # A slipping object may still touch one finger. Open at the measured
        # arm pose first, rather than lifting a half-trapped object with a stale goal.
        self.skill.open_gripper_in_place()
        origin=self.raise_for_drop_search()
        feedback=release_feedback(self.b)
        if feedback['object_contact'] or feedback['tray_contact']:
            raise ReleaseBlocked('Drop recovery has not cleared finger contacts; lateral search remains locked')
        self.skill.release_motion_locked=False
        views=[origin[:2],target.destination_position,np.asarray(target.position)[:2]]
        views += [np.array(xy) for xy in [[.37,-.22],[.49,-.22],[.30,-.12],[.44,-.08],[.56,.02],[.34,.12],[.48,.18]]]
        visited=[]
        for xy in views:
            if any(np.linalg.norm(xy-old)<.025 for old in visited):continue
            visited.append(np.array(xy))
            try:candidates=self.view(xy,.46,recovery=True)
            except (RuntimeError,TimeoutError) as exc:
                self.navigation_events.append(dict(target=target.key,event='drop_view_unavailable',reason=str(exc)))
                continue
            matches=[e for category,e in candidates if category==target.category]
            if not matches:
                self.navigation_events.append(dict(target=target.key,event='search_visible_candidates',items=[dict(category=c,center=np.median(e.points,axis=0).tolist(),extent=np.ptp(e.points,axis=0).tolist()) for c,e in candidates]))
                continue
            estimate=min(matches,key=lambda e:np.linalg.norm(np.median(e.points,axis=0)[:2]-xy))
            tracker=LandingTracker()
            region=landing_region(estimate.points,self.w.cfg['trays'],self.skill.table,self.w.cfg['tray_geometry'])
            self.skill.stage='drop_verify_landing'
            stable=tracker.update(float(self.b.data.time),estimate.points,region)
            visible=True
            for _ in range(16):
                self.skill.hold(.15)
                matches=[e for category,e in self.capture_candidates(recovery=True) if category==target.category]
                if not matches:
                    visible=False;break
                estimate=min(matches,key=lambda e:np.linalg.norm(np.median(e.points,axis=0)-np.median(estimate.points,axis=0)))
                region=landing_region(estimate.points,self.w.cfg['trays'],self.skill.table,self.w.cfg['tray_geometry'])
                stable=tracker.update(float(self.b.data.time),estimate.points,region)
                if stable:break
            if not visible:continue  # Do not grasp a stale location after another bounce/occlusion.
            target.position=np.median(estimate.points,axis=0).tolist()
            target.estimate=estimate
            target.rejected_points=[]
            target.rejected_regions=[]
            target.rejected_grasps=[]  # Its landing orientation/position has changed.
            self.navigation_events.append(dict(target=target.key,event='drop_relocated',
                observed_position=target.position,region=region,stable=bool(stable)))
            if region.startswith('edge:'):
                self.navigation_events.append(dict(target=target.key,event='drop_boundary_uncertain',region=region))
                continue  # Neither assume completion nor disturb a possibly correct landing.
            if region==target.destination:
                if not stable:return 'unseen'  # Wait/reobserve, never rearrange a correct landing.
                if getattr(target,'staging_leg',0)==1:target.staging_final_landed=True
                self.bin_placements[target.destination]+=1
                return 'placed'
            target.recovered_from_drop=True
            target.drop_estimate_ready=True
            return 'retry'
        self.snapshot('drop_target_not_found')
        return 'unseen'


def run_episode(args,seed):
    args.seed=seed
    episode_started=time.monotonic()
    controls=SessionControls(args.mode)
    output=ROOT/'artifacts/harvest'/str(args.seed)
    output.mkdir(parents=True,exist_ok=True)
    world=None
    viewer=None
    report={'status':'failed','seed':args.seed,'language_model':'not_requested'}
    try:
        task=parse_instruction(args.instruction) if args.instruction else None
        if task and task['action']=='clarify':raise ValueError('Instruction needs clarification')
        if task and (task['category'].startswith('cube_') or task['category']=='all_blocks'):
            args.mode='blocks'
            controls.mode='blocks'
            if task['category']=='all_blocks':task['category']='all'
        report['parsed_instruction']=task
        report['language_model']='local_qwen' if task else 'not_requested'
        categories=['cube_red','cube_green','cube_blue'] if args.mode=='blocks' else args.fruits.split(',')
        cfg=configure(args.seed,categories,output)
        world=ImplicitWorld('l2',cfg,output,render=True)
        world.cfg['tasks']['l2']['time_limit_s']=args.time_limit or (180 if args.mode=='blocks' else 300)
        world.reset(args.seed)
        if args.viewer:
            import mujoco.viewer
            viewer=mujoco.viewer.launch_passive(world.backend.model,world.backend.data,key_callback=controls.key,show_left_ui=True,show_right_ui=True)
            viewer.cam.lookat[:]=[.36,-.035,.1]
            viewer.cam.distance=1.05
            viewer.cam.azimuth=135
            viewer.cam.elevation=-40
            viewer.sync()
            print('Live MuJoCo window opened. Close the window to stop.',flush=True)
        skills=FruitSkills(world,output,task)
        skills.viewer=viewer
        skills.controls=controls
        skills.wall_started=episode_started
        harvest=Harvest(skills,max_attempts=None,max_deferrals=None)
        if args.observe_only:
            tracks=skills.overview()
            report.update(status='observed',tracks=[dict(key=t.key,category=t.category,position=t.position) for t in tracks])
        else:report.update(harvest.run())
        report['evaluation']=world.evaluation()
        skills.snapshot('finished')
        if not args.observe_only:
            required=[o for o in world.objects if not task or task['category'] in ('all',o['category'])]
            validated=[o['id'] for o in required if (task['destination'] if task and task['destination']!='mapping'
                else skills.mapping[o['category']]) in world.events[o['id']]['placed']]
            report['validated_objects']=validated
            report['expected_object_count']=len(required)
            report['status']='complete' if len(validated)==len(required) else 'incomplete'
        report['perception']='synthetic_color_baseline_not_real_fruit_classifier'
    except RoundTimedOut:
        report['status']='timed_out'
        report['stop_reason']='competition_time_limit'
    except RestartRound:
        report['status']='restarted_by_user'
    except Exception as exc:
        report['stop_reason']='execution_error'
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if 'harvest' in locals():report.setdefault('events',harvest.events)
        if world and world.backend:
            report.setdefault('evaluation',world.evaluation())
        if 'skills' in locals():
            if not args.observe_only:
                required=[o for o in world.objects if not task or task['category'] in ('all',o['category'])]
                destination=lambda o:task['destination'] if task and task['destination']!='mapping' else skills.mapping[o['category']]
                validated=[o['id'] for o in required if destination(o) in world.events[o['id']]['placed']]
                report['validated_objects']=validated
                report['expected_object_count']=len(required)
                report['pending_categories']=[o['category'] for o in required if o['id'] not in validated]
                if 'harvest' in locals() and hasattr(harvest,'targets'):
                    report.setdefault('completed',getattr(harvest,'completed',[]))
                    report.setdefault('unresolved',[t.key for t in harvest.targets if t.key not in report['completed']])
                skills.round_summary=(f"{report['status']} | placed {len(validated)}/{len(required)} | "
                    f"pending: {', '.join(report['pending_categories']) or 'none'}")
            report['phase_times_s']=skills.phase_times
            report['overview_calls']=skills.overview_calls
            report['local_observations']=skills.navigation_events
            skills.finished_wall=time.monotonic()
            report['wall_time_s']=skills.finished_wall-skills.wall_started
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
        if viewer is not None:
            print('Run ended; the window stays open for inspection. Close it to exit.',flush=True)
            viewer.set_texts((None,None,
                f'Round: {report["status"]} | placed {len(report.get("validated_objects",[]))}/{report.get("expected_object_count","?")}',
                'F5 replay | F6 random | F7 pause | 1 blocks | 2 fruits'))
            while viewer.is_running() and controls.request is None:
                if 'skills' in locals():skills.update_display('Round: '+report['status'])
                viewer.sync()
                time.sleep(.05)
            viewer.close()
        if world:world.close()
    return controls.request

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--mode',choices=['fruits','blocks'],default='fruits')
    p.add_argument('--fruits',default='apple,pear,banana,lemon,orange,strawberry')
    p.add_argument('--instruction',default='')
    p.add_argument('--observe-only',action='store_true')
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--time-limit',type=float,default=None)
    args=p.parse_args()
    if args.time_limit is not None and args.time_limit<=0:p.error("Time limit must be positive")
    seed=args.seed if args.seed is not None else secrets.randbits(32)
    while True:
        if args.mode=='blocks':
            import subprocess
            subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'run-l1.ps1')],check=False)
            break
        print(f'Round seed={seed}, task={args.mode}',flush=True)
        request=run_episode(args,seed)
        if request is None:break
        action,args.mode=request
        if action=='switch':args.instruction=''
        if action!='replay':seed=secrets.randbits(32)

if __name__=='__main__':main()
