"""Sensor-grounded single-object pick/place demo. No simulator object poses in planning."""

from dataclasses import dataclass

import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from piper_sim.pose_preference import natural_tilts,fair_tilts,placement_orientations
from piper_sim.calibration import camera_to_base
from piper_sim.control import target_to_action
from piper_sim.traditional import IK
from piper_sim.planning_trace import PlanningTrace,explain
from piper_sim.route_search import transfer_route,ordered_candidates,joint_space_route,ready_position_route
from piper_sim.release_safety import ReleaseBlocked,release_feedback,opening_clearance


@dataclass
class TargetEstimate:
    center: np.ndarray
    height: float
    yaw: float
    mask: np.ndarray
    points: np.ndarray
    pixel: list
    snap_distance_px: float


def candidates_from_rgbd(obs, calibration, table_top):
    """Visible connected geometry above a configured table; no hidden-instance labels."""
    height, width = obs.rgb.shape[:2]
    v, u = np.indices((height, width))
    z = obs.depth_m
    k = obs.intrinsics
    valid = np.isfinite(z) & (z > 0)
    points = np.stack([(u - k["cx"]) * z / k["fx"], (v - k["cy"]) * z / k["fy"], z], -1)
    base = np.zeros_like(points)
    base[valid] = camera_to_base(points[valid], obs.T_base_end, calibration)
    foreground = valid & (base[..., 2] > table_top + 0.007) & (base[..., 2] < table_top + 0.14)
    foreground &= (
        (base[..., 0] > 0.20)
        & (base[..., 0] < 0.63)
        & (base[..., 1] > -0.14)
        & (base[..., 1] < 0.25)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8))
    result = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 60:
            continue
        mask = labels == label
        cloud = base[mask]
        hi = np.percentile(cloud, 98, axis=0)
        top = float(hi[2])
        top_points = cloud[cloud[:, 2] >= top - 0.004]
        xy = np.median(top_points[:, :2], axis=0)
        if len(top_points) < 15:
            continue
        rect = cv2.minAreaRect(top_points[:, :2].astype(np.float32))
        corners = cv2.boxPoints(rect)
        edge = corners[1] - corners[0]
        yaw = (np.arctan2(edge[1], edge[0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
        ys, xs = np.where(mask)
        result.append(
            TargetEstimate(
                np.r_[xy, (top + table_top) / 2],
                top - table_top,
                float(yaw),
                mask,
                cloud,
                [float(np.mean(xs)), float(np.mean(ys))],
                0,
            )
        )
    return result


def select_pixel(candidates, pixel, max_distance_px=18):
    if not candidates:
        raise ValueError("当前腕部深度图中没有可抓取候选；请换布局或观察姿态。")
    scores = []
    for candidate in candidates:
        y, x = np.where(candidate.mask)
        scores.append(float(np.hypot(x - pixel[0], y - pixel[1]).min()))
    index = int(np.argmin(scores))
    if scores[index] > max_distance_px:
        raise ValueError("模型选点离任何有效深度候选过远，停止执行。")
    target = candidates[index]
    target.snap_distance_px = scores[index]
    return target


def select_rule(candidates, rgb, instruction):
    """Explicit optional CPU baseline, not a substitute presented as Qwen."""
    import re

    colors = {"红": ("red", 0), "绿": ("green", 1), "蓝": ("blue", 2)}
    match = re.fullmatch(
        r"\s*把([红绿蓝])色?方块放(?:到|进)?([红绿蓝])色?(?:托盘|篮子)(?:里|里面)?[。.!！]?\s*",
        instruction,
    )
    if not match:
        raise ValueError(
            "CPU规则模式仅支持“把红色方块放到蓝色托盘”这一类明确颜色指令；自然语言请选千问。"
        )
    source, destination = match.groups()
    channel = colors[source][1]
    options = []
    for candidate in candidates:
        color = np.median(rgb[candidate.mask], axis=0).astype(float)
        other = np.delete(color, channel).max()
        if color[channel] > other * 1.25 and color[channel] > 40:
            options.append(candidate)
    if len(options) != 1:
        raise ValueError("当前画面中没有唯一的指定颜色方块；请调整观察姿态。")
    return options[0], colors[destination][0]


class GraspEmptyError(RuntimeError):
    """Measured gripper width indicates no object is retained."""


class PickPlaceSkill:
    def __init__(self, backend, table_top, callback, config, stepper=None):
        self.b, self.table, self.callback, self.cfg = backend, table_top, callback, config
        self.stepper = stepper or backend.step
        self.stage = "planning"
        self.started = backend.data.time
        self.closed_attempts=0
        self.release_verified=False
        self.retreat_verified=False
        self.release_motion_locked=False
        from piper_sim.drop_recovery import GripMonitor
        self.grip_monitor=GripMonitor(config.get('drop_debounce_s',.10))

    def recenter(self, observation, points):
        from piper_sim.active_view import recenter_camera

        return recenter_camera(
            self.b, observation, points, self.table, self.cfg.get("active_view", {})
        )

    def tick(self, goal):
        if self.release_motion_locked and self.stage not in ('release_open','release_unjam_vertical','release_lift_vertical','drop_raise'):
            raise ReleaseBlocked('Release interlock: only opening or vertical disengagement is permitted')
        if self.b.data.time - self.started > self.cfg.get("max_task_s", 120):
            raise TimeoutError("抓放技能超过时间预算")
        action = target_to_action(goal, self.b.previous_target, self.b.interface)
        if self.cfg.get('ready_transport'):
            self.validate_transfer([self.b.measured()[:6],np.asarray(goal[:6])])
        self.stepper(action)
        self.callback(self.stage)
        if self.grip_monitor.armed and self.cfg.get('grasp_retained'):
            self.grip_monitor.update(float(self.b.data.time),self.cfg['grasp_retained'](),self.stage)

    def move(self, q, width):
        from piper_sim.trajectory import trajectory
        start = self.b.data.time
        initial=self.b.previous_target.copy()
        acceleration=self.motion_profile()
        for goal in trajectory([initial[:6],q],initial[6],width,self.b.interface,acceleration):
            self.tick(goal)
        start=self.b.data.time
        while True:
            self.tick(np.r_[q, width])
            grip_ok = (
                self.b.previous_target[6] < 0.004
                if width == 0
                else abs(self.b.measured()[6] - width) < 0.004
            )
            if np.max(np.abs(self.b.measured()[:6] - q)) < 0.010 and grip_ok:
                return
            if self.b.data.time - start > self.cfg.get('settle_timeout_s',18):
                raise TimeoutError(f"{self.stage}: 关节未在限时内到达目标")

    def hold(self, seconds):
        for _ in range(int(np.ceil(seconds * self.b.interface["control"]["policy_hz"]))):
            self.tick(self.b.previous_target.copy())

    def motion_profile(self):
        profiles=self.cfg.get('motion_profiles')
        if not profiles:return .45
        name=('precision' if self.stage in ('descend','close','place','test_lift','recovery_lower') else
              'loaded' if self.stage in ('lift','transport','transport_detour','transport_ready','plan_from_ready','place_approach','drop_raise') or getattr(getattr(self,'grip_monitor',None),'armed',False) else 'empty')
        profile=profiles[name]
        self.b.interface['control']['max_joint_speed_rad_s']=[profile['speed']]*6
        self.b.interface['control']['max_gripper_speed_m_s']=profile['gripper_speed']
        return profile['acceleration']

    def cartesian_path(self, start_q, destination, yaw, current=None):
        """Validate every intermediate IK pose before commanding any motion."""
        if current is None:
            import mujoco
            data=mujoco.MjData(self.b.model)
            data.qpos[:6]=start_q
            mujoco.mj_kinematics(self.b.model,data)
            current=data.site_xpos[self.b.model.site('tcp').id].copy()
        steps = max(2, int(np.ceil(np.linalg.norm(destination - current) / 0.015)))
        points=[np.array(start_q,copy=True)]
        for point_index,position in enumerate(np.linspace(current, destination, steps + 1)[1:],1):
            if getattr(self,'planning_active',False) and time.monotonic()>=self.planning_deadline:raise RuntimeError('候选搜索预算耗尽，转下一个方案')
            if point_index==1 or point_index==steps or point_index%4==0:self.planning_event('waypoint',index=point_index,total=steps,position=position.tolist())
            q=self.ik.solve(position,yaw,points[-1])
            if np.max(abs(q-points[-1]))>.5:
                error=RuntimeError('IK branch discontinuity; refusing an abrupt joint move')
                error.joint_index=int(np.argmax(abs(q-points[-1])))+1;error.joint_delta_rad=float(np.max(abs(q-points[-1])))
                raise error
            points.append(q)
        if self.cfg.get('ready_transport'):self.validate_transfer(points)
        return points

    def cartesian(self, destination, yaw, width):
        from piper_sim.trajectory import trajectory
        current=self.b.data.site_xpos[self.b.model.site('tcp').id].copy()
        points=self.cartesian_path(self.b.previous_target[:6],destination,yaw,current)
        acceleration=self.motion_profile()
        holding=width==0 and self.grip_monitor.armed
        # A retained object blocks physical closure. Ramp the servo command at
        # its normal rate while moving the arm, instead of timing an impossible
        # finger stroke as part of this Cartesian motion.
        end_width=self.b.previous_target[6] if holding else width
        commands=trajectory(points,self.b.previous_target[6],end_width,self.b.interface,acceleration)
        if self.cfg.get('ready_transport'):self.validate_transfer(commands[:,:6])
        for goal in commands:
            if holding:goal[6]=0
            self.tick(goal)
        # One terminal settling check, never a stop at each Cartesian waypoint.
        start=self.b.data.time
        while np.max(abs(self.b.measured()[:6]-points[-1]))>.010:
            self.tick(np.r_[points[-1],width])
            if self.b.data.time-start>4:raise TimeoutError('Cartesian tracking did not settle')

    def planning_event(self,event,**data):
        deadline_check=self.cfg.get('deadline_check')
        if deadline_check:deadline_check()
        trace=getattr(self,'planning_trace',None)
        if getattr(self,'planning_active',False) and trace:trace.emit(event,**data)

    def planning_phase(self,name):
        self.planning_event('phase',phase=name)

    def plan(self, target, destination, tray_floor):
        self.planning_active=True
        self.direct_transfer_attempts=int(self.cfg.get('direct_attempts_used',0))
        self.planning_trace=PlanningTrace(self.cfg.get('planning_target',''),self.cfg.get('planning_budget_s',20.),self.cfg.get('planning_observer'))
        status='interrupted';error=None
        try:
            result=self.search_plan(target,destination,tray_floor)
            status='found';return result
        except RuntimeError as exc:
            error=exc;status='timeout' if '规划预算耗尽' in str(exc) else 'scheduled_finished'
            raise
        except Exception as exc:error=exc;raise
        finally:
            self.last_planning_report=self.planning_trace.finish(status,error)
            self.planning_active=False

    def search_plan(self, target, destination, tray_floor):
        import copy
        target=copy.copy(target)
        if not hasattr(target,'grasp_yaws'):
            radial=float(np.arctan2(target.center[1],target.center[0]))
            target.grasp_yaws=list(dict.fromkeys([radial,radial+np.pi,radial+np.pi/2,radial-np.pi/2,target.yaw,0.,np.pi]))
        candidates=[target]
        if getattr(target,'category',None)=='bell_pepper':
            target.object_center=getattr(target,'object_center',(np.percentile(target.points,2,axis=0)+np.percentile(target.points,98,axis=0))/2)
        for option in getattr(target,'grasp_quality',{}).get('alternatives',[]):
            alternative=copy.copy(target)
            alternative.center=target.center.copy()
            alternative.center[:len(option['center'])]=option['center']
            alternative.yaw=option['yaw']
            alternative.grasp_yaws=option.get('yaws',[alternative.yaw,alternative.yaw+np.pi])
            candidates.append(alternative)
        rng=getattr(self,'selection_rng',None)
        if rng is None:self.selection_rng=rng=np.random.default_rng(self.cfg.get('random_seed',0))
        current=self.b.data.site_xpos[self.b.model.site('tcp').id].copy()
        # Normalize available geometry scores before travel cost and tie shuffle.
        scores=[float(o.get('score',1.)) for o in getattr(target,'grasp_quality',{}).get('alternatives',[])]
        top=max(scores+[1e-9]) if scores else 1.
        candidates[0].candidate_score=1.
        for c,score in zip(candidates[1:],scores):c.candidate_score=score/top
        failed=getattr(target,'rejected_points',[])
        candidates=[c for c in candidates if not any(np.linalg.norm(c.center-np.asarray(p))<.008 for p in failed)]
        quality=getattr(target,'grasp_quality',{})
        if getattr(target,'category',None) in ('banana','cucumber') and len(target.points)>=3:
            xy=np.asarray(target.points)[:,:2];origin=np.mean(xy,axis=0)
            axis=np.linalg.svd(xy-origin,full_matrices=False)[2][0]
            axial=(xy-origin)@axis;middle=(np.min(axial)+np.max(axial))/2
            radius=float(np.ptp(axial))/6
            for c in candidates:c.priority_tier=0 if abs(float((c.center[:2]-origin)@axis)-middle)<=radius+.003 else 1
        unique=[]
        for c in candidates:
            if not any(np.linalg.norm(c.center-o.center)<.001 and len(c.grasp_yaws)==len(o.grasp_yaws) and np.allclose(c.grasp_yaws,o.grasp_yaws) for o in unique):unique.append(c)
        candidates=ordered_candidates(unique,current,rng)[:16]
        if self.cfg.get('ready_transport') and getattr(target,'category',None) in ('banana','cucumber'):
            candidates=[c for c in candidates if getattr(c,'priority_tier',0)==0]
        if not candidates:raise RuntimeError('当前可见抓取点均已失败，需要新视角或物体姿态变化')
        destinations=[np.asarray(destination)]+[np.asarray(p) for p in getattr(target,'destination_options',[]) if np.linalg.norm(np.asarray(p)-destination)>.005]
        heights=list(dict.fromkeys([self.cfg.get('transport_height_m',.22)]+self.cfg.get('transport_height_candidates_m',[])))
        if self.cfg.get('adaptive_transport_height',False) and not self.cfg.get('ready_transport'):heights=list(dict.fromkeys([max(.10,float(target.center[2])+.05),.14]+heights))
        self.planning_event('scope',grasp_count=len(candidates),landing_count=min(6,len(destinations)),tilt_count=len(self.cfg.get('tilt_candidates',[.15,.35,.6])),height_count=len(heights),combination_upper_bound=sum(len(c.grasp_yaws) for c in candidates)*len(self.cfg.get('tilt_candidates',[.15,.35,.6]))*len(heights)*min(6,len(destinations)),grasps=[dict(index=i+1,center=c.center.tolist(),yaws_deg=[float(np.degrees(y)) for y in c.grasp_yaws],score=c.selection_score,priority_tier=getattr(c,'priority_tier',0)) for i,c in enumerate(candidates)],landings=[p.tolist() for p in destinations[:6]],tilts_deg=[float(np.degrees(t)) for t in self.cfg.get('tilt_candidates',[.15,.35,.6])],heights_m=heights)
        deadline=time.monotonic()+self.cfg.get('planning_budget_s',20.)
        errors=[]
        tilts=self.cfg.get('tilt_candidates',[.15,.35,.6])
        if self.cfg.get('natural_pose_preference'):tilts=fair_tilts(tilts)
        # Middle-circle sections have their own first search pass. Time expiry is
        # evidence of an incomplete search, never proof of unreachability.
        tiers=sorted({getattr(c,'priority_tier',0) for c in candidates})
        timed_out=False
        fallback_plan=None
        for tier in tiers:
            members=[(i,c) for i,c in enumerate(candidates) if getattr(c,'priority_tier',0)==tier]
            pass_deadline=deadline if tier==tiers[-1] else min(deadline,time.monotonic()+self.cfg.get('planning_budget_s',20.)*.7)
            self.planning_event('priority_pass',message=('优先搜索中间圆附近抓点' if tier==0 else '切换其他安全抓点：前一组未找到方案，未证明不可达') if quality.get('method')=='visible_metric_centerline_straight_section' else '搜索可见抓取候选')
            pass_expired=False
            for round_index in range(max(len(c.grasp_yaws) for _,c in members)*len(tilts)*len(heights)):
                for pair_index in range(max(min(6,len(destinations)),len(members))):
                    if time.monotonic()>=pass_deadline:
                        pass_expired=True;timed_out=True;break
                    # Keep the selected half-centre while trying arm/gripper configurations.
                    elapsed=self.cfg.get('planning_budget_s',20.)-(deadline-time.monotonic())
                    primary_budget=self.cfg.get('planning_budget_s',20.)*.65
                    landing_index=0 if elapsed<primary_budget or len(destinations)==1 else 1+(pair_index+round_index)%max(1,min(6,len(destinations))-1)
                    landing=destinations[landing_index]
                    candidate_index,candidate=members[pair_index%len(members)]
                    yaws=getattr(candidate,'grasp_yaws',[candidate.yaw])
                    if round_index>=len(yaws)*len(tilts)*len(heights):continue
                    option=copy.copy(candidate)
                    cycle=int(self.cfg.get('search_cycle',0))
                    option.grasp_yaws=[yaws[(round_index//len(tilts)+round_index%len(tilts)+cycle)%len(yaws)]]
                    tilt=tilts[(round_index+pair_index)%len(tilts)]
                    self.planning_deadline=min(pass_deadline,time.monotonic()+2.)
                    self.planning_event('attempt',grasp_index=candidate_index+1,grasp_count=len(candidates),landing_index=landing_index+1,landing_count=min(6,len(destinations)),grasp_center=option.center.tolist(),landing=landing.tolist(),yaw_deg=float(np.degrees(option.grasp_yaws[0])),tilt_deg=float(np.degrees(tilt)),height_m=heights[(round_index//(len(tilts)*len(yaws))+round_index%len(tilts)+cycle)%len(heights)],priority_tier=tier,grasp_priority_label=('中间圆附近' if tier==0 else '其他安全段') if quality.get('method')=='visible_metric_centerline_straight_section' else '常规可见抓点')
                    try:
                        plan=self._plan_one(option,landing,tray_floor,heights[(round_index//(len(tilts)*len(yaws))+round_index%len(tilts)+cycle)%len(heights)],[tilt])
                        plan.update(destination_position=landing.tolist(),selection_score=candidate.selection_score)
                        if self.cfg.get('natural_pose_preference') and not 0<=tilt<=1.25:
                            if fallback_plan is None:
                                fallback_plan=plan
                                if landing_index==0:destinations=destinations[:1]
                                fallback_plan['carried_cloud']=self.carried_visual_cloud.copy() if hasattr(self,'carried_visual_cloud') else None
                                deadline=min(deadline,time.monotonic()+3.)
                                pass_deadline=min(pass_deadline,deadline)
                            self.planning_event('fallback_available',message='已有可执行备用姿态，短暂继续寻找正向方案；不会仅因偏好而放弃执行')
                            continue
                        self.planning_event('found',detour=plan.get('transfer_waypoints',[]))
                        return plan
                    except RuntimeError as exc:
                        self.planning_event('attempt_failure',**explain(exc));errors.append(str(exc))
                if pass_expired:break
        if fallback_plan is not None:
            if fallback_plan.get('carried_cloud') is not None:self.carried_visual_cloud=fallback_plan['carried_cloud']
            self.planning_event('found',message='正向候选未通过，执行已验证的备用姿态')
            return fallback_plan
        if timed_out:raise RuntimeError('规划预算耗尽；候选搜索未完成，不代表物体不可达。最近原因：'+(errors[-1] if errors else '无'))
        raise RuntimeError('本轮预设组合检查结束，仍未找到路径；未证明整体不可达。最近原因：'+(errors[-1] if errors else '无候选'))

    def validate_transfer(self,points):
        """Robot/self/static checks only; hidden movable-object truth is not used."""
        import mujoco
        model=self.b.model
        if getattr(self,'_transfer_model',None) is not model:
            self._transfer_model=model;self._transfer_data=mujoco.MjData(model)
        data=self._transfer_data
        robot={model.body(n).id for n in ['base_link','gripper_base']+['link'+str(i) for i in range(1,9)]}
        fixtures={model.geom('competition_table').id}|{model.geom('tray_'+c+'_floor').id for c in self.cfg.get('tray_colors',[])}
        installation=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'installation_table')
        if installation>=0:fixtures.add(installation)
        for q in points:
            data.qpos[:]=self.b.data.qpos;data.qpos[:6]=q
            mujoco.mj_forward(model,data)
            # Arm links must clear the desk even before physical penetration occurs.
            arm={model.body('link'+str(i)).id for i in range(2,7)}
            for geom in range(model.ngeom):
                if int(model.geom_bodyid[geom]) not in arm or model.geom_group[geom]==1:continue
                for fixture in fixtures:
                    if mujoco.mj_geomDistance(model,data,geom,fixture,.004,None)<.003:
                        raise RuntimeError('肘臂与桌面净空不足3毫米，拒绝该整臂姿态')
            for contact in data.contact[:data.ncon]:
                g1,g2=int(contact.geom1),int(contact.geom2)
                r1=int(model.geom_bodyid[g1]) in robot;r2=int(model.geom_bodyid[g2]) in robot
                if contact.dist<-.002 and ((r1 and g2 in fixtures) or (r2 and g1 in fixtures) or (r1 and r2)):
                    raise RuntimeError('绕行碰撞：'+str(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g1))+' 与 '+str(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g2))+f'，穿透{-contact.dist*1000:.1f}毫米')

    def transport_path(self,start_q,current,destination,yaw):
        progress=self.planning_event if getattr(self,'planning_active',False) else None
        if self.cfg.get('ready_transport'):
            return self.ready_transport_path(start_q,current,destination,yaw,progress)
        try:
            return transfer_route(self.cartesian_path,start_q,current,destination,yaw,self.validate_transfer,progress=progress)
        except RuntimeError as cartesian_error:
            if getattr(self,'planning_active',False) and time.monotonic()>=self.planning_deadline:raise
            goal=self.ik.solve(destination,yaw,start_q)
            model=self.b.model
            import mujoco
            data=mujoco.MjData(model);sid=model.site('tcp').id
            cloud=getattr(self,'carried_visual_cloud',np.empty((0,3)))
            def bounds(q):
                data.qpos[:]=self.b.data.qpos;data.qpos[:6]=q;mujoco.mj_kinematics(model,data)
                tcp=data.site_xpos[sid].copy();rot=data.site_xmat[sid].reshape(3,3)
                bottom=float(np.min((cloud@rot.T+tcp)[:,2])) if len(cloud) else float(tcp[2]-.04)
                return tcp,bottom
            a,ab=bounds(start_q);b,bb=bounds(goal)
            tcp_floor=max(self.table+.06,min(a[2],b[2])-.003)
            object_floor=max(self.table+.015,min(ab,bb)-.003)
            def validate(points):
                self.validate_transfer(points)
                for i,q in enumerate(points):
                    if i%8==0:self.planning_event('waypoint',index=i+1,total=len(points),position=bounds(q)[0].tolist())
                    tcp,bottom=bounds(q)
                    if tcp[2]<tcp_floor or bottom<object_floor:raise RuntimeError('关节绕行会降低物品净空，拒绝贴桌搬运')
            mids=[]
            center=(np.asarray(current)+np.asarray(destination))/2
            for lift in [.04,.08]:
                point=center.copy();point[2]=min(.34,max(current[2],destination[2])+lift)
                try:mids.append(self.ik.solve(point,yaw,start_q))
                except RuntimeError:continue
            points=joint_space_route(start_q,goal,mids,validate,progress)
            return points,[bounds(points[len(points)//2])[0].tolist()]

    def validate_loaded_route(self,points):
        import mujoco
        self.validate_transfer(points)
        data=mujoco.MjData(self.b.model);sid=self.b.model.site('tcp').id
        for q in points:
            data.qpos[:]=self.b.data.qpos;data.qpos[:6]=q;mujoco.mj_kinematics(self.b.model,data)
            cloud=self.carried_visual_cloud@data.site_xmat[sid].reshape(3,3).T+data.site_xpos[sid]
            if len(cloud) and np.min(cloud[:,2])<self.table+self.cfg.get('minimum_carried_clearance_m',.105)-1e-5:
                raise RuntimeError('持物路线净空不足10cm')

    def final_plan_from_ready(self,plan):
        """Second leg is searched only after physically arriving at ready position."""
        from piper_sim.route_search import ready_position_route
        target=plan['final_target'];floor=plan['tray_floor']
        offset=(np.asarray(getattr(target,'object_center',target.center))-plan['center'])@plan['grasp_rotation']
        self.planning_active=False
        errors=[]
        fresh_deadline=time.monotonic()+8.
        destinations=[plan['final_destination']]+list(getattr(target,'destination_options',[]))
        for destination in destinations[:6]:
            partition=getattr(target,'placement_partition',None)
            if partition and 'center' in partition and np.linalg.norm(np.asarray(destination)-partition['center'])>partition.get('max_center_offset_m',.025):continue
            tilt_options=self.cfg.get('tilt_candidates',[0.,.6,1.2])
            yaws=[float(np.arctan2(destination[1],destination[0])),plan['yaw'],0.,np.pi,np.pi/2,-np.pi/2]
            orientations=placement_orientations(tilt_options,yaws,plan['tilt'],plan['yaw'])
            for tilt,yaw in orientations:
                if time.monotonic()>=fresh_deadline:raise ReleaseBlocked('本轮新空位规划达到8秒，重新观察并换候选，不固守旧数据')
                self.callback('plan_from_ready')
                rotation=Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('y',tilt).as_matrix()@np.diag([1.,-1.,-1.])
                validator3d=getattr(target,'placement_validator_3d',None)
                validator=getattr(target,'placement_validator',None)
                if validator3d:
                    if not validator3d(destination,plan['grasp_rotation'],rotation,plan['center']):continue
                elif validator and not validator(destination,yaw-plan['yaw']):continue
                place=np.r_[np.asarray(destination)-(offset@rotation.T)[:2],floor-float(np.min((self.carried_visual_cloud@rotation.T)[:,2]))+.005]
                above=place.copy();above[2]=max(.22,self.table+self.cfg.get('minimum_carried_clearance_m',.105)-float(np.min((self.carried_visual_cloud@rotation.T)[:,2])))
                try:
                    self.ik=IK(self.b,tilt=tilt)
                    q=self.ik.solve(above,yaw,self.b.measured()[:6])
                    path=ready_position_route(self.b.measured()[:6],q,[q],self.validate_loaded_route)
                    final_q=self.cartesian_path(q,place,yaw,current=above)[-1]
                    if not opening_clearance(self.b,final_q):continue
                    self.check_visual_release(target,plan['center'],plan['grasp_rotation'],final_q,place,yaw-plan['yaw'])
                    plan.update(place=place,above=above,place_yaw=yaw,place_tilt=tilt,
                        destination_position=np.asarray(destination).tolist(),ready_arrived=True,final_transfer_points=path,deferred_final=False)
                    return plan
                except RuntimeError as exc:errors.append(str(exc))
        raise ReleaseBlocked('已持物到达准备位置；目标放置姿态本轮未找到，继续持物规划：'+(errors[-1] if errors else '落区占地约束'))

    def ready_transport_path(self,start_q,current,destination,yaw,progress=None):
        """At most two straight trials per plan, then held-object ready-position route."""
        import mujoco
        from piper_sim.route_search import ready_position_route
        model=self.b.model;data=mujoco.MjData(model);sid=model.site('tcp').id
        cloud=getattr(self,'carried_visual_cloud',np.empty((0,3)))
        floor=self.table+self.cfg.get('minimum_carried_clearance_m',.105)
        def validate(points):
            self.validate_transfer(points)
            for q in points:
                data.qpos[:]=self.b.data.qpos;data.qpos[:6]=q;mujoco.mj_kinematics(model,data)
                tcp=data.site_xpos[sid];rot=data.site_xmat[sid].reshape(3,3)
                bottom=np.min((cloud@rot.T+tcp)[:,2]) if len(cloud) else tcp[2]-.04
                if bottom<floor-1e-5:raise RuntimeError('搬运物体底部未达到桌面以上10cm净空')
        def direct():
            self.direct_transfer_attempts=getattr(self,'direct_transfer_attempts',0)+1
            points=self.cartesian_path(start_q,destination,yaw,current=current)
            validate(points);return points
        errors=[]
        signature=tuple(np.round(np.r_[start_q,destination,yaw,getattr(self.ik,'tilt',0.)],3))
        trials=getattr(self,'direct_route_trials',{})
        self.direct_route_trials=trials
        self.direct_transfer_attempts=trials.get(signature,0)
        while self.direct_transfer_attempts<2:
            try:return direct(),[]
            except RuntimeError as exc:
                errors.append(str(exc));trials[signature]=self.direct_transfer_attempts
                if progress:progress('route_failure',route_label='直线尝试',attempt=self.direct_transfer_attempts,error=str(exc))
        # A short changing-orientation joint route precedes the distant ready fallback.
        goal=self.ik.solve(destination,yaw,start_q)
        try:
            points=joint_space_route(start_q,goal,[],validate,progress)
            return points,[]
        except RuntimeError:pass
        for rise in [.025,.05]:
            points=[np.asarray(start_q).copy()];position=np.asarray(current).copy();via=[]
            try:
                for t in np.linspace(0,1,9)[1:]:
                    point=(1-t)*np.asarray(current)+t*np.asarray(destination)
                    point[2]+=4*rise*t*(1-t)
                    segment=self.cartesian_path(points[-1],point,yaw,current=position)
                    points.extend(segment[1:]);position=point;via.append(point.tolist())
                validate(points)
                return points,via[:-1]
            except RuntimeError:continue
        ready=np.array(self.cfg.get('ready_position_m',[.28,0.,.30]),dtype=float)
        # Ready position is a waypoint while holding; never place on a staging surface.
        ready[2]=max(ready[2],current[2],destination[2])
        goal=self.ik.solve(destination,yaw,start_q)
        candidates=[]
        for tilt in [.7,1.05,.35,0.,-.35]:
            try:candidates.append(IK(self.b,tilt=tilt).solve(ready,np.pi,start_q))
            except RuntimeError:continue
        points=ready_position_route(start_q,goal,candidates,validate,progress)
        return points,[ready.tolist()]

    def _plan_one(self, target, destination, tray_floor,transport_height=None,tilts=None):
        errors = []
        for tilt in (tilts or self.cfg.get('tilt_candidates',[0.6, 0.9, 0.35, 0.15])):
            for yaw in getattr(target,'grasp_yaws',[
                target.yaw + np.pi,
                np.pi,
                target.yaw + np.pi / 2,
                target.yaw,
                target.yaw - np.pi / 2,
            ]):
                if time.monotonic()>=getattr(self,'planning_deadline',float('inf')):raise RuntimeError('本候选搜索预算耗尽，切换下一抓取点或落点')
                if any(np.linalg.norm(np.asarray(c)-target.center)<.008 and abs(np.arctan2(np.sin(yaw-y),np.cos(yaw-y)))<.20 for c,y in getattr(target,'rejected_regions',[])):
                    continue
                if any(abs(tilt-t)<.01 and abs(np.arctan2(np.sin(yaw-y),np.cos(yaw-y)))<.08
                       for t,y in getattr(target,'rejected_grasps',[])):
                    continue
                self.ik = IK(self.b, tilt=tilt)
                rot = (
                    Rotation.from_euler("z", yaw).as_matrix()
                    @ Rotation.from_euler("y", tilt).as_matrix()
                    @ np.diag([1.0, -1.0, -1.0])
                )
                lowest = (rot @ [0, 0, 0.1088 - 0.122])[2] - (np.abs(rot) @ [0.022, 0.003, 0.023])[
                    2
                ]
                center = target.center.copy()
                center[2] = max(center[2], getattr(target,'support_height_m',self.table) - lowest + 0.001)
                hover = center.copy()
                hover[2] = transport_height if transport_height is not None else self.cfg.get("transport_height_m", 0.22)
                if self.cfg.get('ready_transport'):
                    object_bottom=min(float(np.min(np.asarray(target.points)[:,2])),getattr(target,'support_height_m',self.table)) if len(target.points) else self.table
                    hover[2]=max(hover[2],self.table+self.cfg.get('minimum_carried_clearance_m',.105)+center[2]-object_bottom)
                vertical_offset=float(target.center[2]-getattr(target,'object_center',target.center)[2])
                place = np.r_[
                    np.asarray(destination)-(getattr(target,'object_center',target.center)[:2]-target.center[:2]),
                    max(tray_floor + target.height / 2 + vertical_offset + 0.003, tray_floor - lowest + 0.002)+getattr(target,'place_clearance_m',0.),
                ]
                above = place.copy()
                above[2] = hover[2]
                try:
                    self.planning_phase('hover')
                    hover_q=self.ik.solve(hover,yaw,getattr(target,'planning_start_q',self.b.measured()[:6]))
                    near=center.copy();near[2]=min(hover[2],center[2]+.05)
                    # Endpoint reachability alone misses joint limits and branch
                    # switches encountered during the actual descent.
                    self.planning_phase('near')
                    near_q=self.cartesian_path(hover_q,near,yaw)[-1]
                    self.planning_phase('descend')
                    seed=self.cartesian_path(near_q,center,yaw)[-1]
                    self.planning_phase('lift')
                    lift_q=self.cartesian_path(seed,hover,yaw)[-1]
                    # A valid grasp orientation need not also be the tray orientation.
                    self.carried_visual_cloud=(np.asarray(target.points)[::max(1,len(target.points)//150),:3]-center)@rot if len(target.points) else np.empty((0,3))
                    if self.cfg.get('ready_transport'):
                        support=np.r_[getattr(target,'object_center',target.center)[:2],getattr(target,'support_height_m',self.table)]
                        self.carried_visual_cloud=np.vstack([self.carried_visual_cloud,(support-center)@rot])
                    if self.cfg.get('defer_final_until_ready'):
                        ready=np.asarray(self.cfg.get('ready_position_m',[.28,0.,.30]),dtype=float).copy()
                        ready[2]=max(ready[2],hover[2])
                        ready_q=IK(self.b,tilt=.7).solve(ready,np.pi,lift_q)
                        ready_points=ready_position_route(lift_q,ready_q,[ready_q],self.validate_loaded_route)
                        return dict(center=center,grasp_candidate_center=target.center.tolist(),hover=hover,hover_q=hover_q,
                            retreat_q=ready_q,place=place,above=above,yaw=yaw,place_yaw=yaw,place_tilt=tilt,
                            tilt=tilt,height=target.height,transfer_waypoints=[ready.tolist()],
                            deferred_final=True,ready_points=ready_points,ready_position=ready,
                            final_target=target,final_destination=np.asarray(destination).copy(),tray_floor=tray_floor,
                            grasp_rotation=rot)
                    grasp_offset=getattr(target,'object_center',target.center)[:2]-target.center[:2]
                    radial=float(np.arctan2(destination[1],destination[0]))
                    preferred=yaw+getattr(target,'preferred_place_rotation',0.)
                    yaws=list(dict.fromkeys(([preferred,radial+np.pi,radial] if getattr(target,'category',None) in ('banana','cucumber') else [radial+np.pi,preferred,radial])+[yaw]+getattr(target,'preferred_place_yaws',[])+[np.pi,0.,np.pi/2,-np.pi/2]))
                    place_tilts=list(dict.fromkeys([tilt,1.05,1.25,1.45,.9])) if self.cfg.get('adaptive_transport_height',False) else [tilt]
                    if self.cfg.get('ready_transport'):place_tilts=list(dict.fromkeys([tilt]+self.cfg.get('tilt_candidates',[])))
                    orientations=placement_orientations(place_tilts,yaws,tilt,yaw) if self.cfg.get('natural_pose_preference') else [(t,y) for t in place_tilts for y in yaws]
                    for place_tilt,place_yaw in orientations:
                        delta=place_yaw-yaw
                        rotation_xy=np.array([[np.cos(delta),-np.sin(delta)],[np.sin(delta),np.cos(delta)]])
                        place[:2]=np.asarray(destination)-rotation_xy@grasp_offset
                        place_rotation=Rotation.from_euler('z',place_yaw).as_matrix()@Rotation.from_euler('y',place_tilt).as_matrix()@np.diag([1.,-1.,-1.])
                        object_offset=(np.asarray(getattr(target,'object_center',target.center))-center)@rot@place_rotation.T
                        place[:2]=np.asarray(destination)-object_offset[:2]
                        if abs(place_tilt-tilt)>.01:
                            predicted=(np.asarray(target.points)-center)@rot@place_rotation.T
                            hidden=np.r_[getattr(target,'object_center',target.center)[:2],getattr(target,'support_height_m',self.table)]
                            bottom=min(float(np.min(predicted[:,2])),float((hidden-center)@rot@place_rotation.T @ np.array([0.,0.,1.])))
                            placed_lowest=(place_rotation@np.array([0.,0.,.1088-.122]))[2]-(abs(place_rotation)@np.array([.022,.003,.023]))[2]
                            place[2]=max(tray_floor-bottom+.005,tray_floor-placed_lowest+.002)
                        above[:2]=place[:2]
                        above[2]=max(float(hover[2]),float(place[2])+.025)
                        if self.cfg.get('ready_transport') and len(self.carried_visual_cloud):
                            above[2]=max(above[2],self.table+self.cfg.get('minimum_carried_clearance_m',.105)-float(np.min((self.carried_visual_cloud@place_rotation.T)[:,2])))
                        try:
                            validator=getattr(target,'placement_validator',None)
                            validator3d=getattr(target,'placement_validator_3d',None)
                            valid=validator3d(destination,rot,place_rotation,center) if validator3d else validator is None or validator(destination,delta)
                            if not valid:
                                raise RuntimeError('放置方向的视觉占地超出预留边距或与其他物品重叠')
                            self.ik=IK(self.b,tilt=place_tilt)
                            if self.cfg.get('ready_transport') and len(self.carried_visual_cloud):
                                planned_bottom=place[2]+float(np.min((self.carried_visual_cloud@place_rotation.T)[:,2]))
                                if planned_bottom>tray_floor+.008:
                                    raise RuntimeError('该放置姿态受夹爪净空限制，物体仍悬空；换姿态而非高处释放')
                            # Reject impossible destination endpoints before expensive transfer search.
                            self.ik.solve(place,place_yaw,lift_q)
                            self.ik.solve(above,place_yaw,lift_q)
                            self.planning_phase('reorient')
                            pose=lift_q.copy() if self.cfg.get('ready_transport') or (abs(place_yaw-yaw)<1e-6 and abs(place_tilt-tilt)<1e-6) else self.ik.solve(hover,place_yaw,lift_q)
                            self.planning_phase('transfer')
                            transfer_points,transfer_waypoints=self.transport_path(pose,hover,above,place_yaw)
                            pose=transfer_points[-1]
                            preplace=place.copy();preplace[2]=min(above[2],place[2]+.05)
                            self.planning_phase('preplace')
                            pose=self.cartesian_path(pose,preplace,place_yaw)[-1]
                            self.planning_phase('place')
                            place_q=self.cartesian_path(pose,place,place_yaw)[-1]
                            self.planning_phase('opening')
                            if not opening_clearance(self.b,place_q):
                                raise RuntimeError('Full finger opening would hit a tray wall or floor')
                            self.check_visual_release(target,center,rot,place_q,place,place_yaw-yaw)
                            self.planning_phase('retreat')
                            retreat_q=place_q.copy() if self.cfg.get('ready_transport') else self.cartesian_path(place_q,above,place_yaw)[-1]
                            break
                        except RuntimeError as exc:
                            self.planning_event('orientation_failure',place_yaw_deg=float(np.degrees(place_yaw)),**explain(exc));continue
                    else:raise RuntimeError('Grasp is reachable but no tray transfer orientation is reachable')
                except RuntimeError as exc:
                    self.planning_event('pose_failure',**explain(exc))
                    errors.append(str(exc))
                    continue
                break
            else:
                continue
            break
        else:
            raise RuntimeError("没有找到连续可达抓放姿态：" + (errors[-1] if errors else '候选姿态已在接近过程中失败'))
        return {
            "center": center,
            "grasp_candidate_center":target.center.tolist(),
            "hover": hover,
            "hover_q": hover_q,
            "retreat_q":retreat_q,
            "place": place,
            "above": above,
            "yaw": yaw,
            "place_yaw":place_yaw,
            "place_tilt":place_tilt,
            "tilt": tilt,
            "height": target.height,
            "transfer_waypoints":transfer_waypoints,
            "validated_transfer_points":transfer_points,
            "final_target":target,"final_destination":np.asarray(destination).copy(),
            "tray_floor":tray_floor,"grasp_rotation":rot,
        }

    def check_visual_release(self,target,center,grasp_rotation,place_q,place,turn):
        """Reject a planned open pose intersecting the observed object surface."""
        if getattr(target,'category',None)!='bell_pepper':return
        import mujoco
        data=mujoco.MjData(self.b.model);data.qpos[:6]=place_q
        data.qpos[self.b.fingers]=[.05,-.05];mujoco.mj_forward(self.b.model,data)
        points=np.asarray(target.points)
        rotation=Rotation.from_euler('z',turn).as_matrix()
        placed=(points-center)@rotation.T+place
        for name in ['link7_pad','link8_pad']:
            g=self.b.model.geom(name).id
            local=(placed-data.geom_xpos[g])@data.geom_xmat[g].reshape(3,3)
            if np.any(np.all(abs(local)<self.b.model.geom_size[g]+.001,axis=1)):
                raise RuntimeError('可见主体在完全张爪后仍与指垫相交，切换抓点或放置方向')

    def close_until_stable(self):
        """Stop the empty closing stroke after sustained bilateral contact."""
        from piper_sim.trajectory import trajectory
        q=self.b.previous_target[:6].copy();width=self.b.previous_target[6]
        acceleration=self.motion_profile();stable_since=None;previous_width=None
        for goal in trajectory([q,q],width,0,self.b.interface,acceleration):
            self.tick(goal)
            measured=float(self.b.measured()[6])
            accepted=self.cfg.get('grasp_ready',self.cfg.get('grasp_check'))
            stable=accepted and accepted() and measured>.006 and previous_width is not None and abs(measured-previous_width)<.001
            if stable:
                if stable_since is None:stable_since=float(self.b.data.time)
                if self.b.data.time-stable_since>=.10-1e-9:
                    self.hold(.10)
                    if accepted():return
                    stable_since=None
            else:stable_since=None
            previous_width=measured
        self.hold(.30)

    def pick(self, plan):
        self.grip_monitor.disarm()
        self.ik=IK(self.b,tilt=plan['tilt'])
        center, hover, yaw = plan["center"], plan["hover"], plan["yaw"]
        self.stage = "approach"
        self.move(plan['hover_q'] if 'hover_q' in plan else self.ik.solve(hover, yaw, self.b.measured()[:6]), 0.10)
        near=center.copy();near[2]=min(hover[2],center[2]+.05)
        self.stage='near_approach'
        self.cartesian(near,yaw,.10)
        self.stage = "descend"
        self.cartesian(center, yaw, 0.10)
        self.stage = "close"
        self.closed_attempts+=1
        self.close_until_stable()
        if self.b.measured()[6] < 0.006:
            raise GraspEmptyError('闭合后夹爪近乎全闭，立即重观察，跳过搬运。')
        if self.cfg.get('grasp_check') and not self.cfg['grasp_check']():
            raise GraspEmptyError('闭合后的接触不满足双侧稳定夹持，拒绝提升搬运。')
        if self.cfg.get('grasp_retained'):self.grip_monitor.arm()
        self.stage='test_lift'
        test=center.copy();test[2]=min(hover[2],center[2]+.015)
        self.cartesian(test,yaw,0)
        self.hold(.20)
        if self.cfg.get('grasp_check') and not self.cfg['grasp_check']():
            raise GraspEmptyError('Low-height test lift failed bilateral stability check')
        self.stage = "lift"
        self.cartesian(hover, yaw, 0)
        self.hold(self.cfg.get('lift_verify_hold_s',1.1))
        if self.b.measured()[6] < 0.006:
            raise GraspEmptyError("夹爪接近完全闭合：疑似抓空，停止搬运。")
        return {"gripper_width_m": float(self.b.measured()[6]), "holding_estimate": True}

    def place(self, plan):
        self.release_verified=False;self.retreat_verified=False
        if plan.get('deferred_final'):
            self.stage='transport_ready'
            from piper_sim.trajectory import trajectory
            for goal in trajectory(plan['ready_points'],self.b.previous_target[6],0,self.b.interface,self.motion_profile()):self.tick(goal)
            self.hold(.15)
            # No object release or table staging between the two held-object legs.
            self.final_plan_from_ready(plan)
        place, above, yaw = plan["place"], plan["above"], plan.get('place_yaw',plan['yaw'])
        self.stage = "transport"
        self.ik=IK(self.b,tilt=plan.get('place_tilt',plan['tilt']))
        if not self.cfg.get('ready_transport') and (abs(plan.get('place_tilt',plan['tilt'])-plan['tilt'])>.01 or abs(np.arctan2(np.sin(yaw-plan['yaw']),np.cos(yaw-plan['yaw'])))>.05):
            current=self.b.data.site_xpos[self.b.model.site('tcp').id].copy()
            self.move(self.ik.solve(current,yaw,self.b.measured()[:6]),0)
            self.hold(.1)
            if self.cfg.get('grasp_check') and not self.cfg['grasp_check']():
                raise RuntimeError('Bilateral grasp lost during transfer reorientation')
        from piper_sim.trajectory import trajectory
        current=self.b.data.site_xpos[self.b.model.site('tcp').id].copy()
        if plan.get('ready_arrived'):points,waypoints=plan['final_transfer_points'],[]
        elif plan.get('validated_transfer_points') is not None and np.max(abs(self.b.measured()[:6]-plan['validated_transfer_points'][0]))<.04:
            points,waypoints=plan['validated_transfer_points'],plan.get('transfer_waypoints',[])
            self.validate_loaded_route(points)
        else:points,waypoints=self.transport_path(self.b.measured()[:6],current,above,yaw)
        if waypoints:self.stage='transport_detour'
        acceleration=self.motion_profile()
        commands=trajectory(points,self.b.previous_target[6],0,self.b.interface,acceleration)
        if self.cfg.get('ready_transport'):self.validate_loaded_route(commands[:,:6])
        for goal in commands:self.tick(goal)
        start=self.b.data.time
        while np.max(abs(self.b.measured()[:6]-points[-1]))>.010:
            self.tick(np.r_[points[-1],0])
            if self.b.data.time-start>4:raise TimeoutError('搬运绕行末端跟踪超时')
        preplace=place.copy();preplace[2]=min(above[2],place[2]+.05)
        self.stage='place_approach'
        self.cartesian(preplace,yaw,0)
        self.stage = "place"
        try:self.cartesian(place, yaw, 0)
        except (RuntimeError,TimeoutError) as exc:
            from piper_sim.drop_recovery import ObjectDropped
            if isinstance(exc,ObjectDropped):raise
            raise ReleaseBlocked('下放路径受阻：'+str(exc)+'；保持夹持并重新观察落点') from exc
        self.release_and_retreat(plan)
        self.hold(self.cfg.get('place_verify_hold_s',2.2))
        self.stage = "verify"
        return {
            "planned_center_m": plan["center"].tolist(),
            "planned_destination_m": place.tolist(),
            "target_height_estimate_m": plan["height"],
            "grasp_tilt_rad": plan["tilt"],
            "yaw_rad": yaw,
        }

    def vertical_release_move(self,distance,width):
        """Move vertically with measured orientation/XY and validate the path."""
        import mujoco
        from scipy.optimize import least_squares
        from piper_sim.trajectory import trajectory
        data=mujoco.MjData(self.b.model);q=self.b.measured()[:6].copy()
        data.qpos[:6]=q;mujoco.mj_kinematics(self.b.model,data)
        sid=self.b.model.site('tcp').id
        origin=data.site_xpos[sid].copy();rotation=data.site_xmat[sid].reshape(3,3).copy()
        points=[q.copy()];lower,upper=self.b.model.jnt_range[self.b.joints].T
        for dz in np.linspace(0,distance,max(2,int(np.ceil(abs(distance)/.008)))+1)[1:]:
            target=origin+np.array([0,0,dz]);seed=points[-1]
            def residual(candidate):
                data.qpos[:6]=candidate;mujoco.mj_kinematics(self.b.model,data)
                angular=Rotation.from_matrix(rotation@data.site_xmat[sid].reshape(3,3).T).as_rotvec()
                return np.r_[data.site_xpos[sid]-target,.15*angular,.001*(candidate-seed)]
            solution=least_squares(residual,np.clip(seed,lower+1e-5,upper-1e-5),
                bounds=(lower+1e-6,upper-1e-6),max_nfev=100)
            error=residual(solution.x)
            if np.linalg.norm(error[:3])>.0015 or np.linalg.norm(error[3:6])>.004 or max(abs(solution.x-seed))>.5:
                raise ReleaseBlocked('No continuous vertical release clearance path')
            points.append(solution.x)
        self.validate_transfer(points)
        # Discard the stalled arm command; this changes only the servo target.
        self.b.previous_target[:6]=q
        for goal in trajectory(points,self.b.previous_target[6],width,self.b.interface,self.motion_profile()):
            self.tick(goal)
            actual=self.b.data.site_xpos[sid]
            if np.linalg.norm(actual[:2]-origin[:2])>.004:
                self.b.previous_target[:6]=self.b.measured()[:6]
                raise ReleaseBlocked('Lateral tracking drift during vertical release; motion stopped')
        self.hold(.10)
        if abs(self.b.data.site_xpos[sid,2]-(origin[2]+distance))>.004:
            raise ReleaseBlocked('Vertical recovery movement blocked')

    def open_gripper_in_place(self):
        self.grip_monitor.disarm()
        self.release_verified=False;self.retreat_verified=False
        self.release_motion_locked=True
        for attempt in range(4):
            self.stage='release_open'
            q=self.b.measured()[:6].copy()
            self.b.previous_target[:6]=q
            # The loaded command may still be zero while actual fingers hold a
            # 50 mm object. Start opening from measured width, not a fictitious
            # zero-width stroke; keep the same physical speed/acceleration limits.
            self.b.previous_target[6]=float(self.b.measured()[6])
            try:self.move(q,.10)
            except TimeoutError:
                pass  # Explicitly handled below; never authorizes retreat.
            feedback=release_feedback(self.b)
            if feedback['opened']:
                self.hold(.15)
                if release_feedback(self.b)['opened']:
                    self.release_verified=True;break
            if attempt==3:
                raise ReleaseBlocked('Finger opening unconfirmed; lateral motion remains disabled')
            self.stage='release_unjam_vertical'
            self.vertical_release_move(.005,.10)

    def release_diagonal_retreat(self):
        """Prefer vertical; allow verified upward diagonal with open, clear fingers."""
        if not self.release_verified:raise ReleaseBlocked('Opening must be confirmed before retreat')
        feedback=release_feedback(self.b)
        if feedback['object_contact'] or feedback['tray_contact']:
            raise ReleaseBlocked('Contact remains; diagonal retreat is not safe')
        origin=self.b.data.site_xpos[self.b.model.site('tcp').id].copy()
        yaw=float(getattr(self,'release_yaw',0.))
        for dx,dy in [(-.015,0.),(0.,.015),(0.,-.015),(.015,0.)]:
            try:
                goal=origin+np.array([dx,dy,.025])
                points=self.cartesian_path(self.b.measured()[:6],goal,yaw,current=origin)
                self.validate_transfer(points)
            except RuntimeError:continue
            from piper_sim.trajectory import trajectory
            for command in trajectory(points,self.b.previous_target[6],.10,self.b.interface,self.motion_profile()):self.tick(command)
            return
        raise ReleaseBlocked('No safe upward retreat found; continue local release recovery')

    def require_low_release(self,plan):
        if not self.cfg.get('ready_transport'):return
        sid=self.b.model.site('tcp').id
        tcp=self.b.data.site_xpos[sid]
        delta=tcp-np.asarray(plan['place'])
        if np.linalg.norm(delta[:2])>.015 or delta[2]>.008:
            raise ReleaseBlocked('尚未实际到达低位放置姿态，禁止高处张爪')
        cloud=getattr(self,'carried_visual_cloud',np.empty((0,3)))
        if len(cloud):
            rotation=self.b.data.site_xmat[sid].reshape(3,3)
            bottom=float(np.min((cloud@rotation.T+tcp)[:,2]))
            if bottom>self.table+.009:
                raise ReleaseBlocked('物体估计底部离桌面超过9毫米，禁止抛落')

    def correct_release_height(self,plan):
        """Update a stale release height using the same held visual cloud as planning."""
        cloud=getattr(self,'carried_visual_cloud',np.empty((0,3)))
        if not len(cloud):return
        sid=self.b.model.site('tcp').id;rotation=self.b.data.site_xmat[sid].reshape(3,3)
        bottom=float(np.min((cloud@rotation.T+self.b.data.site_xpos[sid])[:,2]))
        if bottom<=self.table+.009:return
        original=plan.setdefault('original_release_z',float(plan['place'][2]))
        correction=min(.004,bottom-(self.table+.005))
        goal=np.asarray(plan['place']).copy();goal[2]-=correction
        if goal[2]<original-.016:
            raise ReleaseBlocked('低位下放修正超过16毫米，需要换放置姿态，保留夹持')
        yaw=plan.get('place_yaw',plan['yaw'])
        points=self.cartesian_path(self.b.measured()[:6],goal,yaw)
        self.validate_transfer(points)
        if not opening_clearance(self.b,points[-1]):
            raise ReleaseBlocked('继续下放会限制张爪，需要更换放置姿态')
        plan['place']=goal

    def held_transition(self,points):
        """Validated holding motion; preserve the current low clearance while raising."""
        import mujoco
        from piper_sim.trajectory import trajectory
        model=self.b.model;data=mujoco.MjData(model);sid=model.site('tcp').id
        cloud=getattr(self,'carried_visual_cloud',np.empty((0,3)))
        def bottom(q):
            data.qpos[:]=self.b.data.qpos;data.qpos[:6]=q;mujoco.mj_kinematics(model,data)
            if not len(cloud):return data.site_xpos[sid,2]-.04
            return float(np.min((cloud@data.site_xmat[sid].reshape(3,3).T+data.site_xpos[sid])[:,2]))
        floor=max(self.table+.002,min(bottom(self.b.measured()[:6]),self.table+.105)-.002)
        commands=trajectory(points,self.b.previous_target[6],self.b.previous_target[6],self.b.interface,self.motion_profile())
        self.validate_transfer(commands[:,:6])
        if any(bottom(q)<floor for q in commands[:,:6]):raise ReleaseBlocked('持物观察路径会降低物体净空')
        for command in commands:
            command[6]=0.;self.tick(command)
        self.hold(.15)

    def lift_for_fresh_view(self,plan):
        # Preserve measured orientation, not the old plan's destination yaw.
        for distance in [.025,.015]:
            try:
                self.vertical_release_move(distance,0.)
                return
            except (RuntimeError,TimeoutError):continue
        origin=self.b.data.site_xpos[self.b.model.site('tcp').id].copy()
        inward=-origin[:2]/max(np.linalg.norm(origin[:2]),1e-6)
        yaw=plan.get('place_yaw',plan['yaw'])
        for back,rise in [(0.,.025),(.025,.04),(.05,.07),(.08,.10),(.12,.14)]:
            goal=origin.copy();goal[:2]+=inward*back;goal[2]+=rise
            try:
                path=self.cartesian_path(self.b.measured()[:6],goal,yaw)
                self.held_transition(path)
                return
            except (RuntimeError,TimeoutError):continue
        # Free the wrist orientation when a fixed-orientation retreat is the blocker.
        for tilt in [.6,.35,.9,1.2]:
            goal=origin.copy();goal[:2]+=inward*.08;goal[2]+=.10
            try:
                q=IK(self.b,tilt=tilt).solve(goal,yaw,self.b.measured()[:6])
                self.held_transition(np.linspace(self.b.measured()[:6],q,30))
                self.ik=IK(self.b,tilt=tilt)
                return
            except (RuntimeError,TimeoutError):continue
        raise ReleaseBlocked('本轮安全抬升/后退均未通过；保留夹持，不重复固定上抬目标')

    def replan_blocked_release(self,plan):
        if 'final_target' not in plan:raise ReleaseBlocked('需要重新观察目标才能更换释放姿态')
        self.stage='release_replan_lift';self.callback(self.stage)
        self.lift_for_fresh_view(plan)
        self.final_plan_from_ready(plan)
        self.place(plan)

    def release_and_retreat(self,plan):
        self.release_yaw=plan.get('place_yaw',plan['yaw'])
        self.require_low_release(plan)
        self.open_gripper_in_place()
        self.stage='release_lift_vertical'
        lift=self.cfg.get('release_lift_m',.06)
        if self.cfg.get('verify_after_vertical_clearance',False) and self.cfg.get('release_to_observation_height',True):
            z=float(self.b.data.site_xpos[self.b.model.site('tcp').id,2])
            lift=max(lift,max(.20,float(plan['above'][2]))-z)
        # Near workspace limits a 6 cm straight lift may be impossible while a
        # short verified vertical disengagement is feasible. Never force a long lift.
        distances=list(dict.fromkeys([lift]+[d for d in [.04,.02,.01] if d<lift]))
        vertical_failed=False
        for index,distance in enumerate(distances):
            try:self.vertical_release_move(distance,.10);break
            except ReleaseBlocked as exc:
                if not str(exc).startswith('No continuous vertical'):raise
                if index==len(distances)-1:
                    if not self.cfg.get('ready_transport'):raise
                    vertical_failed=True
        if vertical_failed:
            self.release_diagonal_retreat()
        feedback=release_feedback(self.b)
        # Wide/tilted objects can still touch a pad after the nominal 6 cm lift.
        # Keep XY/orientation fixed and check feedback after each extra short lift.
        for _ in range(6):
            if not feedback['opened'] or not (feedback['object_contact'] or feedback['tray_contact']):break
            self.stage='release_lift_vertical'
            self.vertical_release_move(.02,.10)
            feedback=release_feedback(self.b)
        if not feedback['opened'] or feedback['object_contact'] or feedback['tray_contact']:
            raise ReleaseBlocked('Finger contact remains after vertical release; lateral motion disabled')
        self.retreat_verified=True
        self.release_motion_locked=False
        if self.cfg.get('verify_after_vertical_clearance',False):
            self.stage='verify'
            return  # The caller moves directly to the nearest depth-valid local view.
        self.stage='retreat'
        # Every lateral/higher move is gated by measured opening and clear fingers.
        self.cartesian(plan['above'],plan.get('place_yaw',plan['yaw']),.10)

    def run(self, target, destination, tray_floor):
        plan = self.plan(target, destination, tray_floor)
        self.pick(plan)
        return self.place(plan)
