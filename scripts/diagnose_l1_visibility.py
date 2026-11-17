"""Diagnose L1 stress-profile visibility on the reported failed seed."""
import json,time
import numpy as np
from web_console import Console,ROOT
from run_l1 import L1Skills
from piper_sim.l1_world import detect_card_targets
from piper_sim.calibration import camera_to_base


def main():
    output=ROOT/'artifacts/l1-visibility-diagnosis';app=Console(output_root=output);app.gap_profile='stress';app.reset('l1',3102957277)
    skills=L1Skills(app.world,app.output,None);skills.record_stage_images=False;skills.record_sensor_frames=False
    original=skills.capture_candidates;frames=[]
    def capture(*args,**kwargs):
        candidates=original(*args,**kwargs);obs=skills.last_observation
        z=obs.depth_m;k=obs.intrinsics;v,u=np.indices(z.shape);valid=np.isfinite(z)
        pixels=np.stack([(u-k['cx'])*z/k['fx'],(v-k['cy'])*z/k['fy'],z],-1)
        points=camera_to_base(pixels[valid],obs.T_base_end,skills.calibration)
        pool=points[abs(points[:,2]-skills.skill.table)<.025]
        r=dict(frame=len(frames)+1,cubes=[c for c,e in candidates],cards=sorted(detect_card_targets(obs,skills.calibration,skills.skill.table)),
            known_cards=sorted(skills.card_targets),plane_z_quantiles=np.percentile(pool[:,2]-skills.skill.table,[10,50,90]).tolist() if len(pool) else [],seen_floor_cells=len(skills.scene_memory.seen_plane))
        frames.append(r);print(json.dumps(r),flush=True)
        return candidates
    skills.capture_candidates=capture
    try:
        skills.scan(['cube_red','cube_green','cube_yellow'],['red','green','yellow'],inventory=True);status='ready'
    except Exception as exc:status=repr(exc)
    finally:
        (output/'result.json').write_text(json.dumps(dict(status=status,frames=frames,navigation=skills.navigation_events),ensure_ascii=False,indent=2),encoding='utf-8')
        print(status,flush=True);app.renderer.close();app.world.close()

if __name__=='__main__':main()
