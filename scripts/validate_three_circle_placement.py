"""RGB-D validation of three-circle reservations on the reported banana layout."""
import json,time
import numpy as np
from web_console import Console,ROOT,WebFruitSkills
from piper_sim.fruit_placement import footprint_radius,visual_footprint


def main():
    output=ROOT/'artifacts/three-circle-validation'
    app=Console(output_root=output);app.reset('l2',1464315472)
    task=json.loads((ROOT/'artifacts/web-console/scene-1-1464315472/task-1.json').read_text(encoding='utf-8'))['parsed']
    skills=WebFruitSkills(app.world,app.output,task);app.bind_callback(skills)
    try:
        start=time.monotonic();targets=skills.overview();records=[]
        for t in targets:
            shape=t.placement_footprint
            base=visual_footprint(t.estimate,t.category)
            distances=np.linalg.norm(t.estimate.points[:,:2,None]-np.asarray(getattr(t.estimate,'object_center',t.estimate.center))[:2,None]-base.offsets.T[None,:,:],axis=1)
            assert np.all(np.min(distances-base.radii,axis=1)<=1e-8)
            records.append(dict(category=t.category,footprint=shape.record(),rotation_rad=t.placement_rotation,
                center=t.destination_position.tolist(),margin_m=t.placement_margin_m,
                landing_count=len(t.destination_options),old_enclosing_radius_m=footprint_radius(t.estimate)))
            for other in targets:
                if t is not other:assert not shape.overlaps(t.destination_position,other.placement_footprint,other.destination_position)
        elapsed=time.monotonic()-start
        target=next(t for t in targets if t.category=='banana')
        before=app.world.backend.measured().copy();time_before=app.world.backend.data.time
        try:skills.prepare_target(target,target.estimate);status='found'
        except RuntimeError as exc:status=str(exc)
        np.testing.assert_equal(before,app.world.backend.measured());assert time_before==app.world.backend.data.time
        report=dict(seed=1464315472,overview_wall_s=elapsed,items=records,banana_plan=status,
            planning_status=app.last_planning_report['status'],planning_scope=app.last_planning_report['scope'],
            no_motion_during_planning=True,physical_pick_tested=False)
        if status=='found':report['selected_rotation_rad']=target.placement_rotation
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True),flush=True)
    finally:app.renderer.close();app.world.close()

if __name__=='__main__':main()
