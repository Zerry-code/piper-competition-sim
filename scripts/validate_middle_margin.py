"""Physical banana/cucumber trials with middle grasp and inset placement."""
import json
import numpy as np
from web_console import Console,ROOT,WebFruitSkills,Harvest

def main():
    results=[]
    for category in ['banana','cucumber']:
        output=ROOT/'artifacts/middle-grasp-margin'/category
        app=Console(output_root=output);app.reset('l2',545426603)
        task=dict(category='all',destination='mapping',assignments=dict(pear='green',cucumber='green',banana='yellow',persimmon='red',potato='yellow'))
        skills=WebFruitSkills(app.world,app.output,task);skills.record_stage_images=False;skills.record_sensor_frames=False
        harvest=Harvest(skills,max_attempts=10,max_deferrals=2,max_preparations_per_target=4)
        try:
            try:
                tracks=skills.overview();target=next(t for t in tracks if t.category==category)
                skills.overview=lambda:[target]
                report=harvest.run()
                obj=next(o for o in app.world.objects if o['category']==category)
                low,high=app.world.bounds(obj['id']);tray=app.world.cfg['trays'][target.destination]
                center=np.array(tray['center_xy_m']);half=np.array(tray['half_size_xy_m'])
                clearance=float(min(np.min(low[:2]-(center-half)),np.min(center+half-high[:2])))
                report.update(category=category,planned_clearance_m=target.placement_margin_m,physical_clearance_m=clearance,sim_s=app.world.elapsed_s)
            except Exception as exc:report=dict(category=category,status='error',error=str(exc))
            report.update(events=harvest.events,navigation=skills.navigation_events,evaluation=app.world.evaluation())
            results.append(report)
            (ROOT/'artifacts/middle-grasp-margin/result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in report.items() if k not in ('navigation','evaluation','events')},ensure_ascii=True),flush=True)
        finally:app.renderer.close();app.world.close()
if __name__=='__main__':main()
