"""Seven measured props, reported seed, grouped physical pick/place verification."""
import json,time
from web_console import Console,ROOT,WebFruitSkills
from piper_sim.harvest import Harvest
out=ROOT/'artifacts/seven-fruit-verified-validation';app=Console(output_root=out);app.reset('l2',3697363034)
assignments={'bell_pepper':'red','persimmon':'red','pear':'green','cucumber':'green','banana':'yellow','potato':'yellow','apple':'yellow'}
skills=WebFruitSkills(app.world,app.output,dict(assignments=assignments,expected_categories=list(assignments)))
skills.record_stage_images=False;skills.record_sensor_frames=False
original=skills.progress
def progress(target,state,reason):
 original(target,state,reason);print(target.key,state,round(app.world.elapsed_s,2),reason,flush=True)
skills.progress=progress
start=time.monotonic();harvest=Harvest(skills,max_attempts=None,max_deferrals=None)
try:result=harvest.run()
except Exception as exc:result=dict(status='error',error=repr(exc))
finally:
 report=dict(result=result,events=harvest.events,navigation=skills.navigation_events,evaluation=app.world.evaluation(),final_bounds={o['category']:[v.tolist() for v in app.world.bounds(o['id'])] for o in app.world.objects},wall_s=time.monotonic()-start)
 (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=True),flush=True);app.renderer.close();app.world.close()
