"""Replay reported pepper release on the same layout through physical controls."""
import json,time
from web_console import Console,ROOT,WebFruitSkills
from piper_sim.harvest import Harvest
out=ROOT/'artifacts/pepper-release-focused-validation';app=Console(output_root=out);app.reset('l2',501903517)
record=json.loads((ROOT/'artifacts/web-console/scene-7-501903517/task-1.json').read_text(encoding='utf-8'))
skills=WebFruitSkills(app.world,app.output,record['parsed']);skills.record_stage_images=False;skills.record_sensor_frames=False
original_overview=skills.overview
def pepper_overview():return [t for t in original_overview() if t.category=='bell_pepper']
skills.overview=pepper_overview
start=time.monotonic();harvest=Harvest(skills,max_attempts=None,max_deferrals=None)
try:result=harvest.run()
except Exception as exc:result=dict(status='error',error=repr(exc))
finally:
 report=dict(result=result,events=harvest.events,navigation=skills.navigation_events,evaluation=app.world.evaluation(),wall_s=time.monotonic()-start)
 (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=True),flush=True);app.renderer.close();app.world.close()
