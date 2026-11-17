import json,time
from web_console import Console,WebFruitSkills,ROOT
from piper_sim.harvest import Harvest
original=json.loads((ROOT/'artifacts/web-console/scene-1-4159889904/task-1.json').read_text(encoding='utf-8'))
out=ROOT/'artifacts/cucumber-round-validation';app=Console(output_root=out);app.reset('l2',4159889904)
s=WebFruitSkills(app.world,out,original['parsed']);s.record_sensor_frames=False;s.record_stage_images=False
start=time.monotonic()
class ValidationStop(BaseException):pass
callback=s.skill.callback
last=[None]
def tick(stage):
 if time.monotonic()-start>260:raise ValidationStop('test-only watchdog')
 if stage!=last[0]:
  if stage in ['plan_from_ready','release_reobserve','close','release_open']:print(round(time.monotonic()-start,1),stage,flush=True)
  last[0]=stage
 callback(stage)
s.skill.callback=tick
h=Harvest(s,max_attempts=None,max_deferrals=None)
try:result=h.run()
except BaseException as exc:result=dict(status='not_complete',error=repr(exc))
finally:
 report=dict(result=result,events=h.events,navigation=s.navigation_events,wall_s=time.monotonic()-start)
 (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=True)[:1000]);app.renderer.close();app.world.close()
