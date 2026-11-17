import time,json
from web_console import Console,WebFruitSkills,ROOT
out=ROOT/'artifacts/fresh-release-physical';app=Console(output_root=out);app.reset('l2',4060984770)
task=dict(action='grouped',category='all',destination='mapping',assignments={'persimmon':'red'},steps=[dict(source='persimmon',destination='red')],scope='selected')
s=WebFruitSkills(app.world,out,task);s.record_stage_images=False;s.record_sensor_frames=False
start=time.monotonic()
class StopValidation(BaseException):pass
original=s.skill.callback
def tick(stage):
 if time.monotonic()-start>100:raise StopValidation('test watchdog only')
 original(stage)
s.skill.callback=tick
try:
 target=s.overview()[0];s.observe_target(target,False)
 assert s.pick(target)
 from piper_sim.release_safety import ReleaseBlocked
 result=dict(status='not_complete')
 for attempt in range(4):
  target.release_retries=2+attempt
  try:
   if s.refresh_held_landing(target):result=dict(status='complete',fresh_attempts=attempt+1);break
  except ReleaseBlocked as exc:print('retry',attempt+1,str(exc),flush=True)
except BaseException as exc:result=dict(status='not_complete',error=repr(exc))
finally:
 report=dict(result=result,wall_s=time.monotonic()-start,navigation=s.navigation_events)
 (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result));app.renderer.close();app.world.close()
