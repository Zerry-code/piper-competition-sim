"""Focused physical pear retry, using the same reported seed and latest table gap."""
import json,time
from web_console import Console,ROOT,WebFruitSkills
from piper_sim.harvest import Harvest
out=ROOT/'artifacts/short-route-potato-validation';app=Console(output_root=out);app.reset('l2',3480991575)
task=dict(action='grouped',category='all',destination='mapping',assignments={'potato':'yellow'},steps=[dict(source='potato',destination='yellow')],scope='selected')
skills=WebFruitSkills(app.world,app.output,task);skills.record_stage_images=False;skills.record_sensor_frames=False
original=skills.overview
def overview():
 targets=[t for t in original() if t.category=='potato']
 return targets
skills.overview=overview
start=time.monotonic()
class ValidationStop(BaseException):pass
callback=skills.skill.callback
def tick(stage):
 if time.monotonic()-start>120:raise ValidationStop('test-only watchdog; controller remains unlimited')
 callback(stage)
skills.skill.callback=tick
skills.skill.cfg['deadline_check']=lambda:tick('validation_clock')
harvest=Harvest(skills,max_attempts=None,max_deferrals=None)
try:result=harvest.run()
except BaseException as exc:result=dict(status='validation_stopped',error=repr(exc))
finally:
 report=dict(result=result,events=harvest.events,navigation=skills.navigation_events,evaluation=app.world.evaluation(),wall_s=time.monotonic()-start,final_bounds={o['category']:[v.tolist() for v in app.world.bounds(o['id'])] for o in app.world.objects})
 (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=True),flush=True);app.renderer.close();app.world.close()
