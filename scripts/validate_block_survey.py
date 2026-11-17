from web_console import Console,ROOT
from run_l1 import L1Skills
import json
out=ROOT/'artifacts/block-survey-fixed';app=Console(initial_mode='l1',output_root=out);app.reset('l1',4202376486)
s=L1Skills(app.world,out,None);s.record_stage_images=False;s.record_sensor_frames=False
try:
 s.scan(['cube_red','cube_green','cube_yellow'],['red','green','yellow'],inventory=True)
 report=dict(status='ready',confirmed=list(s.scene_memory.objects),navigation=s.navigation_events)
except Exception as exc:report=dict(status='failed',error=repr(exc),navigation=s.navigation_events)
finally:
 (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='navigation'}));app.renderer.close();app.world.close()
