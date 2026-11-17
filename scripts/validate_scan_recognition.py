from web_console import Console,WebFruitSkills,ROOT
import json
out=ROOT/'artifacts/scan-recognition-fixed';app=Console(output_root=out);app.reset('l2',3654645402)
task=dict(action='grouped',category='all',destination='mapping',assignments={'bell_pepper':'red','persimmon':'red','pear':'green','cucumber':'green','potato':'yellow','banana':'yellow'},steps=[dict(source=c) for c in ['bell_pepper','persimmon','pear','cucumber','potato','banana']],scope='selected')
skills=WebFruitSkills(app.world,out,task);skills.record_sensor_frames=False;skills.record_stage_images=False
try:
 tracks=skills.overview();report=dict(status='ready',confirmed=[t.category for t in tracks],navigation=skills.navigation_events)
except Exception as exc:report=dict(status='failed',error=str(exc),navigation=skills.navigation_events)
finally:
 (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=True)[:2000]);app.renderer.close();app.world.close()
