import json
from web_console import Console,ROOT,WebFruitSkills
out=ROOT/'artifacts/pair-partition-validation';app=Console(output_root=out);app.reset('l2',3697363034)
task=dict(assignments={'bell_pepper':'red','persimmon':'red','pear':'green','cucumber':'green','banana':'yellow','potato':'yellow','apple':'yellow'},steps=[dict(source=c) for c in ['bell_pepper','persimmon','pear','cucumber']])
skills=WebFruitSkills(app.world,app.output,task);skills.record_stage_images=False;skills.record_sensor_frames=False
try:
 tracks=skills.overview();report=dict(status='ready',visible_objects=[o['category'] for o in app.world.objects],selected=[t.category for t in tracks],unselected=task.get('unselected_categories'),placements=[dict(category=t.category,destination=t.destination,position=t.destination_position.tolist(),partition=t.placement_partition) for t in tracks])
 assert len(report['visible_objects'])==7 and len(tracks)==6
 assert all(sum(t.destination==c for t in tracks)==2 for c in ['red','green','yellow'])
except Exception as exc:report=dict(status='error',error=repr(exc))
finally:
 (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=True));app.renderer.close();app.world.close()
