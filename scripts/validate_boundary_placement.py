import json
from web_console import Console,ROOT,WebFruitSkills,Harvest

def main():
    output=ROOT/'artifacts/banana-boundary-check';app=Console(output_root=output);app.reset('l2',545426603)
    task=dict(category='all',destination='mapping',assignments=dict(pear='green',cucumber='green',banana='yellow',persimmon='red',potato='yellow'))
    skills=WebFruitSkills(app.world,app.output,task);skills.record_stage_images=False;skills.record_sensor_frames=False
    harvest=Harvest(skills,max_attempts=10,max_deferrals=2,max_preparations_per_target=4)
    try:
        tracks=skills.overview();skills.overview=lambda:[next(t for t in tracks if t.category=='banana')]
        try:report=harvest.run()
        except Exception as exc:report=dict(status='error',error=str(exc))
        report.update(events=harvest.events,navigation=skills.navigation_events,evaluation=app.world.evaluation())
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('navigation','evaluation')},ensure_ascii=True),flush=True)
    finally:app.renderer.close();app.world.close()
if __name__=='__main__':main()
