import json
from web_console import Console,ROOT,WebFruitSkills,Harvest

def main():
    app=Console(output_root=ROOT/'artifacts/stress-fruit-progress');app.gap_profile='stress';app.reset('l2',732000359)
    task=dict(category='all',destination='mapping',assignments=dict(banana='yellow',persimmon='yellow',cucumber='green',pear='green',bell_pepper='red',potato='red'))
    skills=WebFruitSkills(app.world,app.output,task);skills.record_stage_images=False
    harvest=Harvest(skills,max_attempts=30,max_deferrals=2)
    original=skills.observe_target
    def observe(target,retry):
        try:return original(target,retry)
        except Exception as exc:print(target.key,str(exc),flush=True);raise
    skills.observe_target=observe
    try:
        try:report=harvest.run()
        except Exception as exc:report=dict(status='error',error=str(exc))
        report.update(events=harvest.events,navigation=skills.navigation_events,evaluation=app.world.evaluation())
        (ROOT/'artifacts/stress-fruit-progress/result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('events','navigation','evaluation')},ensure_ascii=True),flush=True)
    finally:app.renderer.close();app.world.close()
if __name__=='__main__':main()
