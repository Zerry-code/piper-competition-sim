"""Exercise all fruit targets with shuffled three-card positions."""
import json
from web_console import ROOT,DocumentFruitWorld,configure_fruits,WebFruitSkills,Harvest

def main():
    seed=1791236247
    output=ROOT/'artifacts/random-layout-fruit-round';output.mkdir(exist_ok=True)
    world=DocumentFruitWorld('l2',configure_fruits(seed,output),output,render=True)
    world.cfg['tasks']['l2']['time_limit_s']=300;world.reset(seed)
    skills=WebFruitSkills(world,output,dict(category='all',destination='mapping'))
    skills.record_stage_images=False
    harvest=Harvest(skills,max_attempts=30,max_deferrals=None)
    try:
        try:report=harvest.run()
        except Exception as exc:report=dict(status='error',error=f'{type(exc).__name__}: {exc}')
        report.update(events=harvest.events,evaluation=world.evaluation(),navigation=skills.navigation_events)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('navigation','evaluation')},ensure_ascii=True),flush=True)
    finally:world.close()

if __name__=='__main__':main()
