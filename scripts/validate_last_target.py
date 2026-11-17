"""Replay the incomplete six-fruit layout without overwriting the user's run."""
import json
from run_harvest import ROOT,ImplicitWorld,configure,FruitSkills,Harvest,RoundTimedOut


def main():
    output=ROOT/'artifacts/last-target-regression';output.mkdir(exist_ok=True)
    world=ImplicitWorld('l2',configure(387630622,['apple','pear','banana','lemon','orange','strawberry'],output),output,render=True)
    world.cfg['tasks']['l2']['time_limit_s']=300;world.reset(387630622)
    skills=FruitSkills(world,output,None);harvest=Harvest(skills,max_attempts=None,max_deferrals=None)
    try:report=harvest.run()
    except RoundTimedOut:report=dict(status='timed_out',stop_reason='competition_time_limit')
    except Exception as exc:report=dict(status='error',error=f'{type(exc).__name__}: {exc}')
    report.update(seed=387630622,events=harvest.events,evaluation=world.evaluation(),
        navigation=skills.navigation_events,completed=getattr(harvest,'completed',[]))
    skills.snapshot('validation_finished') if world.elapsed_s<300 else None
    (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='navigation'},ensure_ascii=True),flush=True)
    world.close()


if __name__=='__main__':main()
