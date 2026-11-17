"""Physical pear drop followed by an injected local-observation timeout."""
import json,argparse
from web_console import ROOT,DocumentFruitWorld,configure_fruits,WebFruitSkills,Harvest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=1791236247);args=parser.parse_args()
    seed=args.seed
    output=ROOT/('artifacts/pear-retry-recovery' if seed==1791236247 else 'artifacts/pear-retry-reported-layout');output.mkdir(exist_ok=True)
    world=DocumentFruitWorld('l2',configure_fruits(seed,output),output,render=True)
    world.cfg['tasks']['l2']['time_limit_s']=300;world.reset(seed)
    skills=WebFruitSkills(world,output,dict(category='pear',destination='green'))
    skills.record_stage_images=False
    fault=dict(opened=False,started=None,timeout=False)
    hook=world._before_physics_step
    def inject(backend):
        hook(backend)
        z=backend.data.site_xpos[backend.model.site('tcp').id,2]
        if skills.skill.stage=='lift' and z>.15 and not fault['opened']:
            fault.update(opened=True,started=float(backend.data.time))
        if fault['opened'] and backend.data.time-fault['started']<.5:
            backend.data.ctrl[backend.model.actuator('pos_joint7').id]=.035
            backend.data.ctrl[backend.model.actuator('pos_joint8').id]=-.035
    world._before_physics_step=inject
    original_recover=skills.recover_dropped
    def recover(target):
        outcome=original_recover(target)
        if outcome=='retry':target.drop_estimate_ready=False
        return outcome
    skills.recover_dropped=recover
    original_move=skills.skill.move
    def move(q,width):
        if fault['opened'] and skills.skill.stage=='local_observe' and not fault['timeout']:
            fault['timeout']=True
            raise TimeoutError('Injected local observation timeout after pear drop')
        return original_move(q,width)
    skills.skill.move=move
    harvest=Harvest(skills,max_attempts=20,max_deferrals=None)
    try:
        try:report=harvest.run()
        except Exception as exc:report=dict(status='error',error=f'{type(exc).__name__}: {exc}')
        report.update(fault=fault,events=harvest.events,evaluation=world.evaluation(),navigation=skills.navigation_events)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('navigation','evaluation')},ensure_ascii=True),flush=True)
    finally:world.close()


if __name__=='__main__':main()
