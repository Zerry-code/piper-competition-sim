"""Replay release/retreat with measured finger and TCP traces."""
import argparse
import json
from run_harvest import ROOT,ImplicitWorld,configure,FruitSkills,Harvest,RoundTimedOut
from piper_sim.release_safety import release_feedback


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['fruits','blocks','orange'],default='fruits')
    parser.add_argument('--jam-opening',action='store_true')
    parser.add_argument('--seed',type=int,default=3810926668);args=parser.parse_args()
    output=ROOT/'artifacts/release-regression'/(args.mode+('-jammed' if args.jam_opening else ''));output.mkdir(parents=True,exist_ok=True)
    categories=['orange'] if args.mode=='orange' else ['cube_red','cube_green','cube_blue'] if args.mode=='blocks' else ['apple','pear','banana','lemon','orange','strawberry']
    world=ImplicitWorld('l2',configure(args.seed,categories,output),output,render=True)
    world.cfg['tasks']['l2']['time_limit_s']=180 if args.mode=='blocks' else 300;world.reset(args.seed)
    skills=FruitSkills(world,output,None);harvest=Harvest(skills,max_attempts=None,max_deferrals=None)
    fault=dict(triggered=False,started_s=None,commanded_jam_s=8.)
    if args.jam_opening:
        original_hook=world._before_physics_step
        def jam(backend):
            original_hook(backend)
            if skills.skill.stage=='release_open' and not fault['triggered']:
                fault.update(triggered=True,started_s=float(backend.data.time))
            if fault['triggered'] and backend.data.time-fault['started_s']<fault['commanded_jam_s']:
                # Actuator fault only: keep one finger short of full opening.
                backend.data.ctrl[backend.model.actuator('pos_joint7').id]=.020
        world._before_physics_step=jam
    trace=[];callback=skills.skill.callback
    def observe(stage):
        callback(stage)
        if stage.startswith('release_') or stage=='retreat':
            trace.append(dict(stage=stage,time_s=float(skills.b.data.time),
                tcp=skills.b.data.site_xpos[skills.b.model.site('tcp').id].tolist(),
                release_verified=skills.skill.release_verified,retreat_verified=skills.skill.retreat_verified,
                feedback=release_feedback(skills.b)))
    skills.skill.callback=observe
    try:report=harvest.run()
    except RoundTimedOut:report=dict(status='timed_out')
    except Exception as exc:report=dict(status='error',error=f'{type(exc).__name__}: {exc}')
    report.update(seed=args.seed,evaluation=world.evaluation(),events=harvest.events,navigation=skills.navigation_events,trace=trace,fault=fault)
    (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('navigation','trace','evaluation')},ensure_ascii=True),flush=True)
    world.close()


if __name__=='__main__':main()
