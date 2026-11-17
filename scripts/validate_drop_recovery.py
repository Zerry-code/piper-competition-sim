"""Physically induce an unexpected finger opening; never move an object pose."""
import argparse
import json
import numpy as np
from run_harvest import ROOT,ImplicitWorld,configure,FruitSkills,Harvest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenario',choices=['normal','table','correct','wrong'],required=True)
    args=parser.parse_args()
    output=ROOT/'artifacts/drop-recovery'/args.scenario;output.mkdir(parents=True,exist_ok=True)
    world=ImplicitWorld('l2',configure(160,['pear'],output),output,render=True);world.reset(160)
    skills=FruitSkills(world,output,None);harvest=Harvest(skills)
    fault=dict(triggered=False,started=None)
    normal_hook=world._before_physics_step
    # The fault injector knows only the planned fixture zone and the robot's FK.
    # It overrides actuator commands, not fruit positions, contacts or friction.
    goal=np.array(world.cfg['trays']['blue' if args.scenario=='wrong' else 'green']['center_xy_m'])
    if args.scenario=='wrong':
        original_place=skills.skill.place
        def wrong_route(plan):
            diverted=dict(plan);diverted['above']=plan['above'].copy();diverted['above'][:2]=goal
            return original_place(diverted)
        skills.skill.place=wrong_route
    def hook(backend):
        normal_hook(backend)
        position=backend.data.site_xpos[backend.model.site('tcp').id]
        trigger=(args.scenario=='table' and skills.skill.stage=='lift' and position[2]>.15) or (
            args.scenario=='correct' and skills.skill.stage=='place_approach') or (
            args.scenario=='wrong' and skills.skill.stage=='transport' and np.linalg.norm(position[:2]-goal)<.025)
        if trigger and not fault['triggered']:
            fault.update(triggered=True,started=float(backend.data.time))
        if fault['triggered'] and backend.data.time-fault['started']<.5:
            # Inject an unexpected physical opening of both fingers for half a second.
            for name,value in [('pos_joint7',.035),('pos_joint8',-.035)]:
                backend.data.ctrl[backend.model.actuator(name).id]=value
    world._before_physics_step=hook
    try:
        result=harvest.run()
    except Exception as exc:
        result=dict(status='error',error=f'{type(exc).__name__}: {exc}',events=harvest.events)
    result.update(scenario=args.scenario,fault=fault,evaluation=world.evaluation(),navigation=skills.navigation_events)
    result['validation_only_final_bounds']={o['id']:[p.tolist() for p in world.bounds(o['id'])] for o in world.objects}
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('navigation','evaluation')},ensure_ascii=True),flush=True)
    world.close()


if __name__=='__main__':main()
