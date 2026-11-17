"""Training/testing interface: normalized action(7), sensor observation, evaluator reward.

Uses Gymnasium reset/step return conventions without requiring Gymnasium itself.
Not a trained policy. Privileged object state is excluded from observations.
"""
from pathlib import Path
from piper_sim.calibration import ROOT
from piper_sim.fruit_world import ImplicitWorld, configure


class FruitEnv:
    def __init__(self,categories=('apple','pear','banana','lemon','orange','strawberry'),
                 output=None,render=True,horizon_s=300):
        self.categories=tuple(categories)
        self.output=Path(output or ROOT/'artifacts/training')
        self.output.mkdir(parents=True,exist_ok=True)
        self.render=render
        self.horizon_s=horizon_s
        self.world=None
        self.count=0

    def reset(self,*,seed=0,options=None):
        self.close()
        directory=self.output/str(seed)
        directory.mkdir(parents=True,exist_ok=True)
        config=configure(seed,list(self.categories),directory)
        self.world=ImplicitWorld('l2',config,directory,render=self.render)
        self.world.cfg['tasks']['l2']['time_limit_s']=self.horizon_s
        observation=self.world.reset(seed)
        self.count=0
        return observation,{'seed':seed,'reward_source':'privileged_evaluator_only'}

    def step(self,action):
        if self.world is None:raise RuntimeError('Call reset before step')
        import numpy as np
        action=np.asarray(action,dtype=float)
        if action.shape!=(7,) or not np.isfinite(action).all() or np.any(abs(action)>1):
            raise ValueError('Expected finite normalized action(7) in [-1,1]')
        observation=self.world.step(action)
        result=self.world.evaluation()
        count=len(result['correctly_placed'])
        reward=float(count-self.count)
        self.count=count
        terminated=count==len(self.categories)
        truncated=result['timed_out'] and not terminated
        return observation,reward,terminated,truncated,{
            'correctly_placed_count':count,'sim_time_s':result['time_s']}

    def close(self):
        if self.world:self.world.close()
        self.world=None
