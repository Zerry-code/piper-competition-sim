"""Minimal deployment contract; no ROS, CAN or simulator dependency."""
from typing import Protocol
import numpy as np
from piper_sim.observation import Observation

class RobotPort(Protocol):
    def observe(self) -> Observation: ...
    def measured(self) -> np.ndarray: ...
    def step(self,action): ...
    def close(self): ...

# Real adapters must synchronize RGB/depth/joints/pose through
# observation_from_sensors; action units/ranges follow config/interface.yaml.
# Current PickPlaceSkill also uses MuJoCo FK/contact access. Implementing this
# protocol alone does NOT make that controller hardware-ready.
