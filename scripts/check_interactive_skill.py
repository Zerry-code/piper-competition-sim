"""Finite physics check of RGB-D target -> IK grasp -> placement, without model loading."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from piper_sim.calibration import load_config
from piper_sim.competition_world import CompetitionWorld
from piper_sim.demo_skill import PickPlaceSkill, candidates_from_rgbd, select_rule
from piper_sim.wrist_camera import wrist_scene_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", default="把红色方块放到蓝色托盘")
    parser.add_argument("--output", type=Path, default=Path("artifacts/interactive_demo/check"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _, cfg = wrist_scene_config()
    cfg["spawn"].update(x_m=[0.30, 0.47], y_m=[-0.10, 0.10])
    path = args.output / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    world = CompetitionWorld("l1", path, args.output, render=True)
    try:
        world.reset(args.seed)
        b = world.backend
        calibration = load_config()
        candidates = candidates_from_rgbd(b.observe(), calibration, cfg["table"]["top_m"])
        target, color = select_rule(candidates, b.observe().rgb, args.instruction)
        print(
            "sensor target",
            target.center,
            "height",
            target.height,
            "destination",
            color,
            flush=True,
        )
        previous = [None]

        def callback(stage):
            if stage != previous[0]:
                print(stage, round(b.data.time, 2), flush=True)
                previous[0] = stage

        skill = PickPlaceSkill(
            b,
            cfg["table"]["top_m"],
            callback,
            {"transport_height_m": 0.22},
            stepper=lambda action: world.step(action, observe=False),
        )
        error = None
        try:
            plan = skill.run(
                target, cfg["trays"][color]["center_xy_m"], cfg["table"]["top_m"] + 0.004
            )
        except (RuntimeError, TimeoutError) as exc:
            error = str(exc)
            plan = None
        result = {"error": error, "plan": plan, "evaluation": world.evaluation()}
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        events = result["evaluation"]["events"]
        if error or not any(e["grasped"] and color in e["placed"] for e in events.values()):
            sys.exit(1)
    finally:
        world.close()


if __name__ == "__main__":
    main()
