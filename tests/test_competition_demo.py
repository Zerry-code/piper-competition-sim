from types import SimpleNamespace

import yaml

from piper_sim.calibration import ROOT
from piper_sim.competition_demo import round_requests, run_round
from piper_sim.skill_runtime import ManipulationSkills


def config():
    return yaml.safe_load((ROOT / "config/competition/scene.yaml").read_text())


def test_l1_uses_three_public_color_rules_and_l2_uses_visual_classification():
    cfg = config()
    l1 = round_requests("l1", cfg, {})
    assert len(l1) == 3 and all(r["backend"] == "rule" for r in l1)
    assert l1[0]["instruction"] == "把红色方块放到红色托盘"
    l2 = round_requests("l2", cfg, {"l2_max_attempts": 7})
    assert len(l2) == 7 and all(r["classification"] for r in l2)
    assert "苹果→红色托盘" in l2[0]["instruction"]
    assert "不要选择已在托盘内" in l2[0]["instruction"]
    skills = ManipulationSkills(
        SimpleNamespace(settings={"observation_q_rad": [0] * 6}, cfg={}, stage=lambda *args: None),
        None,
    )
    assert skills.interpret_task(l2[0]).status == "ok"


def test_round_continues_after_failed_object_and_never_resets_between_objects(tmp_path):
    calls = []
    world = SimpleNamespace(elapsed_s=0.0)
    completed = []

    def begin():
        calls.append("begin")

    world.begin_attempt = begin
    world.evaluation = lambda: {"correctly_placed": list(completed)}

    def execute(request, new_attempt):
        assert new_attempt is False
        calls.append(request)
        world.elapsed_s += 10
        success = len(calls) > 2
        if success:
            completed.append(str(len(calls)))
        return {"success": success, "message": "verified" if success else "not placed"}

    port = SimpleNamespace(
        task_mode="l1",
        scene_cfg=config(),
        cfg={},
        world=world,
        output=tmp_path,
        seed=42,
        checkpoint=lambda: None,
        update=lambda **kw: None,
        stage=lambda *args: None,
        execute=execute,
    )
    result = run_round(port)
    assert calls.count("begin") == 1 and len(calls) == 4
    assert result["completed"] == 2 and result["target_count"] == 3
    assert not result["success"]
    assert len(list((tmp_path / "rounds").glob("*.json"))) == 1


def test_round_stops_on_model_failure_instead_of_claiming_completion(tmp_path):
    cfg = config()
    calls = []
    world = SimpleNamespace(
        elapsed_s=0.0, begin_attempt=lambda: None, evaluation=lambda: {"correctly_placed": []}
    )

    def execute(request, new_attempt):
        calls.append(request)
        return {"success": False, "failure_code": "model_unavailable", "error": "unavailable"}

    port = SimpleNamespace(
        task_mode="l2",
        scene_cfg=cfg,
        cfg={},
        world=world,
        output=tmp_path,
        seed=42,
        checkpoint=lambda: None,
        update=lambda **kw: None,
        stage=lambda *args: None,
        execute=execute,
    )
    assert not run_round(port)["success"]
    assert len(calls) == 1
