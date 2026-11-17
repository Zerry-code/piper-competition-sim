from types import SimpleNamespace

import numpy as np

from piper_sim.demo_skill import TargetEstimate
from piper_sim.qwen_adapter import parse_prediction
from piper_sim.skill_runtime import (
    ManipulationSkills,
    SkillCoordinator,
    SkillResult,
    destination_text,
)


def test_terse_abstention_never_becomes_ready():
    result = parse_prediction('{"status":"ambiguous"}', (640, 480))
    assert result == {"status": "ambiguous", "target_pixel_xy": None, "destination_color": None}


def test_clipped_target_requires_new_view_and_stale_frame_is_rejected():
    mask = np.zeros((40, 40), dtype=bool)
    mask[:10, 10:20] = True
    target = TargetEstimate(np.array([0.4, 0, 0.08]), 0.04, 0, mask, np.empty((0, 3)), [15, 5], 0)
    port = SimpleNamespace(
        settings={"observation_q_rad": [0] * 6},
        cfg={},
        sequence=4,
        stage=lambda *args: None,
        show_perception=lambda *args: None,
    )
    skills = ManipulationSkills(port, None)
    task = {"frame_id": 4, "target_pixel_xy": [15, 5]}
    detections = {"frame_id": 4, "candidates": [target]}
    assert (
        skills.resolve_target(task, detections, {"frame_id": 4, "observation": None}).code
        == "target_clipped"
    )
    assert skills.resolve_target(task, detections, {"frame_id": 3}).code == "stale_frame"


def test_scheduler_reobserves_and_redetects_before_grasp_after_uncertainty():
    calls = []

    class Skills:
        port = SimpleNamespace(bind_evaluator=lambda target: "eval-only-token")

        def __init__(self):
            self.trace = []

        def call(self, name, **args):
            calls.append(name)
            if name == "observe_scene":
                return SkillResult("ok", {"view": args["view_index"]})
            if name == "interpret_task" and args.get("observation") is None:
                return SkillResult("ok")
            if name == "interpret_task" and calls.count(name) == 2:
                return SkillResult("retryable", code="ambiguous")
            if name == "recover_or_continue":
                return SkillResult("ok", {"next_view": 1})
            if name == "resolve_target":
                return SkillResult("ok", {"target": "sensor-target"})
            if name == "verify_result":
                assert args["binding"] == "eval-only-token"
                return SkillResult("ok", {"success": True})
            if name == "pick_object":
                assert args["target"] == "sensor-target"
                assert "binding" not in args
            return SkillResult("ok")

    assert SkillCoordinator(Skills()).run({})["success"]
    assert calls[1:9] == [
        "observe_scene",
        "detect_objects",
        "interpret_task",
        "recover_or_continue",
        "observe_scene",
        "detect_objects",
        "interpret_task",
        "resolve_target",
    ]
    assert calls[-3:] == ["pick_object", "place_object", "verify_result"]


def test_recovery_is_bounded_for_views_and_empty_grasps():
    port = SimpleNamespace(
        settings={"observation_q_rad": [0] * 6},
        cfg={"observation_search_q_rad": [[0] * 6], "max_grasp_attempts": 2},
        stage=lambda *args: None,
    )
    skills = ManipulationSkills(port, None)
    uncertain = SkillResult("retryable", code="ambiguous")
    assert skills.recover_or_continue(uncertain, 0, 0).data["next_view"] == 1
    assert skills.recover_or_continue(uncertain, 1, 0).status == "failed"
    empty = SkillResult("retryable", code="empty_grasp")
    assert skills.recover_or_continue(empty, 1, 1).status == "ok"
    assert skills.recover_or_continue(empty, 1, 2).status == "failed"


def test_destination_preflight_does_not_guess_from_object_color():
    assert destination_text("把绿苹果放到托盘里") == (None, True)
    assert destination_text("把最右边的水果放到托盘里") == (None, True)
    assert destination_text("把红色方块放到蓝色托盘") == ("blue", False)
    assert destination_text("把柠檬放到托盘里；目的地是绿色托盘。") == ("green", False)


def test_missing_destination_stops_before_camera_or_gpu_calls():
    port = SimpleNamespace(
        settings={"observation_q_rad": [0] * 6},
        cfg={},
        scene_cfg={"trays": {"red": {}, "blue": {}, "green": {}}},
        stage=lambda *args: None,
        checkpoint=lambda: None,
        update=lambda **kw: None,
    )
    skills = ManipulationSkills(port, None)
    result = SkillCoordinator(skills).run({"instruction": "把柠檬放到托盘里", "backend": "qwen"})
    assert result["status"] == "needs_clarification"
    assert result["clarification"]["choices"] == ["red", "blue", "green"]
    assert [row["skill"] for row in skills.trace] == ["interpret_task"]


def test_missing_depth_uses_distance_recovery_instead_of_adjacent_angle():
    port = SimpleNamespace(
        settings={"observation_q_rad": [0] * 6},
        cfg={"observation_search_q_rad": [[0] * 6, [0] * 6], "depth_recovery_view_index": 2},
        stage=lambda *args: None,
    )
    skills = ManipulationSkills(port, None)
    result = skills.recover_or_continue(SkillResult("retryable", code="missing_target_depth"), 0, 0)
    assert result.data == {"decision": "increase_camera_distance", "next_view": 2}


def test_clipped_target_recenter_is_bounded_then_falls_back_to_search():
    port = SimpleNamespace(
        settings={"observation_q_rad": [0] * 6},
        cfg={"observation_search_q_rad": [[0] * 6], "active_view": {"max_attempts": 1}},
        stage=lambda *args: None,
    )
    motion = SimpleNamespace(recenter=lambda obs, pts: (np.ones(6) * 0.1, {"source": "depth"}))
    skills = ManipulationSkills(port, motion)
    clipped = SkillResult(
        "retryable", {"observation": "fresh", "points": "visible"}, code="target_clipped"
    )
    first = skills.recover_or_continue(clipped, 0, 0)
    assert first.data["decision"] == "recenter_camera"
    assert first.data["next_view"] == 0
    np.testing.assert_allclose(skills.pending_view_q, 0.1)
    assert skills.recover_or_continue(clipped, 0, 0).data["next_view"] == 1
    assert skills.recover_or_continue(clipped, 1, 0).status == "failed"


def test_recenter_failure_falls_back_without_using_partial_geometry_for_grasp():
    def reject(*args):
        raise ValueError("unsafe path")

    port = SimpleNamespace(
        settings={"observation_q_rad": [0] * 6},
        cfg={"observation_search_q_rad": [[0] * 6]},
        stage=lambda *args: None,
    )
    skills = ManipulationSkills(port, SimpleNamespace(recenter=reject))
    result = skills.recover_or_continue(
        SkillResult(
            "retryable", {"observation": "fresh", "points": "visible"}, code="target_clipped"
        ),
        0,
        0,
    )
    assert result.data["decision"] == "change_view"
    assert skills.pending_view_q is None


def test_new_instruction_observes_current_pose_without_motion():
    def unexpected(*args):
        raise AssertionError("Must capture first without moving the robot")

    fresh = SimpleNamespace(joint_position_rad=np.zeros(6))
    port = SimpleNamespace(
        settings={"observation_q_rad": [0] * 6},
        cfg={},
        start_from_current=True,
        stage=lambda *args: None,
        sequence=7,
        capture=lambda: fresh,
    )
    skills = ManipulationSkills(port, SimpleNamespace(move=unexpected, stage=""))
    result = skills.observe_scene(0)
    assert result.data["observation"] is fresh
    assert result.data["view_index"] == -1
    assert result.summary["from_current_pose"]
