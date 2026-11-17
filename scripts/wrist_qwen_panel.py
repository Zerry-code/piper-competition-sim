"""Local wrist-camera test panel; CPU simulation and an isolated on-demand GPU worker."""

import argparse
import base64
import json
import os
import select
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image

from piper_sim.calibration import ROOT
from piper_sim.competition_world import CompetitionWorld
from piper_sim.wrist_camera import wrist_scene_config


class CameraPanel:
    def __init__(self, output):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.settings, cfg = wrist_scene_config()
        self.config_path = self.output / "scene_config.yaml"
        self.config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
        self.world = CompetitionWorld(
            self.settings["task"], self.config_path, self.output / "scene", render=True
        )
        self.worker = None
        self.worker_log = None
        self.sequence = 0
        self.requests = 0
        self.seed = self.settings["seed"]
        self.world.reset(self.seed)
        self.capture()

    def capture(self):
        self.sequence += 1
        frame = self.output / "frames" / f"{self.sequence:06d}"
        frame.mkdir(parents=True)
        observation = self.world.observe()
        Image.fromarray(observation.pop("rgb")).save(frame / "wrist_rgb.png")
        np.save(frame / "wrist_depth_m.npy", observation.pop("depth_m"))
        metadata = {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in observation.items()
        }
        metadata.update(
            frame_id=self.sequence,
            seed=self.seed,
            mount="wrist",
            parent_body=self.world.cfg["camera"]["parent_body"],
        )
        (frame / "observation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        b = self.world.backend
        with mujoco.Renderer(b.model, height=480, width=640) as renderer:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.32, -0.02, 0.15]
            camera.distance, camera.azimuth, camera.elevation = 1.05, 135, -35
            option = mujoco.MjvOption()
            option.geomgroup[3] = False
            renderer.update_scene(b.data, camera=camera, scene_option=option)
            Image.fromarray(renderer.render().copy()).save(frame / "overview.png")
        self.frame = frame
        self.metadata = metadata
        return self.state()

    def state(self):
        b = self.world.backend

        def data_url(name):
            return (
                "data:image/png;base64,"
                + base64.b64encode((self.frame / name).read_bytes()).decode()
            )

        return {
            "frame_id": self.sequence,
            "seed": self.seed,
            "mount": "wrist",
            "parent_body": self.world.cfg["camera"]["parent_body"],
            "rgb": data_url("wrist_rgb.png"),
            "overview": data_url("overview.png"),
            "q": b.data.qpos[b.qindices].tolist(),
            "limits": b.model.jnt_range[b.joints].tolist(),
            "T_base_camera": self.metadata["T_base_camera"],
            "model_loaded": self.worker is not None and self.worker.poll() is None,
            "frame_path": str(self.frame),
            "robot_control_enabled": False,
        }

    def reset(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        self.seed = seed
        self.world.reset(seed)
        return self.capture()

    def pose(self, q):
        b = self.world.backend
        q = np.asarray(q, dtype=float)
        limits = b.model.jnt_range[b.joints]
        if (
            q.shape != (6,)
            or not np.isfinite(q).all()
            or np.any(q < limits[:, 0])
            or np.any(q > limits[:, 1])
        ):
            raise ValueError("Six finite joint angles within limits are required")
        # Edit a paused simulation pose, not a physical trajectory or robot command.
        b.data.qpos[b.qindices] = q
        b.data.qvel[b.vindices] = 0
        b.previous_target = b.measured()
        b._set_target(b.previous_target)
        mujoco.mj_forward(b.model, b.data)
        return self.capture()

    def release(self):
        if self.worker is not None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait(timeout=5)
            self.worker = None
        if self.worker_log is not None:
            self.worker_log.close()
            self.worker_log = None

    def infer(self, request):
        if request.get("frame_id") != self.sequence:
            raise ValueError("画面已改变，请刷新后重新提交，避免使用旧图。")
        instruction = request.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 2000:
            raise ValueError("请输入1到2000字的指令")
        if self.worker is None or self.worker.poll() is not None:
            self.release()
            env = dict(os.environ)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env.update(
                PYTHONPATH=str(ROOT / "src"),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            self.worker_log = (self.output / "model_worker.log").open("a")
            self.worker = subprocess.Popen(
                [
                    os.environ.get(
                        "QWEN_PYTHON",
                        str(ROOT / ".local/qwen-runtime/bin/python")
                        if (ROOT / ".local/qwen-runtime/bin/python").exists()
                        else sys.executable,
                    ),
                    str(ROOT / "scripts/qwen_camera_worker.py"),
                    str(ROOT / self.settings["model_config"]),
                ],
                cwd=ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.worker_log,
                text=True,
                bufsize=1,
            )
        start = time.perf_counter()
        # Only this wrist RGB path and text are sent; neither overview nor scene truth.
        self.worker.stdin.write(
            json.dumps(
                {"image": str(self.frame / "wrist_rgb.png"), "instruction": instruction},
                ensure_ascii=False,
            )
            + "\n"
        )
        self.worker.stdin.flush()
        readable, _, _ = select.select([self.worker.stdout], [], [], 90)
        if not readable:
            self.release()
            raise RuntimeError("模型请求超时，已释放该模型进程。")
        line = self.worker.stdout.readline()
        if not line:
            self.release()
            raise RuntimeError("模型进程退出，请查看model_worker.log。")
        result = json.loads(line)
        result.update(
            frame_id=self.sequence,
            frame_path=str(self.frame),
            instruction=instruction,
            wall_time_s=time.perf_counter() - start,
            model_loaded=True,
        )
        if result.get("error"):
            self.release()
            result["model_loaded"] = False
        self.requests += 1
        path = self.output / f"request_{self.requests:04d}.json"
        result["result_path"] = str(path)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def close(self):
        self.release()
        self.world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/vlm/wrist_panel" / str(time.time_ns())
    )
    args = parser.parse_args()
    panel = CameraPanel(args.output)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, code=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                data = (ROOT / "assets/ui/wrist_qwen.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/state":
                self.send_json(panel.state())
            else:
                self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 20000:
                    raise ValueError("Invalid request length")
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("JSON required")
                request = json.loads(self.rfile.read(length))
                if self.path == "/api/capture":
                    result = panel.capture()
                elif self.path == "/api/reset":
                    result = panel.reset(request["seed"])
                elif self.path == "/api/pose":
                    result = panel.pose(request["q"])
                elif self.path == "/api/infer":
                    result = panel.infer(request)
                elif self.path == "/api/release":
                    panel.release()
                    result = {"model_loaded": False}
                else:
                    self.send_json({"error": "Not found"}, 404)
                    return
                self.send_json(result)
            except Exception as exc:  # noqa: BLE001 -- local UI error boundary.
                self.send_json({"error": str(exc)}, 400)

    server = HTTPServer((panel.settings["host"], panel.settings["port"]), Handler)
    print(f"Wrist camera panel: http://{panel.settings['host']}:{server.server_port}/", flush=True)
    print(f"Output: {panel.output}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        panel.close()


if __name__ == "__main__":
    main()
