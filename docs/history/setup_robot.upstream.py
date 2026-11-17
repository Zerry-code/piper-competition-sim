"""Fetch the pinned official PiPER description and generate the local MuJoCo model."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/agilexrobotics/piper_ros.git"
COMMIT = "ac41fcbcdda598f01b51cf6175ed9a24d0dacadc"
target = ROOT / "references/upstream/piper_ros"
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    subprocess.run(["git", "init", str(target)], check=True)
subprocess.run(["git", "-C", str(target), "fetch", "--depth=1", URL, COMMIT], check=True)
subprocess.run(["git", "-C", str(target), "checkout", "--detach", COMMIT], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/build_piper_model.py")], cwd=ROOT, check=True)
