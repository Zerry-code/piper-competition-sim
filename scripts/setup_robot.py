"""Prepare the vendored robot assets. Does not fetch or check out Git repositories."""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    subprocess.run([sys.executable, str(ROOT/'scripts/prepare_assets.py')], check=True, cwd=ROOT)
