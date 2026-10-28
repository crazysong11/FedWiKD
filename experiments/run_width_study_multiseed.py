"""Run the width study for each seed in SEED_POOL."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.launcher import multi_main

SEED_POOL = [42, 43, 44, 45, 46]

if __name__ == "__main__":
    multi_main("width_study", SEED_POOL)
