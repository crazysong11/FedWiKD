"""Student-width study under the homogeneous experiment configuration."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.configuration import load_config
from core.launcher import single_main

WIDTHS = list(range(0, 101, 10))
ENABLED_METHODS = ["fedwikd"]


def build_default_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/width_study.json")


if __name__ == "__main__":
    single_main("width_study", build_default_config, ENABLED_METHODS, WIDTHS)
