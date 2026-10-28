"""Single-seed heterogeneous experiment entry."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.launcher import single_main
from core.configuration import load_config

ENABLED_METHODS = ["fedwikd", "heterofl", "feddf", "fedfd", "local_teacher"]


def build_default_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/heterogeneous.json")


if __name__ == "__main__":
    single_main("heterogeneous", build_default_config, ENABLED_METHODS)
