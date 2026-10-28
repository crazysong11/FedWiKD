"""Read experiment settings, with optional inheritance from another JSON file."""

from copy import deepcopy
import json
from pathlib import Path


def merge_config(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, _parents=()):
    path = Path(path).expanduser().resolve()
    if path in _parents:
        raise ValueError(f"Circular configuration inheritance: {path}")
    settings = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    parent = settings.pop("extends", None)
    if parent is None:
        return settings
    inherited = load_config(path.parent / parent, (*_parents, path))
    return merge_config(inherited, settings)
