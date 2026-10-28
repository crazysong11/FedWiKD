"""Locate completed experiments without bundling or modifying their data."""

from __future__ import annotations
import csv
import json
from pathlib import Path

METHODS = {
    "homogeneous": ("fedwikd", "fedavg", "fedprox", "scaffold"),
    "heterogeneous": ("fedwikd", "heterofl", "feddf", "fedfd", "local_teacher"),
}
SEEDS = (42, 43, 44, 45, 46)


def load_index(data_root):
    """Read an explicit index, or discover runs written by the experiment entries.

    An explicit index may select one run when an experiment was repeated.
    Without one, duplicate completed records are rejected rather than silently
    selecting a favorable run.
    """
    root = Path(data_root).resolve()
    index_path = root / "results_index.csv"
    if index_path.is_file():
        with index_path.open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
    else:
        records = []
        for final_path in sorted(root.rglob("final_result.json")):
            directory = final_path.parent
            config_path = directory / "config.json"
            if not config_path.is_file() or not (directory / "log.csv").is_file():
                continue
            record_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            config = record_config.get("effective_config", {})
            experiment = config.get("paper_experiment")
            if experiment not in (*METHODS, "width_study"):
                continue
            final = json.loads(final_path.read_text(encoding="utf-8-sig"))
            if not final.get("feasible", False) or int(final.get("R_actual", 0)) <= 0:
                continue
            seed = int(config["training_seed"])
            method = record_config["method"]
            width = config.get("paper_width") if experiment == "width_study" else None
            records.append(
                {
                    "id": f"{experiment}_seed{seed}_{method}"
                    + (f"_width{width}" if width is not None else ""),
                    "experiment": experiment,
                    "seed": str(seed),
                    "method": method,
                    "width": str(width) if width is not None else "",
                    "result_directory": str(directory.relative_to(root)),
                    "rounds": str(final["R_actual"]),
                    "final_accuracy_percent": str(final["final_acc"]),
                    "cumulative_energy_J": str(final["total_E_used"]),
                }
            )
    keyed = {}
    for record in records:
        experiment = record["experiment"]
        item = record["width"] if experiment == "width_study" else record["method"]
        key = (experiment, int(record["seed"]), str(item))
        if key in keyed:
            raise ValueError(
                f"Duplicate completed result for {key}. Select one run in results_index.csv."
            )
        keyed[key] = record
    expected = {
        (experiment, seed, method)
        for (experiment, methods) in METHODS.items()
        for seed in SEEDS
        for method in methods
    }
    expected.update(
        (("width_study", seed, str(width)) for seed in SEEDS for width in range(0, 101, 10))
    )
    missing, extra = (expected - keyed.keys(), keyed.keys() - expected)
    if missing or extra:
        raise ValueError(
            f"The paper figure requires the complete five-seed suite. Missing: {sorted(missing)}; unexpected: {sorted(extra)}"
        )
    return [keyed[key] for key in sorted(keyed)]
