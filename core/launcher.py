"""Portable experiment entry points; the simulation cores keep their numerical logic."""

from __future__ import annotations
import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from core.configuration import load_config

ROOT = Path(__file__).resolve().parents[1]


def config_for_seed(base_config, seed):
    """Use the seed mapping of the original multi-seed experiments."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seeds must be integers in [0, 2**32).")
    config = deepcopy(base_config)
    for key in ("data_split_seed", "model_init_seed", "training_seed"):
        config[key] = seed
    config["channel_seed_base"] = (1000 + (seed - 42) * 1000) % 2**32
    config["label_sharding"]["class_latent_seed"] = seed
    if config.get("heterogeneous_model", {}).get("enabled", False):
        config["heterogeneous_model"]["assignment_seed"] = seed
    return config


def validate_selection(seeds, methods, allowed_methods, widths, config):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Specify a nonempty seed pool without duplicates.")
    for seed in seeds:
        config_for_seed(config, seed)
    if not methods or len(methods) != len(set(methods)) or set(methods) - set(allowed_methods):
        raise ValueError(f"Select distinct methods from {allowed_methods}.")
    if widths is not None:
        if (
            not widths
            or len(widths) != len(set(widths))
            or any((type(w) is not int or not 0 <= w < config["N_T"] for w in widths))
        ):
            raise ValueError(
                "Widths must be distinct nonnegative integers below the teacher width."
            )


def prepare_job(experiment, config, methods, width=None):
    """Import the core only when a run or numerical check explicitly needs it."""
    module_name = "heterogeneous" if experiment == "heterogeneous" else "homogeneous"
    engine = importlib.import_module(f"core.{module_name}_engine")
    config = deepcopy(config)
    config["paper_experiment"] = experiment
    config["paper_width"] = width
    engine.ENABLED_BASELINES = list(methods)
    if experiment == "width_study":
        if width is None:
            raise ValueError("Student width study requires an observation width.")
        engine.set_simulation_context(config)
        if width == 0:
            engine.ENABLED_BASELINES = ["local_teacher"]
            engine.ALL_BASELINES["local_teacher"] = {
                "tag": "local_teacher",
                "method": "local_teacher",
                "runner": "LocalTeacherRunner",
            }
        else:
            selected, rounds, _ = engine.run_stage1(config["N_T"], "heuristic_H1", verbose=False)
            if selected is None:
                raise ValueError("Stage 1 found no feasible reference width.")
            profile = engine.compute_comm_profile_for_method(
                "fedkd", width, config["N_T"], check_cantelli=True
            )
            physical = engine.compute_physical_params("fedkd", width, config["N_T"])
            theoretical = (
                engine.compute_R_star_heterogeneous(physical["E_comp"], profile["E_comm_vec"])
                if profile["ok"]
                else None
            )
            config["width_study"] = {
                "objective": "heuristic_H1",
                "train_outside_cantelli_constraint": True,
                "n_S_star": selected,
                "R_star": rounds,
                "point": {
                    "width": width,
                    "is_stage1_selected_width": width == selected,
                    "stage1_communication_constraint_pass": bool(profile["ok"]),
                    "constraint_reason": profile["reason"],
                    "bad_client": profile["bad_client"],
                    "R_theory": theoretical,
                    "training_requested": True,
                },
            }
            engine.ENABLED_BASELINES = ["fedwikd"]
            engine.ALL_BASELINES["fedwikd"] = dict(
                engine.ALL_BASELINES["fedwikd"], runner="WidthStudyRunner"
            )
    return (engine, config)


def run_job(experiment, config, methods, width=None):
    os.chdir(ROOT)
    engine, config = prepare_job(experiment, config, methods, width)
    seed = config["training_seed"]
    output = Path(config["output_dir"]).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output = output / f"seed_{seed}"
    if width is not None:
        output = output / f"width_{width:03d}"
    config["output_dir"] = str(output)
    from core.experiment_runtime import run_experiment

    return run_experiment(vars(engine), config=config)


def execute_pool(experiment, config, methods, seeds, widths, plan_only):
    jobs = []
    for seed in seeds:
        for width in widths if widths is not None else [None]:
            jobs.append(
                {
                    "experiment": experiment,
                    "config": config_for_seed(config, seed),
                    "methods": methods,
                    "width": width,
                }
            )
    if plan_only:
        print(
            json.dumps({"training_started": False, "job_count": len(jobs), "jobs": jobs}, indent=2)
        )
        return
    for i, job in enumerate(jobs, 1):
        print(
            f"[{i}/{len(jobs)}] {experiment}: seed={job['config']['training_seed']}, width={job['width']}, methods={methods}",
            flush=True,
        )
        subprocess.run(
            [sys.executable, "-B", "-m", "core.launcher", "--worker"],
            cwd=ROOT,
            input=json.dumps(job),
            text=True,
            check=True,
        )


def single_main(experiment, config_factory, methods, widths=None):
    parser = argparse.ArgumentParser(description=f"Run the {experiment} experiment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, help="Override the default JSON configuration.")
    parser.add_argument("--methods", nargs="+", default=methods, choices=methods)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print configurations without importing models or training.",
    )
    if widths is not None:
        parser.add_argument("--widths", nargs="+", type=int, default=widths)
    args = parser.parse_args()
    config = load_config(args.config) if args.config else config_factory()
    if args.output_dir:
        config["output_dir"] = str(args.output_dir.expanduser().resolve())
    selected_widths = args.widths if widths is not None else None
    validate_selection([args.seed], args.methods, methods, selected_widths, config)
    execute_pool(experiment, config, args.methods, [args.seed], selected_widths, args.plan_only)


def multi_main(experiment, seed_pool):
    entry = importlib.import_module(f"experiments.{experiment}")
    parser = argparse.ArgumentParser(description=f"Run {experiment} for an editable seed pool.")
    parser.add_argument("--seeds", nargs="+", type=int, default=seed_pool)
    parser.add_argument("--config", type=Path, help="Override the default JSON configuration.")
    parser.add_argument(
        "--methods", nargs="+", choices=entry.ENABLED_METHODS, default=entry.ENABLED_METHODS
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    widths = getattr(entry, "WIDTHS", None)
    if widths is not None:
        parser.add_argument("--widths", nargs="+", type=int, default=widths)
    args = parser.parse_args()
    config = load_config(args.config) if args.config else entry.build_default_config()
    if args.output_dir:
        config["output_dir"] = str(args.output_dir.expanduser().resolve())
    selected_widths = args.widths if widths is not None else None
    validate_selection(args.seeds, args.methods, entry.ENABLED_METHODS, selected_widths, config)
    execute_pool(experiment, config, args.methods, args.seeds, selected_widths, args.plan_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    payload = json.load(sys.stdin)
    run_job(**payload)
