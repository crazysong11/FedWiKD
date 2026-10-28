"""Shared experiment execution and recording."""

from __future__ import annotations
import csv
import hashlib
import json
import random
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

LABELS = {
    "fedwikd": "FedWiKD",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "SCAFFOLD",
    "heterofl": "HeteroFL",
    "feddf": "FedDF",
    "fedfd": "FedFD",
    "local_teacher": "Local training",
}


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value


def save_json(path, value):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(json_value(value), f, ensure_ascii=False, indent=2, allow_nan=False)


def create_run_dir(base):
    base = Path(base).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
    path = base / name
    path.mkdir(exist_ok=False)
    return path


def expected_rounds(ns, config, profile):
    """Existing equal-resource ergodic-energy formula, with the actual payload."""
    physical = config["physical"]
    count = physical["M_TOTAL"] / physical["K"]
    costs = []
    for bits, duration, omega, compute in zip(
        profile["S_data_vec"], profile["T_req_vec"], ns["OMEGA_VEC"], profile["E_comp_vec"]
    ):
        if duration <= 0:
            return None
        target = bits * np.log(2) / (duration * count * physical["W_RB"])
        if target >= ns["exp1"](1e-10):
            return None
        lo, hi = (1e-10, 100.0)
        for _ in range(300):
            mid = (lo + hi) / 2
            if ns["exp1"](mid) > target:
                lo = mid
            else:
                hi = mid
            if (hi - lo) / (lo + 1e-30) < 1e-12:
                break
        x = (lo + hi) / 2
        communication = (
            duration
            * count
            * physical["N_0"]
            * physical["W_RB"]
            * (np.exp(-x) / (x * omega) - ns["exp1"](x) / omega)
        )
        costs.append(compute + communication)
    return int(np.floor(physical["E_TOTAL_K"] / max(costs)))


def set_seed(config):
    seed = config["training_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def profile_for(ns, runner, width, count):
    if hasattr(runner, "get_physical_profile"):
        return runner.get_physical_profile(width)
    profile = ns["compute_physical_params"](runner.method, width, runner.N_T)
    return {key + "_vec": np.full(count, value, dtype=float) for (key, value) in profile.items()}


def client_model(client):
    return client.teacher if hasattr(client, "teacher") else client.model


def evaluate_clients(clients, testset, device, method, batch_size):
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    results = []
    evaluated = clients[:1] if method in ("fedavg", "fedprox", "scaffold") else clients
    for client in evaluated:
        model = client_model(client)
        was_training = model.training
        model.eval()
        total_loss = 0.0
        correct = count = 0
        with torch.no_grad():
            for data, labels in loader:
                data, labels = (data.to(device), labels.to(device))
                logits = model(data)
                total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
                correct += int((logits.argmax(1) == labels).sum().item())
                count += int(labels.numel())
        model.train(was_training)
        if count == 0:
            raise ValueError("Test dataset must not be empty")
        results.append((total_loss / count, 100.0 * correct / count))
    if len(results) == 1 and len(clients) > 1:
        results *= len(clients)
    return results


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for (k, v) in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


def checkpoint(path, clients, server, config, round_number):
    result = {"round": round_number, "config": json_value(config), "clients": []}
    for client in clients:
        item = {"cid": client.cid, "task_model": cpu_tree(client_model(client).state_dict())}
        for name in ("student",):
            model = getattr(client, name, None)
            if isinstance(model, torch.nn.Module):
                item[name] = cpu_tree(model.state_dict())
        if hasattr(client, "control"):
            item["control"] = cpu_tree(client.control)
        result["clients"].append(item)
    result["server"] = {}
    if server is not None:
        for name in ("global_model", "global_student", "projections"):
            model = getattr(server, name, None)
            if isinstance(model, torch.nn.Module):
                result["server"][name] = cpu_tree(model.state_dict())
        for name in ("prototype_models",):
            models = getattr(server, name, None)
            if models is not None:
                result["server"][name] = {
                    str(k): cpu_tree(v.state_dict()) for (k, v) in models.items()
                }
        if getattr(server, "global_control", None) is not None:
            result["server"]["global_control"] = cpu_tree(server.global_control)
    result["checkpoint_kind"] = "evaluation"
    torch.save(result, path)


def run_method(ns, runner, method, config, shared):
    if not config.get("run_id"):
        raise ValueError("Use the experiment main entry to create an independent run archive")
    set_seed(config)
    directory = Path(config["output_dir"]) / method
    directory.mkdir(exist_ok=True)
    runner.exp_dir = directory
    width, planned_rounds, planning = runner.determine_n_S()
    count = config["num_clients"]
    profile = profile_for(ns, runner, width, count)
    if method == "fedwikd":
        if "compute_comm_profile_from_vectors" in ns:
            planning["selected_width_cantelli"] = ns["compute_comm_profile_from_vectors"](
                profile["S_data_vec"], profile["T_req_vec"], check_cantelli=True
            )
        else:
            planning["selected_width_cantelli"] = ns["compute_comm_profile_for_method"](
                "fedkd", width, config["N_T"], check_cantelli=True
            )
    config_record = {
        "effective_config": config,
        "method": method,
        "selected_width": width,
        "R_theory": planned_rounds,
        "planning": planning,
        "physical_profile": profile,
    }
    save_json(directory / "config.json", config_record)
    ecomp = np.asarray(profile["E_comp_vec"], dtype=float)
    duration = np.asarray(profile["T_req_vec"], dtype=float)
    payload = np.asarray(profile["S_data_vec"], dtype=float)
    if any((v.shape != (count,) for v in (ecomp, duration, payload))):
        raise ValueError("A physical profile must contain one entry per client")
    if not all((np.isfinite(v).all() for v in (ecomp, duration, payload))):
        raise ValueError("Non-finite physical profile")
    local_only = method == "local_teacher"
    if (
        not local_only
        and np.any(duration <= 0)
        or np.any(np.asarray(profile["T_comp_vec"]) > config["physical"]["T_MAX"])
    ):
        result = {
            "method": method,
            "R_actual": 0,
            "feasible": False,
            "stop_reason": "computation_exceeds_round_deadline",
        }
        save_json(directory / "final_result.json", result)
        return result
    clients, server = runner.init_components(width)
    set_seed(config)
    physical = config["physical"]
    remaining = np.full(count, physical["E_TOTAL_K"], dtype=float)
    cumulative = uploaded = 0.0
    round_rows = []
    stop_reason = "safety_round_limit"
    schedule_dir = directory / "schedules"
    schedule_dir.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    with (directory / "log.csv").open("w", newline="", encoding="utf-8") as lf, (
        directory / "clients.csv"
    ).open("w", newline="", encoding="utf-8") as cf:
        log_writer = client_writer = None
        for round_idx in range(config["max_rounds_safety"]):
            if np.any(remaining < ecomp - 1e-12):
                stop_reason = "insufficient_computation_energy"
                break
            before = remaining.copy()
            remaining -= ecomp
            power = np.zeros((count, physical["M_TOTAL"]))
            assignment = np.zeros_like(power)
            outage = set()
            if not local_only:
                bits_arg, time_arg = (payload, duration)
                if "compute_comm_profile_from_vectors" not in ns:
                    if not (
                        np.allclose(payload, payload[0]) and np.allclose(duration, duration[0])
                    ):
                        raise ValueError("The homogeneous solver requires identical profiles")
                    bits_arg, time_arg = (float(payload[0]), float(duration[0]))
                _, assignment, outage_list, iterations, power, diagnostics = ns[
                    "solver_dual_decomposition"
                ](
                    shared["channel_seq"][round_idx],
                    bits_arg,
                    time_arg,
                    physical["W_RB"],
                    physical["N_0"],
                    physical["P_MAX"],
                    count,
                    physical["M_TOTAL"],
                    max_iter=config["dual_solver"]["max_iter"],
                    feas_tol=config["dual_solver"]["feas_tol"],
                )
                outage = set((int(i) for i in outage_list))
                np.savez_compressed(
                    schedule_dir / f"round_{round_idx + 1:04d}.npz",
                    assignment=assignment,
                    power=power,
                )
            planned_comm = np.sum(power, axis=1) * duration
            spent_comm = np.zeros(count)
            success, energy_outage = ([], set())
            if not local_only:
                for cid in range(count):
                    if cid in outage:
                        continue
                    if remaining[cid] >= planned_comm[cid]:
                        spent_comm[cid] = planned_comm[cid]
                        remaining[cid] -= spent_comm[cid]
                        success.append(cid)
                    else:
                        spent_comm[cid] = max(0.0, remaining[cid])
                        remaining[cid] = 0.0
                        energy_outage.add(cid)
            round_energy = float(ecomp.sum() + spent_comm.sum())
            cumulative += round_energy
            round_upload = float(payload[success].sum())
            uploaded += round_upload
            runner.train_one_round(clients, server, success)
            metrics = evaluate_clients(
                clients, shared["testset"], ns["device"], method, config["evaluation_batch_size"]
            )
            row = {
                "round": round_idx + 1,
                "avg_loss": float(np.mean([v[0] for v in metrics])),
                "avg_acc": float(np.mean([v[1] for v in metrics])),
                "round_E": round_energy,
                "cumulative_E": cumulative,
                "computation_E_total": float(ecomp.sum()),
                "comm_E_total": float(spent_comm.sum()),
                "min_E_remain": float(remaining.min()),
                "round_uploaded_bits": round_upload,
                "cumulative_uploaded_bits": uploaded,
                "outage_count": len(outage | energy_outage),
                "success_count": len(success),
                "stage2_outage_count": len(outage),
                "energy_outage_count": len(energy_outage),
                "success_clients": ";".join(map(str, success)),
                "stage2_outage_clients": ";".join(map(str, sorted(outage))),
                "energy_outage_clients": ";".join(map(str, sorted(energy_outage))),
                "elapsed_sec": time.perf_counter() - t_start,
            }
            if log_writer is None:
                log_writer = csv.DictWriter(lf, fieldnames=list(row))
                log_writer.writeheader()
            log_writer.writerow(row)
            lf.flush()
            for cid, (loss, accuracy) in enumerate(metrics):
                record = {
                    "round": round_idx + 1,
                    "cid": cid,
                    "test_loss": loss,
                    "test_accuracy_percent": accuracy,
                    "energy_before_J": float(before[cid]),
                    "computation_energy_J": float(ecomp[cid]),
                    "communication_energy_J": float(spent_comm[cid]),
                    "required_communication_energy_J": float(planned_comm[cid]),
                    "energy_after_J": float(remaining[cid]),
                    "upload_applicable": int(not local_only),
                    "stage2_evaluated": int(not local_only),
                    "stage2_outage": int(cid in outage),
                    "energy_outage": int(cid in energy_outage),
                    "upload_success": int(cid in success),
                    "required_upload_bits": float(payload[cid]),
                    "uploaded_bits": float(payload[cid]) if cid in success else 0.0,
                    "upload_window_s": float(duration[cid]),
                    "scheduled_power_W": float(np.sum(power[cid])),
                }
                if client_writer is None:
                    client_writer = csv.DictWriter(cf, fieldnames=list(record))
                    client_writer.writeheader()
                client_writer.writerow(record)
            cf.flush()
            round_rows.append(row)
            print(
                f"[{LABELS[method]} | {round_idx + 1}] acc={row['avg_acc']:.2f}%, E={cumulative:.3f} J, uploads={len(success)}, wireless_outages={len(outage)}, energy_outages={len(energy_outage)}",
                flush=True,
            )
    result = {
        "method": method,
        "feasible": bool(round_rows),
        "R_theory": planned_rounds,
        "R_actual": len(round_rows),
        "selected_width": width,
        "stop_reason": stop_reason,
        "max_rounds_safety": config["max_rounds_safety"],
        "total_E_used": cumulative,
        "total_uploaded_bits": uploaded,
        "remaining_energy_J": remaining,
        "next_round_computation_feasible": bool(np.all(remaining >= ecomp - 1e-12)),
    }
    if round_rows:
        result.update(final_acc=round_rows[-1]["avg_acc"], final_loss=round_rows[-1]["avg_loss"])
    save_json(directory / "final_result.json", result)
    checkpoint(directory / "final_models.pt", clients, server, config, len(round_rows))
    return result


def archive_sources(root, entry):
    directory = root / "source"
    directory.mkdir()
    repository = Path(__file__).resolve().parents[1]
    paths = [
        p
        for folder in ("core", "experiments", "plotting")
        for p in (repository / folder).glob("*.py")
    ]
    paths += list((repository / "configs").glob("*.json"))
    paths += [repository / "README.md", repository / "requirements.txt"]
    hashes = {}
    for path in paths:
        relative = path.relative_to(repository)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(directory / "sha256.json", hashes)


def original_indices(dataset):
    ids = np.arange(len(dataset))
    while hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        ids = np.asarray(dataset.indices)[ids]
        dataset = dataset.dataset
    return ids


def run_experiment(ns, config=None, shared=None):
    """A call always creates a new run and saves its configuration and results."""
    config = json_value(config if config is not None else ns["build_default_config"]())
    methods = list(ns["ENABLED_BASELINES"])
    config["enabled_methods"] = methods
    config["method_labels"] = {name: LABELS[name] for name in methods}
    config["evaluation_batch_size"] = int(config.get("evaluation_batch_size", 50))
    config["evaluation"] = {
        "metric": "unweighted_mean_of_all_client_local_task_models",
        "testset": "full_common_testset",
        "accuracy_unit": "percent",
        "timing": "end_of_round_after_algorithm_updates_and_broadcast",
        "loss": "mean_cross_entropy",
        "test_energy_counted": False,
    }
    config["resource_accounting"] = {
        "model_value_bits": 32,
        "payload": "trainable_parameters",
        "scaffold_upload": (
            "model_parameters_plus_control_delta"
            if config.get("scaffold", {}).get("count_control_upload", True)
            else "model_parameters_only_control_traffic_excluded"
        ),
        "scaffold_control_upload_counted": config.get("scaffold", {}).get(
            "count_control_upload", True
        ),
        "downlink_energy_counted": False,
        "server_energy_counted": False,
        "communication_failure_energy_policy": "consume_remaining_battery_if_insufficient",
    }
    entry = Path(ns["__file__"]).resolve()
    base = Path(config["output_dir"])
    if not base.is_absolute():
        base = entry.parent / base
    root = create_run_dir(base)
    config["output_base_dir"] = str(base)
    config["output_dir"] = str(root)
    config["run_id"] = root.name
    config["entry_script"] = str(entry)
    save_json(root / "config.json", config)
    archive_sources(root, entry)
    save_json(
        root / "environment.json",
        {
            "python": sys.version,
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(ns["device"]),
            "cuda": torch.version.cuda,
            "packages": {
                name: getattr(sys.modules.get(name), "__version__", None)
                for name in ("torchvision", "scipy", "matplotlib", "pandas")
            },
        },
    )
    save_json(root / "status.json", {"status": "running", "started": datetime.now().isoformat()})
    print(f"Run archive: {root}", flush=True)
    results = []
    try:
        ns["set_simulation_context"](config)
        if shared is None:
            shared = ns["prepare_shared_resources"](config)
        np.savez_compressed(
            root / "data_partition.npz",
            **{
                f"client_{i}": original_indices(ds)
                for (i, ds) in enumerate(shared["client_datasets"])
            },
        )
        np.savez_compressed(root / "channels.npz", gains=np.asarray(shared["channel_seq"]))
        for method in methods:
            baseline = dict(ns["ALL_BASELINES"][method])
            baseline["tag"] = method
            runner_class = ns["_resolve_runner_class"](baseline["runner"])
            runner = runner_class(baseline, config, shared)
            result = run_method(ns, runner, method, config, shared)
            results.append(result)
            save_json(root / "summary.json", results)
            del runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        save_json(root / "config.json", config)
        save_json(
            root / "status.json", {"status": "completed", "finished": datetime.now().isoformat()}
        )
    except BaseException:
        save_json(root / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise
    return root
