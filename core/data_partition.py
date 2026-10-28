"""Dataset partitioning and label-distribution diagnostics."""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def summarize_client_label_counts(client_datasets, num_classes=10):
    counts = []
    for ds in client_datasets:
        hist = np.zeros(num_classes, dtype=int)
        for idx in ds.indices:
            _, label = ds.dataset[idx]
            hist[int(label)] += 1
        counts.append(hist)
    return np.stack(counts, axis=0)


def _get_label_from_dataset(dataset, idx):
    if hasattr(dataset, "targets"):
        return int(dataset.targets[idx])
    _, label = dataset[idx]
    return int(label)


def _labels_array(dataset):
    if hasattr(dataset, "targets") and len(dataset.targets) == len(dataset):
        return np.asarray(dataset.targets, dtype=int)
    return np.asarray([_get_label_from_dataset(dataset, i) for i in range(len(dataset))], dtype=int)


def compute_client_label_profiles_from_indices(dataset, client_indices, num_classes=10):
    labels = _labels_array(dataset)
    counts = np.zeros((len(client_indices), num_classes), dtype=int)
    for i, idxs in enumerate(client_indices):
        for idx in idxs:
            c = int(labels[int(idx)])
            if c < 0 or c >= num_classes:
                raise ValueError(f"Label {c} exceeds num_classes={num_classes} range")
            counts[i, c] += 1
    sizes = counts.sum(axis=1).astype(float)
    probs = counts / np.maximum(sizes[:, None], 1.0)
    logC = np.log(num_classes)
    entropy = np.zeros(len(client_indices), dtype=float)
    for i in range(len(client_indices)):
        p = probs[i]
        valid = p > 0
        entropy[i] = -float(np.sum(p[valid] * np.log(p[valid])))
    norm_entropy = entropy / logC if logC > 0 else np.zeros_like(entropy)
    label_skew = 1.0 - norm_entropy
    dominant_class = np.argmax(counts, axis=1).astype(int)
    dominant_ratio = np.max(probs, axis=1)
    return {
        "counts": counts,
        "probs": probs,
        "entropy": entropy,
        "norm_entropy": norm_entropy,
        "label_skew": label_skew,
        "dominant_class": dominant_class,
        "dominant_ratio": dominant_ratio,
        "sizes": sizes.astype(int),
    }


def _label_skew_from_counts(counts, num_classes):
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0:
        return (0.0, 0.0, 0.0)
    p = counts / total
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    norm_entropy = entropy / np.log(num_classes)
    label_skew = 1.0 - norm_entropy
    return (float(label_skew), float(entropy), float(norm_entropy))


def _profiles_to_dataframe(
    profiles, id_name, dist_vec=None, omega_vec=None, assignment=None, num_classes=10
):
    rows = []
    for i in range(len(profiles["counts"])):
        row = {
            id_name: int(i),
            "num_samples": int(profiles["sizes"][i]),
            "entropy": float(profiles["entropy"][i]),
            "norm_entropy": float(profiles["norm_entropy"][i]),
            "label_skew_score": float(profiles["label_skew"][i]),
            "dominant_class": int(profiles["dominant_class"][i]),
            "dominant_ratio": float(profiles["dominant_ratio"][i]),
        }
        if dist_vec is not None:
            row["distance_m"] = float(dist_vec[i])
        if omega_vec is not None:
            row["omega"] = float(omega_vec[i])
        if assignment is not None:
            row["source_partition_id"] = int(assignment[int(i)])
        for c in range(num_classes):
            row[f"class_{c}_count"] = int(profiles["counts"][i, c])
            row[f"class_{c}_ratio"] = float(profiles["probs"][i, c])
        rows.append(row)
    return pd.DataFrame(rows)


def build_label_shards_with_indices(
    dataset, num_clients, slices_per_client=2, rng=None, num_classes=10
):
    if rng is None:
        rng = np.random
    slices_per_client = int(slices_per_client)
    if slices_per_client <= 0:
        raise ValueError("slices_per_client must be a positive integer")
    total_slices = int(num_clients) * slices_per_client
    if total_slices <= 0:
        raise ValueError("total_slices must be positive")
    if total_slices > len(dataset):
        raise ValueError(
            f"total_slices={total_slices} must not exceed the dataset sample count {len(dataset)}"
        )
    labels = _labels_array(dataset)
    sorted_indices = []
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0].astype(int).tolist()
        rng.shuffle(idx_c)
        sorted_indices.extend(idx_c)
    base = len(sorted_indices) // total_slices
    remainder = len(sorted_indices) - base * total_slices
    shard_indices = []
    start = 0
    for sid in range(total_slices):
        size = base + (1 if sid < remainder else 0)
        shard = [int(x) for x in sorted_indices[start : start + size]]
        shard_indices.append(shard)
        start += size
    if start != len(sorted_indices):
        raise RuntimeError("Label shards have inconsistent lengths")
    return shard_indices


def _make_soft_class_latent_positions(num_classes, rng):
    num_classes = int(num_classes)
    base = (np.arange(num_classes, dtype=float) + 0.5) / max(num_classes, 1)
    rng.shuffle(base)
    jitter_scale = 0.35 / max(num_classes, 1)
    jitter = rng.uniform(-jitter_scale, jitter_scale, size=num_classes)
    return np.clip(base + jitter, 0.0, 1.0).astype(float)


def _normalize_distance_coordinates(dist_vec):
    dist_vec = np.asarray(dist_vec, dtype=float)
    d_min = float(np.min(dist_vec))
    d_max = float(np.max(dist_vec))
    if d_max <= d_min:
        return np.zeros_like(dist_vec, dtype=float)
    return ((dist_vec - d_min) / (d_max - d_min)).astype(float)


def _assign_shards_soft_distance_correlated(
    shard_indices,
    shard_profiles,
    dist_vec,
    slices_per_client,
    rng=None,
    num_classes=10,
    sigma=0.25,
    leakage=0.15,
    temperature=1.0,
    no_duplicate_class_per_client=True,
    class_latent_seed=None,
    max_restarts=1000,
):
    if rng is None:
        rng = np.random
    dist_vec = np.asarray(dist_vec, dtype=float)
    num_clients = len(dist_vec)
    slices_per_client = int(slices_per_client)
    total_slices = len(shard_indices)
    expected = num_clients * slices_per_client
    if total_slices != expected:
        raise ValueError(
            f"shard count {total_slices} versus num_clients*slices_per_client={expected} does not match"
        )
    if sigma <= 0:
        raise ValueError("soft_sigma must be positive")
    if leakage < 0:
        raise ValueError("soft_leakage must be nonnegative")
    if temperature <= 0:
        raise ValueError("soft_temperature must be positive")
    if class_latent_seed is None:
        latent_rng = rng
    else:
        latent_rng = np.random.RandomState(int(class_latent_seed))
    class_latent = _make_soft_class_latent_positions(num_classes, latent_rng)
    client_coord = _normalize_distance_coordinates(dist_vec)
    diff = client_coord[:, None] - class_latent[None, :]
    weight = np.exp(-(diff**2) / (2.0 * sigma**2)) + float(leakage)
    if abs(float(temperature) - 1.0) > 1e-12:
        weight = np.power(weight, 1.0 / float(temperature))
    weight = np.maximum(weight, 1e-12)
    dominant_classes = np.asarray(shard_profiles["dominant_class"], dtype=int)
    class_to_slices = {int(c): [] for c in range(num_classes)}
    for sid, c in enumerate(dominant_classes):
        c = int(c)
        if c not in class_to_slices:
            class_to_slices[c] = []
        class_to_slices[c].append(int(sid))
    class_ids = [int(c) for c in range(num_classes) if len(class_to_slices.get(int(c), [])) > 0]
    if no_duplicate_class_per_client:
        for c in class_ids:
            if len(class_to_slices[c]) > num_clients:
                raise ValueError(
                    f"Class {c} shard count {len(class_to_slices[c])} exceeds client count {num_clients}; cannot satisfy no_duplicate_class_per_client=True"
                )
    best = None
    best_score = -np.inf
    success_count = 0
    max_restarts = int(max_restarts)
    for attempt in range(max_restarts):
        cap = np.full(num_clients, slices_per_client, dtype=int)
        client_indices = [[] for _ in range(num_clients)]
        assignment = {int(k): [] for k in range(num_clients)}
        client_classes = [set() for _ in range(num_clients)]
        score = 0.0
        failed = False
        order = list(class_ids)
        rng.shuffle(order)
        order = sorted(order, key=lambda c: -len(class_to_slices[c]))
        for c in order:
            slice_ids = [int(x) for x in class_to_slices[c]]
            rng.shuffle(slice_ids)
            for sid in slice_ids:
                if no_duplicate_class_per_client:
                    candidates = [
                        k for k in range(num_clients) if cap[k] > 0 and c not in client_classes[k]
                    ]
                else:
                    candidates = [k for k in range(num_clients) if cap[k] > 0]
                if len(candidates) == 0:
                    failed = True
                    break
                cand = np.asarray(candidates, dtype=int)
                prob = weight[cand, c] * np.maximum(cap[cand].astype(float), 1.0)
                prob_sum = float(np.sum(prob))
                if prob_sum <= 0 or not np.isfinite(prob_sum):
                    prob = np.ones(len(cand), dtype=float) / len(cand)
                else:
                    prob = prob / prob_sum
                chosen = int(rng.choice(cand, p=prob))
                assignment[chosen].append(int(sid))
                client_indices[chosen].extend(shard_indices[int(sid)])
                client_classes[chosen].add(int(c))
                cap[chosen] -= 1
                score += float(np.log(weight[chosen, c] + 1e-12))
            if failed:
                break
        if failed or np.any(cap != 0):
            continue
        success_count += 1
        diversity_bonus = 0.0
        for k in range(num_clients):
            diversity_bonus += len(client_classes[k]) / max(slices_per_client, 1)
        total_score = score + 0.01 * diversity_bonus
        if total_score > best_score:
            best_score = total_score
            best = (client_indices, assignment, [set(x) for x in client_classes], float(score))
    if best is None:
        raise RuntimeError(
            "No capacity-feasible shard assignment was found within max_assignment_restarts. Increase the restart limit or disable no_duplicate_class_per_client."
        )
    client_indices, assignment, client_classes, raw_score = best
    order_far_to_near = [int(x) for x in np.argsort(-dist_vec)]
    far_half = set(order_far_to_near[: num_clients // 2])
    distance_group = {}
    for k in range(num_clients):
        distance_group[int(k)] = "far_half" if int(k) in far_half else "near_half"
    group_cfg = {
        "soft_distance_correlated": True,
        "sigma": float(sigma),
        "leakage": float(leakage),
        "temperature": float(temperature),
        "no_duplicate_class_per_client": bool(no_duplicate_class_per_client),
        "class_latent_seed": None if class_latent_seed is None else int(class_latent_seed),
        "max_assignment_restarts": int(max_restarts),
        "successful_restarts": int(success_count),
        "best_assignment_log_weight_score": float(raw_score),
        "class_latent_positions": [float(x) for x in class_latent],
        "client_distance_coordinates": [float(x) for x in client_coord],
        "class_client_weight_matrix": weight.astype(float).tolist(),
        "distance_groups_for_diagnostics": dict(distance_group),
    }
    return (client_indices, assignment, distance_group, group_cfg)


def split_label_sharding_soft_distance_correlated_with_indices(
    dataset,
    num_clients,
    slices_per_client=5,
    dist_vec=None,
    rng=None,
    num_classes=10,
    sigma=0.25,
    leakage=0.15,
    temperature=1.0,
    no_duplicate_class_per_client=True,
    class_latent_seed=None,
    max_restarts=1000,
):
    if rng is None:
        rng = np.random
    if dist_vec is None:
        raise ValueError("dist_vec must not be empty")
    shard_indices = build_label_shards_with_indices(
        dataset,
        num_clients=num_clients,
        slices_per_client=slices_per_client,
        rng=rng,
        num_classes=num_classes,
    )
    shard_profiles = compute_client_label_profiles_from_indices(
        dataset, shard_indices, num_classes=num_classes
    )
    client_indices, assignment, distance_group, group_cfg = _assign_shards_soft_distance_correlated(
        shard_indices,
        shard_profiles,
        dist_vec=dist_vec,
        slices_per_client=slices_per_client,
        rng=rng,
        num_classes=num_classes,
        sigma=sigma,
        leakage=leakage,
        temperature=temperature,
        no_duplicate_class_per_client=no_duplicate_class_per_client,
        class_latent_seed=class_latent_seed,
        max_restarts=max_restarts,
    )
    return (client_indices, assignment, distance_group, group_cfg, shard_indices, shard_profiles)


def _slice_profiles_to_dataframe(profiles, id_name="slice_id", num_classes=10):
    rows = []
    for i in range(len(profiles["counts"])):
        row = {
            id_name: int(i),
            "num_samples": int(profiles["sizes"][i]),
            "entropy": float(profiles["entropy"][i]),
            "norm_entropy": float(profiles["norm_entropy"][i]),
            "label_skew_score": float(profiles["label_skew"][i]),
            "dominant_class": int(profiles["dominant_class"][i]),
            "dominant_ratio": float(profiles["dominant_ratio"][i]),
        }
        for c in range(num_classes):
            row[f"class_{c}_count"] = int(profiles["counts"][i, c])
            row[f"class_{c}_ratio"] = float(profiles["probs"][i, c])
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_counts_from_slice_ids(shard_profiles, slice_ids, num_classes=10):
    counts = np.asarray(shard_profiles["counts"], dtype=int)
    if len(slice_ids) == 0:
        return np.zeros(num_classes, dtype=int)
    return counts[[int(x) for x in slice_ids]].sum(axis=0).astype(int)


def _group_distribution_row_from_counts(group_name, counts, num_classes=10):
    counts = np.asarray(counts, dtype=int)
    label_skew, entropy, norm_entropy = _label_skew_from_counts(counts, num_classes=num_classes)
    total = int(np.sum(counts))
    row = {
        "group": group_name,
        "num_samples": total,
        "entropy": float(entropy),
        "norm_entropy": float(norm_entropy),
        "label_skew_score": float(label_skew),
        "dominant_class": int(np.argmax(counts)) if total > 0 else -1,
        "dominant_ratio": float(np.max(counts) / max(total, 1)),
    }
    for c in range(num_classes):
        row[f"class_{c}_count"] = int(counts[c])
        row[f"class_{c}_ratio"] = float(counts[c] / max(total, 1))
    return row


def save_label_sharding_diagnostics(
    output_dir,
    dataset,
    shard_indices,
    client_indices,
    assignment,
    dist_vec,
    omega_vec,
    sharding_cfg,
    split_mode,
    distance_group=None,
    group_cfg=None,
    num_classes=10,
):
    diag_dir = Path(output_dir) / "data_split_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    shard_profiles = compute_client_label_profiles_from_indices(
        dataset, shard_indices, num_classes=num_classes
    )
    after_profiles = compute_client_label_profiles_from_indices(
        dataset, client_indices, num_classes=num_classes
    )
    shard_df = _slice_profiles_to_dataframe(
        shard_profiles, id_name="slice_id", num_classes=num_classes
    )
    after_df = _profiles_to_dataframe(
        after_profiles,
        "physical_client_id",
        dist_vec=dist_vec,
        omega_vec=omega_vec,
        num_classes=num_classes,
    )
    if distance_group is not None:
        after_df["distance_group"] = after_df["physical_client_id"].map(
            lambda x: distance_group.get(int(x), "random")
        )
    assignment_rows = []
    for physical_id in range(len(client_indices)):
        slice_ids = [int(x) for x in assignment[int(physical_id)]]
        slice_dominants = [int(shard_profiles["dominant_class"][sid]) for sid in slice_ids]
        slice_dominant_ratios = [float(shard_profiles["dominant_ratio"][sid]) for sid in slice_ids]
        row = {
            "physical_client_id": int(physical_id),
            "distance_m": float(dist_vec[physical_id]),
            "omega": float(omega_vec[physical_id]),
            "distance_group": (
                distance_group.get(int(physical_id), "random")
                if distance_group is not None
                else "random"
            ),
            "assigned_slice_ids": "|".join((str(x) for x in slice_ids)),
            "assigned_slice_dominant_classes": "|".join((str(x) for x in slice_dominants)),
            "assigned_slice_dominant_ratios": "|".join((f"{x:.4f}" for x in slice_dominant_ratios)),
            "client_num_samples": int(after_profiles["sizes"][physical_id]),
            "client_entropy": float(after_profiles["entropy"][physical_id]),
            "client_norm_entropy": float(after_profiles["norm_entropy"][physical_id]),
            "client_label_skew_score": float(after_profiles["label_skew"][physical_id]),
            "client_dominant_class": int(after_profiles["dominant_class"][physical_id]),
            "client_dominant_ratio": float(after_profiles["dominant_ratio"][physical_id]),
        }
        for c in range(num_classes):
            row[f"client_class_{c}_count"] = int(after_profiles["counts"][physical_id, c])
            row[f"client_class_{c}_ratio"] = float(after_profiles["probs"][physical_id, c])
        assignment_rows.append(row)
    assignment_df = pd.DataFrame(assignment_rows).sort_values("distance_m", ascending=False)
    group_rows = []
    if distance_group is not None:
        group_to_slice_ids = {}
        for physical_id, slice_ids in assignment.items():
            group_name = distance_group.get(int(physical_id), "unassigned")
            group_to_slice_ids.setdefault(str(group_name), [])
            group_to_slice_ids[str(group_name)].extend([int(x) for x in slice_ids])
        for group_name in sorted(group_to_slice_ids.keys()):
            group_rows.append(
                _group_distribution_row_from_counts(
                    group_name,
                    _aggregate_counts_from_slice_ids(
                        shard_profiles, group_to_slice_ids[group_name], num_classes=num_classes
                    ),
                    num_classes=num_classes,
                )
            )
    else:
        all_slice_ids = []
        for slice_ids in assignment.values():
            all_slice_ids.extend([int(x) for x in slice_ids])
        group_rows.append(
            _group_distribution_row_from_counts(
                "all_random_shards",
                _aggregate_counts_from_slice_ids(
                    shard_profiles, all_slice_ids, num_classes=num_classes
                ),
                num_classes=num_classes,
            )
        )
    group_df = pd.DataFrame(group_rows)
    shard_df.to_csv(diag_dir / "shard_distribution.csv", index=False, encoding="utf-8-sig")
    after_df.to_csv(diag_dir / "label_distribution_after.csv", index=False, encoding="utf-8-sig")
    assignment_df.to_csv(diag_dir / "distance_assignment.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(
        diag_dir / "group_aggregate_distribution.csv", index=False, encoding="utf-8-sig"
    )
    extra_files = []
    if group_cfg is not None and group_cfg.get("soft_distance_correlated", False):
        latent_positions = group_cfg.get("class_latent_positions", [])
        if len(latent_positions) > 0:
            latent_df = pd.DataFrame(
                {
                    "class_id": list(range(len(latent_positions))),
                    "latent_position": [float(x) for x in latent_positions],
                }
            )
            latent_df.to_csv(
                diag_dir / "class_latent_positions.csv", index=False, encoding="utf-8-sig"
            )
            extra_files.append("class_latent_positions.csv")
        client_coords = group_cfg.get("client_distance_coordinates", [])
        if len(client_coords) > 0:
            coord_df = pd.DataFrame(
                {
                    "physical_client_id": list(range(len(client_coords))),
                    "distance_m": [float(x) for x in dist_vec],
                    "distance_coordinate": [float(x) for x in client_coords],
                }
            )
            coord_df.to_csv(
                diag_dir / "client_distance_coordinates.csv", index=False, encoding="utf-8-sig"
            )
            extra_files.append("client_distance_coordinates.csv")
        weight_matrix = group_cfg.get("class_client_weight_matrix", None)
        if weight_matrix is not None:
            w = np.asarray(weight_matrix, dtype=float)
            rows = []
            for k in range(w.shape[0]):
                row = {"physical_client_id": int(k), "distance_m": float(dist_vec[k])}
                for c in range(w.shape[1]):
                    row[f"class_{c}_weight"] = float(w[k, c])
                rows.append(row)
            pd.DataFrame(rows).to_csv(
                diag_dir / "soft_assignment_weights.csv", index=False, encoding="utf-8-sig"
            )
            extra_files.append("soft_assignment_weights.csv")
    summary = {
        "data_split_mode": split_mode,
        "num_clients": int(len(client_indices)),
        "num_classes": int(num_classes),
        "slices_per_client": int(sharding_cfg.get("slices_per_client", 2)),
        "total_slices": int(len(shard_indices)),
        "slice_size_min": int(min((len(x) for x in shard_indices))),
        "slice_size_max": int(max((len(x) for x in shard_indices))),
        "sharding_config": dict(sharding_cfg),
        "group_config": group_cfg if group_cfg is not None else {},
        "assignment_physical_to_slices": {
            str(int(k)): [int(x) for x in v] for (k, v) in assignment.items()
        },
        "diagnostic_files": [
            "shard_distribution.csv",
            "label_distribution_after.csv",
            "distance_assignment.csv",
            "group_aggregate_distribution.csv",
        ]
        + extra_files,
    }
    with open(diag_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {
        "diag_dir": str(diag_dir),
        "shard_df": shard_df,
        "after_df": after_df,
        "assignment_df": assignment_df,
        "group_df": group_df,
        "summary": summary,
    }
