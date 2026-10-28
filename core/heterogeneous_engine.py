"""Heterogeneous simulation core. Numerical routines retain their original module context.

Experiment settings are defined in configs/heterogeneous.json. Module-level
initial values below are replaced by set_simulation_context before planning
or training. Keeping this context together preserves the original call order.
"""

from core.models import (
    TeacherCNN as TeacherCNN,
    get_model,
    model_param_count_formula,
    model_forward_flops_formula,
    model_feature_dim_formula,
)
from core.data_partition import (
    summarize_client_label_counts,
    split_label_sharding_soft_distance_correlated_with_indices,
    save_label_sharding_diagnostics,
)
import os
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from scipy.special import exp1
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using device: {device}")
N_T: int = 64
MAX_ROUNDS_SAFETY: int = 200
D_K_EFFECTIVE: float = 50000 / 10
K: int = 10
M_TOTAL: int = 100
W_RB: float = 180000.0
N_0: float = 3.98e-21
P_MAX: float = 0.2
T_FLOP: float = 2e-12
E_FLOP: float = 1e-11
T_MAX: float = 10.0
E_TOTAL_K: float = 1000.0
DELTA: float = 0.1
DIST_VEC: np.ndarray = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 350], dtype=float)
D_REF: float = 100.0
OMEGA_REF: float = 1e-11
PATH_LOSS_EXP: float = 3.5
OMEGA_VEC: np.ndarray = OMEGA_REF * (DIST_VEC / D_REF) ** (-PATH_LOSS_EXP)
DUAL_MAX_ITER: int = 300
DUAL_FEAS_TOL: float = 0.05
ACTIVE_CONFIG: dict = {}
_context_initialized = False


def set_simulation_context(config: dict) -> None:
    global _context_initialized, ACTIVE_CONFIG
    ACTIVE_CONFIG = config
    phys = config["physical"]
    dist_vec = np.asarray(phys["dist_vec"], dtype=float)
    omega_vec = phys["omega_ref"] * (dist_vec / phys["d_ref"]) ** (-phys["path_loss_exp"])
    values_to_inject = {
        "N_T": config["N_T"],
        "MAX_ROUNDS_SAFETY": config["max_rounds_safety"],
        "D_K_EFFECTIVE": float(config.get("total_train_samples", 50000) / phys["K"]),
        "K": phys["K"],
        "M_TOTAL": phys["M_TOTAL"],
        "W_RB": phys["W_RB"],
        "N_0": phys["N_0"],
        "P_MAX": phys["P_MAX"],
        "T_FLOP": phys["T_FLOP"],
        "E_FLOP": phys["E_FLOP"],
        "T_MAX": phys["T_MAX"],
        "E_TOTAL_K": phys["E_TOTAL_K"],
        "DELTA": phys["DELTA"],
        "DIST_VEC": dist_vec,
        "D_REF": phys["d_ref"],
        "OMEGA_REF": phys["omega_ref"],
        "PATH_LOSS_EXP": phys["path_loss_exp"],
        "OMEGA_VEC": omega_vec,
        "DUAL_MAX_ITER": config["dual_solver"]["max_iter"],
        "DUAL_FEAS_TOL": config["dual_solver"]["feas_tol"],
    }
    g = globals()
    for k, v in values_to_inject.items():
        g[k] = v
    _context_initialized = True


class FedKDClient:

    def __init__(
        self,
        cid,
        dataset,
        student_init_state,
        config,
        teacher_width,
        student_width,
        teacher_init_state=None,
    ):
        self.cid = cid
        self.dataset = dataset
        self.loader = DataLoader(
            dataset, batch_size=config.get("batch_size_fedkd", 50), shuffle=True, num_workers=0
        )
        self.config = config
        self.teacher_width = int(teacher_width)
        self.student_width = int(student_width)
        self.teacher = get_model("teacher", teacher_width).to(device)
        if teacher_init_state is not None:
            self.teacher.load_state_dict(teacher_init_state)
        self.student = get_model("student", student_width).to(device)
        load_state_by_overlap(self.student, student_init_state)
        self.teacher_opt = optim.SGD(
            self.teacher.parameters(),
            lr=config.get("teacher_lr_fedkd", 0.001),
            momentum=config.get("teacher_momentum_fedkd", 0.9),
            weight_decay=config.get("teacher_weight_decay_fedkd", 0.0005),
        )
        self.student_opt = optim.SGD(
            self.student.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
        )
        self.criterion = nn.CrossEntropyLoss()
        self.distill_loss = nn.KLDivLoss(reduction="batchmean")
        self.temperature = config.get("temperature_fedkd", 1.0)

    def update_student(self, aggregated_params):
        load_state_by_overlap(self.student, aggregated_params)

    def _compute_teacher_loss(self, teacher_logits, student_logits, labels):
        task_loss = self.criterion(teacher_logits, labels)
        T = self.config.get("teacher_temperature_fedkd", self.temperature)
        distill_loss = self.distill_loss(
            F.log_softmax(teacher_logits / T, dim=1), F.softmax(student_logits.detach() / T, dim=1)
        ) * (T * T)
        alpha = self.config.get("alpha_tea_fedkd", 1.0)
        return (1 - alpha) * task_loss + alpha * distill_loss

    def _compute_student_loss(self, student_logits, teacher_logits, labels):
        task_loss = self.criterion(student_logits, labels)
        T = self.config.get("student_temperature_fedkd", self.temperature)
        distill_loss = self.distill_loss(
            F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits.detach() / T, dim=1)
        ) * (T * T)
        alpha = self.config.get("alpha_stu_fedkd", 1.0)
        return (1 - alpha) * task_loss + alpha * distill_loss

    def train_student(self):
        self.teacher.eval()
        self.student.train()
        for _ in range(self.config["local_epochs_student_fedkd"]):
            for data, labels in self.loader:
                data, labels = (data.to(device), labels.to(device))
                self.student_opt.zero_grad()
                student_logits = self.student(data)
                with torch.no_grad():
                    teacher_logits = self.teacher(data)
                loss = self._compute_student_loss(student_logits, teacher_logits, labels)
                loss.backward()
                self.student_opt.step()
        return {n: p.clone().detach() for (n, p) in self.student.state_dict().items()}

    def train_teacher(self):
        self.teacher.train()
        self.student.eval()
        for _ in range(self.config["local_epochs_teacher_fedkd"]):
            for data, labels in self.loader:
                data, labels = (data.to(device), labels.to(device))
                self.teacher_opt.zero_grad()
                teacher_logits = self.teacher(data)
                with torch.no_grad():
                    student_logits = self.student(data)
                loss = self._compute_teacher_loss(teacher_logits, student_logits, labels)
                loss.backward()
                self.teacher_opt.step()


class LocalTeacherClient:
    """An independent CE-only teacher with the same settings as FedKD's teacher."""

    def __init__(self, cid, dataset, teacher_width, teacher_init_state, config):
        self.cid = cid
        self.dataset = dataset
        self.teacher_width = int(teacher_width)
        self.config = config
        self.loader = DataLoader(
            dataset, batch_size=config.get("batch_size_fedkd", 50), shuffle=True, num_workers=0
        )
        self.teacher = get_model("teacher", self.teacher_width).to(device)
        self.teacher.load_state_dict(teacher_init_state)
        self.teacher_opt = optim.SGD(
            self.teacher.parameters(),
            lr=config.get("teacher_lr_fedkd", 0.001),
            momentum=config.get("teacher_momentum_fedkd", 0.9),
            weight_decay=config.get("teacher_weight_decay_fedkd", 0.0005),
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_teacher(self):
        self.teacher.train()
        for _ in range(self.config["local_epochs_teacher_fedkd"]):
            for data, labels in self.loader:
                data, labels = (data.to(device), labels.to(device))
                self.teacher_opt.zero_grad()
                loss = self.criterion(self.teacher(data), labels)
                loss.backward()
                self.teacher_opt.step()


class FedKDServer:

    def __init__(self, testset, student_width, student_init_state=None):
        self.test_loader = DataLoader(testset, batch_size=50, shuffle=False, num_workers=0)
        self.global_student = get_model("student", student_width).to(device)
        if student_init_state is not None:
            self.global_student.load_state_dict(student_init_state)
        self.criterion = nn.CrossEntropyLoss()

    def aggregate(self, params_list, client_sizes):
        if len(params_list) == 0:
            return {n: p.clone().detach() for (n, p) in self.global_student.state_dict().items()}
        if isinstance(params_list[0], dict) and "state_dict" in params_list[0]:
            aggregated_params = aggregate_state_dicts_by_overlap(
                self.global_student.state_dict(), params_list, client_sizes
            )
            self.global_student.load_state_dict(aggregated_params)
            return aggregated_params
        total_size = sum(client_sizes)
        aggregated_params = {}
        for name in params_list[0]:
            if params_list[0][name].dtype in [torch.int64, torch.long]:
                total_param = torch.zeros_like(params_list[0][name], dtype=torch.float32)
                for i, params in enumerate(params_list):
                    weight = client_sizes[i] / total_size
                    total_param += params[name].float() * weight
                aggregated_params[name] = total_param.to(params_list[0][name].dtype)
            else:
                total_param = torch.zeros_like(params_list[0][name])
                for i, params in enumerate(params_list):
                    weight = client_sizes[i] / total_size
                    total_param += params[name] * weight
                aggregated_params[name] = total_param
        self.global_student.load_state_dict(aggregated_params)
        return aggregated_params

    def evaluate_individual_teachers(self, models):
        results = []
        for model in models:
            model.eval()
            total_correct = 0
            total_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for data, labels in self.test_loader:
                    data, labels = (data.to(device), labels.to(device))
                    logits = model(data)
                    loss = self.criterion(logits, labels)
                    total_loss += loss.item() * data.size(0)
                    total_correct += (logits.argmax(dim=1) == labels).sum().item()
                    total_samples += data.size(0)
            results.append((total_loss / total_samples, total_correct / total_samples))
        return results


def get_baseline_model_spec(method: str):
    if method not in ("feddf", "fedfd"):
        raise ValueError(f"Unsupported baseline: {method}")
    cfg = ACTIVE_CONFIG.get(method, {}) if isinstance(ACTIVE_CONFIG, dict) else {}
    arch = cfg.get("model_arch", "teacher")
    width = cfg.get("model_width", N_T)
    if arch not in ("teacher", "student"):
        raise ValueError(
            f"{method}.model_arch={arch!r} is invalid; expected 'teacher' or 'student'"
        )
    if not isinstance(width, int):
        raise ValueError(f"{method}.model_width={width!r} is invalid; an integer width is required")
    if width <= 0:
        raise ValueError(f"{method}.model_width must be a positive integer, got {width}")
    return (arch, int(width))


def make_model_init_state(arch: str, width: int, seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return deepcopy(get_model(arch, width).to(device).state_dict())


def model_spec_key(arch: str, width: int) -> str:
    return f"{arch}_w{int(width)}"


def validate_model_spec(arch: str, width: int, *, context: str):
    if arch not in ("teacher", "student"):
        raise ValueError(
            f"{context}.model_arch={arch!r} is invalid; expected 'teacher' or 'student'"
        )
    if not isinstance(width, int):
        raise ValueError(
            f"{context}.model_width={width!r} is invalid; an integer width is required"
        )
    if width <= 0:
        raise ValueError(f"{context}.model_width must be a positive integer, got {width}")


def get_default_heterogeneous_widths(config, num_clients):
    hetero_cfg = config.get("heterogeneous_model", {})
    if hetero_cfg.get("client_widths", None) is not None:
        widths = [int(w) for w in hetero_cfg["client_widths"]]
    else:
        base_widths = [int(w) for w in hetero_cfg.get("widths", [64, 128, 192])]
        counts = [int(c) for c in hetero_cfg.get("counts", [3, 4, 3])]
        if len(base_widths) != len(counts):
            raise ValueError("heterogeneous_model.widths and counts must have equal lengths")
        widths = []
        for width, count in zip(base_widths, counts):
            widths.extend([width] * count)
        assignment_policy = hetero_cfg.get("assignment_policy", "fixed_order")
        if assignment_policy in ("random_permutation", "shuffle"):
            seed = int(hetero_cfg.get("assignment_seed", config.get("model_init_seed", 0)))
            rng = np.random.RandomState(seed)
            widths = [int(w) for w in rng.permutation(widths).tolist()]
        elif assignment_policy != "fixed_order":
            raise ValueError(
                f"heterogeneous_model.assignment_policy must be 'fixed_order', 'random_permutation', or 'shuffle'; got {assignment_policy!r}"
            )
    if len(widths) != num_clients:
        raise ValueError(
            f"The number of heterogeneous client widths must equal num_clients={num_clients}, got {len(widths)}"
        )
    return widths


def make_width_heterogeneous_specs(config, method, num_clients, *, default_arch="teacher"):
    method_cfg = config.get(method, {})
    server_arch = method_cfg.get("model_arch", default_arch)
    server_width = int(
        method_cfg.get("model_width", max(get_default_heterogeneous_widths(config, num_clients)))
    )
    validate_model_spec(server_arch, server_width, context=method)
    raw_specs = method_cfg.get("client_model_specs", None)
    if raw_specs is None:
        widths = get_default_heterogeneous_widths(config, num_clients)
        raw_specs = [{"model_arch": server_arch, "model_width": width} for width in widths]
    if len(raw_specs) != num_clients:
        raise ValueError(
            f"{method}.client_model_specs length must equal num_clients={num_clients}, got {len(raw_specs)}"
        )
    specs = []
    for cid, spec in enumerate(raw_specs):
        arch = spec.get("model_arch", server_arch)
        width = int(spec.get("model_width", server_width))
        validate_model_spec(arch, width, context=f"{method}.client_model_specs[{cid}]")
        if arch != server_arch:
            raise ValueError(
                f"Current {method} heterogeneous implementation uses one base architecture with different widths; client {cid}: {arch!r}, server: {server_arch!r}"
            )
        if width > server_width:
            raise ValueError(
                f"{method}.client_model_specs[{cid}].model_width={width} must not exceed server model_width={server_width}"
            )
        specs.append(
            {
                "cid": cid,
                "model_arch": arch,
                "model_width": width,
                "feature_dim": model_feature_dim_formula(arch, width),
                "group_key": model_spec_key(arch, width),
            }
        )
    return specs


def get_fedkd_teacher_specs(config, num_clients):
    hetero_cfg = config.get("heterogeneous_model", {})
    arch = hetero_cfg.get("model_arch", "teacher")
    if arch != "teacher":
        raise ValueError(
            f"FedWiKD requires heterogeneous_model.model_arch to be teacher; got {arch!r}"
        )
    widths = get_default_heterogeneous_widths(config, num_clients)
    specs = []
    for cid, width in enumerate(widths):
        validate_model_spec("teacher", int(width), context=f"fedkd.teacher_specs[{cid}]")
        specs.append(
            {
                "cid": cid,
                "model_arch": "teacher",
                "model_width": int(width),
                "feature_dim": model_feature_dim_formula("teacher", int(width)),
                "group_key": model_spec_key("teacher", int(width)),
            }
        )
    return specs


def get_fedfd_client_specs(config, num_clients):
    fedfd_cfg = config.get("fedfd", {})
    server_arch = fedfd_cfg.get("model_arch", "teacher")
    server_width = int(fedfd_cfg.get("model_width", config.get("N_T", 128)))
    validate_model_spec(server_arch, server_width, context="fedfd")
    raw_specs = fedfd_cfg.get("client_model_specs", None)
    if raw_specs is None and config.get("heterogeneous_model", {}).get("enabled", False):
        widths = get_default_heterogeneous_widths(config, num_clients)
        raw_specs = [{"model_arch": server_arch, "model_width": width} for width in widths]
    elif raw_specs is None:
        raw_specs = [
            {"model_arch": server_arch, "model_width": server_width} for _ in range(num_clients)
        ]
    if len(raw_specs) != num_clients:
        raise ValueError(
            f"fedfd.client_model_specs length must equal num_clients={num_clients}, got {len(raw_specs)}"
        )
    specs = []
    for cid, spec in enumerate(raw_specs):
        arch = spec.get("model_arch", server_arch)
        width = int(spec.get("model_width", server_width))
        validate_model_spec(arch, width, context=f"fedfd.client_model_specs[{cid}]")
        if arch != server_arch:
            raise ValueError(
                f"FedFD width-heterogeneous aggregation requires matching client and server architectures; client {cid}: {arch!r}, server: {server_arch!r}"
            )
        if width > server_width:
            raise ValueError(
                f"fedfd.client_model_specs[{cid}].model_width={width} must not exceed server model_width={server_width}"
            )
        specs.append(
            {
                "cid": cid,
                "model_arch": arch,
                "model_width": width,
                "feature_dim": model_feature_dim_formula(arch, width),
                "group_key": model_spec_key(arch, width),
            }
        )
    return specs


def load_state_by_overlap(model, source_state):
    target_state = model.state_dict()
    updated_state = {}
    for name, target_tensor in target_state.items():
        if name not in source_state:
            updated_state[name] = target_tensor
            continue
        source_tensor = source_state[name].to(target_tensor.device)
        if source_tensor.shape == target_tensor.shape:
            updated_state[name] = source_tensor.clone().detach().to(dtype=target_tensor.dtype)
            continue
        if source_tensor.ndim != target_tensor.ndim:
            updated_state[name] = target_tensor
            continue
        slices = tuple(
            (slice(0, min(s, t)) for (s, t) in zip(source_tensor.shape, target_tensor.shape))
        )
        cloned = target_tensor.clone()
        cloned[slices] = source_tensor[slices].to(dtype=target_tensor.dtype)
        updated_state[name] = cloned
    model.load_state_dict(updated_state)


def aggregate_state_dicts_by_overlap(global_state, upload_records, client_sizes):
    total_size = float(sum(client_sizes))
    new_state = {}
    for name, global_tensor in global_state.items():
        if not torch.is_floating_point(global_tensor):
            same_shape = all(
                (
                    name in record["state_dict"]
                    and record["state_dict"][name].shape == global_tensor.shape
                    for record in upload_records
                )
            )
            if same_shape:
                total_param = torch.zeros_like(global_tensor, dtype=torch.float32)
                for record, client_size in zip(upload_records, client_sizes):
                    weight = float(client_size) / total_size
                    total_param += (
                        record["state_dict"][name].to(global_tensor.device).float() * weight
                    )
                new_state[name] = total_param.to(global_tensor.dtype)
            else:
                new_state[name] = global_tensor.clone().detach()
            continue
        weighted_sum = torch.zeros_like(global_tensor)
        weight_sum = torch.zeros_like(global_tensor)
        for record, client_size in zip(upload_records, client_sizes):
            state = record["state_dict"]
            if name not in state:
                continue
            client_tensor = state[name].to(global_tensor.device)
            if client_tensor.ndim != global_tensor.ndim:
                continue
            slices = tuple(
                (slice(0, min(c, g)) for (c, g) in zip(client_tensor.shape, global_tensor.shape))
            )
            weight = float(client_size) / total_size
            weighted_sum[slices] += client_tensor[slices].to(dtype=global_tensor.dtype) * weight
            weight_sum[slices] += weight
        updated = global_tensor.clone().detach()
        mask = weight_sum > 0
        updated[mask] = weighted_sum[mask] / weight_sum[mask]
        new_state[name] = updated
    return new_state


class UnlabeledCIFAR100(torchvision.datasets.CIFAR100):

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return (img, torch.tensor(-1))


class FedFDClient:

    def __init__(self, cid, dataset, init_state, config, spec):
        self.cid = cid
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=0)
        self.config = config
        self.model_arch = spec["model_arch"]
        self.model_width = int(spec["model_width"])
        self.feature_dim = int(spec["feature_dim"])
        self.group_key = spec["group_key"]
        self.model = get_model(self.model_arch, self.model_width).to(device)
        load_state_by_overlap(self.model, init_state)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("local_lr_fedfd", 0.001),
            momentum=config.get("local_momentum_fedfd", 0.9),
            weight_decay=config.get("local_weight_decay_fedfd", 0.0001),
        )
        self.criterion = nn.CrossEntropyLoss()

    def update_model(self, global_state_dict):
        load_state_by_overlap(self.model, global_state_dict)

    def train_local(self):
        self.model.train()
        for _ in range(self.config["local_epochs_fedfd"]):
            for data, labels in self.loader:
                data, labels = (data.to(device), labels.to(device))
                self.optimizer.zero_grad()
                logits = self.model(data)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
        return {
            "cid": self.cid,
            "state_dict": {n: p.clone().detach() for (n, p) in self.model.state_dict().items()},
            "model_arch": self.model_arch,
            "model_width": self.model_width,
            "feature_dim": self.feature_dim,
            "group_key": self.group_key,
        }


class FedFDOrthogonalProjection(nn.Module):

    def __init__(self, in_dim, out_dim, series_terms=5):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.series_terms = int(series_terms)
        self.raw_W = nn.Parameter(torch.randn(self.in_dim, self.in_dim) * 0.01)

    def _orthogonal_columns(self):
        skew = self.raw_W - self.raw_W.t()
        q = torch.eye(self.in_dim, device=skew.device, dtype=skew.dtype)
        term = torch.eye(self.in_dim, device=skew.device, dtype=skew.dtype)
        for n in range(1, self.series_terms + 1):
            term = term @ skew / float(n)
            q = q + term
        return q[:, : self.out_dim]

    def forward(self, features):
        projection = self._orthogonal_columns()
        return features @ projection


class FedFDServer:

    def __init__(self, testset, unlabeled_dataset, init_state, config, model_arch, model_width):
        self.test_loader = DataLoader(testset, batch_size=50, shuffle=False, num_workers=0)
        self.unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=config.get("distill_batch_size_fedfd", 64),
            shuffle=True,
            num_workers=0,
        )
        self.config = config
        self.model_arch = model_arch
        self.model_width = int(model_width)
        self.feature_dim = model_feature_dim_formula(model_arch, model_width)
        self.base_group_key = model_spec_key(model_arch, model_width)
        self.global_model = get_model(model_arch, model_width).to(device)
        self.global_model.load_state_dict(init_state)
        self.projections = nn.ModuleDict()
        self.criterion = nn.CrossEntropyLoss()
        self.last_diagnostics = {}

    def _projection_key(self, group_key):
        return group_key.replace(":", "_").replace("-", "_")

    def _ensure_projection(self, group_key, out_dim):
        key = self._projection_key(group_key)
        if key not in self.projections:
            terms = int(self.config.get("orthogonal_series_terms_fedfd", 5))
            self.projections[key] = FedFDOrthogonalProjection(
                self.feature_dim, out_dim, series_terms=terms
            ).to(device)
        return self.projections[key]

    def aggregate_init(self, upload_records, client_sizes):
        if len(upload_records) == 0:
            return {n: p.clone().detach() for (n, p) in self.global_model.state_dict().items()}
        global_state = self.global_model.state_dict()
        averaged_state = aggregate_state_dicts_by_overlap(
            global_state, upload_records, client_sizes
        )
        self.global_model.load_state_dict(averaged_state)
        return averaged_state

    def _build_teacher_models(self, upload_records):
        teacher_models = []
        for record in upload_records:
            model = get_model(record["model_arch"], record["model_width"]).to(device)
            model.load_state_dict(record["state_dict"])
            model.eval()
            teacher_models.append((record, model))
        return teacher_models

    def _group_teacher_features(self, teacher_models, data):
        grouped = {}
        for record, model in teacher_models:
            with torch.no_grad():
                features = model.extract_features(data)
            group_key = record["group_key"]
            if group_key not in grouped:
                grouped[group_key] = {
                    "features": [],
                    "feature_dim": int(record["feature_dim"]),
                    "model_arch": record["model_arch"],
                    "model_width": int(record["model_width"]),
                }
            grouped[group_key]["features"].append(features.detach())
        for group in grouped.values():
            group["teacher_feature"] = torch.stack(group["features"], dim=0).mean(dim=0)
            del group["features"]
        return grouped

    def distill(self, upload_records):
        if len(upload_records) == 0:
            return {n: p.clone().detach() for (n, p) in self.global_model.state_dict().items()}
        teacher_models = self._build_teacher_models(upload_records)
        steps = int(self.config.get("distill_steps_fedfd", 500))
        include_base_group = bool(self.config.get("distill_include_base_group_fedfd", False))
        distill_group_dims = {}
        for record in upload_records:
            group_key = record["group_key"]
            if group_key == self.base_group_key and (not include_base_group):
                continue
            distill_group_dims[group_key] = int(record["feature_dim"])
        if len(distill_group_dims) == 0:
            self.last_diagnostics = {
                "fedfd_group_count": len({r["group_key"] for r in upload_records}),
                "fedfd_active_projection_count": 0,
                "fedfd_distill_steps_run": 0,
                "fedfd_last_distill_loss": None,
                "fedfd_homogeneous_noop": True,
            }
            return {n: p.clone().detach() for (n, p) in self.global_model.state_dict().items()}
        for group_key, out_dim in distill_group_dims.items():
            if group_key != self.base_group_key:
                self._ensure_projection(group_key, out_dim)
        params = list(self.global_model.parameters()) + list(self.projections.parameters())
        optimizer = optim.SGD(
            params,
            lr=self.config.get("distill_lr_fedfd", 0.01),
            momentum=self.config.get("distill_momentum_fedfd", 0.0),
            weight_decay=0.0,
        )
        data_iter = iter(self.unlabeled_loader)
        losses = []
        self.global_model.train()
        for _ in range(steps):
            try:
                data, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(self.unlabeled_loader)
                data, _ = next(data_iter)
            data = data.to(device)
            grouped = self._group_teacher_features(teacher_models, data)
            trainable_losses = []
            global_features = self.global_model.extract_features(data)
            for group_key, group in grouped.items():
                if group_key not in distill_group_dims:
                    continue
                target = group["teacher_feature"]
                if group_key == self.base_group_key:
                    projected = global_features
                else:
                    projection = self._ensure_projection(group_key, int(group["feature_dim"]))
                    projected = projection(global_features)
                student_log_prob = F.log_softmax(projected, dim=1)
                student_prob = student_log_prob.exp()
                teacher_log_prob = F.log_softmax(target, dim=1)
                trainable_losses.append(
                    (student_prob * (student_log_prob - teacher_log_prob)).sum(dim=1).mean()
                )
            if len(trainable_losses) == 0:
                self.last_diagnostics = {
                    "fedfd_group_count": len(grouped),
                    "fedfd_active_projection_count": 0,
                    "fedfd_distill_steps_run": 0,
                    "fedfd_last_distill_loss": None,
                    "fedfd_homogeneous_noop": True,
                }
                break
            optimizer.zero_grad()
            loss = torch.stack(trainable_losses).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if len(losses) > 0:
            self.last_diagnostics = {
                "fedfd_group_count": len({r["group_key"] for r in upload_records}),
                "fedfd_active_projection_count": len(self.projections),
                "fedfd_distill_steps_run": len(losses),
                "fedfd_last_distill_loss": losses[-1],
                "fedfd_homogeneous_noop": False,
            }
        return {n: p.clone().detach() for (n, p) in self.global_model.state_dict().items()}

    def evaluate_global(self):
        self.global_model.eval()
        total_correct = 0
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = (data.to(device), labels.to(device))
                logits = self.global_model(data)
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * data.size(0)
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += data.size(0)
        return (total_loss / total_samples, total_correct / total_samples)


def compute_physical_params(method, n_S, N_T):
    r_bp = 2
    E_local = 5
    D_k = D_K_EFFECTIVE
    L_FLOPS = model_forward_flops_formula("teacher", N_T)
    if method == "fedkd":
        P_param = model_param_count_formula("student", n_S)
        S_data = 32 * P_param
        S_FLOPs = model_forward_flops_formula("student", n_S)
        F_total = (2 + r_bp) * E_local * D_k * (S_FLOPs + L_FLOPS)
    elif method in ("feddf", "fedfd"):
        model_arch, model_width = get_baseline_model_spec(method)
        P_param = model_param_count_formula(model_arch, model_width)
        S_data = 32 * P_param
        forward_flops = model_forward_flops_formula(model_arch, model_width)
        local_cfg = ACTIVE_CONFIG.get(method, {}) if isinstance(ACTIVE_CONFIG, dict) else {}
        if method == "feddf":
            local_epochs = int(local_cfg.get("local_epochs_feddf", E_local))
        elif method == "fedfd":
            local_epochs = int(local_cfg.get("local_epochs_fedfd", E_local))
        else:
            local_epochs = E_local
        F_total = (1 + r_bp) * local_epochs * D_k * forward_flops
    else:
        raise ValueError(f"Unknown method: {method}")
    T_comp = T_FLOP * F_total
    E_comp = E_FLOP * F_total
    T_req = T_MAX - T_comp
    return {
        "F_total": F_total,
        "T_comp": T_comp,
        "E_comp": E_comp,
        "T_req": T_req,
        "S_data": S_data,
    }


def _vector_profile_from_lists(F_total_list, S_data_list):
    F_total_vec = np.asarray(F_total_list, dtype=float)
    S_data_vec = np.asarray(S_data_list, dtype=float)
    T_comp_vec = T_FLOP * F_total_vec
    E_comp_vec = E_FLOP * F_total_vec
    T_req_vec = T_MAX - T_comp_vec
    return {
        "F_total_vec": F_total_vec,
        "T_comp_vec": T_comp_vec,
        "E_comp_vec": E_comp_vec,
        "T_req_vec": T_req_vec,
        "S_data_vec": S_data_vec,
        "F_total": float(np.max(F_total_vec)),
        "T_comp": float(np.max(T_comp_vec)),
        "E_comp": float(np.max(E_comp_vec)),
        "T_req": float(np.min(T_req_vec)),
        "S_data": float(np.max(S_data_vec)),
    }


def compute_heterogeneous_physical_profile(method, client_specs, N_T, config):
    r_bp = 2
    D_k = D_K_EFFECTIVE
    F_total_list = []
    S_data_list = []
    if method == "fedkd":
        fedkd_cfg = config.get("fedkd", {})
        local_epochs = int(fedkd_cfg.get("local_epochs_student_fedkd", 5))
        teacher_flops = model_forward_flops_formula("teacher", N_T)
        for spec in client_specs:
            width = int(spec["model_width"])
            student_flops = model_forward_flops_formula("student", width)
            params = model_param_count_formula("student", width)
            S_data_list.append(32 * params)
            F_total_list.append((2 + r_bp) * local_epochs * D_k * (student_flops + teacher_flops))
    elif method in ("fedfd", "feddf"):
        cfg = config.get(method, {})
        if method == "fedfd":
            local_epochs = int(cfg.get("local_epochs_fedfd", 5))
        elif method == "feddf":
            local_epochs = int(cfg.get("local_epochs_feddf", 5))
        elif method == "fedavg":
            local_epochs = int(cfg.get("local_epochs_fedavg", 5))
        else:
            local_epochs = int(cfg.get("local_epochs_fedprox", 5))
        for spec in client_specs:
            arch = spec["model_arch"]
            width = int(spec["model_width"])
            params = model_param_count_formula(arch, width)
            forward_flops = model_forward_flops_formula(arch, width)
            S_data_list.append(32 * params)
            F_total_list.append((1 + r_bp) * local_epochs * D_k * forward_flops)
    else:
        raise ValueError(f"Unknown method for heterogeneous profile: {method}")
    return _vector_profile_from_lists(F_total_list, S_data_list)


def check_student_parameter_count(n_S, teacher_specs):
    validate_model_spec("student", n_S, context="stage1.student")
    student_params = model_param_count_formula("student", int(n_S))
    teacher_params = [
        model_param_count_formula("teacher", int(spec["model_width"])) for spec in teacher_specs
    ]
    if not teacher_params:
        raise ValueError("teacher_specs must not be empty.")
    bad_client = next(
        (i for (i, count) in enumerate(teacher_params) if student_params >= count), None
    )
    return {
        "student_param_count": student_params,
        "teacher_param_counts": teacher_params,
        "bad_client": bad_client,
    }


def compute_fedkd_heterogeneous_teacher_profile(n_S, teacher_specs, config):
    r_bp = 2
    E_local = 5
    D_k = D_K_EFFECTIVE
    size_check = check_student_parameter_count(n_S, teacher_specs)
    if size_check["bad_client"] is not None:
        raise ValueError(
            f"The student must have fewer parameters than every client teacher; n_S={n_S}, student_params={size_check['student_param_count']}, bad_client={size_check['bad_client']}"
        )
    student_flops = model_forward_flops_formula("student", int(n_S))
    student_params = model_param_count_formula("student", int(n_S))
    student_upload_bits = 32 * student_params
    F_total_list = []
    S_data_list = []
    for spec in teacher_specs:
        teacher_width = int(spec["model_width"])
        teacher_flops = model_forward_flops_formula("teacher", teacher_width)
        F_total_list.append((2 + r_bp) * E_local * D_k * (student_flops + teacher_flops))
        S_data_list.append(student_upload_bits)
    return _vector_profile_from_lists(F_total_list, S_data_list)


def analytical_expected_comm_energy_given_omega(method, n_S, N_T, omega_k):
    mp = compute_physical_params(method, n_S, N_T)
    return analytical_expected_comm_energy_given_profile(mp["S_data"], mp["T_req"], omega_k)


def analytical_expected_comm_energy_given_profile(S_data, T_req, omega_k):
    if T_req <= 0:
        return (np.inf, None)
    M_k = M_TOTAL / K
    target = S_data / T_req * np.log(2) / (M_k * W_RB)
    if exp1(1e-10) <= target:
        return (np.inf, None)
    lo, hi = (1e-10, 100.0)
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if exp1(mid) > target:
            lo = mid
        else:
            hi = mid
        if (hi - lo) / (lo + 1e-30) < 1e-12:
            break
    x_star = 0.5 * (lo + hi)
    E_comm = (
        T_req * M_k * N_0 * W_RB * (np.exp(-x_star) / (x_star * omega_k) - exp1(x_star) / omega_k)
    )
    return (float(E_comm), x_star)


def cantelli_feasible_given_omega(method, n_S, N_T, E_comm, x_star, omega_k):
    if x_star is None or not np.isfinite(E_comm):
        return (False, 1.0)
    mp = compute_physical_params(method, n_S, N_T)
    return cantelli_feasible_given_profile(E_comm, x_star, omega_k, mp["T_req"])


def cantelli_feasible_given_profile(E_comm, x_star, omega_k, T_req):
    if x_star is None or not np.isfinite(E_comm):
        return (False, 1.0)
    if T_req <= 0:
        return (False, 1.0)
    M_k = M_TOTAL / K
    C_omega = N_0 * W_RB / omega_k
    mu_req = E_comm / T_req
    t1 = np.exp(-x_star) / x_star**2
    t2 = np.exp(-x_star) / x_star
    t3 = (2.0 / x_star + 1.0) * exp1(x_star)
    E_p2 = C_omega**2 * (t1 + t2 - t3)
    var = max(E_p2 - (mu_req / M_k) ** 2, 0.0)
    sigma2 = M_k * var
    if mu_req >= P_MAX:
        return (False, 1.0)
    prob = sigma2 / (sigma2 + (P_MAX - mu_req) ** 2)
    return (prob <= DELTA, float(prob))


def compute_comm_profile_for_method(method, n_S, N_T, check_cantelli=True):
    E_comm_vec = []
    x_star_vec = []
    prob_vec = []
    for k, omega_k in enumerate(OMEGA_VEC):
        E_comm_k, x_star_k = analytical_expected_comm_energy_given_omega(method, n_S, N_T, omega_k)
        if not np.isfinite(E_comm_k) or x_star_k is None:
            return {
                "ok": False,
                "reason": "rate_infeasible",
                "bad_client": k,
                "E_comm_vec": None,
                "x_star_vec": None,
                "cantelli_prob_vec": None,
                "E_comm_worst": np.inf,
                "x_star_worst": None,
                "worst_client": k,
            }
        if check_cantelli:
            ok_k, prob_k = cantelli_feasible_given_omega(
                method, n_S, N_T, E_comm_k, x_star_k, omega_k
            )
            if not ok_k:
                return {
                    "ok": False,
                    "reason": "cantelli_cutoff",
                    "bad_client": k,
                    "E_comm_vec": None,
                    "x_star_vec": None,
                    "cantelli_prob_vec": None,
                    "E_comm_worst": np.inf,
                    "x_star_worst": None,
                    "worst_client": k,
                }
            prob_vec.append(prob_k)
        else:
            prob_vec.append(np.nan)
        E_comm_vec.append(E_comm_k)
        x_star_vec.append(x_star_k)
    E_comm_vec = np.array(E_comm_vec, dtype=float)
    x_star_vec = np.array(x_star_vec, dtype=float)
    prob_vec = np.array(prob_vec, dtype=float)
    worst_client = int(np.argmax(E_comm_vec))
    return {
        "ok": True,
        "reason": "ok",
        "bad_client": None,
        "E_comm_vec": E_comm_vec,
        "x_star_vec": x_star_vec,
        "cantelli_prob_vec": prob_vec,
        "E_comm_worst": float(E_comm_vec[worst_client]),
        "x_star_worst": float(x_star_vec[worst_client]),
        "worst_client": worst_client,
    }


def compute_comm_profile_from_vectors(S_data_vec, T_req_vec, check_cantelli=True):
    S_data_vec = np.asarray(S_data_vec, dtype=float)
    T_req_vec = np.asarray(T_req_vec, dtype=float)
    if S_data_vec.shape != (K,) or T_req_vec.shape != (K,):
        raise ValueError(
            f"S_data_vec/T_req_vec must have length K={K} vectors; got {S_data_vec.shape}, {T_req_vec.shape}"
        )
    E_comm_vec = []
    x_star_vec = []
    prob_vec = []
    for k, omega_k in enumerate(OMEGA_VEC):
        if T_req_vec[k] <= 0:
            return {
                "ok": False,
                "reason": "T_req<=0",
                "bad_client": k,
                "E_comm_vec": None,
                "x_star_vec": None,
                "cantelli_prob_vec": None,
                "E_comm_worst": np.inf,
                "x_star_worst": None,
                "worst_client": k,
            }
        E_comm_k, x_star_k = analytical_expected_comm_energy_given_profile(
            S_data_vec[k], T_req_vec[k], omega_k
        )
        if not np.isfinite(E_comm_k) or x_star_k is None:
            return {
                "ok": False,
                "reason": "rate_infeasible",
                "bad_client": k,
                "E_comm_vec": None,
                "x_star_vec": None,
                "cantelli_prob_vec": None,
                "E_comm_worst": np.inf,
                "x_star_worst": None,
                "worst_client": k,
            }
        if check_cantelli:
            ok_k, prob_k = cantelli_feasible_given_profile(
                E_comm_k, x_star_k, omega_k, T_req_vec[k]
            )
            if not ok_k:
                return {
                    "ok": False,
                    "reason": "cantelli_cutoff",
                    "bad_client": k,
                    "E_comm_vec": None,
                    "x_star_vec": None,
                    "cantelli_prob_vec": None,
                    "E_comm_worst": np.inf,
                    "x_star_worst": None,
                    "worst_client": k,
                }
            prob_vec.append(prob_k)
        else:
            prob_vec.append(np.nan)
        E_comm_vec.append(E_comm_k)
        x_star_vec.append(x_star_k)
    E_comm_vec = np.array(E_comm_vec, dtype=float)
    x_star_vec = np.array(x_star_vec, dtype=float)
    prob_vec = np.array(prob_vec, dtype=float)
    worst_client = int(np.argmax(E_comm_vec))
    return {
        "ok": True,
        "reason": "ok",
        "bad_client": None,
        "E_comm_vec": E_comm_vec,
        "x_star_vec": x_star_vec,
        "cantelli_prob_vec": prob_vec,
        "E_comm_worst": float(E_comm_vec[worst_client]),
        "x_star_worst": float(x_star_vec[worst_client]),
        "worst_client": worst_client,
    }


def compute_R_star_heterogeneous(E_comp, E_comm_vec):
    if E_comm_vec is None:
        return 0
    rounds = []
    E_comp_vec = np.asarray(E_comp, dtype=float)
    if E_comp_vec.ndim == 0:
        E_comp_vec = np.full(len(E_comm_vec), float(E_comp_vec))
    for e_comp_k, e_comm_k in zip(E_comp_vec, E_comm_vec):
        if e_comp_k + e_comm_k <= 0:
            rounds.append(0)
        else:
            rounds.append(int(np.floor(E_TOTAL_K / (e_comp_k + e_comm_k))))
    if len(rounds) == 0:
        return 0
    return int(min(rounds))


def evaluate_objective(objective, n_s, R):
    if objective != "heuristic_H1":
        raise ValueError(objective)
    return 1.0 / n_s + 1.0 / R


def run_stage1_with_heterogeneous_teachers(N_T, teacher_specs, objective, verbose=True):
    """Offline Width and Round Selection with client-specific teacher costs."""
    teacher_widths = [int(spec["model_width"]) for spec in teacher_specs]
    candidates = list(ACTIVE_CONFIG["stage1_candidate_widths"])
    if (
        not candidates
        or any((type(n) is not int or n <= 0 for n in candidates))
        or len(candidates) != len(set(candidates))
    ):
        raise ValueError("stage1_candidate_widths must contain distinct positive integers.")
    candidates.sort()
    best_ns = None
    best_R = None
    best_obj = np.inf
    cantelli_cutoff = None
    scan_log = []
    for n_s in candidates:
        size_check = check_student_parameter_count(n_s, teacher_specs)
        if size_check["bad_client"] is not None:
            scan_log.append(
                {
                    "n_s": n_s,
                    "feasible": False,
                    "reason": "student_params_not_smaller_than_teacher",
                    "teacher_widths": teacher_widths,
                    **size_check,
                }
            )
            continue
        profile = compute_fedkd_heterogeneous_teacher_profile(n_s, teacher_specs, ACTIVE_CONFIG)
        T_req_vec = np.asarray(profile["T_req_vec"], dtype=float)
        if np.any(T_req_vec <= 0):
            bad_client = int(np.where(T_req_vec <= 0)[0][0])
            scan_log.append(
                {
                    "n_s": n_s,
                    "feasible": False,
                    "reason": "T_req<=0",
                    "bad_client": bad_client,
                    "teacher_widths": teacher_widths,
                }
            )
            continue
        prof = compute_comm_profile_from_vectors(
            profile["S_data_vec"], profile["T_req_vec"], check_cantelli=True
        )
        if not prof["ok"]:
            if prof["reason"] == "cantelli_cutoff" and cantelli_cutoff is None:
                cantelli_cutoff = n_s
            scan_log.append(
                {
                    "n_s": n_s,
                    "feasible": False,
                    "reason": prof["reason"],
                    "bad_client": prof["bad_client"],
                    "teacher_widths": teacher_widths,
                }
            )
            continue
        R_star = compute_R_star_heterogeneous(profile["E_comp_vec"], prof["E_comm_vec"])
        if R_star <= 0:
            scan_log.append(
                {
                    "n_s": n_s,
                    "feasible": False,
                    "reason": "R_star<=0",
                    "E_comp_vec": profile["E_comp_vec"].tolist(),
                    "E_comm_vec": prof["E_comm_vec"].tolist(),
                    "teacher_widths": teacher_widths,
                }
            )
            continue
        obj = evaluate_objective(objective, n_s, R_star)
        scan_log.append(
            {
                "n_s": n_s,
                "feasible": True,
                "E_comp_vec": profile["E_comp_vec"].tolist(),
                "T_req_vec": profile["T_req_vec"].tolist(),
                "S_data_vec": profile["S_data_vec"].tolist(),
                "E_comm_worst": prof["E_comm_worst"],
                "E_comm_vec": prof["E_comm_vec"].tolist(),
                "cantelli_prob_vec": prof["cantelli_prob_vec"].tolist(),
                "worst_client": prof["worst_client"],
                "R_star": R_star,
                "objective": obj,
                "teacher_widths": teacher_widths,
            }
        )
        if obj < best_obj:
            best_obj = obj
            best_ns = n_s
            best_R = R_star
    if best_ns is None:
        raise RuntimeError(
            f"No feasible heterogeneous-teacher Stage 1 width; teacher_widths={teacher_widths}"
        )
    if verbose:
        print(
            f"  Stage 1 [{objective}, hetero teacher]: n_S* = {best_ns}, R_theory = {best_R}, Cantelli cutoff = {cantelli_cutoff}, teacher_widths = {teacher_widths}"
        )
    return (best_ns, best_R, scan_log)


def _compute_min_power_for_rate(g_km, w_rb, n_0, r_req):
    gains = np.asarray(g_km, dtype=float)
    if gains.ndim != 1 or not np.isfinite(gains).all() or np.any(gains < 0):
        raise ValueError("g_km must be a finite nonnegative vector")
    if not (
        np.isfinite(w_rb)
        and w_rb > 0
        and np.isfinite(n_0)
        and (n_0 > 0)
        and np.isfinite(r_req)
        and (r_req >= 0)
    ):
        raise ValueError("Bandwidth/noise must be positive; target rate must be nonnegative")
    powers = np.zeros_like(gains)
    usable = gains > 0
    if r_req == 0 or not np.any(usable):
        return (powers, 0.0)
    noise = n_0 * w_rb
    inv_g = noise / gains[usable]

    def total_rate(level):
        p = np.maximum(level - inv_g, 0.0)
        return float(np.sum(w_rb * np.log1p(p * gains[usable] / noise) / np.log(2.0)))

    lo = 0.0
    hi = max(float(np.max(inv_g)), np.finfo(float).tiny)
    for _ in range(1024):
        if not np.isfinite(hi):
            raise FloatingPointError("Required water level exceeds finite floating-point range")
        if total_rate(hi) >= r_req:
            break
        hi *= 2.0
    else:
        raise FloatingPointError("Could not bracket the required transmission rate")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total_rate(mid) < r_req:
            lo = mid
        else:
            hi = mid
        if (hi - lo) / (hi + np.finfo(float).tiny) < 1e-12:
            break
    powers[usable] = np.maximum(hi - inv_g, 0.0)
    return (powers, float(np.sum(powers)))


def _apply_final_feasibility_cutoff(
    P_matrix, X_matrix, g_matrix, p_max, w_rb, n_0, r_req, rate_tol=1e-06
):
    powers = np.asarray(P_matrix, dtype=float)
    assignment = np.asarray(X_matrix, dtype=float)
    gains = np.asarray(g_matrix, dtype=float)
    if powers.ndim != 2 or assignment.shape != powers.shape or gains.shape != powers.shape:
        raise ValueError("Power, assignment and channel matrices must have matching shapes")
    count = powers.shape[0]
    rates = np.asarray(r_req, dtype=float)
    if rates.ndim == 0:
        rates = np.full(count, float(rates))
    if rates.shape != (count,) or not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("Required rate must be a nonnegative scalar or length-K vector")
    noise = n_0 * w_rb
    outage = []
    for k in range(count):
        if (
            not np.isfinite(powers[k]).all()
            or np.any(powers[k] < 0)
            or (not np.isfinite(assignment[k]).all())
            or (not np.isfinite(gains[k]).all())
            or np.any(gains[k] < 0)
            or (float(np.sum(powers[k])) > p_max + 1e-09)
        ):
            outage.append(k)
            continue
        rbs = np.flatnonzero(assignment[k] > 0.5)
        if len(rbs) == 0:
            outage.append(k)
            continue
        achieved = float(
            np.sum(w_rb * np.log1p(powers[k, rbs] * gains[k, rbs] / noise) / np.log(2.0))
        )
        if not np.isfinite(achieved) or achieved < (1.0 - rate_tol) * rates[k]:
            outage.append(k)
    return outage


def solver_dual_decomposition(
    g_matrix, S_data, T_req_k, w_rb, n_0, p_max, K, M_total, max_iter=300, feas_tol=0.05
):
    S_data_vec = np.asarray(S_data, dtype=float)
    T_req_vec = np.asarray(T_req_k, dtype=float)
    if S_data_vec.ndim == 0:
        S_data_vec = np.full(K, float(S_data_vec))
    if T_req_vec.ndim == 0:
        T_req_vec = np.full(K, float(T_req_vec))
    if S_data_vec.shape != (K,) or T_req_vec.shape != (K,):
        raise ValueError(
            f"S_data/T_req_k must be scalars or vectors of length K={K} vectors; got {S_data_vec.shape}, {T_req_vec.shape}"
        )
    if np.any(T_req_vec <= 0):
        outage_all = list(range(K))
        return (
            np.ones(K) * 1e-12,
            np.zeros((K, M_total)),
            outage_all,
            0,
            np.zeros((K, M_total)),
            None,
        )
    R_req = S_data_vec / T_req_vec
    noise = n_0 * w_rb
    g_mean = float(np.mean(g_matrix))
    M_per_user = M_total / K
    lam_init = np.zeros(K, dtype=float)
    for k in range(K):

        def _rate_from_lam(lv, kk=k):
            p = max(lv * w_rb / (np.log(2) * T_req_vec[kk]) - noise / (g_mean + 1e-30), 0.0)
            return M_per_user * w_rb * np.log2(1.0 + p * g_mean / (noise + 1e-30))

        lam_lo, lam_hi = (1e-12, 0.01)
        for _ in range(200):
            lam_mid = (lam_lo + lam_hi) / 2.0
            if _rate_from_lam(lam_mid) < R_req[k]:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        lam_init[k] = (lam_lo + lam_hi) / 2.0
    lam = lam_init.copy()
    mu = np.zeros(K)
    step_lam_0 = 0.1 * lam_init / (R_req + 1e-30)
    step_mu_0 = 0.1 * lam_init / (p_max + 1e-30)
    X_bar_accum = np.zeros((K, M_total))
    P_bar_accum = np.zeros((K, M_total))
    iters_done = 0
    for it in range(max_iter):
        iters_done = it + 1
        coeff = lam * w_rb * T_req_vec / ((T_req_vec + mu) * np.log(2) + 1e-30)
        P_star = np.maximum(coeff[:, None] - noise / (g_matrix + 1e-30), 0.0)
        rate_mat = w_rb * np.log2(1.0 + P_star * g_matrix / (noise + 1e-30))
        H_mat = (
            P_star * (T_req_vec[:, None] + mu[:, None])
            - lam[:, None] * rate_mat * T_req_vec[:, None]
        )
        best_k_vec = np.argmin(H_mat, axis=0)
        best_H_vec = H_mat[best_k_vec, np.arange(M_total)]
        X_new = np.zeros((K, M_total))
        P_new = np.zeros((K, M_total))
        for m in np.where(best_H_vec < 0)[0]:
            k_m = best_k_vec[m]
            X_new[k_m, m] = 1.0
            P_new[k_m, m] = P_star[k_m, m]
        X_bar_accum += X_new
        P_bar_accum += P_new
        X_bar = X_bar_accum / iters_done
        P_bar = P_bar_accum / iters_done
        step_lam = step_lam_0 / np.sqrt(it + 1)
        step_mu = step_mu_0 / np.sqrt(it + 1)
        rate_actual = w_rb * np.log2(1.0 + P_new * g_matrix / (noise + 1e-30))
        lam = np.maximum(lam + step_lam * (R_req - np.sum(X_new * rate_actual, axis=1)), 1e-12)
        mu = np.maximum(mu + step_mu * (np.sum(P_new, axis=1) - p_max), 0.0)
        rate_bar = w_rb * np.log2(1.0 + P_bar * g_matrix / (noise + 1e-30))
        R_bar_vec = np.sum(X_bar * rate_bar, axis=1)
        P_bar_vec = np.sum(P_bar, axis=1)
        viol_vec = np.abs(R_bar_vec - R_req) / (R_req + 1e-30)
        if float(np.max(viol_vec)) < feas_tol and (not bool(np.any(P_bar_vec > p_max + 1e-09))):
            break
    X_final = np.zeros((K, M_total))
    for m in range(M_total):
        best_k = int(np.argmax(X_bar[:, m]))
        if X_bar[best_k, m] > 1e-06:
            X_final[best_k, m] = 1.0
    P_final = np.zeros((K, M_total))
    for k in range(K):
        rbs_k = np.where(X_final[k, :] > 0.5)[0]
        if len(rbs_k) > 0:
            p_k, _ = _compute_min_power_for_rate(g_matrix[k, rbs_k], w_rb, n_0, R_req[k])
            P_final[k, rbs_k] = p_k
    outage_list = _apply_final_feasibility_cutoff(
        P_final, X_final, g_matrix, p_max, w_rb, n_0, R_req
    )
    P_final[outage_list, :] = 0.0
    return (lam, X_final, outage_list, iters_done, P_final, None)


def generate_channel_sequence(num_rounds, seed_base=1000):
    seq = []
    for r in range(num_rounds):
        rng = np.random.default_rng(seed_base + r)
        g_matrix = rng.exponential(scale=OMEGA_VEC[:, None], size=(K, M_TOTAL))
        seq.append(g_matrix)
    return seq


_cached_trainset = None
_cached_testset = None
_cached_unlabeled_cifar100 = None


def _load_cifar10():
    global _cached_trainset, _cached_testset
    if _cached_trainset is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        _cached_trainset = torchvision.datasets.CIFAR10(
            root="./data/cifar10", train=True, download=True, transform=transform
        )
        _cached_testset = torchvision.datasets.CIFAR10(
            root="./data/cifar10", train=False, download=True, transform=transform
        )
    return (_cached_trainset, _cached_testset)


def _load_unlabeled_cifar100():
    global _cached_unlabeled_cifar100
    if _cached_unlabeled_cifar100 is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        _cached_unlabeled_cifar100 = UnlabeledCIFAR100(
            root="./data/cifar100", train=True, download=True, transform=transform
        )
    return _cached_unlabeled_cifar100


class BaseRunner:

    def __init__(self, baseline_cfg, config, shared):
        self.cfg = baseline_cfg
        self.tag = baseline_cfg["tag"]
        self.method = baseline_cfg["method"]
        self.config = config
        self.N_T = config["N_T"]
        self.training_seed = config["training_seed"]
        self.shared = shared
        self.trainset = shared["trainset"]
        self.testset = shared["testset"]
        self.client_datasets = shared["client_datasets"]
        self.channel_seq = shared["channel_seq"]
        self.teacher_init_state = shared["teacher_init_state"]
        self.exp_dir = Path(config["output_dir"]) / self.tag
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = self.exp_dir / "log.csv"
        self.config_path = self.exp_dir / "config.json"
        self.final_path = self.exp_dir / "final_result.json"
        self.done_flag = self.exp_dir / "done.flag"

    def determine_n_S(self):
        raise NotImplementedError

    def init_components(self, n_S):
        raise NotImplementedError

    def train_one_round(self, clients, server, success_indices):
        raise NotImplementedError

    def evaluate(self, clients, server):
        raise NotImplementedError

    def get_physical_profile(self, n_S):
        mp = compute_physical_params(self.method, n_S, self.N_T)
        return {
            "F_total_vec": np.full(K, mp["F_total"], dtype=float),
            "T_comp_vec": np.full(K, mp["T_comp"], dtype=float),
            "E_comp_vec": np.full(K, mp["E_comp"], dtype=float),
            "T_req_vec": np.full(K, mp["T_req"], dtype=float),
            "S_data_vec": np.full(K, mp["S_data"], dtype=float),
            **mp,
        }


class LocalTeacherRunner(BaseRunner):
    """Local CE training under the existing computation-time and energy budgets."""

    def _teacher_specs(self):
        return get_fedkd_teacher_specs(self.config, self.config["num_clients"])

    def get_physical_profile(self, n_S=None):
        epochs = self.config["fedkd"]["local_epochs_teacher_fedkd"]
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
            raise ValueError("local_epochs_teacher_fedkd must be a positive integer")
        r_bp = 2
        forward_flops = [
            model_forward_flops_formula("teacher", spec["model_width"])
            for spec in self._teacher_specs()
        ]
        total_flops = [(1 + r_bp) * epochs * D_K_EFFECTIVE * f for f in forward_flops]
        return _vector_profile_from_lists(total_flops, [0.0] * len(total_flops))

    def determine_n_S(self):
        profile = self.get_physical_profile()
        r_theory = compute_R_star_heterogeneous(
            profile["E_comp_vec"], np.zeros(self.config["num_clients"])
        )
        return (
            None,
            r_theory,
            {
                "stage1_policy": "not_applicable_no_student",
                "stage2_policy": "not_applicable_no_communication",
                "communication_applicable": False,
                "teacher_specs": self._teacher_specs(),
                "teacher_training_config": deepcopy(self.config["fedkd"]),
                "loss": "CE (coefficient 1.0); no KD",
                "evaluation": "mean teacher loss/accuracy on the full CIFAR-10 test set",
                "round_definition": "local_epochs_teacher_fedkd local teacher epochs",
                "client_T_comp_vec": profile["T_comp_vec"].tolist(),
                "initial_energy_per_client": E_TOTAL_K,
                "max_rounds_safety": MAX_ROUNDS_SAFETY,
                "config_snapshot": deepcopy(self.config),
            },
        )

    def init_components(self, n_S=None):
        states = self.shared["teacher_init_states_by_width"]
        clients = [
            LocalTeacherClient(
                spec["cid"],
                self.client_datasets[spec["cid"]],
                spec["model_width"],
                states[spec["model_width"]],
                self.config["fedkd"],
            )
            for spec in self._teacher_specs()
        ]
        self.test_loader = DataLoader(self.testset, batch_size=50, shuffle=False, num_workers=0)
        self.test_criterion = nn.CrossEntropyLoss()
        return (clients, None)

    def train_one_round(self, clients, server=None, success_indices=None):
        for client in clients:
            client.train_teacher()

    def evaluate(self, clients, server=None):
        self.client_metrics = []
        for client in clients:
            client.teacher.eval()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for data, labels in self.test_loader:
                    data, labels = (data.to(device), labels.to(device))
                    logits = client.teacher(data)
                    loss = self.test_criterion(logits, labels)
                    total_loss += loss.item() * data.size(0)
                    total_correct += (logits.argmax(dim=1) == labels).sum().item()
                    total_samples += data.size(0)
            if total_samples == 0:
                raise ValueError("Local-Teacher evaluation requires a nonempty test set")
            self.client_metrics.append(
                {
                    "cid": client.cid,
                    "teacher_width": client.teacher_width,
                    "test_loss": total_loss / total_samples,
                    "test_acc": total_correct / total_samples * 100.0,
                }
            )
        return (
            float(np.mean([m["test_loss"] for m in self.client_metrics])),
            float(np.mean([m["test_acc"] for m in self.client_metrics])) / 100.0,
        )

    def save_final_teachers(self, clients, round_number, remaining_energy):
        checkpoint_path = self.exp_dir / "final_teachers.pt"
        temporary_path = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "method": self.method,
                "round": round_number,
                "teacher_total_epochs": round_number
                * self.config["fedkd"]["local_epochs_teacher_fedkd"],
                "teacher_specs": self._teacher_specs(),
                "teacher_state_dicts": {
                    client.cid: {
                        name: value.detach().cpu().clone()
                        for (name, value) in client.teacher.state_dict().items()
                    }
                    for client in clients
                },
                "remaining_energy": remaining_energy.tolist(),
                "config": deepcopy(self.config),
            },
            temporary_path,
        )
        os.replace(temporary_path, checkpoint_path)
        return checkpoint_path.name

    def run(self):
        from core.experiment_runtime import run_method

        return run_method(globals(), self, self.tag, self.config, self.shared)


class FedWiKDRunner(BaseRunner):

    def _teacher_specs(self):
        return get_fedkd_teacher_specs(self.config, self.config["num_clients"])

    def get_physical_profile(self, n_S):
        return compute_fedkd_heterogeneous_teacher_profile(n_S, self._teacher_specs(), self.config)

    def determine_n_S(self):
        teacher_specs = self._teacher_specs()
        n_S, R_theory, scan_log = run_stage1_with_heterogeneous_teachers(
            self.N_T, teacher_specs, objective="heuristic_H1", verbose=True
        )
        with open(self.exp_dir / "stage1_scan.json", "w") as f:
            json.dump(scan_log, f, indent=2, ensure_ascii=False)
        return (
            n_S,
            R_theory,
            {
                "stage1_objective": "heuristic_H1",
                "stage1_policy": "stage1_selected_global_student_width_with_heterogeneous_teachers",
                "fedwikd_student_width_policy": "global_homogeneous_student_from_stage1",
                "fedwikd_teacher_width_policy": "local_heterogeneous_teacher_from_config",
                "fedkd_teacher_specs": teacher_specs,
                "fedkd_teacher_widths": [int(spec["model_width"]) for spec in teacher_specs],
            },
        )

    def init_components(self, n_S):
        teacher_specs = self._teacher_specs()
        teacher_init_states_by_width = self.shared.get("teacher_init_states_by_width", {})
        torch.manual_seed(self.config["model_init_seed"])
        torch.cuda.manual_seed_all(self.config["model_init_seed"])
        temp_student = get_model("student", n_S).to(device)
        student_init = deepcopy(temp_student.state_dict())
        del temp_student
        clients = [
            FedKDClient(
                cid,
                ds,
                student_init_state=student_init,
                config=self.config["fedkd"],
                teacher_width=int(teacher_specs[cid]["model_width"]),
                student_width=n_S,
                teacher_init_state=teacher_init_states_by_width[
                    int(teacher_specs[cid]["model_width"])
                ],
            )
            for (cid, ds) in enumerate(self.client_datasets)
        ]
        server = FedKDServer(self.testset, n_S, student_init)
        return (clients, server)

    def train_one_round(self, clients, server, success_indices):
        params_list = [c.train_student() for c in clients]
        sizes = [len(c.dataset) for c in clients]
        if len(success_indices) > 0:
            valid_params = [params_list[i] for i in success_indices]
            valid_sizes = [sizes[i] for i in success_indices]
            agg = server.aggregate(valid_params, valid_sizes)
        else:
            agg = {n: p.clone().detach() for (n, p) in server.global_student.state_dict().items()}
        for c in clients:
            c.update_student(agg)
            c.train_teacher()

    def evaluate(self, clients, server):
        results = server.evaluate_individual_teachers([c.teacher for c in clients])
        avg_loss = float(np.mean([r[0] for r in results]))
        avg_acc = float(np.mean([r[1] for r in results]))
        return (avg_loss, avg_acc)


class FedFDRunner(BaseRunner):

    def _model_spec(self):
        return get_baseline_model_spec("fedfd")

    def get_physical_profile(self, n_S):
        return compute_heterogeneous_physical_profile(
            "fedfd",
            get_fedfd_client_specs(self.config, self.config["num_clients"]),
            self.N_T,
            self.config,
        )

    def _init_state_for_spec(self, model_arch, model_width):
        if model_arch == "teacher" and model_width == self.N_T:
            return self.teacher_init_state
        return make_model_init_state(model_arch, model_width, self.config["model_init_seed"])

    def determine_n_S(self):
        model_arch, model_width = self._model_spec()
        profile = self.get_physical_profile(model_width)
        prof = compute_comm_profile_from_vectors(
            profile["S_data_vec"], profile["T_req_vec"], check_cantelli=False
        )
        if prof["ok"]:
            R_theory = compute_R_star_heterogeneous(profile["E_comp_vec"], prof["E_comm_vec"])
        else:
            R_theory = -1
        fedfd_cfg = self.config.get("fedfd", {})
        client_specs = get_fedfd_client_specs(self.config, self.config["num_clients"])
        distinct_groups = sorted({spec["group_key"] for spec in client_specs})
        return (
            model_width,
            R_theory,
            {
                "model_arch": model_arch,
                "model_width": model_width,
                "local_epochs_fedfd": fedfd_cfg.get("local_epochs_fedfd", None),
                "local_lr_fedfd": fedfd_cfg.get("local_lr_fedfd", None),
                "local_momentum_fedfd": fedfd_cfg.get("local_momentum_fedfd", None),
                "local_weight_decay_fedfd": fedfd_cfg.get("local_weight_decay_fedfd", None),
                "distill_steps_fedfd": fedfd_cfg.get("distill_steps_fedfd", None),
                "distill_batch_size_fedfd": fedfd_cfg.get("distill_batch_size_fedfd", None),
                "distill_lr_fedfd": fedfd_cfg.get("distill_lr_fedfd", None),
                "distill_momentum_fedfd": fedfd_cfg.get("distill_momentum_fedfd", None),
                "orthogonal_series_terms_fedfd": fedfd_cfg.get(
                    "orthogonal_series_terms_fedfd", None
                ),
                "distill_include_base_group_fedfd": fedfd_cfg.get(
                    "distill_include_base_group_fedfd", False
                ),
                "unlabeled_dataset": fedfd_cfg.get("unlabeled_dataset", "CIFAR100_train"),
                "client_model_specs": client_specs,
                "distinct_architecture_groups": distinct_groups,
                "feature_distillation_loss": "KL(log_softmax(projected_global_feature), softmax(group_mean_feature))",
                "orthogonal_projection": "skew_symmetric_taylor_exp_truncated_columns",
                "server_energy_counted": False,
                "communication_object": "full_client_model_state_dict",
                "comm_profile_ok": prof["ok"],
                "comm_profile_reason": prof["reason"],
                "bad_client": prof["bad_client"],
                "worst_client": prof["worst_client"],
            },
        )

    def init_components(self, n_S):
        model_arch, model_width = self._model_spec()
        init_state = self._init_state_for_spec(model_arch, model_width)
        unlabeled_dataset = self.shared.get("unlabeled_dataset", None)
        if unlabeled_dataset is None:
            raise ValueError("FedFD requires shared['unlabeled_dataset']")
        client_specs = get_fedfd_client_specs(self.config, self.config["num_clients"])
        clients = [
            FedFDClient(cid, ds, init_state, self.config["fedfd"], client_specs[cid])
            for (cid, ds) in enumerate(self.client_datasets)
        ]
        server = FedFDServer(
            self.testset,
            unlabeled_dataset,
            init_state,
            self.config["fedfd"],
            model_arch,
            model_width,
        )
        return (clients, server)

    def train_one_round(self, clients, server, success_indices):
        uploads = [c.train_local() for c in clients]
        sizes = [len(c.dataset) for c in clients]
        if len(success_indices) > 0:
            valid_uploads = [uploads[i] for i in success_indices]
            valid_sizes = [sizes[i] for i in success_indices]
            server.aggregate_init(valid_uploads, valid_sizes)
            global_state = server.distill(valid_uploads)
        else:
            global_state = {
                n: p.clone().detach() for (n, p) in server.global_model.state_dict().items()
            }
        for c in clients:
            c.update_model(global_state)

    def evaluate(self, clients, server):
        return server.evaluate_global()


ALL_BASELINES = {
    "local_teacher": {
        "tag": "local_teacher",
        "method": "local_teacher",
        "runner": "LocalTeacherRunner",
    },
    "fedwikd": {"tag": "fedwikd", "method": "fedkd", "runner": "FedWiKDRunner"},
    "feddf": {"tag": "feddf", "method": "feddf", "runner": "SharedBaselineRunner"},
    "fedfd": {"tag": "fedfd", "method": "fedfd", "runner": "FedFDRunner"},
}


class SharedBaselineRunner:

    def __new__(cls, baseline_cfg, config, shared):
        from core.federated_baselines import BaselineRunner

        return BaselineRunner(globals(), baseline_cfg["method"], config, shared)


def _resolve_runner_class(runner_name):
    return {
        "LocalTeacherRunner": LocalTeacherRunner,
        "FedWiKDRunner": FedWiKDRunner,
        "SharedBaselineRunner": SharedBaselineRunner,
        "FedFDRunner": FedFDRunner,
    }[runner_name]


def prepare_shared_resources(config):
    print(f"\n{'=' * 70}")
    print("  Preparing data partitions, initial teacher weights, and channel sequences")
    print(f"{'=' * 70}")
    trainset, testset = _load_cifar10()
    print(f"  CIFAR-10 loaded: training={len(trainset)}, test={len(testset)}")
    unlabeled_dataset = _load_unlabeled_cifar100()
    print(
        f"  FedDF/FedFD unlabeled distillation data loaded: CIFAR-100 train={len(unlabeled_dataset)}"
    )
    data_rng = np.random.RandomState(config["data_split_seed"])
    split_mode = config.get("data_split_mode", "label_sharding_soft_distance_correlated")
    if split_mode != "label_sharding_soft_distance_correlated":
        raise ValueError("Only soft distance label sharding is supported.")
    sharding_cfg = config.get("label_sharding", {})
    slices_per_client = int(sharding_cfg.get("slices_per_client", 5))
    soft_sigma = float(sharding_cfg.get("soft_sigma", 0.25))
    soft_leakage = float(sharding_cfg.get("soft_leakage", 0.15))
    soft_temperature = float(sharding_cfg.get("soft_temperature", 1.0))
    no_duplicate_class_per_client = bool(sharding_cfg.get("no_duplicate_class_per_client", True))
    class_latent_seed = sharding_cfg.get("class_latent_seed", None)
    max_assignment_restarts = int(sharding_cfg.get("max_assignment_restarts", 1000))
    dist_vec = np.asarray(config["physical"]["dist_vec"], dtype=float)
    omega_vec = config["physical"]["omega_ref"] * (dist_vec / config["physical"]["d_ref"]) ** (
        -config["physical"]["path_loss_exp"]
    )
    client_indices, assignment, distance_group, group_cfg, shard_indices, shard_profiles = (
        split_label_sharding_soft_distance_correlated_with_indices(
            trainset,
            num_clients=config["num_clients"],
            slices_per_client=slices_per_client,
            dist_vec=dist_vec,
            rng=data_rng,
            num_classes=10,
            sigma=soft_sigma,
            leakage=soft_leakage,
            temperature=soft_temperature,
            no_duplicate_class_per_client=no_duplicate_class_per_client,
            class_latent_seed=class_latent_seed,
            max_restarts=max_assignment_restarts,
        )
    )
    client_datasets = [Subset(trainset, idxs) for idxs in client_indices]
    if sharding_cfg.get("save_diagnostics", True):
        diagnostics = save_label_sharding_diagnostics(
            config["output_dir"],
            trainset,
            shard_indices,
            client_indices,
            assignment,
            dist_vec,
            omega_vec,
            sharding_cfg=sharding_cfg,
            split_mode="label_sharding_soft_distance_correlated",
            distance_group=distance_group,
            group_cfg=group_cfg,
            num_classes=10,
        )
        print(f"  Distance-correlated label-sharding diagnostics saved: {diagnostics['diag_dir']}")
        print(
            f"  Soft sharding parameters: slices_per_client={slices_per_client}, sigma={soft_sigma}, leakage={soft_leakage}, temperature={soft_temperature}, no_duplicate_class_per_client={no_duplicate_class_per_client}"
        )
        print(
            "  Class latent positions:",
            [round(float(x), 4) for x in group_cfg.get("class_latent_positions", [])],
        )
        print("  Client-to-shard mapping (decreasing distance):")
        print(
            diagnostics["assignment_df"][
                [
                    "physical_client_id",
                    "distance_m",
                    "distance_group",
                    "assigned_slice_ids",
                    "assigned_slice_dominant_classes",
                    "client_label_skew_score",
                    "client_dominant_class",
                    "client_dominant_ratio",
                ]
            ].to_string(index=False)
        )
    split_desc = f"Label-Sharding-Soft-Distance-Correlated(slices_per_client={slices_per_client}, sigma={soft_sigma}, leakage={soft_leakage}, temperature={soft_temperature}, no_duplicate_class_per_client={no_duplicate_class_per_client}, total_slices={config['num_clients'] * slices_per_client}, total_train_samples={len(trainset)})"
    sizes = [len(ds) for ds in client_datasets]
    label_counts = summarize_client_label_counts(client_datasets)
    print(
        f"  Client data partition complete (mode={split_desc}, seed={config['data_split_seed']}）"
    )
    print(f"  Samples per client = {sizes}")
    print("  Client label counts rows=client, cols=class0..9:")
    print(label_counts)
    channel_seq = generate_channel_sequence(
        config["max_rounds_safety"], seed_base=config["channel_seed_base"]
    )
    print(f"  Channel sequence generated: {len(channel_seq)} rounds x shape={channel_seq[0].shape}")
    teacher_widths = sorted(
        set(get_default_heterogeneous_widths(config, config["num_clients"]) + [config["N_T"]])
    )
    teacher_init_states_by_width = {}
    for width in teacher_widths:
        torch.manual_seed(config["model_init_seed"])
        torch.cuda.manual_seed_all(config["model_init_seed"])
        teacher_init_states_by_width[int(width)] = deepcopy(
            get_model("teacher", int(width)).to(device).state_dict()
        )
    teacher_init_state = teacher_init_states_by_width[int(config["N_T"])]
    num_params = sum(
        (
            v.numel()
            for v in teacher_init_state.values()
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
        )
    )
    print(
        f"  Initial teacher weights generated: N_T={config['N_T']}, floating-point parameter count={num_params:,}, model_init_seed={config['model_init_seed']}"
    )
    print(f"  Widths of initialized heterogeneous teachers = {teacher_widths}")
    return {
        "trainset": trainset,
        "testset": testset,
        "client_datasets": client_datasets,
        "channel_seq": channel_seq,
        "teacher_init_state": teacher_init_state,
        "teacher_init_states_by_width": teacher_init_states_by_width,
        "unlabeled_dataset": unlabeled_dataset,
    }


ALL_BASELINES["heterofl"] = {
    "tag": "heterofl",
    "method": "heterofl",
    "runner": "SharedBaselineRunner",
}
ENABLED_BASELINES = ["fedwikd", "heterofl", "feddf", "fedfd", "local_teacher"]
