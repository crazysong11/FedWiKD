"""Shared federated baseline implementations."""

from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def clone_state(model):
    return {k: v.detach().clone() for (k, v) in model.state_dict().items()}


def average_same_shape(states, weights):
    if not states:
        raise ValueError("Cannot average an empty prototype group")
    total = float(sum(weights))
    output = {}
    for key, first in states[0].items():
        if any((s[key].shape != first.shape for s in states)):
            raise ValueError("FedDF only averages parameters within identical prototypes")
        if first.is_floating_point():
            output[key] = sum((s[key] * (w / total) for (s, w) in zip(states, weights)))
        else:
            output[key] = torch.stack([s[key] for s in states]).max(dim=0).values
    return output


def extract_subnetwork(global_state, local_model):
    """Fixed, nested channel indices; reject unsupported tensor mappings."""
    output = {}
    for key, local in local_model.state_dict().items():
        source = global_state[key]
        if source.ndim != local.ndim or any((a < b for (a, b) in zip(source.shape, local.shape))):
            raise ValueError(f"Not a nested subnetwork: {key}")
        output[key] = source[tuple((slice(0, n) for n in local.shape))].clone()
    return output


def aggregate_subnetworks(global_state, states, label_sets=None):
    """HeteroFL coordinate-wise mean over clients containing each parameter.

    The output classifier is aggregated only over clients observing its label,
    as in the official HeteroFL implementation. No padding contributes weight.
    """
    output = deepcopy(global_state)
    for key, old in global_state.items():
        if not old.is_floating_point():
            raise ValueError("HeteroFL sBN must not upload running-statistic buffers")
        total = torch.zeros_like(old)
        count = torch.zeros_like(old)
        for i, state in enumerate(states):
            value = state[key]
            if value.ndim != old.ndim or any((a > b for (a, b) in zip(value.shape, old.shape))):
                raise ValueError(f"Unsupported subnetwork shape: {key}")
            sl = tuple((slice(0, n) for n in value.shape))
            if key in ("fc2.weight", "fc2.bias") and label_sets is not None:
                for label in label_sets[i]:
                    idx = (int(label),) + sl[1:]
                    total[idx] += value[int(label)]
                    count[idx] += 1
            else:
                total[sl] += value
                count[sl] += 1
        mask = count > 0
        output[key][mask] = total[mask] / count[mask]
    return output


def make_heterofl_model(base_class, width, global_width):
    """Project three-convolution CNN with HeteroFL normalization and scaling."""

    class HeteroFLCNN(base_class):

        def __init__(self):
            super().__init__(width)
            self.rate = width / global_width
            self.bn1 = nn.BatchNorm2d(width, momentum=None, track_running_stats=False)
            self.bn2 = nn.BatchNorm2d(width, momentum=None, track_running_stats=False)
            self.bn3 = nn.BatchNorm2d(width, momentum=None, track_running_stats=False)

        def extract_features(self, x):
            for conv, bn in (
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3),
            ):
                x = conv(x)
                if self.training:
                    x = x / self.rate
                x = self.pool(F.relu(bn(x)))
            return F.relu(self.fc1(x.flatten(1)))

    return HeteroFLCNN()


class LocalClient:

    def __init__(self, cid, dataset, model, config, method, device):
        self.cid, self.dataset, self.model = (cid, dataset, model)
        self.config, self.method, self.device = (config, method, device)
        self.loader = DataLoader(
            dataset, batch_size=config.get("batch_size", 50), shuffle=True, num_workers=0
        )
        self.lr = float(config.get(f"local_lr_{method}", 0.001))
        self.epochs = int(config.get(f"local_epochs_{method}", 5))
        momentum = float(config.get(f"local_momentum_{method}", 0.9))
        decay = float(config.get(f"local_weight_decay_{method}", 0.0005))
        if self.lr <= 0 or self.epochs < 1:
            raise ValueError("Local learning rate and epochs must be positive")
        if method == "scaffold" and (momentum != 0 or decay != 0):
            raise ValueError("SCAFFOLD option II uses plain SGD in this implementation")
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=self.lr, momentum=momentum, weight_decay=decay
        )
        self.control = {k: torch.zeros_like(v) for (k, v) in model.named_parameters()}
        self.pending_control = None
        self.labels = []
        if method == "heterofl":
            indices = np.arange(len(dataset))
            base = dataset
            while hasattr(base, "indices") and hasattr(base, "dataset"):
                indices = np.asarray(base.indices)[indices]
                base = base.dataset
            if hasattr(base, "targets"):
                labels = np.asarray(base.targets)[indices]
            else:
                labels = [int(dataset[i][1]) for i in range(len(dataset))]
            self.labels = sorted(set((int(v) for v in labels)))

    def train_local(self, global_control=None):
        self.model.train()
        initial = {k: v.detach().clone() for (k, v) in self.model.named_parameters()}
        steps = 0
        for _ in range(self.epochs):
            for data, labels in self.loader:
                data, labels = (data.to(self.device), labels.to(self.device))
                self.optimizer.zero_grad()
                logits = self.model(data)
                if self.method == "heterofl" and self.config.get("masked_cross_entropy", True):
                    mask = torch.zeros(logits.shape[1], device=logits.device, dtype=torch.bool)
                    mask[self.labels] = True
                    logits = logits.masked_fill(~mask, 0.0)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                if global_control is not None:
                    for key, param in self.model.named_parameters():
                        param.grad.add_(global_control[key] - self.control[key])
                self.optimizer.step()
                steps += 1
        if global_control is not None:
            if not steps:
                raise ValueError("SCAFFOLD requires a nonempty client dataset")
            self.pending_control = {
                key: self.control[key]
                - global_control[key]
                + (initial[key] - param.detach()) / (steps * self.lr)
                for (key, param) in self.model.named_parameters()
            }
        return clone_state(self.model)


class BaselineRunner:

    def __init__(self, namespace, method, config, shared):
        self.ns, self.method, self.config, self.shared = (namespace, method, config, shared)
        self.exp_dir = Path(config["output_dir"]) / method
        self.exp_dir.mkdir(exist_ok=True)
        self.device = namespace["device"]
        self.N_T = int(config["N_T"])
        self.datasets = shared["client_datasets"]
        self.cfg = config[method]
        if method == "scaffold":
            if (
                self.cfg.get("server_learning_rate", 1.0) != 1.0
                or self.cfg.get("control_option", "II") != "II"
            ):
                raise ValueError("This SCAFFOLD configuration supports server step 1 and option II")
        if method == "heterofl":
            if (
                self.cfg.get("normalization") != "static_batch_norm"
                or self.cfg.get("scaler") is not True
            ):
                raise ValueError("The configured HeteroFL baseline requires sBN and Scaler")
            if self.cfg.get("inference_statistics") != "batch_statistics_no_global_calibration":
                raise ValueError("Unsupported HeteroFL inference-statistics setting")
        if method in ("heterofl", "feddf") and "make_width_heterogeneous_specs" in namespace:
            self.specs = namespace["make_width_heterogeneous_specs"](
                config, method, len(self.datasets), default_arch="teacher"
            )
        else:
            self.specs = [{"model_arch": "teacher", "model_width": self.N_T} for _ in self.datasets]
        if any((s["model_arch"] != "teacher" for s in self.specs)):
            raise ValueError("This experiment uses the predefined teacher CNN family")
        self.global_width = max((s["model_width"] for s in self.specs))
        self.prototype_models = {}
        self.global_control = None

    def get_physical_profile(self, unused=None):
        flops, bits = ([], [])
        epochs = int(self.cfg[f"local_epochs_{self.method}"])
        for ds, spec in zip(self.datasets, self.specs):
            width = int(spec["model_width"])
            p = self.ns["model_param_count_formula"]("teacher", width)
            forward = self.ns["model_forward_flops_formula"]("teacher", width)
            work = 3 * epochs * len(ds) * forward
            payload = 32 * p
            if self.method == "scaffold":
                steps = epochs * int(np.ceil(len(ds) / self.cfg.get("batch_size", 50)))
                work += (2 * steps + 3) * p
                if self.cfg.get("count_control_upload", True):
                    payload += 32 * p
            flops.append(work)
            bits.append(payload)
        physical = self.config["physical"]
        work = np.asarray(flops, dtype=float)
        comp_time = work * physical["T_FLOP"]
        return {
            "F_total_vec": work,
            "T_comp_vec": comp_time,
            "E_comp_vec": work * physical["E_FLOP"],
            "T_req_vec": physical["T_MAX"] - comp_time,
            "S_data_vec": np.asarray(bits, dtype=float),
        }

    def determine_n_S(self):
        profile = self.get_physical_profile()
        from core.experiment_runtime import expected_rounds

        rounds = expected_rounds(self.ns, self.config, profile)
        return (
            self.global_width,
            rounds,
            {
                "client_model_specs": self.specs,
                "algorithm": self.method,
                "control_update": (
                    "option_II_acknowledged_uploads_only" if self.method == "scaffold" else None
                ),
                "wireless_policy": "all_compute_successful_uploads_aggregate_all_receive",
            },
        )

    def _new_model(self, width):
        torch.manual_seed(self.config["model_init_seed"])
        if self.method == "heterofl":
            model = make_heterofl_model(self.ns["TeacherCNN"], width, self.global_width)
        else:
            model = self.ns["get_model"]("teacher", width)
        return model.to(self.device)

    def init_components(self, unused=None):
        self.clients = []
        if self.method != "feddf":
            self.global_model = self._new_model(self.global_width)
            if self.method == "scaffold":
                self.global_model.load_state_dict(self.shared["teacher_init_state"])
            initial = clone_state(self.global_model)
        for cid, (dataset, spec) in enumerate(zip(self.datasets, self.specs)):
            width = int(spec["model_width"])
            model = self._new_model(width)
            if self.method == "feddf":
                key = ("teacher", width)
                if key not in self.prototype_models:
                    prototype = self._new_model(width)
                    states = self.shared.get("teacher_init_states_by_width", {})
                    if width in states:
                        prototype.load_state_dict(states[width])
                    self.prototype_models[key] = prototype
                model.load_state_dict(self.prototype_models[key].state_dict())
            elif self.method == "heterofl":
                model.load_state_dict(extract_subnetwork(initial, model))
            else:
                model.load_state_dict(initial)
            client = LocalClient(cid, dataset, model, self.cfg, self.method, self.device)
            client.model_width = width
            self.clients.append(client)
        if self.method == "scaffold":
            self.global_control = {
                k: torch.zeros_like(v) for (k, v) in self.global_model.named_parameters()
            }
        return (self.clients, self)

    def _distill_prototype(self, model, teachers):
        dataset = self.shared["unlabeled_dataset"]
        loader = DataLoader(
            dataset, batch_size=self.cfg["distill_batch_size_feddf"], shuffle=True, num_workers=0
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg["distill_lr_feddf"])
        temperature = float(self.cfg["distill_temperature_feddf"])
        data_iter = iter(loader)
        model.train()
        for _ in range(int(self.cfg["distill_steps_feddf"])):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            data = batch[0].to(self.device)
            with torch.no_grad():
                target_logits = torch.stack([teacher(data) for teacher in teachers]).mean(0)
                target = F.softmax(target_logits / temperature, dim=1)
            optimizer.zero_grad()
            loss = (
                F.kl_div(
                    F.log_softmax(model(data) / temperature, dim=1), target, reduction="batchmean"
                )
                * temperature**2
            )
            loss.backward()
            optimizer.step()

    def train_one_round(self, clients, server, success_indices):
        states = [c.train_local(self.global_control) for c in clients]
        success = list(success_indices)
        if self.method == "scaffold":
            if success:
                self.global_model.load_state_dict(
                    average_same_shape([states[i] for i in success], [1.0] * len(success))
                )
                for key in self.global_control:
                    delta = sum(
                        (clients[i].pending_control[key] - clients[i].control[key] for i in success)
                    )
                    self.global_control[key].add_(delta / len(clients))
                for i in success:
                    clients[i].control = clients[i].pending_control
            for client in clients:
                client.pending_control = None
                client.model.load_state_dict(self.global_model.state_dict())
        elif self.method == "heterofl":
            if success:
                state = aggregate_subnetworks(
                    clone_state(self.global_model),
                    [states[i] for i in success],
                    [clients[i].labels for i in success],
                )
                self.global_model.load_state_dict(state)
            for client in clients:
                client.model.load_state_dict(
                    extract_subnetwork(self.global_model.state_dict(), client.model)
                )
        elif self.method == "feddf":
            if success:
                teachers = []
                for i in success:
                    teacher = deepcopy(clients[i].model).eval()
                    for p in teacher.parameters():
                        p.requires_grad_(False)
                    teachers.append(teacher)
                for key, prototype in self.prototype_models.items():
                    group = [i for i in success if clients[i].model_width == key[1]]
                    if group:
                        prototype.load_state_dict(
                            average_same_shape(
                                [states[i] for i in group], [len(clients[i].dataset) for i in group]
                            )
                        )
                    self._distill_prototype(prototype, teachers)
            for client in clients:
                client.model.load_state_dict(
                    self.prototype_models["teacher", client.model_width].state_dict()
                )
        else:
            raise ValueError(self.method)
