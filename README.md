# FedWiKD

Code for **Joint Model Capacity and Transmission Optimization for Wireless Federated Knowledge Distillation**.

FedWiKD combines offline student-width selection with online wireless resource allocation to balance model capacity and training rounds under client resource constraints.

## Requirements

- Python 3.10
- PyTorch 2.9.0 and torchvision 0.24.0
- CUDA 12.6 for the GPU environment
- Other dependencies: see `requirements.txt`

```bash
pip install -r requirements.txt
```

## Datasets

- Source: [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html).
- **CIFAR-10**: local training and evaluation; stored in `data/cifar10/`.
- **CIFAR-100**: unlabeled distillation data for FedDF and FedFD; stored in `data/cifar100/`.

The experiment entries download missing data and perform preprocessing automatically.

## Quick Start

Check the installation, models, and offline planning without downloading data or training:

```bash
python -m unittest discover -s tests -v
```

To start a single-seed FedWiKD experiment:

```bash
python experiments/homogeneous.py --seed 42 --methods fedwikd
```

Run commands from the repository root. The smoke test is a quick check; the full experiments require model training.

## Code Structure

```text
├── configs
│   ├── homogeneous.json                # Homogeneous experiment settings
│   ├── heterogeneous.json              # Heterogeneous experiment settings
│   └── width_study.json                # Student width study; inherits homogeneous settings
├── experiments
│   ├── homogeneous.py                  # Homogeneous comparison entry
│   ├── heterogeneous.py                # Heterogeneous comparison entry
│   ├── width_study.py               # Student width study under homogeneous settings
│   ├── run_homogeneous_multiseed.py     # Homogeneous comparison across multiple seeds
│   ├── run_heterogeneous_multiseed.py   # Heterogeneous comparison across multiple seeds
│   └── run_width_study_multiseed.py  # Student width study across multiple seeds
├── core
│   ├── models.py                       # Teacher and student CNNs and operation counts
│   ├── data_partition.py               # Data partitioning and distribution diagnostics
│   ├── homogeneous_engine.py           # Models, data partitioning, planning, and training
│   ├── heterogeneous_engine.py         # Corresponding routines for heterogeneous teachers
│   ├── federated_baselines.py          # Shared baseline implementations
│   ├── experiment_runtime.py           # Evaluation, resource accounting, and logging
│   ├── configuration.py                # JSON configuration loading and inheritance
│   └── launcher.py                     # Experiment execution and seed management
├── plotting
│   ├── reproduce_paper.py              # Generate Figure 1 and Table I
│   ├── plot_final_results.py           # Plot curves with moving averages and error bands
│   ├── export_table.py                 # Export Table I from computed statistics
│   └── result_index.py                 # Locate and validate experiment records
├── tests
│   └── test_smoke.py                   # Configuration, model, and planning checks
├── requirements.txt                    # Python dependencies
└── README.md
```

## Running Experiments

- Settings: edit `configs/*.json`, or pass `--config PATH` to an experiment entry.
- Seeds: edit `SEED_POOL` in a multi-seed entry, or pass `--seeds 42 43 44 45 46`.
- Results are summarized across seeds as means and sample standard deviations; numerical variation can occur across hardware and software environments.

**Homogeneous teachers:** FedWiKD, FedAvg, FedProx, and SCAFFOLD.

```bash
python experiments/run_homogeneous_multiseed.py
```

**Heterogeneous teachers:** FedWiKD, HeteroFL, FedDF, FedFD, and local training.

```bash
python experiments/run_heterogeneous_multiseed.py
```

**Student width study:** evaluates the importance of student width optimization over widths 0, 10, ..., 100. Width 0 denotes local training without a student.

```bash
python experiments/run_width_study_multiseed.py
```

To run a single seed or select methods:

```bash
python experiments/homogeneous.py --seed 42 --methods fedwikd fedavg
python experiments/heterogeneous.py --seed 42
python experiments/width_study.py --seed 42
```

Results are saved under `results/`, with separate directories for each run. Each method records `log.csv`, `clients.csv`, `config.json`, and `final_result.json`.

## Reproducing Figures and Tables

| Paper output | Experiment command |
|---|---|
| Figure 1(a): Homogeneous teachers | `python experiments/run_homogeneous_multiseed.py` |
| Figure 1(b): Heterogeneous teachers | `python experiments/run_heterogeneous_multiseed.py` |
| Figure 1(c): Student width study | `python experiments/run_width_study_multiseed.py` |
| Table I: Accuracy and Training Statistics | Homogeneous and heterogeneous commands above |

After completing all three experiments for seeds 42--46, generate Figure 1 and Table I:

```bash
python plotting/reproduce_paper.py --data-root results --output-dir figures
```

- **Figure 1(a)--(b):** centered five-round moving averages, with the mean and sample standard deviation across seeds.
- **Figure 1(c) and Table I:** last-five-round accuracy averaged within each seed, then summarized across seeds.
- Outputs are saved to `figures/`. Use one completed record per seed and method/width to avoid duplicate-run errors.
