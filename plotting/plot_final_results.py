"""Reproduce Fig. 1 and its statistics from the final, unchanged experiment logs.

Python 3.10 or newer with numpy and matplotlib. No training or source-data writes occur.
Example (from any directory):
  python plot_final_results.py --data-root PATH --output-dir PATH
Each trajectory is smoothed over five centered rounds (shortened at boundaries),
then aligned to the five seeds' common energy interval without extrapolation.
Shaded bands show mean +/- sample SD across the smoothed seed trajectories.
Panels (a) and (b) display only the first 3000 J and 4000 J, respectively.
The table and width panel average each seed's last five completed rounds before
computing the across-seed mean and sample SD. Width zero (pure local training) is
connected to the other widths. Raw final-round statistics are retained in the audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
from result_index import load_index

ACCURACY_TAIL_ROUNDS = 5

# Subfigure labels and captions are managed by LaTeX, not embedded in panels.
FIGURE1_LATEX = r"""\begin{figure*}[!t]
\centering
\setlength{\abovecaptionskip}{2pt}
\captionsetup[subfloat]{captionskip=2pt}
\subfloat[Homogeneous teachers]{%
\includegraphics[width=0.32\textwidth]{figures/redrawn/simulation-homogeneous.pdf}%
\label{fig:simulation-homogeneous}}\hfill
\subfloat[Heterogeneous teachers]{%
\includegraphics[width=0.32\textwidth]{figures/redrawn/simulation-heterogeneous.pdf}%
\label{fig:simulation-heterogeneous}}\hfill
\subfloat[Student width study]{%
\includegraphics[width=0.32\textwidth]{figures/redrawn/simulation-width.pdf}%
\label{fig:simulation-width}}
\caption{Test accuracy. Panels (a)--(b) use centered moving averages over up to five rounds per run; (c) averages each run's last five rounds. Lines and shading show means and $\pm$ one sample standard deviation across seeds.}
\label{fig:simulation-results}
\end{figure*}
"""


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def centered_moving_average(values, window):
    """Use available neighbors at the edges, without padding or extrapolation."""
    half = window // 2
    return np.array(
        [
            np.mean(values[max(0, i - half) : min(len(values), i + half + 1)])
            for i in range(len(values))
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Odd number of training rounds; applied only to panels (a) and (b).",
    )
    args = parser.parse_args()
    if args.smoothing_window < 1 or args.smoothing_window % 2 != 1:
        parser.error("--smoothing-window must be a positive odd integer")
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(data_root)
    assert len(index) == 100
    groups = {}
    provenance = []
    per_seed_accuracy = []
    for record in index:
        directory = data_root / record["result_directory"]
        logs = read_csv(directory / "log.csv")
        clients = read_csv(directory / "clients.csv")
        final = json.loads((directory / "final_result.json").read_text(encoding="utf-8-sig"))
        rounds = int(record["rounds"])
        assert len(logs) == rounds == int(final["R_actual"]), record["id"]
        assert [int(row["round"]) for row in logs] == list(range(1, rounds + 1))
        assert len(clients) == 10 * rounds
        accuracy = float(record["final_accuracy_percent"])
        assert abs(accuracy - float(logs[-1]["avg_acc"])) < 1e-7
        assert abs(accuracy - float(final["final_acc"])) < 1e-7
        assert rounds >= ACCURACY_TAIL_ROUNDS
        tail_rows = logs[-ACCURACY_TAIL_ROUNDS:]
        tail_accuracy = float(np.mean([float(row["avg_acc"]) for row in tail_rows]))
        energy = np.array([float(row["cumulative_E"]) for row in logs])
        assert np.all(np.diff(energy) > 0)
        assert abs(energy[-1] - float(record["cumulative_energy_J"])) < 1e-6
        key = record["width"] if record["experiment"] == "width_study" else record["method"]
        groups.setdefault((record["experiment"], key), []).append(
            {
                "seed": int(record["seed"]),
                "accuracy": accuracy,
                "accuracy_last5": tail_accuracy,
                "rounds": rounds,
                "energy": energy,
                "curve": np.array([float(row["avg_acc"]) for row in logs]),
                "uploads": float(np.mean([float(row["success_count"]) for row in logs])),
            }
        )
        per_seed_accuracy.append(
            {
                "id": record["id"],
                "experiment": record["experiment"],
                "method_or_width": key,
                "seed": int(record["seed"]),
                "first_averaged_round": int(tail_rows[0]["round"]),
                "last_averaged_round": rounds,
                "round_count": ACCURACY_TAIL_ROUNDS,
                "accuracy_last5_mean": tail_accuracy,
                "raw_final_round_accuracy": accuracy,
                "source_directory": record["result_directory"],
            }
        )
        provenance.append(
            {
                "id": record["id"],
                "directory": record["result_directory"],
                "sha256": {
                    name: digest(directory / name)
                    for name in ("config.json", "log.csv", "clients.csv", "final_result.json")
                },
            }
        )
    summary = {}
    flat = []
    for (experiment, key), runs in groups.items():
        runs.sort(key=lambda run: run["seed"])
        assert [run["seed"] for run in runs] == [42, 43, 44, 45, 46]
        values = dict(
            accuracy_mean=float(np.mean([r["accuracy"] for r in runs])),
            accuracy_sample_sd=float(np.std([r["accuracy"] for r in runs], ddof=1)),
            accuracy_last5_mean=float(np.mean([r["accuracy_last5"] for r in runs])),
            accuracy_last5_sample_sd=float(np.std([r["accuracy_last5"] for r in runs], ddof=1)),
            rounds_mean=float(np.mean([r["rounds"] for r in runs])),
            successful_uploads_per_round_mean=float(np.mean([r["uploads"] for r in runs])),
            final_energy_mean_J=float(np.mean([r["energy"][-1] for r in runs])),
        )
        summary.setdefault(experiment, {})[key] = values
        flat.append(dict(experiment=experiment, method_or_width=key, **values))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.30))
    fig.subplots_adjust(left=0.056, right=0.996, bottom=0.23, top=0.89, wspace=0.32)
    definitions = [
        (
            "homogeneous",
            "(a) Homogeneous teachers",
            [
                ("fedwikd", "FedWiKD", "#0072B2", "-", "o"),
                ("fedavg", "FedAvg", "#D55E00", "--", "s"),
                ("fedprox", "FedProx", "#009E73", "-.", "^"),
                ("scaffold", "SCAFFOLD", "#CC79A7", ":", "D"),
            ],
        ),
        (
            "heterogeneous",
            "(b) Heterogeneous teachers",
            [
                ("fedwikd", "FedWiKD", "#0072B2", "-", "o"),
                ("heterofl", "HeteroFL", "#D55E00", "--", "s"),
                ("feddf", "FedDF", "#009E73", "-.", "^"),
                ("fedfd", "FedFD", "#CC79A7", ":", "D"),
                ("local_teacher", "Local training", "#777777", (0, (4, 1, 1, 1)), "v"),
            ],
        ),
    ]
    curve_ranges = {}
    curve_records = []
    for ax, (experiment, title, styles) in zip(axes[:2], definitions):
        for method, label, color, linestyle, marker in styles:
            runs = groups[(experiment, method)]
            start = max(r["energy"][0] for r in runs)
            stop = min(r["energy"][-1] for r in runs)
            grid = np.linspace(start, stop, 240)
            values = np.array(
                [
                    np.interp(
                        grid,
                        r["energy"],
                        centered_moving_average(r["curve"], args.smoothing_window),
                    )
                    for r in runs
                ]
            )
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1)
            ax.fill_between(
                grid / 1000, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0, zorder=1
            )
            ax.plot(
                grid / 1000,
                mean,
                label=label,
                color=color,
                linestyle=linestyle,
                zorder=3 if method == "fedwikd" else 2,
            )
            curve_records.extend(
                {
                    "experiment": experiment,
                    "method": method,
                    "energy_J": float(x),
                    "smoothed_accuracy_mean": float(y),
                    "sample_sd": float(s),
                    "seed_count": len(runs),
                }
                for x, y, s in zip(grid, mean, sd)
            )
            curve_ranges[f"{experiment}/{method}"] = [start, stop]
        ax.set_title(title, pad=5)
        ax.set_xlabel("Cumulative client energy (kJ)", labelpad=2)
        ax.set_ylabel("Test accuracy (%)", labelpad=2)
        ax.set_xlim(0, 3 if experiment == "homogeneous" else 4)
        ax.set_ylim(20, 65)
        ax.set_yticks([20, 30, 40, 50, 60])
        ax.set_xticks([0, 1, 2, 3] if experiment == "homogeneous" else [0, 1, 2, 3, 4])
        ax.legend(
            loc="lower right",
            framealpha=0.95,
            edgecolor="0.8",
            ncol=2,
            fontsize=6.2,
            borderpad=0.35,
            labelspacing=0.3,
            handlelength=1.6,
            columnspacing=0.7,
            handletextpad=0.4,
        )

    ax = axes[2]
    widths = list(range(0, 101, 10))
    acc = np.array([summary["width_study"][str(w)]["accuracy_last5_mean"] for w in widths])
    sd = np.array([summary["width_study"][str(w)]["accuracy_last5_sample_sd"] for w in widths])
    ax.fill_between(widths, acc - sd, acc + sd, color="#0072B2", alpha=0.18, linewidth=0, zorder=1)
    ax.plot(widths, acc, "o-", color="#0072B2", markersize=3, label="FedWiKD", zorder=3)
    ax.annotate(
        "Reduces to\nlocal training",
        xy=(0, acc[0]),
        xytext=(8, 32),
        fontsize=7,
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "color": "0.3",
            "lw": 0.7,
            "connectionstyle": "arc3,rad=-0.15",
        },
    )
    ax.axvline(40, color="#555555", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(43, 47, "Selected\n" + r"$n_{\mathrm{S}}^*=40$", fontsize=7, va="center")
    ax.set_title("(c) Student width study", pad=5)
    ax.set_xlabel(r"Student width $n_{\mathrm{S}}$", labelpad=2)
    ax.set_ylabel("Accuracy over last 5 rounds (%)", labelpad=2, fontsize=7.5)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([30, 40, 50, 60])
    ax.set_xlim(-5, 105)
    ax.set_ylim(28, 65)
    for ax in axes:
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.tick_params(length=2.5, width=0.6, pad=2)
    fig.savefig(
        output_dir / "simulation-results-row.pdf",
        metadata={
            "Title": "Final five-seed simulation results",
            "Author": "",
            "Subject": "Centered five-round moving averages for energy curves; last-five-round averages with across-seed sample SD for width study and table",
        },
    )
    fig.savefig(output_dir / "simulation-results-row.png", dpi=300)
    # The combined row is a preview; the manuscript uses these three PDFs.
    # Equal crop margins and heights keep axes aligned after LaTeX placement.
    for ax in axes:
        ax.set_title("")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inches = fig.dpi_scale_trans.inverted()
    frames = [ax.get_window_extent(renderer).transformed(inches) for ax in axes]
    bounds = [ax.get_tightbbox(renderer).transformed(inches) for ax in axes]
    left = max(frame.x0 - bound.x0 for frame, bound in zip(frames, bounds))
    right = max(bound.x1 - frame.x1 for frame, bound in zip(frames, bounds))
    bottom = min(bound.y0 for bound in bounds)
    top = max(bound.y1 for bound in bounds)
    pad = 0.025
    panel_names = ("simulation-homogeneous", "simulation-heterogeneous", "simulation-width")
    for name, frame, panel in zip(panel_names, frames, axes):
        # Exclude neighboring axes from the PDF content, not just its page box.
        for other in axes:
            other.set_visible(other is panel)
        bbox = Bbox.from_extents(
            frame.x0 - left - pad, bottom - pad, frame.x1 + right + pad, top + pad
        )
        fig.savefig(
            output_dir / f"{name}.pdf",
            bbox_inches=bbox,
            pad_inches=0,
            metadata={"Title": name, "Author": ""},
        )
        fig.savefig(output_dir / f"{name}.png", bbox_inches=bbox, pad_inches=0, dpi=300)
    (output_dir / "figure1.tex").write_text(FIGURE1_LATEX, encoding="utf-8")
    plt.close(fig)
    audit = {
        "seeds": [42, 43, 44, 45, 46],
        "record_count": len(index),
        "accuracy_units": "percent",
        "dispersion": "sample standard deviation (ddof=1)",
        "curve_method": "Centered moving average within each seed before linear interpolation on the common energy interval of all five runs; no extrapolation. Lines and shaded bands use the mean and sample SD across smoothed seeds. Display ranges are limited to 3000 J and 4000 J; exported curves and final statistics retain the complete runs.",
        "display_energy_limits_J": {"homogeneous": 3000, "heterogeneous": 4000},
        "smoothing": {
            "window_rounds": args.smoothing_window,
            "type": "centered arithmetic moving average",
            "boundary_rule": "Use available neighbors within the centered window; no padding",
            "applied_to": "Training trajectories in panels (a) and (b) only",
            "summary_and_width_statistics": "Separate fixed last-five-round average within each seed, followed by across-seed mean and sample SD",
        },
        "reported_accuracy": {
            "within_seed": "Arithmetic mean of the last five completed rounds",
            "round_count": ACCURACY_TAIL_ROUNDS,
            "across_seeds": "Equal-weight mean and sample standard deviation of five seed-level averages; not 25 independent observations",
            "mean_field": "accuracy_last5_mean",
            "sd_field": "accuracy_last5_sample_sd",
            "raw_final_round_fields_retained": ["accuracy_mean", "accuracy_sample_sd"],
        },
        "width_panel": "Last-five-round accuracy means +/- sample SD across seeds; widths 0 to 100 connected in one curve without smoothing across widths. Width 0 is pure local training without a student.",
        "uploads_method": "Average uploads per round within each seed, then equal average across five seeds.",
        "curve_energy_intervals_J": curve_ranges,
        "summary": summary,
        "index_sha256": hashlib.sha256(
            json.dumps(index, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "provenance": provenance,
    }
    (output_dir / "final-results-statistics.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "final-results-statistics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    with (output_dir / "smoothed-energy-curves.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_records[0]))
        writer.writeheader()
        writer.writerows(curve_records)
    with (output_dir / "accuracy-last5-per-seed.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed_accuracy[0]))
        writer.writeheader()
        writer.writerows(per_seed_accuracy)
    print(
        json.dumps(
            {
                "records_verified": len(index),
                "summary": summary,
                "output_directory": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
