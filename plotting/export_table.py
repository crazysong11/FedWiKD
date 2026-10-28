"""Export the nine Table I rows from plot_final_results.py statistics (standard library only)."""

from pathlib import Path
import argparse
import csv
import json
import math

GROUPS = [
    (
        "homogeneous",
        "Homogeneous",
        [
            ("fedwikd", "FedWiKD"),
            ("fedavg", "FedAvg"),
            ("fedprox", "FedProx"),
            ("scaffold", "SCAFFOLD"),
        ],
    ),
    (
        "heterogeneous",
        "Heterogeneous",
        [
            ("fedwikd", "FedWiKD"),
            ("heterofl", "HeteroFL"),
            ("feddf", "FedDF"),
            ("fedfd", "FedFD"),
            ("local_teacher", "Local training"),
        ],
    ),
]


def export_table(stats_path, output_dir):
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8-sig"))
    if stats.get("record_count") != 100 or stats.get("seeds") != [42, 43, 44, 45, 46]:
        raise ValueError("Expected the final 100-record, five-seed statistics.")
    if stats.get("reported_accuracy", {}).get("mean_field") != "accuracy_last5_mean":
        raise ValueError(
            "Expected last-five-round accuracy statistics, not raw final-round accuracy."
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, scenario, methods in GROUPS:
        for tag, name in methods:
            value = stats["summary"][key][tag]
            row = {
                "scenario": scenario,
                "method": name,
                "accuracy_mean_percent": value["accuracy_last5_mean"],
                "accuracy_sample_sd_percent": value["accuracy_last5_sample_sd"],
                "rounds_mean": value["rounds_mean"],
                "successful_uploads_per_round_mean": value["successful_uploads_per_round_mean"],
            }
            if not all(math.isfinite(v) for v in list(row.values())[2:]):
                raise ValueError(f"Non-finite statistics for {scenario}/{name}")
            rows.append(row)
    with (output / "table1_raw.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    display = [
        {
            "Scenario": r["scenario"],
            "Method": r["method"],
            "Last 5 rounds accuracy (%)": f"{r['accuracy_mean_percent']:.2f} ± {r['accuracy_sample_sd_percent']:.2f}",
            "Rounds": f"{r['rounds_mean']:.1f}",
            "Successful uploads/round": f"{r['successful_uploads_per_round_mean']:.2f}",
        }
        for r in rows
    ]
    with (output / "table1.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(display[0]))
        writer.writeheader()
        writer.writerows(display)
    headers = list(display[0])
    markdown = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    markdown.extend("| " + " | ".join(r.values()) + " |" for r in display)
    (output / "table1.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    latex = [
        r"\begin{table}[!tb]",
        r"\caption{Accuracy and Training Statistics}",
        r"\label{tab:comparison}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Method & Last $5$ rounds accuracy (\%) & Rounds & \begin{tabular}[c]{@{}c@{}}Successful\\uploads/round\end{tabular} \\",
        r"\hline",
    ]
    for _, scenario, _ in GROUPS:
        members = [r for r in rows if r["scenario"] == scenario]
        best_acc = max(r["accuracy_mean_percent"] for r in members)
        best_uploads = max(r["successful_uploads_per_round_mean"] for r in members)
        latex += [r"\multicolumn{4}{c}{\raisebox{-0.5pt}{" + scenario + r"}} \\", r"\hline"]
        for row in members:
            acc = f"{row['accuracy_mean_percent']:.2f}"
            uploads = f"{row['successful_uploads_per_round_mean']:.2f}"
            if row["accuracy_mean_percent"] == best_acc:
                acc = r"\mathbf{" + acc + "}"
            if row["successful_uploads_per_round_mean"] == best_uploads:
                uploads = r"\mathbf{" + uploads + "}"
            latex.append(
                row["method"]
                + " & $"
                + acc
                + r"\pm"
                + f"{row['accuracy_sample_sd_percent']:.2f}"
                + "$ & $"
                + f"{row['rounds_mean']:.1f}"
                + "$ & $"
                + uploads
                + r"$ \\"
            )
        latex.append(r"\hline")
    latex += [r"\end{tabular}", r"\end{table}"]
    (output / "table1.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statistics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    output = args.output_dir or args.statistics.resolve().parent
    rows = export_table(args.statistics, output)
    print(f"Exported {len(rows)} Table I rows to {output}")


if __name__ == "__main__":
    main()
