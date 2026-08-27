"""Plot raw constraint-violation rate against the number of active constraints."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_paths = sorted(glob.glob(args.reports_glob))
    if not report_paths:
        raise FileNotFoundError(
            f"no constraint reports matched {args.reports_glob!r}"
        )

    rows = []
    for report_path in report_paths:
        with open(report_path, "r", encoding="utf-8") as stream:
            report = json.load(stream)
        active_columns = report.get("active_numerical_columns", [])
        hit_rate = float(report["raw_joint_hit_rate"])
        num_samples = int(report["num_samples"])
        standard_error = math.sqrt(hit_rate * (1.0 - hit_rate) / num_samples)
        training_hit_rate = report.get("training_partial_hit_rate")
        rows.append(
            {
                "constraint_count": len(active_columns),
                "active_columns": ",".join(str(value) for value in active_columns),
                "num_samples": num_samples,
                "raw_joint_hit_rate": hit_rate,
                "raw_violation_rate": 1.0 - hit_rate,
                "violation_ci95_low": max(0.0, 1.0 - hit_rate - 1.96 * standard_error),
                "violation_ci95_high": min(1.0, 1.0 - hit_rate + 1.96 * standard_error),
                "training_partial_hit_rate": training_hit_rate,
                "training_violation_rate": (
                    None if training_hit_rate is None else 1.0 - float(training_hit_rate)
                ),
                "report": report_path,
            }
        )

    rows.sort(key=lambda row: row["constraint_count"])
    counts = [row["constraint_count"] for row in rows]
    if len(set(counts)) != len(counts):
        raise ValueError("expected exactly one report for each constraint count")
    expected = list(range(min(counts), max(counts) + 1))
    if counts != expected:
        raise ValueError(f"constraint-count reports are incomplete: found {counts}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "constraint_violation_by_count.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "constraint_violation_by_count.json"
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    generated = [row["raw_violation_rate"] for row in rows]
    lower_error = [
        row["raw_violation_rate"] - row["violation_ci95_low"] for row in rows
    ]
    upper_error = [
        row["violation_ci95_high"] - row["raw_violation_rate"] for row in rows
    ]

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.errorbar(
        counts,
        generated,
        yerr=[lower_error, upper_error],
        marker="o",
        linewidth=2,
        capsize=3,
        label="Doob-guided generations",
    )
    if all(row["training_violation_rate"] is not None for row in rows):
        axis.plot(
            counts,
            [row["training_violation_rate"] for row in rows],
            marker="s",
            linestyle="--",
            label="Training-data prevalence baseline",
        )
    axis.set_xlabel("Number of active numerical constraints")
    axis.set_ylabel("Joint constraint violation rate")
    axis.set_xticks(counts)
    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plot_path = output_dir / "constraint_violation_by_count.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    print(f"Read {len(rows)} reports covering constraint counts {counts}")
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
