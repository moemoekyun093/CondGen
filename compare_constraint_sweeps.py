"""Compare Doob and HARPOON over identical nested numerical constraints."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import pandas as pd
import torch

from tabdiff.doob_h_evaluation import raw_constraint_report
from tabdiff.metrics import TabMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doob-samples-glob", required=True)
    parser.add_argument("--harpoon-samples-glob", required=True)
    parser.add_argument("--real-data", default="synthetic/shoppers/real.csv")
    parser.add_argument("--info-file", default="data/shoppers/info.json")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def density_metrics(
    reference_path: Path,
    samples: pd.DataFrame,
    info: dict,
) -> dict[str, float]:
    evaluator = TabMetrics(
        str(reference_path),
        str(reference_path),
        None,
        info,
        torch.device("cpu"),
        metric_list=["density"],
        include_density_diagnostic=False,
    )
    metrics, _ = evaluator.evaluate(samples.copy())
    return {name: float(value) for name, value in metrics.items()}


def canonical_columns(query: dict) -> tuple[tuple[str, float | None, float | None], ...]:
    return tuple(
        (
            str(column["name"]),
            None if column.get("raw_lower") is None else float(column["raw_lower"]),
            None if column.get("raw_upper") is None else float(column["raw_upper"]),
        )
        for column in query["columns"]
    )


def discover(pattern: str) -> dict[int, tuple[Path, dict]]:
    discovered = {}
    for filename in sorted(glob.glob(pattern)):
        samples_path = Path(filename)
        query_path = samples_path.with_suffix(".query.json")
        if not query_path.is_file():
            raise FileNotFoundError(query_path)
        with open(query_path, "r", encoding="utf-8") as stream:
            query = json.load(stream)
        count = len(query.get("columns", []))
        if count <= 0:
            raise ValueError(f"query has no active columns: {query_path}")
        if count in discovered:
            raise ValueError(f"multiple samples found for constraint count {count}")
        discovered[count] = (samples_path, query)
    if not discovered:
        raise FileNotFoundError(f"no samples matched {pattern!r}")
    return discovered


def main() -> None:
    args = parse_args()
    real_path = Path(args.real_data)
    info_path = Path(args.info_file)
    for path in (real_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    doob = discover(args.doob_samples_glob)
    harpoon = discover(args.harpoon_samples_glob)
    if set(doob) != set(harpoon):
        raise ValueError(
            "Doob and HARPOON levels differ: "
            f"Doob={sorted(doob)}, HARPOON={sorted(harpoon)}"
        )

    real = pd.read_csv(real_path)
    with open(info_path, "r", encoding="utf-8") as stream:
        info = json.load(stream)
    output_dir = Path(args.output_dir)
    reference_dir = output_dir / "conditional_real_references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for count in sorted(doob):
        doob_path, doob_query = doob[count]
        harpoon_path, harpoon_query = harpoon[count]
        if canonical_columns(doob_query) != canonical_columns(harpoon_query):
            raise ValueError(
                f"methods use different constraints at level {count}: "
                f"{doob_path} versus {harpoon_path}"
            )

        _, real_mask = raw_constraint_report(real, doob_query)
        conditional_real = real.loc[real_mask].reset_index(drop=True)
        if len(conditional_real) < 2:
            raise ValueError(
                f"fewer than two real rows satisfy level {count}"
            )
        reference_path = reference_dir / f"k{count:02d}.csv"
        conditional_real.to_csv(reference_path, index=False)

        for method, samples_path, query in (
            ("Doob", doob_path, doob_query),
            ("HARPOON", harpoon_path, harpoon_query),
        ):
            samples = pd.read_csv(samples_path)
            if list(samples.columns) != list(real.columns):
                raise ValueError(
                    f"column mismatch between {samples_path} and {real_path}"
                )
            constraint_report, _ = raw_constraint_report(samples, query)
            metrics = density_metrics(reference_path, samples, info)
            hit_rate = float(constraint_report["joint_hit_rate"])
            sample_count = len(samples)
            standard_error = math.sqrt(
                hit_rate * (1.0 - hit_rate) / sample_count
            )
            rows.append(
                {
                    "method": method,
                    "constraint_count": count,
                    "active_columns": ",".join(
                        column["name"] for column in query["columns"]
                    ),
                    "generated_rows": sample_count,
                    "conditional_real_rows": len(conditional_real),
                    "raw_joint_hit_rate": hit_rate,
                    "violation_rate": 1.0 - hit_rate,
                    "violation_ci95_low": max(
                        0.0, 1.0 - hit_rate - 1.96 * standard_error
                    ),
                    "violation_ci95_high": min(
                        1.0, 1.0 - hit_rate + 1.96 * standard_error
                    ),
                    "shape": metrics["density/Shape"],
                    "trend": metrics["density/Trend"],
                    "overall": metrics["density/Overall"],
                    "samples": str(samples_path),
                }
            )

    csv_path = output_dir / "doob_vs_harpoon_by_constraint_count.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "doob_vs_harpoon_by_constraint_count.json"
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    colors = {"Doob": "tab:blue", "HARPOON": "tab:orange"}
    for method in ("Doob", "HARPOON"):
        method_rows = [row for row in rows if row["method"] == method]
        counts = [row["constraint_count"] for row in method_rows]
        violations = [row["violation_rate"] for row in method_rows]
        lower = [
            row["violation_rate"] - row["violation_ci95_low"]
            for row in method_rows
        ]
        upper = [
            row["violation_ci95_high"] - row["violation_rate"]
            for row in method_rows
        ]
        axes[0].errorbar(
            counts,
            violations,
            yerr=[lower, upper],
            marker="o",
            linewidth=2,
            capsize=3,
            color=colors[method],
            label=method,
        )
        axes[1].plot(
            counts,
            [row["shape"] for row in method_rows],
            marker="o",
            linewidth=2,
            color=colors[method],
            label=method,
        )
        axes[2].plot(
            counts,
            [row["trend"] for row in method_rows],
            marker="o",
            linewidth=2,
            color=colors[method],
            label=method,
        )

    axes[0].set_title("Constraint violations (lower is better)")
    axes[0].set_ylabel("Joint violation rate")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_title("Column Shape (higher is better)")
    axes[1].set_ylabel("Shape score")
    axes[2].set_title("Column-pair Trend (higher is better)")
    axes[2].set_ylabel("Trend score")
    counts = sorted(doob)
    for axis in axes:
        axis.set_xlabel("Number of active numerical constraints")
        axis.set_xticks(counts)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    plot_path = output_dir / "doob_vs_harpoon_shape_trend_violation.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    print(f"Compared levels {sorted(doob)} against filtered real references")
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
