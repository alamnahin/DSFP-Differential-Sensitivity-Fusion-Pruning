"""
utils/logging_utils.py  (imported as utils.logging in the project)
Structured logging setup and a CSV-backed ResultsLogger for reproducible
experiment tracking.

ResultsLogger writes:
  results.csv          – one row per (seed × method × pruning_rate)
  ablation.csv         – one row per ablation variant
  summary.csv          – mean ± std across seeds per (method × rate)
"""

from __future__ import annotations
import csv
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: str, level: str = "INFO") -> logging.Logger:
    """
    Configure root logger with both a StreamHandler (stdout) and a
    FileHandler.  Returns the root logger so callers can use it directly.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    return logging.getLogger()


# ---------------------------------------------------------------------------
# ResultsLogger
# ---------------------------------------------------------------------------

_RESULTS_FIELDS = [
    "timestamp", "seed", "method", "pruning_rate",
    "acc_pruned", "acc_kd", "acc_retention_pct",
    "params_M", "params_reduction_pct",
    "flops_M", "flops_reduction_pct",
    "filters",
]

_ABLATION_FIELDS = [
    "timestamp", "seed", "variant",
    "acc_kd", "acc_retention_pct",
    "params_M", "flops_M",
]

_SUMMARY_FIELDS = [
    "method", "pruning_rate",
    "acc_kd_mean", "acc_kd_std",
    "acc_retention_mean", "acc_retention_std",
    "params_M_mean", "flops_M_mean",
    "flops_reduction_mean",
    "n_seeds",
]


class ResultsLogger:
    """
    Append-mode CSV logger for experiment results.

    Files are created on first write with a header row.
    """

    def __init__(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir     = output_dir
        self._results_path  = os.path.join(output_dir, "results.csv")
        self._ablation_path = os.path.join(output_dir, "ablation.csv")
        self._summary_path  = os.path.join(output_dir, "summary.csv")
        self._json_path     = os.path.join(output_dir, "results.json")
        self._all_rows: list[dict] = []
        self._ensure_headers()

    # ── CSV helpers ───────────────────────────────────────────────────────────

    def _ensure_headers(self) -> None:
        for path, fields in [
            (self._results_path,  _RESULTS_FIELDS),
            (self._ablation_path, _ABLATION_FIELDS),
        ]:
            if not os.path.exists(path):
                with open(path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writeheader()

    def _append_row(self, path: str, fields: list[str], row: dict) -> None:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields,
                                    extrasaction="ignore")
            writer.writerow(row)

    # ── Public write methods ──────────────────────────────────────────────────

    def log_entry(self, seed: int, **kwargs: Any) -> None:
        """Log one (seed × method × rate) result row."""
        row = {"timestamp": datetime.utcnow().isoformat(), "seed": seed,
               **kwargs}
        self._append_row(self._results_path, _RESULTS_FIELDS, row)
        self._all_rows.append(row)

    def log_ablation(self, seed: int, results: list[dict]) -> None:
        """Log ablation study rows for one seed."""
        for entry in results:
            row = {"timestamp": datetime.utcnow().isoformat(),
                   "seed": seed, **entry}
            self._append_row(self._ablation_path, _ABLATION_FIELDS, row)

    def summarize(self, all_results: list[dict]) -> None:
        """
        Aggregate across seeds: compute mean ± std per (method, rate),
        write summary.csv and results.json.
        """
        import numpy as np

        # Group rows by (method, rate)
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for seed_result in all_results:
            for pr in seed_result.get("pruning_results", []):
                key = (pr["method"], pr["pruning_rate"])
                groups[key].append(pr)

        summary_rows = []
        for (method, rate), rows in sorted(groups.items()):
            acc_kd_vals  = [r["acc_kd"] for r in rows]
            ret_vals     = [r["acc_retention_pct"] for r in rows]
            params_vals  = [r["params_M"] for r in rows]
            flops_vals   = [r["flops_M"] for r in rows]
            flops_r_vals = [r["flops_reduction_pct"] for r in rows]
            summary_rows.append({
                "method":                method,
                "pruning_rate":          rate,
                "acc_kd_mean":           round(float(np.mean(acc_kd_vals)), 4),
                "acc_kd_std":            round(float(np.std(acc_kd_vals)), 4),
                "acc_retention_mean":    round(float(np.mean(ret_vals)), 2),
                "acc_retention_std":     round(float(np.std(ret_vals)), 2),
                "params_M_mean":         round(float(np.mean(params_vals)), 4),
                "flops_M_mean":          round(float(np.mean(flops_vals)), 2),
                "flops_reduction_mean":  round(float(np.mean(flops_r_vals)), 2),
                "n_seeds":               len(rows),
            })

        with open(self._summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows)

        # Full JSON dump for programmatic use
        with open(self._json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger = logging.getLogger(__name__)
        logger.info(f"Summary written to {self._summary_path}")
        logger.info(f"Full results JSON written to {self._json_path}")

        # Print summary table to stdout
        _print_summary_table(summary_rows)


def _print_summary_table(rows: list[dict]) -> None:
    """Pretty-print the summary table to stdout."""
    if not rows:
        return
    header = (f"{'Method':<10} {'Rate%':>6} {'Acc_KD':>9} "
               f"{'±std':>6} {'Retention%':>11} {'FLOPs_M':>9} "
               f"{'FLOP↓%':>7} {'Seeds':>6}")
    print("\n" + "=" * len(header))
    print("DSFP — Final Results Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['method']:<10} {r['pruning_rate']:>6.0f} "
            f"{r['acc_kd_mean']:>9.4f} {r['acc_kd_std']:>6.4f} "
            f"{r['acc_retention_mean']:>11.2f} "
            f"{r['flops_M_mean']:>9.2f} "
            f"{r['flops_reduction_mean']:>7.2f} "
            f"{r['n_seeds']:>6}"
        )
    print("=" * len(header) + "\n")
