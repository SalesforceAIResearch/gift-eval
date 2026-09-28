#!/usr/bin/env python3
"""Validate a GIFT-Eval leaderboard result submission.

Run from the repository root with, for example:

    python scripts/validate_results.py results/my_model
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


DATASET_CONFIGURATIONS_PATH = Path(__file__).with_name("dataset_configurations.csv")


def _load_dataset_configurations() -> dict[str, tuple[str, int]]:
    with DATASET_CONFIGURATIONS_PATH.open(encoding="utf-8", newline="") as handle:
        return {
            row["dataset"]: (row["domain"], int(row["num_variates"]))
            for row in csv.DictReader(handle)
        }


EXPECTED_DATASETS = _load_dataset_configurations()
EXPECTED_DATA_ROWS = len(EXPECTED_DATASETS)
EXPECTED_COLUMNS = (
    "dataset",
    "model",
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain",
    "num_variates",
)
METRIC_COLUMNS = tuple(
    column for column in EXPECTED_COLUMNS if column.startswith("eval_metrics/")
)
REQUIRED_CONFIG_FIELDS = (
    "model",
    "model_type",
    "testdata_leakage",
    "replication_code_available",
)
VALID_MODEL_TYPES = {
    "statistical",
    "deep-learning",
    "agentic",
    "pretrained",
    "fine-tuned",
    "zero-shot",
}
VALID_YES_NO = {"Yes", "No"}
FOLDER_NAME_EXCEPTIONS = {
    # A slash cannot be represented inside a single filesystem folder name.
    "STRIDE w/ Synapse": "STRIDE_w_Synapse",
}


def _load_config(config_path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"config.json must be valid UTF-8 JSON: {exc}")
        return None

    if not isinstance(config, dict):
        errors.append("config.json must contain a JSON object")
        return None

    for field in REQUIRED_CONFIG_FIELDS:
        if field not in config:
            errors.append(f"config.json is missing required field {field!r}")
        elif not isinstance(config[field], str) or not config[field].strip():
            errors.append(f"config.json field {field!r} must be a non-empty string")

    model_type = config.get("model_type")
    if model_type not in VALID_MODEL_TYPES:
        errors.append(
            f"config.json model_type must be one of {sorted(VALID_MODEL_TYPES)}; "
            f"got {model_type!r}"
        )

    for field in ("testdata_leakage", "replication_code_available"):
        value = config.get(field)
        if value not in VALID_YES_NO:
            errors.append(
                f"config.json {field} must be one of {sorted(VALID_YES_NO)}; "
                f"got {value!r}"
            )

    return config


def _load_results(results_path: Path, errors: list[str]) -> list[dict[str, str]]:
    try:
        with results_path.open(encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.reader(handle, strict=True))
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"all_results.csv must be valid UTF-8 CSV: {exc}")
        return []

    if not raw_rows:
        errors.append("all_results.csv must not be empty")
        return []

    header = tuple(raw_rows[0])
    if header != EXPECTED_COLUMNS:
        errors.append(
            "all_results.csv must have exactly these 15 columns in this order: "
            + ", ".join(EXPECTED_COLUMNS)
        )

    data_rows = raw_rows[1:]
    if len(data_rows) != EXPECTED_DATA_ROWS:
        errors.append(
            f"all_results.csv must contain {EXPECTED_DATA_ROWS} data rows "
            f"({EXPECTED_DATA_ROWS + 1} lines including the header); got "
            f"{len(data_rows)} data rows"
        )

    for line_number, row in enumerate(data_rows, start=2):
        if len(row) != len(EXPECTED_COLUMNS):
            errors.append(
                f"all_results.csv line {line_number} has {len(row)} columns; "
                f"expected {len(EXPECTED_COLUMNS)}"
            )

    if header != EXPECTED_COLUMNS or any(
        len(row) != len(EXPECTED_COLUMNS) for row in data_rows
    ):
        return []

    return [dict(zip(EXPECTED_COLUMNS, row)) for row in data_rows]


def _validate_rows(
    rows: list[dict[str, str]], expected_model: Any, errors: list[str]
) -> None:
    datasets: dict[str, int] = {}

    for line_number, row in enumerate(rows, start=2):
        dataset = row["dataset"].strip()
        if not dataset:
            errors.append(f"all_results.csv line {line_number} has an empty dataset")
        elif dataset not in EXPECTED_DATASETS:
            errors.append(
                f"all_results.csv line {line_number} has unknown dataset "
                f"configuration {dataset!r}"
            )
        elif dataset in datasets:
            errors.append(
                f"all_results.csv line {line_number} duplicates dataset {dataset!r} "
                f"from line {datasets[dataset]}"
            )
        else:
            datasets[dataset] = line_number

        if row["model"] != expected_model:
            errors.append(
                f"all_results.csv line {line_number} model must be "
                f"{expected_model!r}; got {row['model']!r}"
            )

        if dataset in EXPECTED_DATASETS:
            expected_domain, expected_num_variates = EXPECTED_DATASETS[dataset]
            if row["domain"] != expected_domain:
                errors.append(
                    f"all_results.csv line {line_number} domain for {dataset!r} "
                    f"must be {expected_domain!r}; got {row['domain']!r}"
                )
        else:
            expected_num_variates = None

        try:
            num_variates = float(row["num_variates"])
            if (
                not math.isfinite(num_variates)
                or not num_variates.is_integer()
                or num_variates <= 0
            ):
                raise ValueError
        except ValueError:
            errors.append(
                f"all_results.csv line {line_number} num_variates must be a "
                f"positive integer; got {row['num_variates']!r}"
            )
        else:
            if (
                expected_num_variates is not None
                and num_variates != expected_num_variates
            ):
                errors.append(
                    f"all_results.csv line {line_number} num_variates for "
                    f"{dataset!r} must be {expected_num_variates}; got "
                    f"{row['num_variates']!r}"
                )

        for column in METRIC_COLUMNS:
            value = row[column]
            if not value.strip():
                continue
            try:
                number = float(value)
            except ValueError:
                errors.append(
                    f"all_results.csv line {line_number} {column} must be numeric; "
                    f"got {value!r}"
                )
                continue
            if not math.isfinite(number):
                errors.append(
                    f"all_results.csv line {line_number} {column} must be finite; "
                    f"got {value!r}"
                )
            elif number < 0:
                errors.append(
                    f"all_results.csv line {line_number} {column} must be "
                    f"non-negative; got {value!r}"
                )

    missing_datasets = sorted(EXPECTED_DATASETS.keys() - datasets.keys())
    if missing_datasets:
        errors.append(
            "all_results.csv is missing dataset configurations: "
            + ", ".join(missing_datasets)
        )


def validate_submission(submission_dir: Path) -> list[str]:
    """Return all validation errors for one ``results/<MODEL_NAME>`` folder."""

    submission_dir = Path(submission_dir)
    errors: list[str] = []

    if not submission_dir.is_dir():
        return [f"submission path is not a directory: {submission_dir}"]

    files = sorted(path.name for path in submission_dir.iterdir())
    expected_files = ["all_results.csv", "config.json"]
    if files != expected_files:
        errors.append(
            f"submission folder must contain exactly {expected_files}; got {files}"
        )

    config_path = submission_dir / "config.json"
    results_path = submission_dir / "all_results.csv"
    if not config_path.is_file() or not results_path.is_file():
        return errors

    config = _load_config(config_path, errors)
    rows = _load_results(results_path, errors)
    if config is None:
        return errors

    config_model = config.get("model")
    folder_model = submission_dir.name
    expected_folder = (
        FOLDER_NAME_EXCEPTIONS.get(config_model, config_model)
        if isinstance(config_model, str)
        else config_model
    )
    if folder_model != expected_folder:
        errors.append(
            f"submission folder name and config.json model must match exactly: "
            f"{folder_model!r} != {config_model!r}"
        )

    if rows:
        expected_row_model = (
            config_model
            if isinstance(config_model, str)
            and FOLDER_NAME_EXCEPTIONS.get(config_model) == folder_model
            else folder_model
        )
        _validate_rows(
            rows,
            expected_row_model,
            errors,
        )

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate one or more results/<MODEL_NAME> submission folders."
    )
    parser.add_argument("submission_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)

    failed = False
    for submission_dir in args.submission_dirs:
        errors = validate_submission(submission_dir)
        if errors:
            failed = True
            print(f"FAIL {submission_dir}", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        else:
            print(f"PASS {submission_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
