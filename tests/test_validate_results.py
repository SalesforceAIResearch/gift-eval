from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.validate_results import (
    EXPECTED_COLUMNS,
    EXPECTED_DATASETS,
    EXPECTED_DATA_ROWS,
    VALID_MODEL_TYPES,
    main,
    validate_submission,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_SUBMISSIONS = sorted(
    path
    for path in (REPOSITORY_ROOT / "results").iterdir()
    if (path / "config.json").is_file()
)


def make_submission(tmp_path: Path, model: str = "example-model") -> Path:
    submission_dir = tmp_path / model
    submission_dir.mkdir()
    config = {
        "model": model,
        "model_type": "zero-shot",
        "model_dtype": "float32",
        "model_link": "https://example.com/model",
        "code_link": "https://example.com/code",
        "org": "Example Org",
        "testdata_leakage": "No",
        "replication_code_available": "Yes",
    }
    (submission_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with (submission_dir / "all_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPECTED_COLUMNS)
        for dataset, (domain, num_variates) in EXPECTED_DATASETS.items():
            writer.writerow(
                [dataset, model]
                + ["1.0"] * 11
                + [domain, str(num_variates)]
            )

    return submission_dir


def update_config(submission_dir: Path, **updates: str) -> None:
    config_path = submission_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(updates)
    config_path.write_text(json.dumps(config), encoding="utf-8")


def update_csv_cell(
    submission_dir: Path, row_index: int, column: str, value: str
) -> None:
    results_path = submission_dir / "all_results.csv"
    with results_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    rows[row_index][EXPECTED_COLUMNS.index(column)] = value
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_valid_submission_passes(tmp_path: Path) -> None:
    submission_dir = make_submission(tmp_path)

    assert validate_submission(submission_dir) == []
    assert main([str(submission_dir)]) == 0


@pytest.mark.parametrize("model_type", sorted(VALID_MODEL_TYPES))
def test_all_documented_model_types_are_valid(
    tmp_path: Path, model_type: str
) -> None:
    submission_dir = make_submission(tmp_path)
    update_config(submission_dir, model_type=model_type)

    assert validate_submission(submission_dir) == []


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_type": "foundation"}, "model_type must be one of"),
        ({"testdata_leakage": "false"}, "testdata_leakage must be one of"),
        (
            {"replication_code_available": "N/A"},
            "replication_code_available must be one of",
        ),
    ],
)
def test_invalid_config_labels_fail(
    tmp_path: Path, updates: dict[str, str], message: str
) -> None:
    submission_dir = make_submission(tmp_path)
    update_config(submission_dir, **updates)

    assert any(message in error for error in validate_submission(submission_dir))


def test_folder_config_and_every_csv_model_must_match(tmp_path: Path) -> None:
    submission_dir = make_submission(tmp_path)
    update_config(submission_dir, model="different-config-model")
    update_csv_cell(submission_dir, 3, "model", "different-row-model")

    errors = validate_submission(submission_dir)

    assert any("folder name and config.json model" in error for error in errors)
    assert any("line 4 model must be 'example-model'" in error for error in errors)


def test_wrong_row_count_and_columns_fail(tmp_path: Path) -> None:
    submission_dir = make_submission(tmp_path)
    results_path = submission_dir / "all_results.csv"
    with results_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    rows[0].append("unexpected")
    rows.pop()
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    errors = validate_submission(submission_dir)

    assert any("exactly these 15 columns" in error for error in errors)
    assert any("must contain 97 data rows (98 lines" in error for error in errors)


def test_duplicate_dataset_and_invalid_values_fail(tmp_path: Path) -> None:
    submission_dir = make_submission(tmp_path)
    first_dataset = next(iter(EXPECTED_DATASETS))
    update_csv_cell(submission_dir, 3, "dataset", first_dataset)
    update_csv_cell(submission_dir, 4, "num_variates", "0")
    update_csv_cell(submission_dir, 5, "eval_metrics/MSE[mean]", "nan")
    update_csv_cell(submission_dir, 6, "eval_metrics/MAE[0.5]", "-1")

    errors = validate_submission(submission_dir)

    assert any(f"duplicates dataset {first_dataset!r}" in error for error in errors)
    assert any("num_variates must be a positive integer" in error for error in errors)
    assert any("eval_metrics/MSE[mean] must be finite" in error for error in errors)
    assert any("eval_metrics/MAE[0.5] must be non-negative" in error for error in errors)


def test_integer_valued_float_num_variates_is_valid(tmp_path: Path) -> None:
    submission_dir = make_submission(tmp_path)
    expected_num_variates = list(EXPECTED_DATASETS.values())[1][1]
    update_csv_cell(
        submission_dir, 2, "num_variates", f"{expected_num_variates}.0"
    )

    assert validate_submission(submission_dir) == []


def test_empty_metric_value_is_allowed(tmp_path: Path) -> None:
    submission_dir = make_submission(tmp_path)
    update_csv_cell(submission_dir, 2, "eval_metrics/MAPE[0.5]", "")

    assert validate_submission(submission_dir) == []


def test_dataset_domain_and_num_variates_must_match_manifest(tmp_path: Path) -> None:
    submission_dir = make_submission(tmp_path)
    update_csv_cell(submission_dir, 2, "domain", "Wrong domain")
    update_csv_cell(submission_dir, 3, "num_variates", "999")
    update_csv_cell(submission_dir, 4, "dataset", "unknown/F/short")

    errors = validate_submission(submission_dir)

    assert any("domain for" in error for error in errors)
    assert any("num_variates for" in error for error in errors)
    assert any("unknown dataset configuration" in error for error in errors)
    assert any("is missing dataset configurations" in error for error in errors)


def test_only_the_two_submission_files_are_allowed(tmp_path: Path) -> None:
    submission_dir = make_submission(tmp_path)
    (submission_dir / "notes.txt").write_text("extra", encoding="utf-8")

    errors = validate_submission(submission_dir)

    assert any("must contain exactly" in error for error in errors)


def test_cli_reports_all_errors_and_returns_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    submission_dir = make_submission(tmp_path)
    update_config(submission_dir, testdata_leakage="Maybe")
    update_csv_cell(submission_dir, 2, "model", "wrong")

    assert main([str(submission_dir)]) == 1
    stderr = capsys.readouterr().err
    assert "FAIL" in stderr
    assert "testdata_leakage" in stderr
    assert "line 3 model" in stderr


@pytest.mark.parametrize(
    "submission_dir",
    COMMITTED_SUBMISSIONS,
    ids=lambda path: path.name,
)
def test_committed_submission_is_valid(submission_dir: Path) -> None:
    errors = validate_submission(submission_dir)

    assert errors == [], "\n".join(errors)
