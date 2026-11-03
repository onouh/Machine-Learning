"""Tests for the HTML experiment report: generation, parsing, artifact checks."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hotel_booking.regression.run import _CSV_FIELDS
from hotel_booking.reporting.report import (
    REPORT_PATH,
    RUNS_DIR,
    build_report,
    validate_report,
)

_ROWS = [
    {
        "label": "after_Ridge_tuned", "variant": "after", "model": "Ridge",
        "tuned": True, "drop_adr_zero": False, "rmse": 55.5, "mae": 40.1,
        "r2": 0.35, "best_cv_rmse": 55.9, "search_n_iter": 20,
        "best_params": "{'alpha': 3.2}", "leak_check_passed": True,
    },
    {
        "label": "after_Ridge_baseline", "variant": "after", "model": "Ridge",
        "tuned": False, "drop_adr_zero": False, "rmse": 56.0, "mae": 40.4,
        "r2": 0.34, "leak_check_passed": True,
    },
    {
        "label": "before_Ridge_tuned", "variant": "before", "model": "Ridge",
        "tuned": True, "drop_adr_zero": False, "rmse": 54.9, "mae": 39.8,
        "r2": 0.36, "best_cv_rmse": 55.3, "search_n_iter": 20,
        "best_params": "{'alpha': 2.7}", "leak_check_passed": True,
    },
]


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    path = tmp_path / "results" / "regression"
    path.mkdir(parents=True)
    with (path / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in _ROWS:
            writer.writerow({field: row.get(field, "") for field in _CSV_FIELDS})
    # Per-run JSON artifacts for every row label must exist for validation.
    for row in _ROWS:
        (path / f"{row['label']}.json").write_text("{}", encoding="utf-8")
    return path


def test_build_report_generates_html(runs_dir: Path, tmp_path: Path) -> None:
    out = build_report(runs_csv=runs_dir / "runs.csv", out_html=tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "after_Ridge_tuned" in html
    assert "javascript" not in html.lower()  # static: no JS, no CDN


def test_validate_report_passes_on_generated_file(runs_dir: Path, tmp_path: Path) -> None:
    out = build_report(runs_csv=runs_dir / "runs.csv", out_html=tmp_path / "r.html")
    assert validate_report(html_path=out, runs_csv=runs_dir / "runs.csv") == []


def test_validate_report_flags_per_run_artifacts(runs_dir: Path, tmp_path: Path) -> None:
    # Delete one per-run JSON → the artifact check must flag it.
    (runs_dir / "after_Ridge_tuned.json").unlink()
    out = build_report(runs_csv=runs_dir / "runs.csv", out_html=tmp_path / "r.html")
    problems = validate_report(html_path=out, runs_csv=runs_dir / "runs.csv")
    assert any("artifact missing" in p for p in problems)
    assert any("after_Ridge_tuned.json" in p for p in problems)


def test_report_parser_matches_and_closes_tags(runs_dir: Path, tmp_path: Path) -> None:
    from hotel_booking.reporting.report import _ReportChecker

    out = build_report(runs_csv=runs_dir / "runs.csv", out_html=tmp_path / "r.html")
    parser = _ReportChecker()
    parser.feed(out.read_text(encoding="utf-8"))
    parser.close()
    assert parser.errors == []
    assert not parser.stack
    headers = [c.strip() for c in parser.h2_texts]  # exposed by checker
    assert "Before vs after truncation" in headers


def test_report_defaults_point_to_outputs() -> None:
    assert str(REPORT_PATH) == str(Path("outputs/report/regression_report.html"))
    assert str(RUNS_DIR) == str(Path("outputs/results/regression"))
