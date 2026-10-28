"""Tests for dataset loading, validation, and profiling."""

import pandas as pd

from hotel_booking.data.load import load_dataset
from hotel_booking.data.profiling import profile_dataset
from hotel_booking.data.validate import EXPECTED_COLUMNS, validate_schema


def test_validate_schema_accepts_expected_columns_and_binary_target() -> None:
    frame = pd.DataFrame({column: [0] for column in EXPECTED_COLUMNS})

    assert validate_schema(frame) == []


def test_validate_schema_reports_missing_columns_and_bad_target() -> None:
    frame = pd.DataFrame({"IsCanceled": [0, 2]})

    issues = validate_schema(frame)

    assert any("Missing expected columns" in issue for issue in issues)
    assert any("Unexpected IsCanceled values" in issue for issue in issues)


def test_load_dataset_strips_whitespace_and_normalizes_missing_tokens(tmp_path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("category,count\n  Online TA  ,1\nNULL,2\n", encoding="utf-8")

    frame = load_dataset(path)

    assert frame.loc[0, "category"] == "Online TA"
    assert pd.isna(frame.loc[1, "category"])


def test_profile_dataset_reports_duplicates_missingness_and_labels() -> None:
    frame = pd.DataFrame(
        {
            "IsCanceled": [0, 1, 1],
            "Country": ["PRT", None, "PRT"],
        }
    )

    profile = profile_dataset(frame)

    assert profile["rows"] == 3
    assert profile["columns"] == 2
    assert profile["exact_duplicate_rows"] == 0
    assert profile["is_canceled_counts"] == {"1": 2, "0": 1}
    country = next(column for column in profile["column_profile"] if column["name"] == "Country")
    assert country["missing"] == 1
    assert country["cardinality"] == 1
