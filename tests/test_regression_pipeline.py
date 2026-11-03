"""Tests for the rewritten preprocessing pipeline (rules + train-only fitting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from hotel_booking.leakage.checker import (
    LEAKAGE_FORBIDDEN_FEATURES,
    LeakageError,
    ManifestFitter,
    assert_no_leakage,
    fingerprint_rows,
)
from hotel_booking.regression.pipeline import (
    BOOKING_TIME_CATEGORICAL,
    BOOKING_TIME_NUMERIC,
    build_preprocessor,
    clean_for_regression,
    feature_names,
    invalid_row_mask,
)
from tests.conftest import synthetic_adr_frame

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10,
    "November": 11, "December": 12,
}


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    return synthetic_adr_frame()


def _removed_indices(frame: pd.DataFrame) -> np.ndarray:
    mask, _ = invalid_row_mask(frame)
    return np.flatnonzero(mask.to_numpy())


def test_invalid_row_rules_remove_expected_rows(synthetic) -> None:
    cleaned, log = clean_for_regression(synthetic, variant="after")
    counts = log["invalid_row_rules"]["counts"]

    # From conftest: row 0 ADR=5400, row 3 Adults=0 (also triggers the guests
    # rule since its random child count is 0), row 4 zero nights, row 5 Babies=10.
    assert counts["adr_eq_5400_single_extreme"] == 1
    assert counts["adults_eq_0"] == 1
    assert counts["total_nights_le_0"] == 1
    assert counts["babies_in_9_10"] == 1
    assert counts["total_guests_le_0"] == 1
    assert log["rows_removed_by_invalid_rules"] == 4
    assert len(cleaned) == len(synthetic) - 4


def test_adr_zero_kept_by_default_dropped_on_flag(synthetic) -> None:
    cleaned_keep, log_keep = clean_for_regression(synthetic, variant="after")
    _, log_drop = clean_for_regression(synthetic, variant="after", drop_adr_zero=True)
    assert log_keep["adr_zero_rule"].startswith("kept")
    assert log_drop["adr_zero_rule"] == "dropped (sensitivity run)"
    assert log_keep["adr_zero_rows"] == 2
    assert log_keep["rows_after_cleaning"] - log_drop["rows_after_cleaning"] == 2


def test_arrival_reconstruction_matches_dates(synthetic) -> None:
    cleaned, _ = clean_for_regression(synthetic, variant="after")
    month = synthetic["ArrivalDateMonth"].map(MONTH_MAP)
    frame = pd.DataFrame(
        {
            "year": synthetic["ArrivalDateYear"],
            "month": month,
            "day": synthetic["ArrivalDateDayOfMonth"],
        }
    )
    dates = pd.to_datetime(frame)
    expected_days = (dates - dates.min()).dt.days.to_numpy()
    removed = _removed_indices(synthetic)
    keep = np.ones(len(synthetic), dtype=bool)
    keep[removed] = False
    assert np.allclose(cleaned["arrival_days"].to_numpy(), expected_days[keep])
    assert cleaned["arrival_month"].between(1, 12).all()
    assert cleaned["arrival_dow"].between(1, 7).all()


def test_agent_null_becomes_sentinel_and_country_unknown(synthetic) -> None:
    cleaned, log = clean_for_regression(synthetic, variant="after")
    assert log["Agent_fillna_sentinel"] == {"value": "0", "rows": 1}
    assert (cleaned["Agent"].astype(str) == "0").any()
    assert "Unknown" in set(cleaned["Country"].astype(str))


def test_variant_after_has_no_post_booking_columns(synthetic) -> None:
    cleaned, _ = clean_for_regression(synthetic, variant="after")
    forbidden_predictors = set(LEAKAGE_FORBIDDEN_FEATURES) - {"ADR"}  # ADR is the target
    assert not forbidden_predictors & set(cleaned.columns)
    assert set(BOOKING_TIME_NUMERIC) <= set(cleaned.columns)
    assert set(BOOKING_TIME_CATEGORICAL) <= set(cleaned.columns)


def test_variant_before_includes_pre_truncation_columns(synthetic) -> None:
    cleaned, _ = clean_for_regression(synthetic, variant="before")
    for extra in (
        "IsCanceled", "BookingChanges", "DaysInWaitingList",
        "AssignedRoomType", "Company",
    ):
        assert extra in cleaned.columns


def test_dtypes_are_sklearn_safe(synthetic) -> None:
    cleaned, _ = clean_for_regression(synthetic, variant="after")
    for column in BOOKING_TIME_NUMERIC:
        assert pd.api.types.is_float_dtype(cleaned[column]), column
    for column in BOOKING_TIME_CATEGORICAL:
        assert cleaned[column].dtype == object, column


def test_preprocessor_fits_on_train_rows_only(synthetic) -> None:
    cleaned, _ = clean_for_regression(synthetic, variant="after")
    X = cleaned.drop(columns=["ADR"])
    n_train = int(len(X) * 0.7)
    X_tr, X_te = X.iloc[:n_train], X.iloc[n_train:]

    pipe = Pipeline([("manifest", ManifestFitter()), ("prep", build_preprocessor("after"))])
    pipe.fit(X_tr)
    report = assert_no_leakage(
        train_fingerprints=fingerprint_rows(X_tr),
        test_fingerprints=fingerprint_rows(X_te),
        feature_columns=list(X_tr.columns),
        fitted_pipeline=pipe,
    )
    assert report["passed"]

    # Deliberately leaky: refit on train+test; the manifest must catch it.
    pipe_leaky = Pipeline([("manifest", ManifestFitter()), ("prep", build_preprocessor("after"))])
    pipe_leaky.fit(pd.concat([X_tr, X_te], axis=0).reset_index(drop=True))
    with pytest.raises(LeakageError):
        assert_no_leakage(
            train_fingerprints=fingerprint_rows(X_tr),
            test_fingerprints=fingerprint_rows(X_te),
            feature_columns=list(X_tr.columns),
            fitted_pipeline=pipe_leaky,
        )


def test_feature_names_exposed(synthetic) -> None:
    cleaned, _ = clean_for_regression(synthetic, variant="after")
    X = cleaned.drop(columns=["ADR"])
    preproc = build_preprocessor("after").fit(X)
    names = feature_names(preproc)
    assert len(names) == preproc.transform(X).shape[1]
    assert any(name.startswith("cat__Agent_") for name in names)
    assert any(name.startswith("num__LeadTime") for name in names)
