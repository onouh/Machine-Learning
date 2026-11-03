"""Tests for the four-path leakage checker (positive + failing variants)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from hotel_booking.leakage.checker import (
    LeakageError,
    ManifestFitter,
    assert_no_leakage,
    check_column_allowlist,
    check_row_disjointness,
    fingerprint_rows,
)
from hotel_booking.regression.pipeline import build_preprocessor, clean_for_regression
from tests.conftest import synthetic_adr_frame


def _train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned, _ = clean_for_regression(synthetic_adr_frame(), variant="after")
    X = cleaned.drop(columns=["ADR"])
    n = int(len(X) * 0.7)
    return X.iloc[:n], X.iloc[n:]


def test_fingerprint_row_disjoint_positive() -> None:
    X_tr, X_te = _train_test()
    report = check_row_disjointness(fingerprint_rows(X_tr), fingerprint_rows(X_te))
    assert report["passed"]
    assert report["overlapping_rows"] == 0


def test_fingerprint_deterministic_for_same_content() -> None:
    X_tr, X_te = _train_test()
    one, two = X_tr.iloc[:10], X_tr.iloc[:10].copy()
    np.testing.assert_array_equal(fingerprint_rows(one), fingerprint_rows(two))


def test_row_overlap_detected() -> None:
    X_tr, _ = _train_test()
    report = check_row_disjointness(fingerprint_rows(X_tr), fingerprint_rows(X_tr))
    assert not report["passed"]
    assert report["overlapping_rows"] == len(X_tr)


def test_column_allowlist_blocks_forbidden_features() -> None:
    fine = check_column_allowlist(["num__LeadTime", "cat__Meal_BB"], ["BookingChanges"])
    assert fine["passed"]
    bad = check_column_allowlist(["num__BookingChanges", "cat__Meal_BB"], ["BookingChanges"])
    assert not bad["passed"]
    assert bad["forbidden_hits"] == ["num__BookingChanges"]


def test_manifest_clean_fit_passes_checker() -> None:
    X_tr, X_te = _train_test()
    pipe = Pipeline(
        [("manifest", ManifestFitter()), ("prep", build_preprocessor("after"))]
    )
    pipe.fit(X_tr)
    report = assert_no_leakage(
        train_fingerprints=fingerprint_rows(X_tr),
        test_fingerprints=fingerprint_rows(X_te),
        feature_columns=list(X_tr.columns),
        fitted_pipeline=pipe,
    )
    assert report["passed"]
    assert report["fit_manifest"]["rows_seen_outside_train"] == 0
    assert report["tuning_purity"]["passed"]


def test_manifest_catches_test_rows_in_fit_when_leaked() -> None:
    X_tr, X_te = _train_test()
    pipe = Pipeline(
        [("manifest", ManifestFitter()), ("prep", build_preprocessor("after"))]
    )
    pipe.fit(pd.concat([X_tr, X_te]).reset_index(drop=True))  # leaked fit
    with pytest.raises(LeakageError):
        assert_no_leakage(
            train_fingerprints=fingerprint_rows(X_tr),
            test_fingerprints=fingerprint_rows(X_te),
            feature_columns=list(X_tr.columns),
            fitted_pipeline=pipe,
        )


def test_forbidden_column_reaches_feature_matrix() -> None:
    X_tr, X_te = _train_test()
    leaked_columns = list(X_tr.columns) + ["BookingChanges"]
    with pytest.raises(LeakageError):
        assert_no_leakage(
            train_fingerprints=fingerprint_rows(X_tr),
            test_fingerprints=fingerprint_rows(X_te),
            feature_columns=leaked_columns,
            fitted_pipeline=Pipeline([("manifest", ManifestFitter())]).fit(X_tr),
        )
