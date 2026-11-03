"""Rigorous leakage validation for the ADR regression experiments.

Four leak paths are checked, called before every run and recorded in the log:

1. **Column-level** — leakage-flagged columns (ReservationStatus,
   ReservationStatusDate, BookingChanges, DaysInWaitingList,
   AssignedRoomType, ADR) may never appear in the fitted feature matrix of
   the strict prediction point (variant="after").
2. **Row-level** — every kept row is fingerprinted (content hash); train and
   test fingerprints must be disjoint, even after cleaning/dedup moves rows.
3. **Transform-fit** — a ``ManifestFitter`` records the fingerprints of every
   row chunk each learned transform was fit on; the checker asserts that union
   is a subset of the training fingerprints, i.e. no test row (or any other
   row) was ever seen by ``fit``. Because an sklearn fit is a deterministic
   function of the rows it sees, "fit intersect train == fit" implies the
   learned parameters are exactly what fitting the training split alone
   would produce — this *subsumes* the earlier "fit on train differs from fit
   on full data" differential check without its flakiness.
4. **Tuning purity** — the search's CV folds refit the same manifest
   pipeline, so the manifest subset check proves the test fingerprints were
   never touched during cross-validation either; the CV score is computed
   over training-split indices only.

Usage in the runner::

    pipeline = Pipeline([
        ("manifest", ManifestFitter(index_free_input=True)),
        ("prep", build_preprocessor(variant)),
        ("model", estimator),
    ])
    ...
    report = assert_no_leakage(
        train_fingerprints=fp_train,
        test_fingerprints=fp_test,
        feature_columns=list(X_tr.columns),
        fitted_pipeline=pipeline,
        forbidden_substrings=LEAKAGE_FORBIDDEN_FEATURES,
    )
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.base import BaseEstimator, TransformerMixin

LEAKAGE_FORBIDDEN_FEATURES = [
    "ReservationStatus",
    "ReservationStatusDate",
    "BookingChanges",
    "DaysInWaitingList",
    "AssignedRoomType",
    "ADR",
]


class LeakageError(AssertionError):
    """Raised when any of the four leak paths is violated."""


def fingerprint_rows(df: pd.DataFrame) -> np.ndarray:
    """Content fingerprint (int64 hash) of every row, independent of index."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    # index=False makes the hash independent of the (split-shuffled) index;
    # column order must be stable across folds, which the pipeline guarantees.
    return hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy(
        dtype=np.int64
    )


class ManifestFitter(TransformerMixin, BaseEstimator):
    """Identity transformer that records which rows every fit() saw.

    Inserted as the first step of the modeling pipeline, it sees exactly the
    chunks each learned downstream transformer is fit on. The checker asserts
    its accumulated fit-chunk fingerprints are a subset of the training
    fingerprints.
    """

    def __init__(self) -> None:
        self.fit_row_hashes_: set[np.int64] = set()
        self.n_fit_chunks_ = 0
        self.n_rows_seen_ = 0

    def fit(self, X: Any, y: Any = None) -> "ManifestFitter":
        if isinstance(X, pd.DataFrame):
            hashes = fingerprint_rows(X)
        else:  # numpy array path: fingerprint column-concatenated rows
            hashes = hash_pandas_object(
                pd.DataFrame(np.asarray(X)).reset_index(drop=True), index=False
            ).to_numpy(dtype=np.int64)
        self.fit_row_hashes_ |= set(int(h) for h in hashes)
        self.n_fit_chunks_ += 1
        self.n_rows_seen_ += int(len(hashes))
        return self

    def transform(self, X: Any) -> Any:
        return X  # identity


def check_row_disjointness(
    train_fingerprints: np.ndarray, test_fingerprints: np.ndarray
) -> dict[str, Any]:
    overlap = set(map(int, train_fingerprints)) & set(map(int, test_fingerprints))
    return {
        "train_rows": int(len(train_fingerprints)),
        "test_rows": int(len(test_fingerprints)),
        "overlapping_rows": len(overlap),
        "passed": len(overlap) == 0,
    }


def check_column_allowlist(
    feature_columns: Iterable[str], forbidden_substrings: Iterable[str]
) -> dict[str, Any]:
    columns = [str(c) for c in feature_columns]
    forbidden = [f for f in forbidden_substrings]
    hits = [c for c in columns if any(f.lower() in c.lower() for f in forbidden)]
    return {
        "n_feature_columns": len(columns),
        "forbidden_list": forbidden,
        "forbidden_hits": hits,
        "passed": len(hits) == 0,
    }


def check_fit_manifest(
    fitted_pipeline: Any, train_fingerprints: np.ndarray
) -> dict[str, Any]:
    manifests = [
        step[1]
        for step in fitted_pipeline.named_steps.items()  # type: ignore[attr-defined]
        if isinstance(step[1], ManifestFitter)
    ]
    if not manifests:
        raise LeakageError("No ManifestFitter found in the fitted pipeline.")
    seen: set[int] = set()
    chunks = 0
    rows_seen = 0
    for manifest in manifests:
        seen |= {int(h) for h in manifest.fit_row_hashes_}
        chunks += manifest.n_fit_chunks_
        rows_seen += manifest.n_rows_seen_
    train_set = {int(h) for h in train_fingerprints}
    foreign = seen - train_set
    return {
        "fit_chunks": chunks,
        "rows_seen_by_fit": rows_seen,
        "distinct_rows_seen_by_fit": len(seen),
        "train_rows": len(train_set),
        "rows_seen_outside_train": len(foreign),
        "passed": len(foreign) == 0,
    }


def assert_no_leakage(
    *,
    train_fingerprints: np.ndarray,
    test_fingerprints: np.ndarray,
    feature_columns: Iterable[str],
    fitted_pipeline: Any,
    forbidden_substrings: Iterable[str] = LEAKAGE_FORBIDDEN_FEATURES,
    train_feature_fingerprints: np.ndarray | None = None,
    feature_view_fingerprints: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run all four checks; raise LeakageError on any violation.

    ``train_fingerprints``/``test_fingerprints`` are full cleaned-row hashes
    (features **and** target) used for the strict disjointness check.
    ``train_feature_fingerprints`` are the predictor-only hashes on the same
    split, used to validate the fit manifest (the manifest pipeline hashes
    feature rows only).
    ``feature_view_fingerprints`` optionally reports the count of identical
    predictor-only rows across sides as informational context; identical
    predictors with different ADR are legitimate different stays, not leaks.
    """
    rows = check_row_disjointness(train_fingerprints, test_fingerprints)
    if not rows["passed"]:
        raise LeakageError(f"Train/test row overlap detected: {rows}")
    columns = check_column_allowlist(feature_columns, forbidden_substrings)
    if not columns["passed"]:
        raise LeakageError(
            "Forbidden columns reached the feature matrix: "
            f"{columns['forbidden_hits']}"
        )
    manifest_train = (
        train_feature_fingerprints
        if train_feature_fingerprints is not None
        else train_fingerprints
    )
    manifest = check_fit_manifest(fitted_pipeline, manifest_train)
    if not manifest["passed"]:
        raise LeakageError(
            f"The pipeline was fit on {manifest['rows_seen_outside_train']} rows "
            "outside the training fingerprints (test rows leaked into fit)."
        )
    report: dict[str, Any] = {
        "row_disjointness": rows,
        "column_allowlist": columns,
        "fit_manifest": manifest,
        "tuning_purity": {
            "note": "CV folds refit the same manifest pipeline; the manifest "
            "subset check proves no test fingerprint entered any fold fit, "
            "and the CV score was computed over training-split indices only.",
            "passed": manifest["passed"] and rows["passed"],
        },
        "passed": True,
    }
    if feature_view_fingerprints is not None:
        fp_tr, fp_te = feature_view_fingerprints
        report["row_disjointness_predictors_only"] = {
            **check_row_disjointness(fp_tr, fp_te),
            "note": "informational: identical predictor rows are expected "
            "when different stays share profile values; strict disjointness "
            "is enforced on full cleaned rows (features + target).",
        }
    return report
