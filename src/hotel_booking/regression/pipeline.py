"""ADR regression preprocessing — strict booking-creation prediction point.

Prediction point
----------------
ADR is fixed when the booking is made. Only fields knowable at booking
creation may be predictors. Post-booking fields are dropped with a recorded
reason (see ``DROP_WITH_REASON``); measured evidence for every decision lives
in ``docs/evidence/adr_analysis_evidence.md``, regenerable with::

    .venv/bin/python -m hotel_booking.data.adr_evidence

Leakage is enforced two ways:
1. an explicit allowlist — the model never sees a post-booking column, and
   ``assert_no_leakage`` re-checks after fitting (``hotel_booking.leakage``),
2. every learned transform (imputer, rare-category folding, one-hot, scaler)
   lives in an sklearn pipeline fitted only on training folds.

Variants
--------
``variant="after"``  : truncation applied — booking-time-only features.
``variant="before"`` : pre-truncation feature set (post-booking fields and
Company/IsCanceled included) used to quantify how the removed data skewed
ADR. The shared cleaning (invalid rows, ADR==5400 removal) is identical
across variants so the comparison is controlled.
"""

from __future__ import annotations

import calendar
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_TARGET = "ADR"
PREDICTION_POINT = "booking creation"
VARIANTS = ("before", "after")

# ---------------------------------------------------------------------------
# Booking-time feature plan (variant="after")
# ---------------------------------------------------------------------------
BOOKING_TIME_NUMERIC = [
    "LeadTime",
    "arrival_days",
    "arrival_month",
    "arrival_dow",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]
BOOKING_TIME_CATEGORICAL = [
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "DepositType",
    "CustomerType",
    "Agent",
]

# Columns that appear ONLY in the "before" variant (unknowable at booking time).
PRE_TRUNCATION_NUMERIC_EXTRA = ["IsCanceled", "BookingChanges", "DaysInWaitingList"]
PRE_TRUNCATION_CATEGORICAL_EXTRA = ["AssignedRoomType", "Company"]

DROP_WITH_REASON: dict[str, str] = {
    "ADR": "target — never a predictor",
    "IsCanceled": "outcome known only later; variant 'after' drops it, "
    "'before' includes it to show the truncation effect",
    "ReservationStatus": "direct cancellation outcome (measured 100% agreement with IsCanceled)",
    "ReservationStatusDate": "post-booking status lifecycle timestamp "
    "(precedes arrival on 31,631 rows)",
    "BookingChanges": "accrues after booking creation (10,268 rows with changes); "
    "unknown when ADR is fixed",
    "DaysInWaitingList": "realized after the booking request (3,443 rows with waiting days)",
    "AssignedRoomType": "assigned at/after check-in (differs from ReservedRoomType on 7,192 rows)",
    "Company": "75,641/79,330 rows NULL; raw company ID, not a booking-time rate driver",
    "ArrivalDateWeekNumber": "replaced by the reconstructed arrival signal, dropped after rebuild",
    "ArrivalDateYear": "absorbed by arrival-days reconstruction",
    "ArrivalDateMonth": "absorbed by arrival-days reconstruction",
    "ArrivalDateDayOfMonth": "absorbed by arrival-days reconstruction",
}

# Deterministic (non-learned) fills — they encode schema knowledge, not data.
CATEGORICAL_FILL: dict[str, str] = {
    "Agent": "0",  # NULL token at load; '0' sentinel = no agent recorded
    "Country": "Unknown",
    "Company": "Unknown",  # used by the 'before' variant; 95% of rows are NULL
}

LEAKAGE_FORBIDDEN_FEATURES = [
    "ReservationStatus",
    "ReservationStatusDate",
    "BookingChanges",
    "DaysInWaitingList",
    "AssignedRoomType",
    "ADR",
]

MONTH_MAP = {calendar.month_name[m].lower(): m for m in range(1, 13)}


def _month_to_number(series: pd.Series) -> pd.Series:
    mapped = series.astype("string").str.strip().str.lower().map(MONTH_MAP)
    if mapped.isna().any():
        bad = series[mapped.isna()].dropna().unique()[:10]
        raise ValueError(f"Unmappable ArrivalDateMonth values: {bad}")
    return mapped.astype(np.int64)


def _arrival_dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        pd.DataFrame(
            {
                "year": pd.to_numeric(df["ArrivalDateYear"], errors="raise"),
                "month": _month_to_number(df["ArrivalDateMonth"]),
                "day": pd.to_numeric(df["ArrivalDateDayOfMonth"], errors="raise"),
            },
            index=df.index,
        ),
        errors="raise",
    )


def invalid_row_mask(df: pd.DataFrame) -> tuple[pd.Series, dict[str, dict[str, Any]]]:
    """Rule mask plus per-rule counts/reasons for stays that are not valid."""
    total_guests = (
        df["Adults"].fillna(0)
        + pd.to_numeric(df["Children"], errors="raise").fillna(0)
        + df["Babies"].fillna(0)
    )
    total_nights = (
        pd.to_numeric(df["StaysInWeekendNights"], errors="raise").fillna(0)
        + pd.to_numeric(df["StaysInWeekNights"], errors="raise").fillna(0)
    )
    rules: dict[str, pd.Series] = {
        "adr_eq_5400_single_extreme": df["ADR"].eq(5400),
        "total_guests_le_0": total_guests.le(0),
        "total_nights_le_0": total_nights.le(0),
        "adults_eq_0": df["Adults"].eq(0),
        "babies_in_9_10": df["Babies"].isin([9, 10]),
    }
    counts = {name: int(bools.fillna(False).sum()) for name, bools in rules.items()}
    reasons = {
        "adr_eq_5400_single_extreme": "single extreme ADR row (max non-5400 is 510) "
        "— not a business rate",
        "total_guests_le_0": "no guests: not a valid stay",
        "total_nights_le_0": "no nights: ADR not meaningfully defined",
        "adults_eq_0": "EDA outlier rule — a booking needs at least one adult",
        "babies_in_9_10": "EDA outlier rule — 9/10 babies in one booking",
    }
    combined = pd.Series(False, index=df.index)
    for mask in rules.values():
        combined |= mask.fillna(False)
    return combined, {"counts": counts, "reasons": reasons}


def clean_for_regression(
    df: pd.DataFrame,
    *,
    variant: str = "after",
    drop_adr_zero: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean rows and keep the exact predictor columns for the variant.

    Returns (frame, log). The frame holds the variant's raw predictor columns
    plus the target; encoding/scaling happens later inside
    ``build_preprocessor`` so every learned transform stays out of the split.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    log: dict[str, Any] = {
        "prediction_point": PREDICTION_POINT,
        "variant": variant,
        "drop_adr_zero": drop_adr_zero,
    }
    frame = df.copy()

    # --- text normalization at the load boundary -----------------------------
    for column in frame.columns:
        dtype = frame[column].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            values = frame[column].astype("string").str.strip()
            is_token = values.str.upper().eq("NULL").fillna(False)
            if column == "Children":
                is_token |= values.str.upper().eq("NA").fillna(False)
            values.loc[is_token] = pd.NA
            frame[column] = values

    # --- numeric coercion ----------------------------------------------------
    numeric_cols = [
        "IsCanceled", "LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
        "Adults", "Children", "Babies", "IsRepeatedGuest",
        "PreviousCancellations", "PreviousBookingsNotCanceled", "BookingChanges",
        "DaysInWaitingList", "ADR", "RequiredCarParkingSpaces",
        "TotalOfSpecialRequests",
    ]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")

    # --- arrival reconstruction (before dropping the raw date parts) --------
    arrival = _arrival_dates(frame)
    log["arrival_min"] = str(arrival.min().date())
    log["arrival_max"] = str(arrival.max().date())
    frame["arrival_days"] = (arrival - arrival.min()).dt.days.astype(np.int64)
    frame["arrival_month"] = arrival.dt.month.astype(np.int64)
    frame["arrival_dow"] = arrival.dt.day_of_week.astype(np.int64) + 1

    # --- invalid rows (shared across variants; per-rule evidence logged) -----
    invalid, rule_log = invalid_row_mask(frame)
    log["invalid_row_rules"] = rule_log
    log["rows_removed_by_invalid_rules"] = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    log["rows_after_invalid_rules"] = int(len(frame))

    # --- ADR==0 handling: keep (complimentary stays) unless sensitivity flag --
    zero_mask = frame["ADR"].eq(0).fillna(False)
    log["adr_zero_rows"] = int(zero_mask.sum())
    if drop_adr_zero:
        log["adr_zero_rule"] = "dropped (sensitivity run)"
        frame = frame.loc[~zero_mask].copy()
    else:
        log["adr_zero_rule"] = "kept (complimentary stays; see evidence doc section 4)"

    # Dedup is performed at the END of cleaning (sentinel fills + dtype casts
    # done, and on the FINAL returned column set) — see the tail of this
    # function.

    assert not frame["ADR"].isna().any(), "ADR cannot be missing after cleaning"
    assert (frame["ADR"] >= 0).all(), "negative ADR present after cleaning"

    # --- deterministic categorical normalization -----------------------------
    for column, fill_value in CATEGORICAL_FILL.items():
        if column in frame.columns:
            missing = int(frame[column].isna().sum())
            filled = frame[column].astype("string").fillna(fill_value)
            frame[column] = filled.astype(object)
            log[f"{column}_fillna_sentinel"] = {"value": fill_value, "rows": missing}

    # --- feature columns per variant -----------------------------------------
    num_features = BOOKING_TIME_NUMERIC + (
        PRE_TRUNCATION_NUMERIC_EXTRA if variant == "before" else []
    )
    cat_features = BOOKING_TIME_CATEGORICAL + (
        PRE_TRUNCATION_CATEGORICAL_EXTRA if variant == "before" else []
    )
    log["numeric_features"] = num_features
    log["categorical_features"] = cat_features
    log["kept_columns"] = sorted(set(num_features) | set(cat_features))
    if variant == "before":
        log["variant_note"] = (
            "pre-truncation feature set: post-booking fields (IsCanceled, "
            "BookingChanges, DaysInWaitingList, AssignedRoomType) and Company kept"
        )
    else:
        log["variant_note"] = "truncated booking-time-only feature set"

    frame["Adults"] = pd.to_numeric(frame["Adults"], errors="raise")
    frame["Children"] = pd.to_numeric(frame["Children"], errors="raise")
    frame["Babies"] = pd.to_numeric(frame["Babies"], errors="raise")

    # --- final dtype normalization for sklearn interop -----------------------
    # Nullable pandas dtypes / pd.NA break sklearn's object-dtype NaN masks;
    # convert to plain numpy-backed dtypes with np.nan as the only missingness
    # token. No values change — this is a dtype cast only.
    for column in num_features:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    for column in cat_features:
        series = frame[column]
        frame[column] = series.astype(object).where(series.notna(), np.nan)

    out_columns = num_features + cat_features + [RAW_TARGET]
    out = frame[out_columns].copy()
    duplicate_mask = out.duplicated()
    log["exact_duplicates_removed"] = int(duplicate_mask.sum())
    out = out.drop_duplicates(keep="first").reset_index(drop=True)
    log["rows_after_cleaning"] = int(len(out))
    return out, log


def build_preprocessor(variant: str = "after") -> ColumnTransformer:
    """Learned transforms — fit on the training split / CV folds only."""
    numeric_cols = BOOKING_TIME_NUMERIC + (
        PRE_TRUNCATION_NUMERIC_EXTRA if variant == "before" else []
    )
    categorical_cols = BOOKING_TIME_CATEGORICAL + (
        PRE_TRUNCATION_CATEGORICAL_EXTRA if variant == "before" else []
    )

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="0")),
            # min_frequency folds rare training-fold categories into one
            # 'infrequent_sklearn' bucket per feature — learned per fold.
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Readable output column names of a *fitted* preprocessor."""
    return [str(name) for name in preprocessor.get_feature_names_out()]


def load_h2_for_regression(path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset via the shared strict loader."""
    from hotel_booking.data.load import load_dataset

    return load_dataset(path if path is not None else Path("data/raw/H2.csv"))
