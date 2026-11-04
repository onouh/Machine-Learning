import json
from datetime import datetime
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CLEANED_PATH = REPO_ROOT / "data" / "processed" / "H2_cleaned.csv"
LOG_PATH = REPO_ROOT / "outputs" / "results" / "regression" / "data_audit.jsonl"
TARGET = "ADR"

NUMERIC_FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
]
CATEGORICAL_FEATURES = [
    "ArrivalDateMonth",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "ReservedRoomType",
    "DepositType",
    "Agent",
    "CustomerType",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FORBIDDEN = [
    "IsCanceled",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "AssignedRoomType",
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "ReservationStatus",
    "ReservationStatusDate",
]
DERIVED = ["ArrivalDateYear", "ArrivalDateMonth"]
FOLD_COLUMNS = ["Country", "Agent"]

def add_arrival_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ArrivalDateYear and ArrivalDateMonth from the date column."""
    if "date" not in df.columns:
        raise ValueError("Column 'date' is missing; cannot derive arrival year and month.")
    already = [c for c in DERIVED if c in df.columns]
    if already:
        raise ValueError(f"Input already has {already}; refusing to overwrite them.")

    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        dates = df["date"]
    else:
        dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any():
        raise ValueError(f"'date' has {int(dates.isna().sum())} missing values.")

    df["ArrivalDateYear"] = dates.dt.year
    df["ArrivalDateMonth"] = dates.dt.month_name()
    return df

def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the approved features X and the target y (ADR)."""
    if df.columns.has_duplicates:
        raise ValueError("Input has duplicate column names.")
    needed = [c for c in FEATURES if c not in DERIVED] + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = add_arrival_features(df)
    X = df[FEATURES].copy()
    validate_features(X)

    y = pd.to_numeric(df[TARGET], errors="raise")
    if y.isna().any():
        raise ValueError(f"{TARGET} has {int(y.isna().sum())} missing values.")
    return X, y.rename(TARGET)

def _whole_number_text(series: pd.Series) -> pd.Series:
    """Turn 9, 9.0 and '9' into the text '9'; missing stays missing."""
    nums = pd.to_numeric(series, errors="raise")
    if ((nums.dropna() % 1) != 0).any():
        raise ValueError(f"{series.name or 'column'} has non-whole numbers.")
    return nums.round().astype("Int64").astype("string")

def validate_features(X: pd.DataFrame) -> None:
    """Raise a clear error unless X has exactly the approved features."""
    if X.columns.has_duplicates:
        dupes = sorted(X.columns[X.columns.duplicated()].unique())
        raise ValueError(f"Duplicate column names in X: {dupes}")
    present = set(X.columns)
    forbidden = sorted(present & set(FORBIDDEN))
    if forbidden:
        raise ValueError(f"Forbidden columns found in X: {forbidden}")
    if TARGET in present:
        raise ValueError(f"The target {TARGET} must not be in X.")
    missing = [c for c in FEATURES if c not in present]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    unexpected = sorted(present - set(FEATURES))
    if unexpected:
        raise ValueError(f"Unapproved columns in X: {unexpected}")


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Tidy types and category labels. Nothing here is learned from data."""
    validate_features(X)
    X = X[FEATURES].copy()

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="raise")

    agent = _whole_number_text(X["Agent"])
    X["Agent"] = agent.mask(agent == "0").fillna("No agent")

    X["IsRepeatedGuest"] = _whole_number_text(X["IsRepeatedGuest"]).fillna("Unknown")

    for col in CATEGORICAL_FEATURES:
        if col in ("Agent", "IsRepeatedGuest"):
            continue
        text = X[col].astype("string").str.strip()
        text = text.mask(text.str.upper().isin(["", "NULL"]))
        X[col] = text.fillna("Unknown")

    return X

def build_preprocessor(scale_numeric: bool = False, min_frequency: int | None = 10) -> Pipeline:
    """Unfitted preprocessor. Nothing is learned until .fit() is called on training rows."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    other_categoricals = [c for c in CATEGORICAL_FEATURES if c not in FOLD_COLUMNS]

    encode = ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "cat_small",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                other_categoricals,
            ),
            (
                "cat_fold",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=True,
                ),
                FOLD_COLUMNS,
            ),
        ]
    )
    return Pipeline(
        [
            ("prepare", FunctionTransformer(prepare_features, feature_names_out="one-to-one")),
            ("encode", encode),
        ]
    )


def audit_frame(df: pd.DataFrame, label: str, features: list[str]) -> dict:
    """Summarise the data for one run and append it to the audit log."""
    cols = features + [TARGET]
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "rows": len(df),
        "n_features": len(features),
        "features": features,
        "duplicates_all_columns": int(df.duplicated().sum()),
        "duplicates_features_plus_target": int(df.duplicated(subset=cols).sum()),
        "adr_zero_rows": int((df[TARGET] == 0).sum()),
        "adr_mean": round(float(df[TARGET].mean()), 2),
        "adr_median": round(float(df[TARGET].median()), 2),
        "adr_max": round(float(df[TARGET].max()), 2),
        "zero_night_rows": int(((df["StaysInWeekendNights"] + df["StaysInWeekNights"]) == 0).sum()),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


if __name__ == "__main__":
    X, y = get_xy(pd.read_csv(CLEANED_PATH))
    print("X shape:", X.shape, "| y shape:", y.shape)
    print("X columns:", list(X.columns))
    print("ADR in X?", TARGET in X.columns)
    print("Month values:", sorted(X["ArrivalDateMonth"].unique()))
    print("Year values:", sorted(X["ArrivalDateYear"].unique()))
    full = add_arrival_features(pd.read_csv(CLEANED_PATH))
    audit = audit_frame(full, "audited_19", FEATURES)
    print(json.dumps({k: v for k, v in audit.items() if k != "features"}, indent=2))