"""Milestone 1, step 2: load and verify H2.csv.

Suggested home: split into src/data/loader.py (load_h2) and
src/data/audit.py (audit). Run directly: python h2_load_and_audit.py data/H2.csv
"""
import sys
import numpy as np
import pandas as pd

EXPECTED_ROWS, EXPECTED_COLS = 79_330, 31
EXPECTED_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
]


def load_h2(path: str) -> pd.DataFrame:
    """Load H2 with only *structural* cleaning (no data-dependent decisions)."""
    # keep_default_na=False so the literal string "NULL" is not silently
    # turned into NaN for some columns and left as text in others.
    df = pd.read_csv(path, keep_default_na=False, na_values=[], low_memory=False)
    # Several text columns are space-padded ('HB       ', 'A               ').
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            df[c] = df[c].str.strip()
    # Make missingness explicit and consistent.
    for c in ["Agent", "Company", "Country", "Children"]:
        df[c] = df[c].replace({"NULL": np.nan, "NA": np.nan, "": np.nan})
    df["Children"] = pd.to_numeric(df["Children"])
    df["ReservationStatusDate"] = pd.to_datetime(df["ReservationStatusDate"])
    return df


def audit(df: pd.DataFrame) -> None:
    print("== Schema ==")
    print("shape:", df.shape, "| expected:", (EXPECTED_ROWS, EXPECTED_COLS))
    assert list(df.columns) == EXPECTED_COLUMNS, "column names differ from Section 4"

    print("\n== Missing values ==")
    miss = df.isna().sum()
    print(miss[miss > 0].to_string())

    print("\n== Duplicates ==")
    n_dup = df.duplicated().sum()
    print(f"exact duplicate rows: {n_dup} ({n_dup / len(df):.1%})")
    print(f"IsCanceled rate: all={df.IsCanceled.mean():.3f}  "
          f"deduplicated={df.drop_duplicates().IsCanceled.mean():.3f}")

    print("\n== Unusual values ==")
    nights = df.StaysInWeekendNights + df.StaysInWeekNights
    guests = df.Adults + df.Children.fillna(0) + df.Babies
    print("ADR == 0:", (df.ADR == 0).sum(), "| ADR > 500:", (df.ADR > 500).sum(),
          "| ADR max:", df.ADR.max())
    print("zero total nights:", (nights == 0).sum(), "| zero guests:", (guests == 0).sum())
    print("LeadTime skew:", round(df.LeadTime.skew(), 2), "| max:", df.LeadTime.max())

    print("\n== Cardinality ==")
    for c in df.select_dtypes(include=["object", "string"]).columns:
        print(f"{c:24s}{df[c].nunique():5d}")

    print("\n== Arrival years (note 2017 is partial) ==")
    print(df.ArrivalDateYear.value_counts().sort_index().to_string())

    print("\n== Leakage probes for IsCanceled (explore, don't train on these) ==")
    print(pd.crosstab(df.ReservationStatus, df.IsCanceled))
    changed_room = df.ReservedRoomType != df.AssignedRoomType
    print("cancel rate | room changed:", round(df[changed_room].IsCanceled.mean(), 3),
          "| not changed:", round(df[~changed_room].IsCanceled.mean(), 3))
    print("cancel rate | BookingChanges>0:",
          round(df[df.BookingChanges > 0].IsCanceled.mean(), 3),
          "| ==0:", round(df[df.BookingChanges == 0].IsCanceled.mean(), 3))
    dup_extra = df.duplicated()                      # True for every copy after the first
    print(dup_extra.sum(), "extra copies")

    # Are duplicates concentrated among agent bookings, compared with the whole dataset?
    print(df["Agent"].notna().mean())                # share of all rows with an agent
    print(df[dup_extra]["Agent"].notna().mean())     # share among the duplicates

    # How big are the duplicate groups?
    counts = df.value_counts(dropna=False)
    print(counts[counts > 1].describe())

    # Which segments do they come from?
    print(df[dup_extra]["MarketSegment"].value_counts(normalize=True).head())
    # Is Groups over-represented among duplicates? Compare with the whole dataset
    print(df["MarketSegment"].value_counts(normalize=True))

    # What are the biggest duplicate groups?
    counts = df.value_counts(dropna=False)
    print(counts[counts > 1].sort_values(ascending=False).head(10))

    # How many duplicate groups have more than 10 copies?
    print((counts > 10).sum())

    # Do the duplicates differ in cancellation rate?
    print(df.duplicated().groupby(df["IsCanceled"]).mean())
    n_before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"removed {n_before - len(df)} duplicates, {len(df)} rows left")

if __name__ == "__main__":
    audit(load_h2(sys.argv[1] if len(sys.argv) > 1 else "data/H2.csv"))