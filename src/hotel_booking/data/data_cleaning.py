import argparse
import calendar
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_ROWS = 79_330
EXPECTED_COLS = 31

EXPECTED_COLUMNS = [
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "AssignedRoomType",
    "BookingChanges",
    "DepositType",
    "Agent",
    "Company",
    "DaysInWaitingList",
    "CustomerType",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "ReservationStatus",
    "ReservationStatusDate",
]

NUMERIC_COLUMNS = [
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]


def load_h2(path: Path) -> pd.DataFrame:
    """Load H2, check its columns and clean text values."""
    df = pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("The CSV columns do not match the expected H2 schema.")

    if df.shape != (EXPECTED_ROWS, EXPECTED_COLS):
        raise ValueError(
            f"Unexpected H2 shape {df.shape}; "
            f"expected {(EXPECTED_ROWS, EXPECTED_COLS)}."
        )

    for column in df.columns:
        dtype = df[column].dtype
        is_text = (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        )

        if is_text:
            values = df[column].astype("string").str.strip()
            values = values.replace("", pd.NA)

            null_token = values.str.upper().eq("NULL").fillna(False)

            if column == "Children":
                null_token |= values.str.upper().eq("NA").fillna(False)

            values.loc[null_token] = pd.NA
            df[column] = values

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df["ReservationStatusDate"] = pd.to_datetime(
        df["ReservationStatusDate"],
        format="%Y-%m-%d",
        errors="raise",
    )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by handling missing children, creating dates, and removing rows and duplicates."""
    df = df.copy()

    missing_children = int(df["Children"].isna().sum())
    df["Children"] = df["Children"].fillna(0)
    print(f"Children missing values changed to 0: {missing_children}")
        
    # CN and CHN both mean China; standardise to the 3-letter ISO code
    cn_mask = df["Country"].eq("CN").fillna(False)
    print(f"Country 'CN' changed to 'CHN': {int(cn_mask.sum())}")
    df.loc[cn_mask, "Country"] = "CHN"
    month_to_number = {
        calendar.month_name[month].lower(): month
        for month in range(1, 13)
    }
    month_numbers = df["ArrivalDateMonth"].str.lower().map(month_to_number)

    date_parts = pd.DataFrame(
        {
            "year": df["ArrivalDateYear"],
            "month": month_numbers,
            "day": df["ArrivalDateDayOfMonth"],
        },
        index=df.index,
    )

    arrival_dates = pd.to_datetime(date_parts, errors="coerce")
    invalid_dates = arrival_dates.isna()

    if invalid_dates.any():
        examples = df.loc[
            invalid_dates,
            [
                "ArrivalDateYear",
                "ArrivalDateMonth",
                "ArrivalDateDayOfMonth",
            ],
        ].head()
        raise ValueError(
            "Could not create a valid arrival date. Example rows:\n"
            + examples.to_string(index=False)
        )

    date_text = arrival_dates.dt.strftime("%Y-%m-%d")
    date_position = df.columns.get_loc("ArrivalDateYear")
    df.insert(date_position, "date", date_text)

    columns_to_drop = [
        "Company",
        "ArrivalDateYear",
        "ArrivalDateMonth",
        "ArrivalDateDayOfMonth",
        "ArrivalDateWeekNumber",
    ]
    df = df.drop(columns=columns_to_drop)

    print("\nColumns removed:")
    for column in columns_to_drop:
        print(f"  {column}")

    removal_rules = {
        "ADR == 5400": df["ADR"].eq(5400),
        "Babies is 9 or 10": df["Babies"].isin([9, 10]),
        "Adults == 0": df["Adults"].eq(0),
    }

    print("\nRows matching each removal rule:")
    rows_to_remove = pd.Series(False, index=df.index)

    for rule_name, mask in removal_rules.items():
        mask = mask.fillna(False)
        print(f"  {rule_name}: {int(mask.sum())}")
        rows_to_remove |= mask

    print("Unique rows removed by these rules:", int(rows_to_remove.sum()))
    df = df.loc[~rows_to_remove].copy()

    extra_copies = int(df.duplicated().sum())
    rows_in_duplicate_groups = int(df.duplicated(keep=False).sum())

    print("\nDuplicates after cleaning:")
    print("  Extra duplicate copies:", extra_copies)
    print("  Rows in duplicate groups:", rows_in_duplicate_groups)

    #change blank agents to id 0
    df['Agent'] = df['Agent'].fillna(0).astype(int)
    #change blank countries to unknown
    df['Country'] = df['Country'].fillna('Unknown')
    #change date into datetimem
    df['date'] = pd.to_datetime(df['date'])

    return df.drop_duplicates(keep="first").reset_index(drop=True)


def resolve_from_repo(path_text: str) -> Path:
    """Resolve relative paths from the project root."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean H2.csv"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/raw/H2.csv",
        help="Input CSV path (default: data/raw/H2.csv)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="data/processed/H2_cleaned.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    input_path = resolve_from_repo(args.input)
    output_path = resolve_from_repo(args.output)

    print("Input file:", input_path)
    df = load_h2(input_path)
    print("Verified input shape:", df.shape)

    cleaned = clean_data(df)

    # Validate date column
    if not pd.api.types.is_datetime64_any_dtype(cleaned["date"]):
        raise TypeError("Date column is not datetime.")

    if cleaned["date"].isna().any():
        raise ValueError("Date column contains missing or invalid dates.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    print("\nFinal cleaned shape:", cleaned.shape)
    print("Final columns:", cleaned.columns.tolist())

    missing = cleaned.isna().sum()
    missing = missing[missing > 0]
    print("\nRemaining missing values:")
    print(missing.to_string() if not missing.empty else "None")

    print("\nSaved cleaned file to:", output_path)


if __name__ == "__main__":
    main()