"""Schema checks for the City Hotel H2 dataset."""

from collections.abc import Sequence

import pandas as pd

EXPECTED_COLUMNS = (
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
)


def validate_schema(
    frame: pd.DataFrame,
    expected_columns: Sequence[str] = EXPECTED_COLUMNS,
) -> list[str]:
    """Return schema and target-label issues; an empty list means checks passed."""
    issues: list[str] = []
    if frame.empty:
        issues.append("Dataset has no rows.")
    if frame.columns.has_duplicates:
        issues.append("Dataset has duplicate column names.")

    missing = sorted(set(expected_columns) - set(frame.columns))
    if missing:
        issues.append(f"Missing expected columns: {', '.join(missing)}")

    if "IsCanceled" in frame:
        observed = set(frame["IsCanceled"].dropna().astype(str).str.strip())
        unexpected = observed - {"0", "1"}
        if unexpected:
            issues.append(f"Unexpected IsCanceled values: {sorted(unexpected)}")
        if frame["IsCanceled"].isna().any():
            issues.append("IsCanceled contains missing values.")

    return issues
