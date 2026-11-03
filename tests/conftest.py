"""Shared synthetic fixtures for fast, deterministic tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_adr_frame(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """A tiny H2-shaped frame covering every column the ADR pipeline needs."""
    rng = np.random.default_rng(seed)
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    frame = pd.DataFrame(
        {
            "IsCanceled": rng.integers(0, 2, n),
            "LeadTime": rng.integers(1, 200, n),
            "ArrivalDateYear": rng.choice([2015, 2016], n),
            "ArrivalDateMonth": rng.choice(months, n),
            "ArrivalDateDayOfMonth": rng.integers(1, 28, n),
            "ArrivalDateWeekNumber": rng.integers(1, 53, n),
            "StaysInWeekendNights": rng.integers(0, 2, n),
            "StaysInWeekNights": rng.integers(1, 5, n),
            "Adults": rng.integers(1, 3, n),
            "Children": rng.integers(0, 2, n).astype(float),
            "Babies": rng.integers(0, 2, n),
            "Meal": rng.choice(["BB", "HB", "SC", "FB"], n),
            "Country": rng.choice(["PRT", "DEU", "GBR", "FRA", "CN"], n),
            "MarketSegment": rng.choice(
                ["Online TA", "Offline TA/TO", "Corporate", "Groups"], n
            ),
            "DistributionChannel": rng.choice(["TA/TO", "Direct", "Corporate"], n),
            "IsRepeatedGuest": rng.integers(0, 2, n),
            "PreviousCancellations": rng.integers(0, 3, n),
            "PreviousBookingsNotCanceled": rng.integers(0, 4, n),
            "ReservedRoomType": rng.choice(["A", "D", "E"], n),
            "AssignedRoomType": rng.choice(["A", "D", "E"], n),
            "BookingChanges": rng.integers(0, 3, n),
            "DepositType": rng.choice(["No Deposit", "Refundable"], n),
            "Agent": rng.choice(["1", "9", "7"], n).astype(object),
            "Company": np.where(rng.random(n) < 0.9, np.nan, rng.integers(1, 20, n)).astype(object),
            "DaysInWaitingList": rng.integers(0, 10, n),
            "CustomerType": rng.choice(["Transient", "Contract"], n),
            "ADR": np.round(np.abs(rng.normal(100, 30, n)), 2),
            "RequiredCarParkingSpaces": rng.integers(0, 2, n),
            "TotalOfSpecialRequests": rng.integers(0, 4, n),
            "ReservationStatus": rng.choice(["Check-Out", "Canceled"], n),
            "ReservationStatusDate": "2016-01-01",
        }
    )
    # Inject rule-triggering rows so the cleaning rules are exercised:
    frame.loc[0, "ADR"] = 5400.0
    frame.loc[1, "ADR"] = 0.0
    frame.loc[2, "ADR"] = 0.0
    frame.loc[3, "Adults"] = 0
    frame.loc[4, "StaysInWeekNights"] = 0
    frame.loc[4, "StaysInWeekendNights"] = 0  # force the zero-nights rule
    frame.loc[5, "Babies"] = 10
    # Inject literal missing tokens as the raw CSV would:
    frame.loc[6, "Agent"] = "NULL"
    frame.loc[7, "Country"] = "NULL"
    return frame


def booking_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce a synthetic frame to the variant-after predictor columns."""
    from hotel_booking.regression.pipeline import (
        BOOKING_TIME_CATEGORICAL,
        BOOKING_TIME_NUMERIC,
    )

    return frame[BOOKING_TIME_NUMERIC + BOOKING_TIME_CATEGORICAL].copy()
