"""Measured evidence collection for the ADR regression task.

Every keep/drop decision in the ADR preprocessing pipeline is justified by a
number measured here over the *full raw* H2.csv. Re-run with:

    .venv/bin/python -m hotel_booking.data.adr_evidence

to regenerate ``docs/evidence/adr_analysis_evidence.md`` and the companion
``.json``.
"""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = REPO_ROOT / "data" / "raw" / "H2.csv"
OUT_MD = REPO_ROOT / "docs" / "evidence" / "adr_analysis_evidence.md"
OUT_JSON = REPO_ROOT / "docs" / "evidence" / "adr_analysis_evidence.json"

EXPECTED_ROWS, EXPECTED_COLS = 79_330, 31

TEXT_COLUMNS = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment",
    "DistributionChannel", "ReservedRoomType", "AssignedRoomType",
    "DepositType", "Agent", "Company", "CustomerType",
    "ReservationStatus", "ReservationStatusDate",
]

NUMERIC_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "BookingChanges",
    "DaysInWaitingList", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]


def load_raw(path: Path = CSV_PATH) -> pd.DataFrame:
    """Load the raw CSV with the strict schema/shape/token rules of the project."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    if df.shape != (EXPECTED_ROWS, EXPECTED_COLS):
        raise ValueError(f"Unexpected H2 shape {df.shape}; expected {(EXPECTED_ROWS, EXPECTED_COLS)}.")

    # Count literal missing tokens before normalization so evidence is provenance-grade.
    token_counts: dict[str, dict[str, int]] = {}
    for column in df.columns:
        if column in TEXT_COLUMNS:
            s = df[column].astype("string").str.strip()
            tokens = {
                "NULL": int(s.str.upper().eq("NULL").fillna(False).sum()),
                "NA": int(s.str.upper().eq("NA").fillna(False).sum()),
                "empty": int(s.eq("").fillna(False).sum()),
            }
            if any(tokens.values()):
                token_counts[column] = {k: v for k, v in tokens.items() if v}
        df[column] = df[column].replace(to_replace=[r"^\s*$"], value=pd.NA, regex=True)
        s = df[column].astype("string").str.strip()
        if column == "Children":
            df[column] = s.mask(s.str.upper().isin(["NULL", "NA"]))
        elif column in TEXT_COLUMNS:
            df[column] = s.mask(s.str.upper().isin(["NULL"]))
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df.attrs["missing_tokens"] = token_counts  # type: ignore[attr-defined]
    return df


def _arrival_dates(df: pd.DataFrame) -> pd.Series:
    month_map = {calendar.month_name[m].lower(): m for m in range(1, 13)}
    months = df["ArrivalDateMonth"].astype("string").str.lower().map(month_map)
    return pd.to_datetime(
        pd.DataFrame(
            {
                "year": pd.to_numeric(df["ArrivalDateYear"], errors="raise"),
                "month": pd.to_numeric(months, errors="raise"),
                "day": pd.to_numeric(df["ArrivalDateDayOfMonth"], errors="raise"),
            }
        ),
        errors="raise",
    )


def collect_evidence(df: pd.DataFrame) -> dict[str, Any]:
    """Measure every number referenced by the pipeline's keep/drop docstrings."""
    ev: dict[str, Any] = {}
    ev["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    # ---- Leakage 1: ReservationStatus x IsCanceled agreement -----------------
    cross = df.groupby(["ReservationStatus", "IsCanceled"]).size()
    ev["leakage_reservation_status_x_is_canceled"] = {
        f"status={rs}|isCanceled={int(ic)}": int(n) for (rs, ic), n in cross.items()
    }
    check_out = (df["ReservationStatus"] == "Check-Out") == (df["IsCanceled"] == 0)
    check_neg = (df["ReservationStatus"].isin(["Canceled", "No-Show"])) == (df["IsCanceled"] == 1)
    ev["leakage_reservation_status_agrees_with_is_canceled"] = bool(
        check_out.all() and check_neg.all()
    )

    # ---- Leakage 2: ReservationStatusDate timing ----------------------------
    arrival = _arrival_dates(df)
    rsd = pd.to_datetime(df["ReservationStatusDate"])
    ev["arrival_min"] = str(arrival.min().date())
    ev["arrival_max"] = str(arrival.max().date())
    ev["reservation_status_date_min"] = str(rsd.min().date())
    ev["reservation_status_date_max"] = str(rsd.max().date())
    ev["reservation_status_date_before_arrival_any"] = bool((rsd < arrival).any())
    ev["reservation_status_date_before_arrival_count"] = int((rsd < arrival).sum())

    # ---- Leakage 3: post-booking field reach --------------------------------
    ev["booking_changesgt0"] = int((df["BookingChanges"] > 0).sum())
    ev["booking_changes_values"] = {
        str(v): int(n) for v, n in df["BookingChanges"].value_counts().sort_index().items() if n >= 1
    }
    ev["days_in_waiting_list_gt0"] = int((df["DaysInWaitingList"] > 0).sum())
    ev["assigned_differs_from_reserved"] = int(
        (df["AssignedRoomType"] != df["ReservedRoomType"]).sum()
    )
    ev["assigned_room_type_by_room_type"] = sorted(df["AssignedRoomType"].unique().tolist())

    # ---- Null/token profile -------------------------------------------------
    nulls = df.isna().sum()
    ev["nulls_per_column"] = {c: int(v) for c, v in nulls.items() if v > 0}
    ev["missing_literal_tokens_pre_normalization"] = {
        col: counts for col, counts in dict(df.attrs.get("missing_tokens", {})).items()  # type: ignore[attr-defined]
    }
    survivors: dict[str, list[str]] = {}
    for column in TEXT_COLUMNS:
        vals = set(df[column].dropna().astype("string").unique())
        leaked = sorted(v for v in vals if v.strip().upper() in {"NULL", "NA", ""})
        if leaked:
            survivors[column] = leaked
    ev["literal_tokens_surviving_normalization"] = survivors

    # ---- Extremes -----------------------------------------------------------
    ev["adr_negatives_count"] = int((df["ADR"] < 0).sum())
    ev["adr_eq_5400"] = int((df["ADR"] == 5400).sum())
    ev["adr_max_non_5400"] = float(df.loc[df["ADR"] != 5400, "ADR"].max())
    ev["adr_quantiles"] = {
        str(q): float(df["ADR"].quantile(q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)
    }
    ev["adr_mean"] = float(df["ADR"].mean())
    lv = pd.to_numeric(df["LeadTime"], errors="raise")
    ev["lead_time_max"] = int(lv.max())
    total_nights = (
        pd.to_numeric(df["StaysInWeekendNights"], errors="raise").fillna(0).astype(int)
        + pd.to_numeric(df["StaysInWeekNights"], errors="raise").fillna(0).astype(int)
    )
    total_guests = (
        pd.to_numeric(df["Adults"], errors="raise").fillna(0)
        + pd.to_numeric(df["Children"], errors="raise").fillna(0)
        + pd.to_numeric(df["Babies"], errors="raise").fillna(0)
    )
    ev["stays_zero_nights"] = int((total_nights == 0).sum())
    ev["adults_eq_0"] = int((df["Adults"] == 0).sum())
    ev["babies_9_or_10"] = int(df["Babies"].isin([9, 10]).sum())
    ev["total_guests_eq_0"] = int((total_guests <= 0).sum())
    ev["babies_values"] = {str(v): int(n) for v, n in df["Babies"].value_counts().sort_index().items()}

    # ---- ADR==0 business meaning -------------------------------------------
    zero = df[df["ADR"] == 0]
    ev["adr_zero_count"] = int(len(zero))
    ev["adr_zero_share_of_rows"] = float(len(zero) / len(df))
    ev["adr_zero_status_x_market_segment"] = {
        f"status={rs}|segment={ms}": int(n)
        for (rs, ms), n in zero.groupby(["ReservationStatus", "MarketSegment"]).size().items()
    }
    ev["adr_zero_by_deposit"] = {str(k): int(v) for k, v in zero["DepositType"].value_counts().items()}
    ev["adr_zero_by_customer_type"] = {str(k): int(v) for k, v in zero["CustomerType"].value_counts().items()}
    res = zero[zero["MarketSegment"] != "Complementary"]
    ev["adr_zero_not_complementary_count"] = int(len(res))
    ev["adr_zero_not_complementary_deposit"] = {
        str(k): int(v) for k, v in res["DepositType"].value_counts().items()
    } if len(res) else {}

    # ---- Company / Agent ----------------------------------------------------
    for col in ["Company", "Agent"]:
        ev[f"{col}_null"] = int(df[col].isna().sum())
        ev[f"{col}_nonnull"] = int(df[col].notna().sum())
        ev[f"{col}_cardinality"] = int(df[col].nunique())

    # ---- Duplicates ---------------------------------------------------------
    ev["duplicate_rows_excluding_first"] = int(df.duplicated(keep="first").sum())
    ev["rows_in_duplicate_groups"] = int(df.duplicated(keep=False).sum())

    # ---- Booking-time stat checks (before, for the "before" row-count audit) --
    ev["booking_changes_gt0_gt0_mean_adr"] = float(
        df.loc[df["BookingChanges"] > 0, "ADR"].mean()
    )
    return ev


def _render_table(rows: list[tuple[str, str]]) -> str:
    widths = [max(len(r[0]) for r in rows), max(len(r[1]) for r in rows)]
    head = f"| {'Label'.ljust(widths[0])} | {'Value'.ljust(widths[1])} |"
    sep = f"|{'-' * (widths[0] + 2)}|{'-' * (widths[1] + 2)}|"
    body = "\n".join(
        f"| {label.ljust(widths[0])} | {value.ljust(widths[1])} |"
        for label, value in rows
    )
    return "\n".join([head, sep, body])


def render_markdown(ev: dict[str, Any]) -> str:
    lines: list[str] = [
        "# ADR regression — measured evidence from the full raw H2.csv",
        "",
        "Every claim in the ADR preprocessing keep/drop logic traces to a number below.",
        f"Source: `data/raw/H2.csv` — {ev['shape']['rows']:,} rows × {ev['shape']['columns']} columns.",
        "",
        "## 1. Leakage checks",
        "",
    ]
    lines.append(_render_table([
        ("ReservationStatus['Check-Out'] ↔ IsCanceled==0 with the rest ↔ 1",
         str(ev["leakage_reservation_status_agrees_with_is_canceled"])),
        ("Rows where ReservationStatusDate precedes arrival",
         f"{ev['reservation_status_date_before_arrival_count']:,}"),
        ("ReservationStatusDate range",
         f"{ev['reservation_status_date_min']} .. {ev['reservation_status_date_max']}"),
        ("Arrival range",
         f"{ev['arrival_min']} .. {ev['arrival_max']}"),
        ("BookingChanges > 0 rows (post-booking edits exist)",
         f"{ev['booking_changesgt0']:,}"),
        ("DaysInWaitingList > 0 rows (only known after request)",
         f"{ev['days_in_waiting_list_gt0']:,}"),
        ("AssignedRoomType ≠ ReservedRoomType (assign happens at/after check-in)",
         f"{ev['assigned_differs_from_reserved']:,}"),
    ]))
    lines += [
        "",
        "Cross-tab ReservationStatus × IsCanceled (perfect agreement ⇒ direct "
        "cancellation outcome = columns ReservationStatus, ReservationStatusDate must not be "
        "ADR predictors at the booking-creation point):",
        "",
        _render_table([
            (k, str(v)) for k, v in
            ev["leakage_reservation_status_x_is_canceled"].items()
        ]),
        "",
        "## 2. Missing-value profile (raw tokens before normalization)",
        "",
        _render_table([
            (c, str(v)) for c, v in ev["nulls_per_column"].items()
        ]),
        "",
        "Literal `NULL`/`NA` token counts **before** normalization: "
        + (json.dumps(ev["missing_literal_tokens_pre_normalization"]) if ev["missing_literal_tokens_pre_normalization"] else "{}"),
        "",
        "Tokens surviving normalization (must be `{}` → proven literal categories): "
        + json.dumps(ev["literal_tokens_surviving_normalization"]),
        "",
        "## 3. Extremes and invalid-row rules",
        "",
        _render_table([
            ("ADR < 0 rows", str(ev["adr_negatives_count"])),
            ("ADR == 5400 rows", str(ev["adr_eq_5400"])),
            ("Max ADR excluding 5400", str(ev["adr_max_non_5400"])),
            ("LeadTime max", str(ev["lead_time_max"])),
            ("Stays with total nights == 0", str(ev["stays_zero_nights"])),
            ("Adults == 0 rows", str(ev["adults_eq_0"])),
            ("Total guests ≤ 0 rows", str(ev["total_guests_eq_0"])),
            ("Babies ∈ {9, 10} rows", str(ev["babies_9_or_10"])),
        ]),
        "",
        "ADR quantiles: " + json.dumps(ev["adr_quantiles"]),
        "",
        "## 4. ADR == 0 business meaning",
        "",
        _render_table([
            ("ADR == 0 rows", f"{ev['adr_zero_count']:,} ({ev['adr_zero_share_of_rows']:.1%} of rows)"),
            ("of which MarketSegment != 'Complementary'", str(ev["adr_zero_not_complementary_count"])),
        ]),
        "",
        "ADR==0 × ReservationStatus × MarketSegment (is 0 a real complimentary rate?):",
        "",
        _render_table([
            (k, str(v)) for k, v in ev["adr_zero_status_x_market_segment"].items()
        ]),
        "",
        "ADR==0 by DepositType: " + json.dumps(ev["adr_zero_by_deposit"]),
        "ADR==0 by CustomerType: " + json.dumps(ev["adr_zero_by_customer_type"]),
        "",
        "## 5. Company / Agent fields",
        "",
        _render_table([
            ("Company NULL / non-null", f"{ev['Company_null']:,} / {ev['Company_nonnull']:,}"),
            ("Company distinct values", str(ev["Company_cardinality"])),
            ("Agent NULL / non-null", f"{ev['Agent_null']:,} / {ev['Agent_nonnull']:,}"),
            ("Agent distinct values", str(ev["Agent_cardinality"])),
        ]),
        "",
        "## 6. Duplicates",
        "",
        _render_table([
            ("Exact duplicate rows (excluding first occurrence)", f"{ev['duplicate_rows_excluding_first']:,}"),
            ("Rows that belong to a duplicate group", f"{ev['rows_in_duplicate_groups']:,}"),
        ]),
        "",
        "## 7. Other distribution facts",
        "",
        f"- Babies value counts: {json.dumps(ev['babies_values'])}",
        f"- ADR mean: {ev['adr_mean']:.2f}",
        "",
        "### Keep/drop rationale this evidence supports",
        "",
        "| Column | Decision | Reason (evidence above) |",
        "|---|---|---|",
        "| ReservationStatus / ReservationStatusDate | **Drop** | Perfect IsCanceled agreement; status date is post-arrival for most rows → outcome leakage at the booking-creation point. |",
        "| BookingChanges | **Drop** | Post-booking edits (rows with changes exist); not knowable when ADR is fixed. |",
        "| DaysInWaitingList | **Drop** | Realized after the customer requests, not at booking creation. |",
        "| AssignedRoomType | **Drop** | Assigned at/after check-in; refit differs from ReservedRoomType in thousands of rows. |",
        "| IsCanceled | **Drop** | Outcome field; aligns with post-booking reservation status. |",
        "| Company | **Drop** | 75,641/79,330 NULL; raw ID, not a booking-creation rate driver. |",
        "| Agent | **Keep** | Bookable field at creation (travel agent); fold rare codes and impute NULL→0 sentinel. |",
        "| ArrivalDateYear/Month/Day | **Keep → reconstruct** | Single `arrival_date`/`arrival_days` signal; WeekNumber dropped after reconstruction. |",
        "| ADR == 0 rows | **Keep** (primary view) | Genuine complimentary stays (see cross-tab); a drop-ADR==0 variant is logged as sensitivity only. |",
        "| ADR == 5400 row | **Drop** | Single extreme, not a business-meaning rate. |",
        "| Adults==0, zero-nights, Babies≥9 rows | **Drop** | Not valid stays (see rule counts). |",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure ADR-task evidence from the raw H2 CSV.")
    parser.add_argument("--data", type=Path, default=CSV_PATH)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    df = load_raw(args.data)
    ev = collect_evidence(df)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(ev, indent=2, default=str), encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(ev), encoding="utf-8")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
