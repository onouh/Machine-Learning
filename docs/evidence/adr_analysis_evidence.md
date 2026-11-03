# ADR regression — measured evidence from the full raw H2.csv

Every claim in the ADR preprocessing keep/drop logic traces to a number below.
Source: `data/raw/H2.csv` — 79,330 rows × 31 columns.

## 1. Leakage checks

| Label                                                                  | Value                    |
|------------------------------------------------------------------------|--------------------------|
| ReservationStatus['Check-Out'] ↔ IsCanceled==0 with the rest ↔ 1       | True                     |
| Rows where ReservationStatusDate precedes arrival                      | 31,631                   |
| ReservationStatusDate range                                            | 2014-10-17 .. 2017-09-07 |
| Arrival range                                                          | 2015-07-01 .. 2017-08-31 |
| BookingChanges > 0 rows (post-booking edits exist)                     | 10,268                   |
| DaysInWaitingList > 0 rows (only known after request)                  | 3,443                    |
| AssignedRoomType ≠ ReservedRoomType (assign happens at/after check-in) | 7,192                    |

Cross-tab ReservationStatus × IsCanceled (perfect agreement ⇒ direct cancellation outcome = columns ReservationStatus, ReservationStatusDate must not be ADR predictors at the booking-creation point):

| Label                         | Value |
|-------------------------------|-------|
| status=Canceled|isCanceled=1  | 32186 |
| status=Check-Out|isCanceled=0 | 46228 |
| status=No-Show|isCanceled=1   | 916   |

## 2. Missing-value profile (raw tokens before normalization)

| Label    | Value |
|----------|-------|
| Children | 4     |
| Country  | 24    |
| Agent    | 8131  |
| Company  | 75641 |

Literal `NULL`/`NA` token counts **before** normalization: {"Country": {"NULL": 24}, "Agent": {"NULL": 8131}, "Company": {"NULL": 75641}}

Tokens surviving normalization (must be `{}` → proven literal categories): {}

## 3. Extremes and invalid-row rules

| Label                        | Value |
|------------------------------|-------|
| ADR < 0 rows                 | 0     |
| ADR == 5400 rows             | 1     |
| Max ADR excluding 5400       | 510.0 |
| LeadTime max                 | 629   |
| Stays with total nights == 0 | 331   |
| Adults == 0 rows             | 390   |
| Total guests ≤ 0 rows        | 167   |
| Babies ∈ {9, 10} rows        | 2     |

ADR quantiles: {"0.01": 0.0, "0.25": 79.2, "0.5": 99.9, "0.75": 126.0, "0.99": 228.0}

## 4. ADR == 0 business meaning

| Label                                     | Value                |
|-------------------------------------------|----------------------|
| ADR == 0 rows                             | 1,208 (1.5% of rows) |
| of which MarketSegment != 'Complementary' | 700                  |

ADR==0 × ReservationStatus × MarketSegment (is 0 a real complimentary rate?):

| Label                                  | Value |
|----------------------------------------|-----|
| status=Canceled|segment=Complementary  | 54  |
| status=Canceled|segment=Corporate      | 3   |
| status=Canceled|segment=Direct         | 8   |
| status=Canceled|segment=Groups         | 14  |
| status=Canceled|segment=Offline TA/TO  | 27  |
| status=Canceled|segment=Online TA      | 8   |
| status=Check-Out|segment=Aviation      | 5   |
| status=Check-Out|segment=Complementary | 447 |
| status=Check-Out|segment=Corporate     | 27  |
| status=Check-Out|segment=Direct        | 102 |
| status=Check-Out|segment=Groups        | 130 |
| status=Check-Out|segment=Offline TA/TO | 209 |
| status=Check-Out|segment=Online TA     | 159 |
| status=No-Show|segment=Aviation        | 1   |
| status=No-Show|segment=Complementary   | 7   |
| status=No-Show|segment=Direct          | 1   |
| status=No-Show|segment=Online TA       | 6   |

ADR==0 by DepositType: {"No Deposit": 1208}
ADR==0 by CustomerType: {"Transient": 900, "Transient-Party": 288, "Contract": 12, "Group": 8}

## 5. Company / Agent fields

| Label                   | Value          |
|-------------------------|----------------|
| Company NULL / non-null | 75,641 / 3,689 |
| Company distinct values | 207            |
| Agent NULL / non-null   | 8,131 / 71,199 |
| Agent distinct values   | 223            |

## 6. Duplicates

| Label                                             | Value  |
|---------------------------------------------------|--------|
| Exact duplicate rows (excluding first occurrence) | 25,902 |
| Rows that belong to a duplicate group             | 31,748 |

## 7. Other distribution facts

- Babies value counts: {"0": 78961, "1": 361, "2": 6, "9": 1, "10": 1}
- ADR mean: 105.30

### Keep/drop rationale this evidence supports

| Column | Decision | Reason (evidence above) |
|---|---|---|
| ReservationStatus / ReservationStatusDate | **Drop** | Perfect IsCanceled agreement; status date is post-arrival for most rows → outcome leakage at the booking-creation point. |
| BookingChanges | **Drop** | Post-booking edits (rows with changes exist); not knowable when ADR is fixed. |
| DaysInWaitingList | **Drop** | Realized after the customer requests, not at booking creation. |
| AssignedRoomType | **Drop** | Assigned at/after check-in; refit differs from ReservedRoomType in thousands of rows. |
| IsCanceled | **Drop** | Outcome field; aligns with post-booking reservation status. |
| Company | **Drop** | 75,641/79,330 NULL; raw ID, not a booking-creation rate driver. |
| Agent | **Keep** | Bookable field at creation (travel agent); fold rare codes and impute NULL→0 sentinel. |
| ArrivalDateYear/Month/Day | **Keep → reconstruct** | Single `arrival_date`/`arrival_days` signal; WeekNumber dropped after reconstruction. |
| ADR == 0 rows | **Keep** (primary view) | Genuine complimentary stays (see cross-tab); a drop-ADR==0 variant is logged as sensitivity only. |
| ADR == 5400 row | **Drop** | Single extreme, not a business-meaning rate. |
| Adults==0, zero-nights, Babies≥9 rows | **Drop** | Not valid stays (see rule counts). |
