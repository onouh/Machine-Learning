# Project Memory

Use this file for concise, durable decisions and verified project updates. Put actionable future work in [TASK.md](TASK.md); keep full methodology and results in `docs/`.

## Project decisions

- Use the City Hotel dataset `H2.csv` only; do not combine it with H1.
- Keep reusable Python implementation in `src/hotel_booking/`; notebooks are optional exploration, not the submitted solution.
- Provide both a CLI and a Streamlit UI over shared service functions.
- Use GitHub PRs for changes to `main`, with one required approval and required lightweight tests/lint checks.
- Ignore local datasets and generated outputs by default. Ignore generated model artifacts by default too; force-add only explicitly selected artifacts when justified. Keep fitted models out of the submission ZIP.
- Keep exact source column labels documented if code normalizes them.

## Verified dataset observations

These observations were measured from the supplied H2.csv; re-run profiling if the source file changes.

- 79,330 rows and 31 columns.
- 25,902 excess exact full-row duplicates; preserve and investigate rather than assuming they are invalid bookings.
- `IsCanceled` positive rate is about 41.7%.
- `ReservationStatus` agrees perfectly with `IsCanceled` in this file; it is direct classification leakage.
- Missing tokens include 4 `Children`, 24 `Country`, 8,131 `Agent`, and 75,641 `Company` entries.
- ADR has 1,208 zero values and a maximum of 5,400. Zero ADR meaning remains unresolved; compare analyses with zeros included and excluded.
- No customer identifier is present, so clustering can describe booking profiles, not longitudinal customer segments.
- The CSV headers use `ArrivalDateYear`, `ArrivalDateMonth`, `ArrivalDateWeekNumber`, and `ArrivalDateDayOfMonth`; the assignment handout renders some of these with spaces. Document any code mapping.

## Open assumptions

- Define the ADR prediction point before selecting regression predictors.
- Cancellation prediction is at booking creation; exclude fields only known or updated later, and verify feature availability against the real booking workflow.
- Exact duplicate handling needs sensitivity analysis because no reservation key is supplied.
