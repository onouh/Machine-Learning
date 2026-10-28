"""Dataset summary helpers for the CLI and UI."""

from typing import Any

import pandas as pd


def profile_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return row/column counts and per-column quality summaries."""
    columns: list[dict[str, Any]] = []
    for name in frame.columns:
        series = frame[name]
        columns.append(
            {
                "name": name,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "cardinality": int(series.nunique(dropna=True)),
            }
        )

    cancellation_counts: dict[str, int] = {}
    if "IsCanceled" in frame:
        cancellation_counts = {
            str(label): int(count)
            for label, count in frame["IsCanceled"].value_counts(dropna=False).items()
        }

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "column_profile": columns,
        "is_canceled_counts": cancellation_counts,
    }
