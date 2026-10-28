"""Dataset loading and lightweight normalization helpers."""

from pathlib import Path

import pandas as pd

MISSING_TOKENS = {"NULL", "NA", "N/A", ""}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV and normalize whitespace and common missing-value tokens."""
    frame = pd.read_csv(path, dtype_backend="numpy_nullable")
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].astype("string").str.strip()
        frame[column] = values.mask(values.str.upper().isin(MISSING_TOKENS))
    return frame
