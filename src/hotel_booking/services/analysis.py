"""Shared dataset analysis service."""

from pathlib import Path
from typing import Any

import pandas as pd

from hotel_booking.data.load import load_dataset
from hotel_booking.data.profiling import profile_dataset
from hotel_booking.data.validate import validate_schema


def analyze_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Load, validate, and profile a dataset without fitting transformations."""
    frame = load_dataset(path)
    return frame, profile_dataset(frame), validate_schema(frame)
