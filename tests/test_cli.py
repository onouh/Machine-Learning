"""CLI smoke tests that do not require the source dataset."""

import pandas as pd

from hotel_booking.cli import main
from hotel_booking.data.validate import EXPECTED_COLUMNS


def test_cli_returns_actionable_code_when_dataset_is_missing(tmp_path, monkeypatch, capsys) -> None:
    missing_path = tmp_path / "missing.csv"
    monkeypatch.setattr("sys.argv", ["hotel-booking", "validate", "--data", str(missing_path)])

    result = main()

    assert result == 2
    assert "Dataset not found" in capsys.readouterr().out


def test_cli_validates_synthetic_h2_csv(tmp_path, monkeypatch, capsys) -> None:
    data_path = tmp_path / "h2.csv"
    frame = pd.DataFrame({column: [0, 1] for column in EXPECTED_COLUMNS})
    frame.to_csv(data_path, index=False)
    monkeypatch.setattr("sys.argv", ["hotel-booking", "validate", "--data", str(data_path)])

    result = main()

    assert result == 0
    assert "Schema validation passed" in capsys.readouterr().out
