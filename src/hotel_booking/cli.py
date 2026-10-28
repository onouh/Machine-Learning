"""Command-line entry point for the hotel-booking project."""

import argparse
import json
from pathlib import Path

from hotel_booking.services.analysis import analyze_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser."""
    parser = argparse.ArgumentParser(
        prog="hotel-booking",
        description="Analyze and validate the City Hotel H2 dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("analyze", "Print a compact data-quality profile."),
        ("validate", "Check the dataset schema and target labels."),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument(
            "--data",
            type=Path,
            default=Path("data/raw/H2.csv"),
            help="Path to H2 CSV (default: data/raw/H2.csv).",
        )
    return parser


def main() -> int:
    """Run a CLI command and return a process exit code."""
    args = build_parser().parse_args()
    if not args.data.is_file():
        print(f"Dataset not found: {args.data}. See data/README.md for setup instructions.")
        return 2

    _, profile, issues = analyze_dataset(args.data)
    if args.command == "analyze":
        print(json.dumps(profile, indent=2))
    else:
        if issues:
            print("Schema validation failed:")
            for issue in issues:
                print(f"- {issue}")
            return 1
        print(f"Schema validation passed for {args.data}.")
        print(json.dumps({"rows": profile["rows"], "columns": profile["columns"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
