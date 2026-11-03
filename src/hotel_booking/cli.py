"""Command-line entry point for the hotel-booking project."""

import argparse
from pathlib import Path

from hotel_booking.regression import pipeline as regression_pipeline
from hotel_booking.regression.run import run_suite
from hotel_booking.reporting import report as reporting_report
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
            required=False,
            default=Path("data/raw/H2.csv"),
            help="Path to H2 CSV (default: data/raw/H2.csv).",
        )

    regression_cmd = subparsers.add_parser(
        "regression", help="Run ADR regression experiments (baseline + CV-tuned)."
    )
    regression_cmd.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/H2.csv"),
        help="Path to H2 CSV (default: data/raw/H2.csv).",
    )
    regression_cmd.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/results/regression"),
        help="Directory for runs.csv, per-run JSON, and conclusions.",
    )
    regression_cmd.add_argument(
        "--variant",
        choices=("before", "after", "both"),
        default="both",
        help="Feature-set variant: truncated ('after') / pre-truncation ('before') / both.",
    )
    regression_cmd.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model subset (default: the full model grid).",
    )
    regression_cmd.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip CV hyperparameter search (baseline-config runs only).",
    )
    regression_cmd.add_argument(
        "--drop-adr-zero",
        action="store_true",
        help="Sensitivity: drop ADR==0 rows before modeling (logged variant).",
    )
    regression_cmd.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (default: config.yaml).",
    )

    report_cmd = subparsers.add_parser(
        "report", help="Regenerate and validate the HTML report from the latest runs."
    )
    report_cmd.add_argument(
        "--runs",
        type=Path,
        default=None,
        help="Path to runs.csv (default: outputs/results/regression/runs.csv).",
    )
    report_cmd.add_argument(
        "--out",
        type=Path,
        default=None,
        help="HTML report path (default: outputs/report/regression_report.html).",
    )
    return parser


def main() -> int:
    """Run a CLI command and return a process exit code."""
    args = build_parser().parse_args()
    if args.command in {"analyze", "validate"} and not args.data.is_file():
        print(f"Dataset not found: {args.data}. See data/README.md for setup instructions.")
        return 2

    if args.command == "analyze":
        _, profile, issues = analyze_dataset(args.data)
        import json

        print(json.dumps(profile, indent=2))
        return 0
    if args.command == "validate":
        _, profile, issues = analyze_dataset(args.data)
        if issues:
            print("Schema validation failed:")
            for issue in issues:
                print(f"- {issue}")
            return 1
        print(f"Schema validation passed for {args.data}.")
        import json

        print(json.dumps({"rows": profile["rows"], "columns": profile["columns"]}, indent=2))
        return 0
    if args.command == "regression":
        variants: tuple[str, ...]
        if args.variant == "both":
            variants = regression_pipeline.VARIANTS
        else:
            variants = (args.variant,)
        models = tuple(m.strip() for m in args.models.split(",") if m.strip()) or None
        run_suite(
            data_path=args.data,
            out_dir=args.out,
            variants=variants,
            models=models,
            tuned=not args.no_tune,
            drop_adr_zero=args.drop_adr_zero,
            seed=args.seed,
        )
        return 0
    if args.command == "report":
        return reporting_report.main(runs_csv=args.runs, out_html=args.out, validate=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
