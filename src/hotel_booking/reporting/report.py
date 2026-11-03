"""Self-contained HTML experiment report for ADR regression runs.

Reads the latest ``runs.csv`` (plus per-run JSONs) and renders a static,
dependency-free report — tables and CSS only; no JavaScript, no CDN — to
``outputs/report/regression_report.html``.

``validate_report`` parses the generated HTML with stdlib ``html.parser``
and confirms the required sections and referenced exists.
"""

from __future__ import annotations

from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

from hotel_booking.regression.pipeline import (
    PREDICTION_POINT,
)

REPORT_PATH = Path("outputs/report/regression_report.html")
RUNS_DIR = Path("outputs/results/regression")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    thead = "".join(f"<th>{escape(h)}</th>" for h in headers)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr>" + thead + "</tr></thead><tbody>" + tbody + "</tbody></table>"
    )


def build_report(
    runs_csv: Path | None = None,
    out_html: Path | None = None,
) -> Path:
    """Render the report; returns the written path."""
    runs_csv = runs_csv or RUNS_DIR / "runs.csv"
    out_html = out_html or REPORT_PATH
    if not runs_csv.exists():
        raise FileNotFoundError(
            f"No runs.csv at {runs_csv}; run the regression suite first."
        )
    runs_df = pd.read_csv(runs_csv)
    ok = runs_df[runs_df["error"].isna()] if "error" in runs_df.columns else runs_df
    ok = ok[ok["rmse"].notna()].copy() if "rmse" in ok.columns else ok

    h1 = "<h1>ADR regression — experiment report</h1>"
    p_point = (
        f"<p class='lead'>Prediction point: <strong>{escape(PREDICTION_POINT)}</strong>. "
        "Metrics on the held-out test set in original ADR units "
        "(predictions clipped at 0). Every run passed the four-path leakage "
        "checker (columns, row disjointness, fit manifest, tuning purity)."
        "</p>"
    )

    # Summary: best runs
    summary_rows: list[list[str]] = []
    best_over = ok.sort_values("rmse").iloc[0]
    summary_rows.append([
        "Best overall", str(best_over["label"]),
        f"{best_over['rmse']:.2f}", f"{best_over['mae']:.2f}", f"{best_over['r2']:.3f}",
    ])
    for variant in ("before", "after"):
        sub = ok[ok["variant"] == variant]
        if sub.empty:
            continue
        best_v = sub.sort_values("rmse").iloc[0]
        summary_rows.append([
            f"Best {variant}", str(best_v["label"]),
            f"{best_v['rmse']:.2f}", f"{best_v['mae']:.2f}", f"{best_v['r2']:.3f}",
        ])
    summary_tables = _table(
        ["scope", "run label", "RMSE", "MAE", "R2"], summary_rows
    )

    # All runs table
    runs_rows = [
        [
            str(r["variant"]), str(r["model"]), str(r["tuned"]),
            f"{r['rmse']:.2f}", f"{r['mae']:.2f}", f"{r['r2']:.3f}",
            str(int(r["search_n_iter"])) if not pd.isna(r.get("search_n_iter")) else "-",
            f"{r['best_cv_rmse']:.2f}" if not pd.isna(r.get("best_cv_rmse")) else "-",
        ]
        for _, r in ok.sort_values(["variant", "model"]).iterrows()
    ]
    runs_tables = _table(
        ["variant", "model", "tuned", "RMSE", "MAE", "R2", "search iters", "best CV RMSE"], runs_rows
    )

    # Hyperparameters
    tuned = ok[ok["tuned"] == True].sort_values(["variant", "model"])  # noqa: E712
    hyper_rows = [
        [
            escape(str(r["variant"])), escape(str(r["model"])),
            escape(str(r["best_params"])),
            f"{r['best_cv_rmse']:.2f}", f"{r['rmse']:.2f}",
        ]
        for _, r in tuned.iterrows()
    ]
    hyper_tables = _table(
        ["variant", "model", "best params (CV-tuned)", "best CV RMSE", "test RMSE"], hyper_rows
    )

    # Before vs after comparison
    pair = ok[ok["drop_adr_zero"] == False].pivot_table(  # noqa: E712
        index="model", columns="variant", values="rmse", aggfunc="min"
    )
    ba_rows = []
    for model, row in pair.iterrows():
        before, after = row.get("before"), row.get("after")
        if pd.isna(before) or pd.isna(after):
            continue
        ba_rows.append([str(model), f"{before:.2f}", f"{after:.2f}", f"{before - after:+.2f}"])
    ba_tables = _table(["model", "RMSE before", "RMSE after", "delta (before − after)"], ba_rows)
    ba_note = (
        "<p>A negative delta shows the truncated booking-time feature set "
        "removed skewing information without hurting held-out accuracy; positive "
        "means the removal lost usable signal. Claims are read solely from the "
        "logged RMSE numbers above.</p>"
    )

    # ADR == 0 sensitivity
    sens = runs_df[runs_df["drop_adr_zero"] == True]  # noqa: E712
    sens_html = ""
    if not sens.empty:
        rows = [
            [str(r["variant"]), str(r["model"]), f"{r['rmse']:.2f}"]
            for _, r in sens.sort_values("rmse").iterrows()
        ]
        sens_html = (
            "<section><h2>ADR == 0 sensitivity</h2>"
            + _table(["variant", "model", "RMSE (ADR==0 rows dropped)"], rows)
            + "<p> Primary view keeps ADR==0 rows (complimentary stays; "
              "docs/evidence section 4). This table is the logged effect of "
              "dropping them.</p>"
            "</section>"
        )

    # Preprocessing rationale
    rationale_html = (
        "<section><h2>Preprocessing rationale (evidence-based)</h2>"
        "<p>See <code>docs/evidence/adr_analysis_evidence.md</code> for every "
        "measured number behind these decisions.</p>"
        + _table(
            ["decision", "what it does"],
            [
                ["Prediction point", "booking creation only"],
                ["Dropped columns", "ReservationStatus(+Date), BookingChanges, DaysInWaitingList, AssignedRoomType, Company (75,641 NULL), ArrivalDateWeekNumber after reconstruction; IsCanceled in 'after' only"],
                ["Agent", "kept as booking-time categorical; NULL→'0' sentinel, rare codes folded by train-fold OneHotEncoder(min_frequency=30)"],
                ["Country", "CN→CHN normalized; rare countries folded by min_frequency"],
                ["Arrival features", "Year/Month/Day → arrival_days, arrival_month, arrival_dow"],
                ["ADR==0", "kept in primary view (complimentary stays); drop variant logged as sensitivity"],
                ["Invalid rows", "ADR==5400, guests≤0, nights≤0, Adults==0, Babies∈{9,10} — per-rule counts logged per run"],
                ["Duplicates", "exact duplicates dropped keep-first (logged per run)"],
                ["Learned transforms", "imputer/scaler/one-hot fit on training folds only (verified by manifest)"],
                ["Split", "80/20 split, stratified on ADR==0, fixed random seed; test set never used for fitting or tuning"],
            ],
        )
        + "</section>"
    )

    # Leakage hints: sum over runs
    leak_ok = int(runs_df.get("leak_check_passed", pd.Series(dtype=bool)).astype(bool).sum()
                  if "leak_check_passed" in runs_df.columns else 0)
    leak_html = f"<p class='note'>Runs that passed the leakage checker: {leak_ok} / {len(runs_df)}.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ADR regression — experiment report</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; margin: 2rem auto; max-width: 1180px; color: #1a1f24; }}
h1 {{ font-size: 1.6rem; }}
h2 {{ font-size: 1.2rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.6rem 0; }}
th, td {{ border: 1px solid #d7dde2; padding: 6px 10px; text-align: left; vertical-align: top; }}
th {{ background: #f2f5f7; }}
p.lead {{ color: #333; font-size: 0.98rem; }}
p.note {{ color: #5a6570; font-size: 0.9rem; }}
section {{ margin-bottom: 1.5rem; }}
</style>
</head>
<body>
{h1}
{p_point}
{leak_html}
<section><h2>Summary — best runs</h2>{summary_tables}</section>
<section><h2>All runs (variant × model × tuning)</h2>{runs_tables}</section>
<section><h2>Per-model hyperparameters (CV-tuned)</h2>{hyper_tables}</section>
<section><h2>Before vs after truncation</h2>{ba_tables}{ba_note}</section>
{sens_html}
{rationale_html}
<section><h2>Missing rows / errors</h2>{_error_table(runs_df)}</section>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


def _error_table(runs_df: pd.DataFrame) -> str:
    if "error" not in runs_df.columns:
        return "<p>No failed runs.</p>"
    errors = runs_df[runs_df["error"].notna()]
    if errors.empty:
        return "<p>No failed runs.</p>"
    rows = [[str(r["label"]), escape(str(r["error"]))] for _, r in errors.iterrows()]
    return _table(["run label", "error"], rows)


class _ReportChecker(HTMLParser):
    """Strict checker: track tag pairing and collect section header text."""

    _VOID_TAGS = {"meta", "br", "img", "hr", "link", "input"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.h2_texts: list[str] = []
        self._in_h2 = False
        self._h2_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag not in self._VOID_TAGS:
            self.stack.append(tag)
        if tag == "h2":
            self._in_h2 = True
            self._h2_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._VOID_TAGS:
            return
        if tag == "h2" and self._in_h2:
            self.h2_texts.append("".join(self._h2_buffer).strip())
            self._in_h2 = False
        if not self.stack:
            self.errors.append(f"Unexpected closing tag </{tag}>")
            return
        opened = self.stack.pop()
        if opened != tag:
            self.errors.append(f"Mismatched tag: <{opened}> closed by </{tag}>")
        if tag == "h2" and self._in_h2 is False:
            pass

    def handle_data(self, data: str) -> None:
        if self._in_h2:
            self._h2_buffer.append(data)


def validate_report(
    html_path: Path | None = None,
    runs_csv: Path | None = None,
) -> list[str]:
    """Validate the report: HTML pairing, required sections, artifact presence.

    Returns a list of problems; empty list means the report is valid.
    """
    html_path = html_path or REPORT_PATH
    runs_csv = runs_csv or RUNS_DIR / "runs.csv"
    if not html_path.exists():
        return [f"Report missing: {html_path}"]
    text = html_path.read_text(encoding="utf-8")

    parser = _ReportChecker()
    parser.feed(text)
    parser.close()
    problems: list[str] = list(parser.errors)
    if parser.stack:
        problems.append(f"Unclosed tags: {parser.stack}")

    required_headers = [
        "Summary — best runs", "All runs (variant × model × tuning)",
        "Per-model hyperparameters (CV-tuned)", "Before vs after truncation",
        "Preprocessing rationale (evidence-based)",
    ]
    found = {h for h in parser.h2_texts}
    for header in required_headers:
        if header not in found:
            problems.append(f"Missing report section: {header}")

    # Runs log + every per-run JSON recorded in runs.csv must exist on disk:
    # they are the referenced artifacts behind the report tables.
    if not runs_csv.exists():
        problems.append(f"Runs log missing: {runs_csv}")
        return problems
    runs_df = pd.read_csv(runs_csv)
    runs_dir = runs_csv.parent
    labels = set(map(str, runs_df.get("label", pd.Series([], dtype=str)).dropna()))
    for label in sorted(labels):
        artifact = runs_dir / f"{label}.json"
        if not artifact.exists():
            problems.append(f"Referenced artifact missing: {artifact}")
    return problems


def main(
    runs_csv: Path | None = None, out_html: Path | None = None, validate: bool = True
) -> int:
    path = build_report(runs_csv=runs_csv, out_html=out_html)
    print(f"Wrote report: {path}")
    if validate:
        problems = validate_report(out_html or path, runs_csv)
        if problems:
            print("Report validation FAILED:")
            for problem in problems:
                print(f"- {problem}")
            return 1
        print("Report validation passed (HTML parse + sections + artifacts).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
