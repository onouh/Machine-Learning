"""Backward-compat shim for :mod:`hotel_booking.regression_models.run`.

All new regression work lives in the enclosing :mod:`regression_models`
subdirectory. This module re-exports the real implementation so that any
remaining callers (tests, scripts, docs) continue to resolve without
changes.
"""

from hotel_booking.regression_models.run import (
    _CSV_FIELDS,
    _append_runs_csv,
    _flatten_record,
    build_conclusions,
    main,
    run_once,
    run_suite,
    DEFAULT_TEST_SIZE,
    __all__ as _shim_all,
)

__all__ = list(_shim_all)

DEFAULT_TEST_SIZE = 0.2


def _resolve_paths() -> dict[str, Path]:
    cfg = load_config()
    paths_cfg = cfg.get("paths", {})
    results = Path(paths_cfg.get("results", "outputs/results"))
    return {
        "data": Path(cfg["data"]["path"]),
        "results": results,
        "runs": results / "regression",
        "report": Path("outputs/report"),
    }


def run_once(
    *,
    data_path: Path,
    variant: str,
    model_key: str,
    spec: ModelSpec,
    tuned: bool,
    drop_adr_zero: bool,
    seed: int,
    test_size: float = DEFAULT_TEST_SIZE,
) -> tuple[dict[str, Any], Any | None]:
    """One experiment cell; returns (run_record, best_estimator_for_reuse)."""
    t0 = time.perf_counter()
    label = (
        f"{variant}_{model_key}_{'tuned' if tuned else 'baseline'}"
        + ("" if not drop_adr_zero else "_no_adr0")
    )

    df = load_raw(data_path)
    cleaned, clean_log = clean_for_regression(
        df, variant=variant, drop_adr_zero=drop_adr_zero
    )
    X = cleaned.drop(columns=["ADR"])
    y = cleaned["ADR"].to_numpy(dtype=float)

    idx_train, idx_test = train_test_split(
        np.arange(len(X)),
        test_size=test_size,
        random_state=seed,
        stratify=(y == 0).astype(int),
    )
    X_tr, X_te = X.iloc[idx_train], X.iloc[idx_test]
    y_tr, y_te = y[idx_train], y[idx_test]
    # Strict disjointness on FULL cleaned rows (features + target): exact
    # full-row duplicates are removed pre-split, so a nonzero overlap means a
    # pipeline regression (e.g. split-before-dedup) — hard failure.
    cleaned_train = cleaned.iloc[idx_train]
    cleaned_test = cleaned.iloc[idx_test]
    fp_train_full = fingerprint_rows(cleaned_train)
    fp_test_full = fingerprint_rows(cleaned_test)
    fp_train_X = fingerprint_rows(X_tr)
    fp_test_X = fingerprint_rows(X_te)

    tuned_log: dict[str, Any] = {}
    if tuned:
        search_log = tune_model(spec, X_tr, y_tr, seed=seed, variant=variant)
        tuned_log = {k: v for k, v in search_log.items() if k != "best_estimator"}
        if search_log.get("best_estimator") is not None:
            # Refit by the search (on the capped subsample when spec.subsample
            # applies; recorded in search_subsample_used).
            pipeline = search_log["best_estimator"]
            fit_seconds = float(search_log.get("refit_seconds") or 0.0)
        else:  # pragma: no cover - RandomizedSearchCV always refits
            pipeline = build_full_pipeline(spec, variant)
            fit_start = time.perf_counter()
            pipeline.fit(X_tr, y_tr)
            fit_seconds = time.perf_counter() - fit_start
    else:
        pipeline = build_full_pipeline(spec, variant)
        fit_start = time.perf_counter()
        pipeline.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - fit_start

    y_pred = np.clip(np.asarray(pipeline.predict(X_te), dtype=float), 0.0, None)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mae = float(mean_absolute_error(y_te, y_pred))
    r2 = float(r2_score(y_te, y_pred))

    leak_report = assert_no_leakage(
        train_fingerprints=fp_train_full,
        test_fingerprints=fp_test_full,
        train_feature_fingerprints=fp_train_X,
        feature_view_fingerprints=(fp_train_X, fp_test_X),
        feature_columns=list(X_tr.columns),
        fitted_pipeline=pipeline,
        forbidden_substrings=LEAKAGE_FORBIDDEN_FEATURES if variant == "after" else [],
    )

    preproc = pipeline.named_steps["prep"]
    record: dict[str, Any] = {
        "label": label,
        "run_id": f"{label}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": "ADR",
        "metric_in": "original ADR units (predictions clipped at >= 0)",
        "prediction_point": PREDICTION_POINT,
        "variant": variant,
        "variant_note": clean_log.get("variant_note", ""),
        "truncated_columns_dropped_after": sorted(
            c for c in DROP_WITH_REASON if variant == "after" and c not in {"ADR"}
        ),
        "model": model_key,
        "tuned": tuned,
        "scoring": SCORING if tuned else "default params (baseline)",
        "drop_adr_zero": drop_adr_zero,
        "random_seed": seed,
        "test_size": test_size,
        "rows_total_after_cleaning": clean_log["rows_after_cleaning"],
        "rows_training": int(len(X_tr)),
        "rows_test": int(len(X_te)),
        "rows_removed_by_invalid_rules": clean_log["rows_removed_by_invalid_rules"],
        "invalid_row_rule_counts": clean_log["invalid_row_rules"]["counts"],
        "adr_zero_rows": clean_log["adr_zero_rows"],
        "adr_zero_rule": clean_log["adr_zero_rule"],
        "arrival_window": f"{clean_log['arrival_min']} .. {clean_log['arrival_max']}",
        "n_preprocessor_features": len(feature_names(preproc)),
        "feature_names": feature_names(preproc),
        "best_params": tuned_log.get("best_params", {}),
        "search_space_as_params": tuned_log.get("search_space_as_params", {}),
        "best_cv_rmse": tuned_log.get("best_cv_rmse"),
        "search_n_iter": tuned_log.get("n_iter"),
        "search_cv_folds": tuned_log.get("cv_folds"),
        "search_subsample_used": tuned_log.get("subsample_used", 0),
        "search_seconds": tuned_log.get("search_seconds"),
        "refit_seconds": tuned_log.get("refit_seconds"),
        "fit_seconds": round(fit_seconds, 3),
        "total_seconds": round(time.perf_counter() - t0, 3),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "test_adr_mean": float(np.mean(y_te)),
        "test_adr_median": float(np.median(y_te)),
        "test_adr_max": float(np.max(y_te)),
        "leak_check": leak_report,
    }
    return record, pipeline


def run_suite(
    *,
    data_path: Path | None = None,
    out_dir: Path | None = None,
    variants: tuple[str, ...] = VARIANTS,
    models: tuple[str, ...] | None = None,
    tuned: bool = True,
    baseline: bool = True,
    drop_adr_zero: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run the experiment matrix and log every cell (JSON + runs.csv)."""
    paths = _resolve_paths()
    data_path = data_path or paths["data"]
    out_dir = out_dir or paths["runs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "runs.csv"
    if seed is None:
        seed = int(load_config().get("random_seed", 42))
    specs = regressors(seed)
    selected = list(models) if models else list(specs)

    rows: list[dict[str, Any]] = []
    total = len(variants) * len(selected) * ((1 if baseline else 0) + (1 if tuned else 0))
    done = 0
    for variant in variants:
        for model_key in selected:
            spec = specs[model_key]
            for is_tuned in (
                [True, False] if tuned and baseline else ([True] if tuned else [False])
            ):
                done += 1
                print(f"[{done:02d}/{total:02d}] {variant} {model_key} "
                      f"{'tuned' if is_tuned else 'baseline'} ...", flush=True)
                try:
                    record, _est = run_once(
                        data_path=data_path,
                        variant=variant,
                        model_key=model_key,
                        spec=spec,
                        tuned=is_tuned,
                        drop_adr_zero=drop_adr_zero,
                        seed=seed,
                    )
                except LeakageError as exc:
                    record = {
                        "label": f"{variant}_{model_key}_leakage_failure",
                        "variant": variant,
                        "model": model_key,
                        "tuned": is_tuned,
                        "error": f"LeakageError: {exc}",
                        "leak_check_passed": False,
                    }
                    print(f"  !! leakage check FAILED: {exc}")
                except Exception as exc:  # keep the matrix running; log the error
                    record = {
                        "label": f"{variant}_{model_key}_{'tuned' if is_tuned else 'baseline'}",
                        "variant": variant,
                        "model": model_key,
                        "tuned": is_tuned,
                        "error": repr(exc),
                    }
                    print(f"  !! error: {exc!r}")
                rows.append(record)
                (out_dir / f"{record['label']}.json").write_text(
                    json.dumps(record, indent=2, default=str), encoding="utf-8"
                )
                _append_runs_csv(runs_csv, record)
            print(f"    rmse={record.get('rmse', 'n/a')} "
                  f"best_params={record.get('best_params', '-')}", flush=True)

    runs_df = pd.DataFrame(rows)

    # Fail the suite if any run leaked (error records are reported separately).
    if "leak_check" in runs_df.columns:
        leak_records = runs_df[runs_df["leak_check"].notna()]
        passed = leak_records["leak_check"].map(
            lambda report: isinstance(report, dict) and report.get("passed", False)
        )
        if not bool(passed.all()):
            raise LeakageError(
                "At least one run failed the leakage checker; see its run JSON."
            )
    return runs_df


_CSV_FIELDS = [
    "label", "timestamp", "target", "metric_in", "prediction_point",
    "variant", "variant_note", "model", "tuned", "scoring",
    "drop_adr_zero", "random_seed", "test_size",
    "rows_total_after_cleaning", "rows_training", "rows_test",
    "rows_removed_by_invalid_rules", "invalid_row_rule_counts",
    "adr_zero_rows", "adr_zero_rule", "arrival_window",
    "n_preprocessor_features", "feature_names",
    "best_params", "search_space_as_params", "best_cv_rmse",
    "search_n_iter", "search_cv_folds", "search_subsample_used",
    "search_seconds", "refit_seconds",
    "fit_seconds", "total_seconds",
    "rmse", "mae", "r2",
    "test_adr_mean", "test_adr_median", "test_adr_max",
    "leak_check", "leak_check_passed", "error",
]


def _append_runs_csv(runs_csv: Path, record: dict[str, Any]) -> None:
    """Append one row with a fixed field list so multi-invocation logs align."""
    flat = _flatten_record(record)
    new_file = not runs_csv.exists()
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    with runs_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({field: flat.get(field, "") for field in _CSV_FIELDS})


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat = dict(record)
    for key in ("feature_names", "best_params", "search_space_as_params",
                "leak_check", "invalid_row_rule_counts"):
        if key in flat:
            flat[key] = json.dumps(flat[key], default=str)
    flat["leak_check_passed"] = bool(record.get("leak_check", {}).get("passed", False))
    return flat


def build_conclusions(runs_dir: Path | None = None) -> str:
    """Conclusions markdown from the logged runs (per-model hyperparameters +
    before-vs-after comparison + ADR==0 sensitivity)."""
    paths = _resolve_paths()
    runs_dir = runs_dir or paths["runs"]
    runs_csv = runs_dir / "runs.csv"
    if not runs_csv.exists():
        raise FileNotFoundError(f"No runs.csv at {runs_csv}; run the suite first.")
    runs_df = pd.read_csv(runs_csv)
    if runs_df.empty:
        raise RuntimeError(f"No runs logged in {runs_csv}")
    ok = runs_df[runs_df["error"].isna()] if "error" in runs_df.columns else runs_df
    ok = ok[ok["rmse"].notna()]
    lines: list[str] = []
    lines.append("# ADR regression — experiment conclusions")
    lines.append("")
    lines.append(f"Prediction point: **{PREDICTION_POINT}**. All metrics are reported in "
                 "original ADR units on the held-out test set (predictions clipped at ≥ 0).")
    lines.append("")
    lines.append("## Run matrix")
    lines.append("")
    lines.append(f"- Runs logged: {len(runs_df)} ({len(ok)} successful, "
                 f"{len(runs_df) - len(ok)} failed).")
    lines.append(f"- Variants: {', '.join(runs_df['variant'].unique())}")
    lines.append(f"- Models: {', '.join(str(m) for m in ok['model'].unique())}")
    lines.append("")
    lines.append("## Per-model results (original ADR units)")
    lines.append("")
    lines.append("| metric | best run | variant | tuned | RMSE | MAE | R2 |")
    lines.append("|---|---|---|---|---|---|---|")
    best = ok.sort_values("rmse").iloc[0]
    lines.append(
        f"| best overall | `{best['label']}` | {best['variant']} | {best['tuned']} | "
        f"{best['rmse']:.2f} | {best['mae']:.2f} | {best['r2']:.3f} |"
    )
    for variant in VARIANTS:
        for tuned_flag in (True, False):
            sub = ok[(ok["variant"] == variant) & (ok["tuned"] == tuned_flag)]
            if sub.empty:
                continue
            sub_best = sub.sort_values("rmse").iloc[0]
            lines.append(
                f"| best {variant} {'tuned' if tuned_flag else 'baseline'} "
                f"| `{sub_best['label']}` | "
                f"{sub_best['variant']} | {sub_best['tuned']} | {sub_best['rmse']:.2f} | "
                f"{sub_best['mae']:.2f} | {sub_best['r2']:.3f} |"
            )
    lines.append("")
    lines.append("### All runs")
    lines.append("")
    lines.append(
        "| variant | model | tuned | RMSE | MAE | R2 | rows_train | rows_test | n_features |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, row in ok.sort_values(["variant", "model", "tuned"]).iterrows():
        lines.append(
            f"| {row['variant']} | {row['model']} | {row['tuned']} | "
            f"{row['rmse']:.2f} | {row['mae']:.2f} | {row['r2']:.3f} | "
            f"{row['rows_training']:,} | {row['rows_test']:,} | "
            f"{row['n_preprocessor_features']} |"
        )
    lines.append("")

    lines.append("## Per-model best hyperparameters (CV-tuned)")
    lines.append("")
    lines.append(
        "| variant | model | best params | CV RMSE | test RMSE | CV→test gap | search sec |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for _, row in ok[ok["tuned"] == True].sort_values(["variant", "model"]).iterrows():  # noqa: E712
        gap = row["rmse"] - row["best_cv_rmse"]
        lines.append(
            f"| {row['variant']} | {row['model']} | `{row['best_params']}` | "
            f"{row['best_cv_rmse']:.2f} | {row['rmse']:.2f} | {gap:+.2f} | "
            f"{row['search_seconds']} |"
        )
    lines.append("")

    # --- before vs after comparison (from logged numbers only) ---------------
    lines.append("## Before vs after truncation (columns removed at booking time)")
    lines.append("")
    pairable = ok[ok["drop_adr_zero"] == False].pivot_table(  # noqa: E712
        index="model", columns="variant", values="rmse", aggfunc="min"
    )
    lines.append("| model | RMSE before | RMSE after | delta (before − after) |")
    lines.append("|---|---|---|---|")
    for model, row in pairable.iterrows():
        before, after = row.get("before"), row.get("after")
        if pd.isna(before) or pd.isna(after):
            continue
        lines.append(
            f"| {row.name} | {before:.2f} | {after:.2f} | {before - after:+.2f} |"
        )
    lines.append("")
    lines.append("A positive delta means the truncated (leak-free) feature set "
                 "**lost** information; a negative delta means removing the "
                 "post-booking data did not hurt (or actually improved) the "
                 "heterogeneous prediction on the held-out test set.")
    lines.append("")

    # --- ADR == 0 sensitivity -------------------------------------------------
    sens = runs_df[runs_df["drop_adr_zero"] == True]  # noqa: E712
    if not sens.empty and "rmse" in sens.columns:
        lines.append("## ADR == 0 sensitivity (rows dropped)")
        lines.append("")
        lines.append("| variant | model | RMSE without ADR==0 rows |")
        lines.append("|---|---|---|")
        for _, row in sens.sort_values("rmse").iterrows():
            lines.append(f"| {row['variant']} | {row['model']} | {row['rmse']:.2f} |")
        base = ok[
            (ok["drop_adr_zero"] == False)  # noqa: E712
            & (ok["model"] == sens["model"].iloc[0])
        ].sort_values("rmse").iloc[0]
        lines.append("")
        lines.append(
            f"Comparable ADR==0-kept run: `{base['label']}` RMSE "
            f"{base['rmse']:.2f} → the drop/remove of "
            f"{int(runs_df['adr_zero_rows'].iloc[0])} "
            "complimentary-stay rows changes the target mean and thus "
            "RMSE levels; see evidence doc section 4 for why the rows "
            "are kept in the primary view."
        )
        lines.append("")

    lines.append("## Preprocessing decisions (measured evidence)")
    lines.append("")
    lines.append("- Prediction point: booking creation; every column listed in "
                 "`DROP_WITH_REASON` is excluded from the strict feature set.")
    lines.append("- Learned transforms (imputer, rare-category fold, one-hot, scaler) "
                 "are fit on training folds only, inside both CV folds and the "
                 "final refit.")
    lines.append("- ADR==0 rows are kept in the primary view (complimentary stays).")
    lines.append("- Invalid rows removed (per-rule counts logged per run).")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```")
    lines.append(".venv/bin/python -m hotel_booking.regression.run  # full matrix")
    lines.append('.venv/bin/python main.py regression --variant after --models RandomForest')
    lines.append(".venv/bin/python -m hotel_booking.reporting.report  # HTML report")
    lines.append("```")
    lines.append("")
    lines.append("All runs logged in `outputs/results/regression/runs.csv` and one "
                 "JSON per run in the same directory.")
    return "\n".join(lines) + "\n"


def main(
    *,
    data_path: Path | None = None,
    out_dir: Path | None = None,
    variants: tuple[str, ...] = VARIANTS,
    models: tuple[str, ...] | None = None,
    tuned: bool = True,
    baseline: bool = True,
    drop_adr_zero: bool = False,
    skip_conclusions: bool = False,
) -> pd.DataFrame:
    runs_df = run_suite(
        data_path=data_path,
        out_dir=out_dir,
        variants=variants,
        models=models,
        tuned=tuned,
        baseline=baseline,
        drop_adr_zero=drop_adr_zero,
    )
    if not skip_conclusions:
        md = build_conclusions(out_dir)
        out_path = (out_dir or _resolve_paths()["runs"]) / "conclusions.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"Wrote {out_path}")
    return runs_df


if __name__ == "__main__":
    main()
