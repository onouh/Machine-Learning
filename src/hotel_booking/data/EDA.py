import argparse
import difflib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load import load_dataset

# Continuous / count columns worth looking at for skew and outliers.
NUMERIC = [
    "LeadTime", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "BookingChanges",
    "DaysInWaitingList", "ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
]
PLACEHOLDERS = {"NULL", "NA", "N/A", "NAN", "", "?", "UNDEFINED", "UNKNOWN"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DTYPE_NOTES = {
    "Children": "has text tokens ('NA') in the raw file -> should be an integer count",
    "ReservationStatusDate": "stored as text -> parse to datetime",
    "Agent": "ID stored as text -> categorical, never numeric",
    "Company": "ID stored as text -> categorical, never numeric",
    "IsCanceled": "binary target stored as int -> treat as a class label",
    "IsRepeatedGuest": "binary flag stored as int -> treat as categorical",
    "ArrivalDateMonth": "month names as text -> ordered categorical / month number",
    "ArrivalDateYear": "numeric, but really a period marker (2017 is partial)",
}


# helper functions
def grid(n, ncols=4, cell=(4.0, 3.2)):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def save(fig, out, name, show):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    if not show:
        plt.close(fig)


def text_columns(raw):
    return [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]


# Checks missing values
def missing_values(clean, raw, out, show):
    print("\n[01] MISSING VALUES")
    pct = (clean.isna().mean() * 100).sort_values(ascending=False)
    pct = pct[pct > 0]
    print("missing after normalising NULL/NA tokens:")
    print(clean.isna().sum()[clean.isna().sum() > 0].to_string())

    print("\nplaceholder tokens hiding in the raw text columns:")
    found = False
    for c in text_columns(raw):
        s = raw[c].astype(str).str.strip()
        hits = s[s.str.upper().isin(PLACEHOLDERS)].value_counts()
        for token, n in hits.items():
            print(f"  {c:22s} '{token}': {n}")
            found = True
    if not found:
        print("  none")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(pct.index[::-1], pct.values[::-1], color="tab:red")
    for i, v in enumerate(pct.values[::-1]):
        ax.text(float(v), i, f" {v:.2f}%", va="center")
    ax.set_xlabel("% of rows missing")
    ax.set_title("Missing values by column")
    ax.set_xlim(0, float(pct.max()) * 1.2 if len(pct) else 1)
    save(fig, out, "01_missing_values.png", show)


# duplicate records
def duplicates(clean, out, show):
    print("\n[02] DUPLICATE RECORDS")
    dup = clean.duplicated()
    n_dup = int(dup.sum())
    rate_all = float(clean.IsCanceled.mean())
    rate_dedup = float(clean.drop_duplicates().IsCanceled.mean())
    print(f"exact duplicate rows: {n_dup} ({n_dup / len(clean):.1%})")
    print(f"IsCanceled rate: all={rate_all:.3f}  deduplicated={rate_dedup:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(["unique rows", "duplicate rows"], [int((~dup).sum()), n_dup],
                color=["tab:blue", "tab:orange"])
    axes[0].set_title("Duplicate rows")
    axes[1].bar(["all rows", "deduplicated"], [rate_all * 100, rate_dedup * 100],
                color=["tab:gray", "tab:green"])
    axes[1].set_ylabel("% canceled")
    axes[1].set_title("Cancel rate changes after dropping duplicates")
    save(fig, out, "02_duplicates.png", show)


# incorrect data types
def data_types(clean, raw):
    print("\n[03] INCORRECT DATA TYPES")
    rows = [(c, str(raw[c].dtype), str(clean[c].dtype), DTYPE_NOTES[c])
            for c in raw.columns if c in DTYPE_NOTES]
    print(pd.DataFrame(rows, columns=["column", "raw dtype", "after load_h2", "issue"])
          .to_string(index=False))


# inconsistent formats
def formats(raw):
    print("\n[05] INCONSISTENT FORMATS")
    rows = []
    for c in text_columns(raw):
        s = raw[c].astype(str)
        stripped = s.str.strip()
        padded = int((s != stripped).sum())
        case_variants = int(stripped.nunique() - stripped.str.lower().nunique())
        if padded or case_variants:
            rows.append((c, padded, case_variants))
    print("columns with padding / case variants:")
    print(pd.DataFrame(rows, columns=["column", "padded values", "case variants"])
          .to_string(index=False) if rows else "  none")

    d = raw["ReservationStatusDate"].astype(str).str.strip()
    ok = d.str.match(r"^\d{4}-\d{2}-\d{2}$").mean()
    print(f"\nReservationStatusDate rows in YYYY-MM-DD format: {ok:.1%}")


# typos
def typos(clean):
    print("\n[06] SPELLING AND TYPOGRAPHICAL ERRORS")
    bad_months = sorted(set(clean.ArrivalDateMonth.dropna()) - set(MONTHS))
    print("invalid month names:", bad_months or "none")

    country = clean.Country.dropna()
    bad_codes = sorted(set(country[~country.str.match(r"^[A-Z]{3}$")]))
    print("country values not in 3-letter-uppercase format:", bad_codes or "none")
    rare = country.value_counts()
    rare_codes = rare[rare <= 3]
    print(f"{len(rare_codes)} country codes appear <= 3 times (rare categories, "
          f"mostly valid ISO codes): {rare_codes.index[:8].tolist()} ...")
    for code in bad_codes:
        print(f"  '{code}' appears {int((country == code).sum())} times "
              "(rest of the column uses 3-letter ISO codes, e.g. CHN)")

    print("\nnear-duplicate category labels (similarity >= 0.85):")
    found = False
    for c in ["Meal", "MarketSegment", "DistributionChannel", "ReservedRoomType",
              "AssignedRoomType", "DepositType", "CustomerType", "ReservationStatus"]:
        vals = sorted(clean[c].dropna().unique())
        for i, a in enumerate(vals):
            for b in vals[i + 1:]:
                if difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.85:
                    print(f"  {c}: '{a}' ~ '{b}'")
                    found = True
    if not found:
        print("  none")

    print("\nvague labels that mean 'unknown' (not typos, but need a decision):")
    for c in ["Meal", "MarketSegment", "DistributionChannel"]:
        n = int((clean[c] == "Undefined").sum())
        if n:
            print(f"  {c}: 'Undefined' x {n}")


# Generates graphs to show skewness of data
def skewness(clean, out, show):
    print("\n[DISTRIBUTIONS AND SKEWNESS]")
    skew = clean[NUMERIC].skew().sort_values()
    print(skew.round(2).to_string())
    skew.round(3).to_csv(out / "skewness.csv", header=["skew"])

    fig, axes = grid(len(NUMERIC))
    for ax, c in zip(axes, NUMERIC):
        ax.hist(clean[c].dropna(), bins=40, color="tab:blue", edgecolor="white")
        ax.set_title(f"{c}\nskew = {float(skew[c]):.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Distributions of numeric columns", y=1.0)
    save(fig, out, "03_distributions.png", show)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:red" if abs(v) > 1 else "tab:green" for v in skew.values]
    ax.barh(skew.index, skew.values, color=colors)
    ax.axvline(1, ls="--", c="gray")
    ax.axvline(-1, ls="--", c="gray")
    ax.set_xlabel("skewness (red = |skew| > 1, strongly skewed)")
    ax.set_title("Skewness of numeric columns")
    save(fig, out, "04_skewness.png", show)

    # Does a log transform fix the worst continuous columns?
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for row, c in enumerate(["LeadTime", "ADR"]):
        x = clean[c].dropna()
        axes[row, 0].hist(x, bins=40, color="tab:blue", edgecolor="white")
        axes[row, 0].set_title(f"{c}: skew = {float(x.skew()):.2f}")
        lx = np.log1p(x.clip(lower=0))
        axes[row, 1].hist(lx, bins=40, color="tab:green", edgecolor="white")
        axes[row, 1].set_title(f"log1p({c}): skew = {float(lx.skew()):.2f}")
    save(fig, out, "05_log_transform.png", show)


# Used to catch outliers by creating csv file then generating box plots
def outliers(clean, out, show):
    print("\n[04] OUTLIERS (IQR rule: outside Q1 - 1.5*IQR, Q3 + 1.5*IQR)")
    rows = []
    for c in NUMERIC:
        x = clean[c].dropna()
        q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((x < lo) | (x > hi)).sum())
        rows.append((c, round(lo, 2), round(hi, 2), n, round(100 * n / len(x), 2),
                     float(x.max())))
    table = pd.DataFrame(rows, columns=["column", "lower fence", "upper fence",
                                        "n outliers", "% outliers", "max"])
    print(table.to_string(index=False))
    table.to_csv(out / "outliers_iqr.csv", index=False)

    fig, axes = grid(len(NUMERIC))
    for ax, c in zip(axes, NUMERIC):
        ax.boxplot(clean[c].dropna(), vert=False, flierprops={"markersize": 2})
        ax.set_title(c, fontsize=9)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)
    fig.suptitle("Boxplots (points beyond the whiskers are IQR outliers)", y=1.0)
    save(fig, out, "06_boxplots.png", show)

    # ADR has a few extreme values that squash the whole plot; show it twice.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].boxplot(clean.ADR, flierprops={"markersize": 2})
    axes[0].set_title("ADR, full range (max = %.0f)" % float(clean.ADR.max()))
    axes[1].boxplot(clean.ADR[clean.ADR <= 500], flierprops={"markersize": 2})
    axes[1].set_title("ADR, rows with ADR <= 500")
    save(fig, out, "07_adr_boxplot.png", show)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/H2.csv")
    ap.add_argument("--out", default="outputs/eda")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.path, keep_default_na=False, na_values=[], low_memory=False)
    clean = load_dataset(args.path)
    if args.dedup:
        clean = clean.drop_duplicates().reset_index(drop=True)
    print(f"rows analysed: {len(clean)} (dedup={args.dedup})")

    missing_values(clean, raw, out, args.show)
    duplicates(clean, out, args.show)
    data_types(clean, raw)
    outliers(clean, out, args.show)
    formats(raw)
    typos(clean)
    skewness(clean, out, args.show)

    print(f"\nfigures and tables saved to {out.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()