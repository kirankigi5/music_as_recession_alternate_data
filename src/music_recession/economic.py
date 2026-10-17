"""Validate local FRED snapshots and prepare retrospective six-month targets.

No network access, imputation, model fitting, or real-time availability claims.
Missing monthly dates are fatal; explicitly missing values are audited and
excluded where needed. Decimal arithmetic preserves source yield precision.
"""

import argparse
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


FRED_COLUMNS = ("observation_date", "USREC", "GS10", "TB3MS")
GS10_COLUMNS = ("observation_date", "GS10")
OUTPUT_COLUMNS = (
    "predictor_month", "target_month", "USREC_t", "GS10_t", "TB3MS_t",
    "spread_t", "target_t",
)
HORIZON_MONTHS = 6
MISSING_MARKERS = frozenset(("", ".", "NA", "N/A"))


class ValidationError(ValueError):
    """An input violates the monthly economic data contract."""


@dataclass
class MonthlySource:
    path: Path
    sha256: str
    columns: Tuple[str, ...]
    observations: Dict[date, Dict[str, Optional[Decimal]]]


def add_months(month: date, months: int) -> date:
    """Advance a first-of-month date by calendar months, including year rolls."""
    if month.day != 1:
        raise ValueError("Expected a first-of-month date")
    ordinal = month.year * 12 + month.month - 1 + months
    return date(ordinal // 12, ordinal % 12 + 1, 1)


def parse_value(raw: str, column: str, location: str) -> Optional[Decimal]:
    value = raw.strip()
    if value in MISSING_MARKERS:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValidationError(f"{location}: invalid {column} value {raw!r}") from exc
    if not number.is_finite():
        raise ValidationError(f"{location}: non-finite {column} value {raw!r}")
    if column == "USREC" and number not in (Decimal(0), Decimal(1)):
        raise ValidationError(f"{location}: USREC must be 0 or 1, got {raw!r}")
    return number


def load_monthly(path: Path, columns: Sequence[str]) -> MonthlySource:
    """Read, hash and strictly validate one complete monthly calendar.

    All columns are required, in the documented order. Duplicate dates,
    unsorted dates, calendar gaps, malformed records and invalid values fail
    before any output is written. Leading or internal missing cells are
    retained as None for the subsequent exclusion audit.
    """
    path = Path(path)
    content = path.read_bytes()
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{path}: expected UTF-8 CSV") from exc
    observations = {}
    try:
        reader = csv.DictReader(io.StringIO(decoded, newline=""), strict=True)
        if reader.fieldnames != list(columns):
            raise ValidationError(
                f"{path}: expected columns {list(columns)}, got {reader.fieldnames}"
            )
        for row in reader:
            location = f"{path}:{reader.line_num}"
            if None in row or any(value is None for value in row.values()):
                raise ValidationError(f"{location}: malformed CSV row width")
            raw_month = row["observation_date"]
            try:
                month = date.fromisoformat(raw_month)
            except ValueError as exc:
                raise ValidationError(f"{location}: invalid date {raw_month!r}") from exc
            if raw_month != month.isoformat() or month.day != 1:
                raise ValidationError(f"{location}: expected YYYY-MM-01 monthly date")
            if month in observations:
                raise ValidationError(f"{location}: duplicate date {month}")
            observations[month] = {
                column: parse_value(row[column], column, location)
                for column in columns[1:]
            }
    except csv.Error as exc:
        raise ValidationError(f"{path}: malformed CSV: {exc}") from exc
    if not observations:
        raise ValidationError(f"{path}: no observations")
    dates = list(observations)
    if dates != sorted(dates):
        raise ValidationError(f"{path}: dates must be in ascending order")
    missing_dates = []
    month = dates[0]
    while month < dates[-1]:
        if month not in observations:
            missing_dates.append(month.isoformat())
        month = add_months(month, 1)
    if missing_dates:
        raise ValidationError(f"{path}: missing monthly dates: {', '.join(missing_dates)}")
    return MonthlySource(
        path, hashlib.sha256(content).hexdigest(), tuple(columns), observations
    )


def coverage(months: Sequence[date]) -> dict:
    return {
        "start": min(months).isoformat() if months else None,
        "end": max(months).isoformat() if months else None,
        "count": len(months),
    }


def describe_source(source: MonthlySource) -> dict:
    dates = list(source.observations)
    series = {}
    for column in source.columns[1:]:
        available = [d for d in dates if source.observations[d][column] is not None]
        internal_missing = [
            d.isoformat() for d in dates
            if available and available[0] <= d <= available[-1]
            and source.observations[d][column] is None
        ]
        series[column] = {
            "non_missing_coverage": coverage(available),
            "missing_values": len(dates) - len(available),
            "internal_missing_months": internal_missing,
        }
    return {
        "path": source.path.as_posix(),
        "sha256": source.sha256,
        "schema": {
            column: "YYYY-MM-01" if column == "observation_date" else
            "nullable binary 0/1" if column == "USREC" else
            "nullable finite decimal; percent"
            for column in source.columns
        },
        "monthly_coverage": coverage(dates),
        "series": series,
        "validation": {
            "status": "passed", "ascending_dates": True,
            "duplicate_dates": 0, "missing_monthly_dates": [],
            "invalid_values": 0,
        },
        "provenance": {
            "description": "Existing local FRED-format snapshot; not independently authenticated",
            "series_reference_urls": {
                c: f"https://fred.stlouisfed.org/series/{c}" for c in source.columns[1:]
            },
            "retrieved_at": None,
            "vintage_date": None,
            "historical_release_times": None,
        },
    }


def cross_check_gs10(primary: MonthlySource, check: MonthlySource) -> dict:
    """Require identical available GS10 months and exact decimal values."""
    def available(source: MonthlySource) -> dict:
        return {
            month: row["GS10"] for month, row in source.observations.items()
            if row["GS10"] is not None
        }

    left, right = available(primary), available(check)
    if not left or not right:
        raise ValidationError("GS10 cross-check: no non-missing values to compare")
    if left.keys() != right.keys():
        only_primary = sorted(d.isoformat() for d in left.keys() - right.keys())
        only_check = sorted(d.isoformat() for d in right.keys() - left.keys())
        raise ValidationError(
            f"GS10 cross-check coverage mismatch: only primary={only_primary}; "
            f"only check={only_check}"
        )
    mismatches = [d.isoformat() for d in left if left[d] != right[d]]
    if mismatches:
        raise ValidationError(f"GS10 cross-check value mismatch: {', '.join(mismatches)}")
    return {
        "status": "passed", "matched_months": len(left), "mismatches": 0,
        "coverage": coverage(list(left)), "comparison": "exact decimal equality",
        "limitation": "Agreement between local files does not establish independent provenance or vintage accuracy",
    }


def build_monthly(source: MonthlySource) -> Tuple[List[dict], dict]:
    """Join each predictor month to USREC exactly six calendar months later."""
    rows = []
    excluded_predictors = []
    excluded_target_month = []
    excluded_target_value = []
    for month, values in source.observations.items():
        target_month = add_months(month, HORIZON_MONTHS)
        if values["GS10"] is None or values["TB3MS"] is None:
            excluded_predictors.append(month)
            continue
        future = source.observations.get(target_month)
        if future is None:
            excluded_target_month.append(month)
            continue
        if future["USREC"] is None:
            excluded_target_value.append(month)
            continue
        rows.append({
            "predictor_month": month.isoformat(),
            "target_month": target_month.isoformat(),
            "USREC_t": int(values["USREC"]) if values["USREC"] is not None else "",
            "GS10_t": format(values["GS10"], "f"),
            "TB3MS_t": format(values["TB3MS"], "f"),
            "spread_t": format(values["GS10"] - values["TB3MS"], "f"),
            "target_t": int(future["USREC"]),
        })
    audit = {
        "input_rows": len(source.observations),
        "retained_rows": len(rows),
        "excluded_rows": len(source.observations) - len(rows),
        "reason_priority": ["missing_predictors", "absent_target_month", "missing_target_value"],
        "exclusions_are_mutually_exclusive": True,
        "exclusions": {
            "missing_predictors": coverage(excluded_predictors),
            "absent_target_month": coverage(excluded_target_month),
            "missing_target_value": coverage(excluded_target_value),
        },
        "imputed_values": 0,
    }
    return rows, audit


def prepare(fred: Path, gs10_check: Path, output: Path, manifest: Path) -> dict:
    """Validate all inputs, then write deterministic CSV and JSON artifacts."""
    fred, gs10_check, output, manifest = map(Path, (fred, gs10_check, output, manifest))
    # Prevent CLI path mistakes from overwriting either input or mixing formats.
    if output.resolve() == manifest.resolve():
        raise ValidationError("Output CSV and manifest must be different paths")
    protected = {fred.resolve(), gs10_check.resolve()}
    for destination, suffix in ((output, ".csv"), (manifest, ".json")):
        if destination.resolve() in protected:
            raise ValidationError("Output paths must not overwrite source files")
        if destination.suffix.lower() != suffix:
            raise ValidationError(f"Expected {suffix} output path, got {destination}")
    primary = load_monthly(fred, FRED_COLUMNS)
    check = load_monthly(gs10_check, GS10_COLUMNS)
    comparison = cross_check_gs10(primary, check)
    rows, exclusions = build_monthly(primary)
    if not rows:
        raise ValidationError("No usable predictor/target rows after validation")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode("utf-8")
    report = {
        "manifest_version": 1,
        "dataset_status": "retrospective; historical real-time availability not established",
        "horizon_calendar_months": HORIZON_MONTHS,
        "reproduce_from_repository_root": "python -m music_recession.economic",
        "sources": [describe_source(primary), describe_source(check)],
        "cross_check_gs10": comparison,
        "preparation": exclusions,
        "output": {
            "path": output.as_posix(),
            "sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "rows": len(rows), "columns": list(OUTPUT_COLUMNS),
            "schema": {
                "predictor_month": "YYYY-MM-01; observation month, not release timestamp",
                "target_month": "YYYY-MM-01; predictor_month + 6 calendar months",
                "USREC_t": "nullable binary 0/1; reference only, not a predictor",
                "GS10_t": "finite decimal; percent",
                "TB3MS_t": "finite decimal; percent",
                "spread_t": "finite decimal; percentage points; GS10_t - TB3MS_t",
                "target_t": "binary 0/1; source USREC at target_month",
            },
            "predictor_coverage": coverage([date.fromisoformat(r["predictor_month"]) for r in rows]),
            "target_coverage": coverage([date.fromisoformat(r["target_month"]) for r in rows]),
            "target_counts": {str(k): sum(r["target_t"] == k for r in rows) for k in (0, 1)},
        },
        "limitations": [
            "Source retrieval dates, publication times and historical vintages are unknown.",
            "USREC reflects retrospective NBER dating; target month is not label availability date.",
            "Source authenticity is not established by local integrity and agreement checks.",
            "No modeling or backtesting is implemented; no study-period restriction applied.",
        ],
    }
    json_bytes = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(csv_bytes)
    manifest.write_bytes(json_bytes)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fred", type=Path, default=Path("fredgraph.csv"))
    parser.add_argument("--gs10-check", type=Path, default=Path("GS10.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/economic_monthly.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/data_manifest.json"))
    args = parser.parse_args(argv)
    try:
        report = prepare(args.fred, args.gs10_check, args.output, args.manifest)
    except (ValidationError, OSError) as exc:
        parser.exit(1, f"Economic preparation failed: {exc}\n")
    result = report["output"]
    print(
        f"Prepared {result['rows']} rows x {len(result['columns'])} columns: "
        f"{result['predictor_coverage']['start']} through {result['predictor_coverage']['end']}\n"
        f"GS10 cross-check: {report['cross_check_gs10']['matched_months']} matching months\n"
        f"CSV: {args.output}\nManifest: {args.manifest}\n"
        "Retrospective dataset; historical real-time availability is not established."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
