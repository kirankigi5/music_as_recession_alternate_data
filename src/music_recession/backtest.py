"""Retrospective expanding-window evaluation of the unchanged Step 3 Probit.

Every scheduled calendar month is retained, including unsuccessful forecasts.
Metrics compare model and expanding-prevalence probabilities on paired months.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.stats import rankdata

from .baseline import (
    BaselineConfig, BaselineError, Observation, fit_probit, load_economic,
    parse_month, select_training, validate_observations,
)
from .economic import add_months


START = date(1973, 10, 1)
END = date(2026, 3, 1)
SCHEMA_VERSION = 1
LOG_EPSILON = float(np.finfo(float).eps)
FORECAST_COLUMNS = (
    "schema_version", "model_id", "status", "error_code", "error_message",
    "predictor_month", "forecast_cutoff", "target_month", "spread_t",
    "predicted_probability", "actual_label", "reference_probability",
    "latest_allowed_training_target_month", "training_sample_count",
    "training_predictor_start", "training_predictor_end", "training_target_start",
    "training_target_end", "training_nonrecession_count", "training_recession_count",
    "training_recession_prevalence", "training_months_sha256", "coefficients_json",
    "intercept", "spread_coefficient", "converged", "iterations",
    "optimizer_warnflag", "log_likelihood", "max_abs_average_score",
    "design_condition_number", "information_min_eigenvalue",
    "information_condition_number", "warnings_json", "diagnostics_json",
)
CALIBRATION_COLUMNS = (
    "bin_lower", "bin_upper", "upper_inclusive", "count",
    "mean_predicted_probability", "observed_recession_frequency",
)


def _diagnostic_json(value) -> str:
    # Numerical failures can themselves contain infinity/NaN diagnostics.
    # Preserve these explicitly as strings instead of losing the failed month
    # to a JSON serialization error or emitting nonstandard JSON numbers.
    def safe(item):
        if isinstance(item, dict):
            return {key: safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(val) for val in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        return item
    return json.dumps(safe(value), sort_keys=True, allow_nan=False)


def _vectors(actual, probability):
    try:
        y, p = np.asarray(actual, dtype=float), np.asarray(probability, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Labels and probabilities must be numeric") from exc
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or y.shape != p.shape:
        raise ValueError("Expected nonempty, equally sized one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Actual labels must be finite binary values")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return y, p


def brier_score(actual, probability) -> float:
    y, p = _vectors(actual, probability)
    return float(np.mean((p - y) ** 2))


def brier_skill_score(model_score: float, reference_score: float) -> Optional[float]:
    if not all(np.isfinite(s) and 0 <= s <= 1 for s in (model_score, reference_score)):
        raise ValueError("Brier Scores must be finite values in [0, 1]")
    return None if reference_score == 0 else float(1 - model_score / reference_score)


def roc_auc(actual, probability) -> Optional[float]:
    """Mann–Whitney rank formulation, giving tied probabilities half credit."""
    y, p = _vectors(actual, probability)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    ranks = rankdata(p, method="average")
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) /
                 (positives * negatives))


def log_loss(actual, probability) -> dict:
    """Natural-log loss; floor only zero likelihoods of the observed outcome."""
    y, p = _vectors(actual, probability)
    likelihood = np.where(y == 1, p, 1 - p)
    zero = likelihood == 0
    safe = np.where(zero, LOG_EPSILON, likelihood)
    return {"value": float(-np.log(safe).mean()), "clipped_observations": int(zero.sum()),
            "epsilon": LOG_EPSILON,
            "policy": "Only zero observed-outcome likelihoods are floored to float64 epsilon; saved probabilities, Brier and AUC are unchanged"}


def calibration_table(actual, probability) -> list:
    """Ten fixed equal-width bins; empty bins remain with null frequencies."""
    y, p = _vectors(actual, probability)
    edges = np.arange(11, dtype=float) / 10
    indices = np.minimum(np.searchsorted(edges, p, side="right") - 1, 9)
    result = []
    for index in range(10):
        selected = indices == index
        count = int(selected.sum())
        result.append({
            "bin_lower": round(float(edges[index]), 1),
            "bin_upper": round(float(edges[index + 1]), 1),
            "upper_inclusive": index == 9, "count": count,
            "mean_predicted_probability": float(p[selected].mean()) if count else None,
            "observed_recession_frequency": float(y[selected].mean()) if count else None,
        })
    return result


def calendar_months(start: date, end: date) -> list:
    if start.day != 1 or end.day != 1 or end < start:
        raise ValueError("Forecast bounds must be ordered first-of-month dates")
    result = []
    month = start
    while month <= end:
        result.append(month)
        month = add_months(month, 1)
    return result


def _training_metadata(training: Sequence[Observation]) -> dict:
    positives = sum(r.target for r in training)
    prevalence = positives / len(training)
    months = "\n".join(r.predictor_month.isoformat() for r in training)
    return {
        "training_sample_count": len(training),
        "training_predictor_start": training[0].predictor_month.isoformat(),
        "training_predictor_end": training[-1].predictor_month.isoformat(),
        "training_target_start": training[0].target_month.isoformat(),
        "training_target_end": training[-1].target_month.isoformat(),
        "training_nonrecession_count": len(training) - positives,
        "training_recession_count": positives,
        "training_recession_prevalence": prevalence,
        "reference_probability": prevalence,
        "training_months_sha256": hashlib.sha256(months.encode("utf-8")).hexdigest(),
    }


def run_backtest(rows: Sequence[Observation], config: BaselineConfig,
                 start: date = START, end: date = END) -> list:
    """Fit from scratch for every scheduled month; retain failures as records.

    Malformed input is a global validation error. Missing forecast dates, missing
    outcomes and fit failures are explicit per-month records. No warm starts,
    optimizer retries, tuning or implicit model changes occur.
    """
    validate_observations(rows)
    by_month = {r.predictor_month: r for r in rows}
    records = []
    for month in calendar_months(start, end):
        observation = by_month.get(month)
        record = dict.fromkeys(FORECAST_COLUMNS)
        record.update(
            schema_version=SCHEMA_VERSION, model_id="A", status="failed",
            predictor_month=month.isoformat(),
            forecast_cutoff=add_months(month, config.forecast_cutoff_month_offset).replace(day=config.forecast_cutoff_day).isoformat(),
            target_month=add_months(month, config.horizon_months).isoformat(),
            latest_allowed_training_target_month=add_months(month, -config.training_target_lag_months).isoformat(),
            spread_t=observation.spread if observation else None,
            actual_label=observation.target if observation else None,
            warnings_json="[]", diagnostics_json="{}",
        )
        try:
            training = select_training(rows, month, config)
            record.update(_training_metadata(training))
            if observation is None:
                raise BaselineError("missing_forecast", f"No observation for predictor month {month}")
            fitted = fit_probit(training, config)
            probability = float(fitted.predict_probability([observation.spread])[0])
            # Reject an invalid returned probability; never clip or silently omit it.
            _vectors([0], [probability])
            fit = fitted.report
            diagnostics = fit["diagnostics"]
            record.update(
                predicted_probability=probability,
                coefficients_json=json.dumps(fit["coefficients"], sort_keys=True, allow_nan=False),
                intercept=fit["coefficients"]["intercept"], spread_coefficient=fit["coefficients"]["spread_t"],
                status="ok" if observation.target is not None else "missing_actual",
                error_code=None if observation.target is not None else "missing_actual",
                error_message=None if observation.target is not None else "Future outcome is unavailable; forecast not evaluated",
            )
            record.update({k: diagnostics[k] for k in (
                "converged", "iterations", "optimizer_warnflag", "log_likelihood",
                "max_abs_average_score", "design_condition_number",
                "information_min_eigenvalue", "information_condition_number",
            )})
            record["warnings_json"] = json.dumps(diagnostics["warnings"], sort_keys=True)
            record["diagnostics_json"] = json.dumps(diagnostics, sort_keys=True, allow_nan=False)
        except BaselineError as exc:
            record.update(status="failed", error_code=exc.code, error_message=str(exc),
                          converged=exc.diagnostics.get("converged"),
                          iterations=exc.diagnostics.get("iterations"),
                          optimizer_warnflag=exc.diagnostics.get("optimizer_warnflag"),
                          warnings_json=_diagnostic_json(exc.diagnostics.get("warnings", [])),
                          diagnostics_json=_diagnostic_json(exc.diagnostics))
        except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as exc:
            record.update(status="failed", error_code="unexpected_numerical_failure", error_message=f"{type(exc).__name__}: {exc}")
        records.append(record)
    return records


def summarize(records: Sequence[dict]) -> dict:
    if not records:
        raise ValueError("No forecast records supplied")
    evaluated = [r for r in records if r["status"] == "ok"]
    not_evaluated = [{"predictor_month": r["predictor_month"], "status": r["status"],
                      "reason": r["error_code"], "message": r["error_message"]}
                     for r in records if r["status"] != "ok"]
    coverage = {
        "scheduled_forecasts": len(records),
        "successful_forecasts": sum(r["status"] in ("ok", "missing_actual") for r in records),
        "failed_forecasts": sum(r["status"] == "failed" for r in records),
        "forecasts_missing_actual": sum(r["status"] == "missing_actual" for r in records),
        "evaluated_forecasts": len(evaluated),
        "full_evaluation_coverage": len(evaluated) == len(records),
        "scheduled_predictor_start": records[0]["predictor_month"],
        "scheduled_predictor_end": records[-1]["predictor_month"],
        "evaluated_predictor_start": evaluated[0]["predictor_month"] if evaluated else None,
        "evaluated_predictor_end": evaluated[-1]["predictor_month"] if evaluated else None,
        "evaluated_target_start": evaluated[0]["target_month"] if evaluated else None,
        "evaluated_target_end": evaluated[-1]["target_month"] if evaluated else None,
        "not_evaluated_months": not_evaluated,
        "forecasts_with_warnings": sum(bool(json.loads(r["warnings_json"])) for r in records),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "retrospective pseudo-out-of-sample evaluation; not historically executable real-time forecasts",
        "coverage": coverage,
        "comparison_policy": "Model and expanding-training-prevalence reference are evaluated on identical status=ok months only; every other month is retained and reported",
        "brier_score": None, "reference_brier_score": None, "brier_skill_score": None,
        "roc_auc": None, "log_loss": None, "calibration": [],
        "limitations": [
            "Final FRED observations and retrospective NBER classifications contain historical hindsight.",
            "The t+1 day-15 cutoff and t-1 training-label boundary do not establish historical publication or announcement availability.",
            "Monthly outcomes and forecast errors are dependent; few independent recession episodes limit inference.",
            "No uncertainty interval or causal interpretation is provided.",
            "Model B must be compared after rerunning both models on compatible training windows and identical evaluation months.",
        ],
    }
    if not evaluated:
        result["metric_note"] = "No paired evaluable forecasts; all metrics are undefined"
        return result
    y = [r["actual_label"] for r in evaluated]
    p = [r["predicted_probability"] for r in evaluated]
    ref = [r["reference_probability"] for r in evaluated]
    model_bs, reference_bs = brier_score(y, p), brier_score(y, ref)
    skill = brier_skill_score(model_bs, reference_bs)
    result.update(
        brier_score=model_bs, reference_brier_score=reference_bs, brier_skill_score=skill,
        brier_skill_interpretation=("undefined: reference Brier Score is zero" if skill is None else
                                    "Model A improves on expanding prevalence" if skill > 0 else
                                    "Model A matches expanding prevalence" if skill == 0 else
                                    "Model A is worse than expanding prevalence"),
        roc_auc=roc_auc(y, p),
        roc_auc_note="Undefined when evaluated labels contain only one class; ties receive half credit",
        log_loss=log_loss(y, p),
        mean_predicted_probability=float(np.mean(p)),
        mean_reference_probability=float(np.mean(ref)),
        observed_recession_frequency=float(np.mean(y)),
        actual_recession_count=int(sum(y)), actual_nonrecession_count=len(y) - int(sum(y)),
        calibration=calibration_table(y, p),
        calibration_policy="Ten fixed bins [0,.1), ... [.9,1]; empty bins have count 0 and null frequencies",
    )
    return result


def _write_csv(path: Path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(records: Sequence[dict], metrics: dict, directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "team5-baseline-v1"})
    subtitle = (f"Retrospective pseudo-out-of-sample | {metrics['coverage']['evaluated_forecasts']} / "
                f"{len(records)} months evaluated")

    def save(fig, stem):
        fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
        fig.savefig(directory / f"{stem}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)

    dates = [date.fromisoformat(r["target_month"]) for r in records]
    predicted = [r["predicted_probability"] if r["status"] in ("ok", "missing_actual") else np.nan for r in records]
    reference = [r["reference_probability"] if r["reference_probability"] is not None else np.nan for r in records]
    fig, ax = plt.subplots(figsize=(14, 4.8), layout="constrained")
    shaded = False
    for r, target in zip(records, dates):
        if r["actual_label"] == 1:
            ax.axvspan(target, add_months(target, 1), color="#d9dde3", alpha=.85,
                       linewidth=0, label="Actual recession target months" if not shaded else None)
            shaded = True
    ax.plot(dates, predicted, color="#195caa", lw=1.6, label="Model A: Treasury-spread Probit")
    ax.plot(dates, reference, color="#af6826", lw=1.2, ls="--", label="Eligible-training recession prevalence")
    failed = [d for r, d in zip(records, dates) if r["status"] != "ok"]
    if failed:
        ax.scatter(failed, [0] * len(failed), marker="x", color="#b22222", label="Not evaluated", zorder=5)
    ax.set(title="Six-month-ahead recession probabilities\n" + subtitle,
           xlabel="Target month (predictor month + 6); assumed issuance on day 15 of predictor month + 1",
           ylabel="Recession probability", ylim=(0, 1), xlim=(dates[0], add_months(dates[-1], 1)))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "probability_timeline")

    fig, (ax, counts) = plt.subplots(2, 1, figsize=(7.4, 8), height_ratios=[3, 1],
                                    sharex=True, layout="constrained")
    ax.plot([0, 1], [0, 1], "--", color="#777777", label="Perfect calibration")
    populated = [b for b in metrics["calibration"] if b["count"]]
    ax.plot([b["mean_predicted_probability"] for b in populated],
            [b["observed_recession_frequency"] for b in populated],
            "o-", color="#195caa", label="Model A (fixed 10% bins)")
    for b in populated:
        ax.annotate(f"n={b['count']}", (b["mean_predicted_probability"], b["observed_recession_frequency"]),
                    xytext=(5, 8), textcoords="offset points", fontsize=8)
    if not populated:
        ax.text(.5, .5, "No evaluable forecasts", ha="center")
    ax.set(title="Calibration / reliability\n" + subtitle, ylabel="Observed recession frequency", ylim=(-.03, 1.08))
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    bins = metrics["calibration"]
    counts.bar([(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins],
               [b["count"] for b in bins], width=.085, color="#195caa")
    counts.set(xlabel="Mean predicted probability within each bin (top); probability bins (bottom)",
               ylabel="Months", xlim=(0, 1))
    counts.xaxis.set_major_formatter(PercentFormatter(1))
    counts.set_xticks(np.linspace(0, 1, 11))
    save(fig, "calibration")

    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    values = [metrics["brier_score"], metrics["reference_brier_score"]]
    if all(v is not None for v in values):
        bars = ax.bar(["Model A\nTreasury-spread Probit", "Reference\nExpanding training prevalence"],
                      values, color=["#195caa", "#af6826"], width=.6)
        ax.bar_label(bars, labels=[f"{v:.6f}" for v in values], padding=5)
        ax.set_ylim(0, max(max(values) * 1.25, .01))
    else:
        ax.text(.5, .5, "No paired evaluable forecasts", ha="center", transform=ax.transAxes)
    ax.set(title="Brier Score comparison (lower is better)\n" + subtitle, ylabel="Mean squared probability error")
    ax.grid(axis="y", alpha=.2)
    save(fig, "brier_comparison")


def write_reports(records, metrics, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory / "forecasts.csv", FORECAST_COLUMNS, records)
    _write_csv(directory / "calibration.csv", CALIBRATION_COLUMNS, metrics["calibration"])
    failures = [r for r in records if r["status"] != "ok"]
    (directory / "failures.json").write_text(json.dumps(failures, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_plots(records, metrics, directory)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/economic_monthly.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/baseline.json"))
    parser.add_argument("--start", default=START.isoformat())
    parser.add_argument("--end", default=END.isoformat())
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baseline_backtest"))
    args = parser.parse_args(argv)
    try:
        config = BaselineConfig.load(args.config)
        rows = load_economic(args.data)
        start, end = parse_month(args.start), parse_month(args.end)
        records = run_backtest(rows, config, start, end)
        metrics = summarize(records)
        metrics["config"] = asdict(config)
        # This inclusive calendar arithmetic independently checks loop length.
        expected = (end.year - start.year) * 12 + end.month - start.month + 1
        available = sum(start <= r.predictor_month <= end for r in rows)
        metrics["calendar_audit"] = {
            "expected_months_by_calendar_arithmetic": expected,
            "scheduled_records": len(records), "available_input_months": available,
            "record_count_discrepancy": len(records) - expected,
            "input_month_discrepancy": available - expected,
            "project_expected_months": 630,
            "uses_project_evaluation_bounds": (start, end) == (START, END),
        }
        metrics["provenance"] = {
            "inputs": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for key, path in (("economic_csv", args.data), ("baseline_config", args.config))},
            "implementation_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("baseline.py", "backtest.py", "economic.py")
            },
            "software": {"python": platform.python_version(), **{
                p: importlib.metadata.version(p) for p in ("numpy", "scipy", "statsmodels", "pandas", "matplotlib")}},
        }
        write_reports(records, metrics, args.output_dir)
    except (BaselineError, ValueError, OSError) as exc:
        parser.exit(1, f"Backtest aborted: {exc}\n")
    print(json.dumps({"coverage": metrics["coverage"], "brier_score": metrics["brier_score"],
                      "reference_brier_score": metrics["reference_brier_score"],
                      "brier_skill_score": metrics["brier_skill_score"], "output_dir": str(args.output_dir)}, indent=2))
    # Artifacts still include every month when coverage is incomplete.
    return 0 if metrics["coverage"]["full_evaluation_coverage"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
