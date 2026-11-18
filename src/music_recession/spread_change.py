"""SUPPLEMENTARY: spread-only Probit versus spread plus month-to-month spread change.

Not part of the original Model A/B experiment. This module is new so that the
files hashed in existing manifests (baseline.py, backtest.py, economic.py,
music_model.py, ...) stay byte-for-byte unchanged.

    delta_spread_t = spread_t - spread_(t-1)

is defined only when the previous *calendar* month is observed. It is never
computed against the previous row when a month is missing. Both models are
refit from scratch at every forecast origin on the identical common sample:
the rows where delta_spread is defined. The six-month target, expanding
window, t-1 training-label rule, 240-month minimum and probabilistic output
are those of config/baseline.json, unchanged.
"""

import csv
import hashlib
import json
import warnings
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.special import ndtr
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning, PerfectSeparationError, PerfectSeparationWarning,
)

from .backtest import _diagnostic_json, _vectors, calendar_months
from .baseline import (
    BaselineConfig, BaselineError, Observation, fit_probit, parse_month,
    select_training, validate_observations,
)
from .economic import add_months, parse_value

START = date(1973, 10, 1)
END = date(2026, 3, 1)
LABEL = "SUPPLEMENTARY - not part of the original experiment"
PREDICTION_COLUMNS = (
    "predictor_month", "forecast_cutoff", "target_month", "status", "error_code",
    "error_message", "spread_t", "spread_prev_month", "delta_spread_t", "actual_label",
    "baseline_probability", "spread_change_probability", "training_count",
    "training_start", "training_end", "training_target_end", "training_recession_count",
    "training_months_sha256", "baseline_fit_json", "spread_change_fit_json",
)


def spread_changes(path: Path) -> Dict[date, Dict[str, Decimal]]:
    """Exact decimal spread changes from the prepared economic CSV.

    Returns {month: {"spread": s_t, "previous": s_(t-1), "delta": s_t - s_(t-1)}}
    only for months whose previous calendar month is present in the file.
    """
    spreads = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream, strict=True):
            month = parse_month(raw["predictor_month"])
            value = parse_value(raw["spread_t"], "spread", str(path))
            if value is None or month in spreads:
                raise BaselineError("invalid_input", f"Missing or duplicate spread at {month}")
            spreads[month] = value
    return changes_from_spreads(spreads)


def changes_from_spreads(spreads: Dict[date, Decimal]) -> Dict[date, Dict[str, Decimal]]:
    """Calendar-aware differences: month t pairs only with add_months(t, -1)."""
    result = {}
    for month, value in sorted(spreads.items()):
        previous = spreads.get(add_months(month, -1))
        if previous is not None:
            result[month] = {"spread": value, "previous": previous, "delta": value - previous}
    return result


def fit_spread_change_probit(training: Sequence[Observation], deltas: Dict[date, float],
                             config: BaselineConfig) -> dict:
    """Unscaled Probit on [1, spread_t, delta_spread_t]; failures raise, no fallback."""
    validate_observations(training, require_labels=True)
    if len(training) < config.min_training_months:
        raise BaselineError("insufficient_training", "Too few eligible training months")
    try:
        x = np.asarray([[r.spread, deltas[r.predictor_month]] for r in training], dtype=float)
    except KeyError as exc:
        raise BaselineError("missing_input", f"No spread change for training month {exc}") from exc
    y = np.asarray([r.target for r in training], dtype=float)
    if not np.isfinite(x).all():
        raise BaselineError("missing_input", "Nonfinite training features")
    if np.unique(y).size != 2:
        raise BaselineError("single_class", "Probit training requires both recession classes")
    if np.any(np.ptp(x, axis=0) <= 0):
        raise BaselineError("constant_predictor", "Constant training feature")
    design = np.column_stack((np.ones(len(x)), x))
    condition = float(np.linalg.cond(design))
    if np.linalg.matrix_rank(design) < 3 or condition > config.max_condition_number:
        raise BaselineError("ill_conditioned", "Rank-deficient or ill-conditioned design")
    # Same bounded signed-margin linear program as music_model: a positive
    # optimum means complete or quasi-complete separation.
    signed = (2 * y - 1)[:, None] * design
    objective = signed.sum(axis=0)
    k = design.shape[1]
    lp = linprog(np.r_[-objective, objective],
                 A_ub=np.vstack([np.column_stack((-signed, signed)), np.ones((1, 2 * k))]),
                 b_ub=np.r_[np.zeros(len(x)), 1.], bounds=(0, None), method="highs")
    if not lp.success:
        raise BaselineError("separation_check_failure", "Could not establish nonseparation")
    if -lp.fun > 1e-7:
        raise BaselineError("separation", "Complete or quasi-complete separation")
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = Probit(y, design, missing="raise", check_rank=True)
            fitted = model.fit(start_params=np.zeros(k), method=config.optimizer,
                               maxiter=config.maxiter, tol=config.tolerance, disp=False,
                               full_output=True, warn_convergence=True)
    except PerfectSeparationError as exc:
        raise BaselineError("separation", str(exc)) from exc
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise BaselineError("optimization_failure", str(exc)) from exc
    diagnostics = dict(
        converged=bool(fitted.mle_retvals.get("converged", False)),
        iterations=int(fitted.mle_retvals.get("iterations", 0)),
        optimizer_warnflag=int(fitted.mle_retvals.get("warnflag", 0)),
        design_condition_number=condition,
        warnings=[{"category": w.category.__name__, "message": str(w.message)} for w in caught])
    if any(issubclass(w.category, PerfectSeparationWarning) for w in caught):
        raise BaselineError("separation", "Optimizer reported separation", diagnostics)
    if (not diagnostics["converged"] or diagnostics["optimizer_warnflag"]
            or any(issubclass(w.category, ConvergenceWarning) for w in caught)):
        raise BaselineError("nonconvergence", "Probit optimizer did not converge", diagnostics)
    if caught:
        raise BaselineError("numerical_failure", "Optimizer warning; no automatic fallback", diagnostics)
    params = np.asarray(fitted.params)
    information = -np.asarray(model.hessian(params))
    score = np.asarray(model.score(params)) / len(x)
    if not all(np.isfinite(v).all() for v in (params, fitted.bse, information, score, fitted.llf)):
        raise BaselineError("numerical_failure", "Nonfinite fit diagnostics", diagnostics)
    info_condition = float(np.linalg.cond(information))
    diagnostics.update(
        log_likelihood=float(fitted.llf), max_abs_average_score=float(np.abs(score).max()),
        information_condition_number=info_condition,
        information_min_eigenvalue=float(np.linalg.eigvalsh(information).min()),
        separation_check="bounded signed-margin linear program")
    if diagnostics["information_min_eigenvalue"] <= 0 or info_condition > config.max_condition_number:
        raise BaselineError("ill_conditioned", "Invalid information matrix", diagnostics)
    if diagnostics["max_abs_average_score"] > config.max_average_score:
        raise BaselineError("nonconvergence", "Likelihood score exceeds threshold", diagnostics)
    names = ("intercept", "spread_t", "delta_spread_t")
    return dict(params=params, report=dict(
        formula="Phi(intercept + b1 * spread_t + b2 * delta_spread_t)",
        coefficients=dict(zip(names, params.tolist())),
        standard_errors=dict(zip(names, np.asarray(fitted.bse).tolist())),
        standard_errors_note="Model-based diagnostics only; no time-series inference claimed",
        standardization="none; both predictors in percentage points",
        diagnostics=diagnostics))


def spread_change_probability(fit: dict, spread: float, delta: float) -> float:
    value = float(ndtr(np.array([1.0, spread, delta]) @ fit["params"]))
    _vectors([0], [value])
    return value


def evaluate(rows: Sequence[Observation], changes: Dict[date, dict], config: BaselineConfig,
             start: date = START, end: date = END) -> list:
    """One record per scheduled calendar month; both models on the common sample."""
    validate_observations(rows)
    deltas = {m: float(v["delta"]) for m, v in changes.items()}
    common = [r for r in rows if r.predictor_month in deltas]
    by_month = {r.predictor_month: r for r in rows}
    records = []
    for month in calendar_months(start, end):
        observation = by_month.get(month)
        change = changes.get(month)
        record = dict.fromkeys(PREDICTION_COLUMNS, "")
        record.update(
            predictor_month=month.isoformat(),
            forecast_cutoff=add_months(month, config.forecast_cutoff_month_offset)
            .replace(day=config.forecast_cutoff_day).isoformat(),
            target_month=add_months(month, config.horizon_months).isoformat(),
            spread_t=observation.spread if observation else "",
            spread_prev_month=float(change["previous"]) if change else "",
            delta_spread_t=float(change["delta"]) if change else "",
            actual_label=observation.target if observation and observation.target is not None else "",
            training_count=0)
        try:
            if observation is None:
                raise BaselineError("missing_forecast", f"No observation for {month}")
            if change is None:
                raise BaselineError("missing_delta", f"Previous calendar month of {month} unobserved")
            training = select_training(common, month, config)
            record.update(
                training_count=len(training),
                training_start=training[0].predictor_month.isoformat(),
                training_end=training[-1].predictor_month.isoformat(),
                training_target_end=training[-1].target_month.isoformat(),
                training_recession_count=sum(r.target for r in training),
                training_months_sha256=hashlib.sha256("\n".join(
                    r.predictor_month.isoformat() for r in training).encode()).hexdigest())
            baseline = fit_probit(training, config)
            record["baseline_probability"] = float(baseline.predict_probability([observation.spread])[0])
            record["baseline_fit_json"] = _diagnostic_json(
                {k: baseline.report[k] for k in ("coefficients", "standard_errors", "diagnostics")})
            extended = fit_spread_change_probit(training, deltas, config)
            record["spread_change_probability"] = spread_change_probability(
                extended, observation.spread, deltas[month])
            record["spread_change_fit_json"] = _diagnostic_json(extended["report"])
            record["status"] = "ok" if observation.target is not None else "missing_actual"
        except BaselineError as exc:
            record.update(status="warmup" if exc.code == "insufficient_training" else
                          exc.code if exc.code in ("missing_delta", "missing_forecast") else "failed",
                          error_code=exc.code, error_message=str(exc))
        records.append(record)
    return records
