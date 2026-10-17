"""Model A: an intercept and yield spread in a retrospective statsmodels Probit.

Reusable fitting and single-forecast functions only; no backtest or scoring.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning, HessianInversionWarning,
    PerfectSeparationError, PerfectSeparationWarning,
)

from .economic import OUTPUT_COLUMNS, ValidationError, add_months, parse_value


class BaselineError(ValueError):
    """Explicit validation/fit failure; never replaced with another estimator."""

    def __init__(self, code: str, message: str, diagnostics: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics or {}


def parse_month(value: str) -> date:
    try:
        month = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise BaselineError("invalid_date", f"Invalid monthly date: {value!r}") from exc
    if month.day != 1 or month.isoformat() != value:
        raise BaselineError("invalid_date", f"Expected YYYY-MM-01, got {value!r}")
    return month


@dataclass(frozen=True)
class BaselineConfig:
    model: str = "probit"
    horizon_months: int = 6
    training_start: str = "1953-04-01"
    min_training_months: int = 240
    forecast_cutoff_month_offset: int = 1
    forecast_cutoff_day: int = 15
    training_target_lag_months: int = 1
    standardize: bool = False
    optimizer: str = "newton"
    maxiter: int = 100
    tolerance: float = 1e-10
    max_average_score: float = 1e-6
    max_condition_number: float = 1e12

    def __post_init__(self):
        parse_month(self.training_start)
        if self.model != "probit" or self.optimizer != "newton":
            raise BaselineError("invalid_config", "Only Probit with the Newton optimizer is implemented")
        if type(self.horizon_months) is not int or self.horizon_months != 6:
            raise BaselineError("invalid_config", "This study requires a six-calendar-month horizon")
        if self.standardize is not False:
            raise BaselineError("invalid_config", "Spread standardization is not implemented; use false")
        for name, minimum, maximum in (
            ("min_training_months", 3, None), ("maxiter", 1, None),
            ("forecast_cutoff_month_offset", 1, 1),
            ("forecast_cutoff_day", 1, 28), ("training_target_lag_months", 1, None),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
                raise BaselineError("invalid_config", f"Invalid configuration value for {name}: {value!r}")
        for name in ("tolerance", "max_average_score", "max_condition_number"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise BaselineError("invalid_config", f"{name} must be finite and positive")

    @classmethod
    def load(cls, path: Path):
        try:
            values = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise TypeError("configuration must be a JSON object")
            return cls(**values)
        except (TypeError, json.JSONDecodeError) as exc:
            raise BaselineError("invalid_config", str(exc)) from exc


@dataclass(frozen=True)
class Observation:
    predictor_month: date
    target_month: date
    spread: float
    target: Optional[int]


def validate_observations(rows: Sequence[Observation], require_labels: bool = False) -> None:
    if not rows:
        raise BaselineError("missing_input", "No economic observations supplied")
    previous = None
    for row in rows:
        if not isinstance(row.predictor_month, date) or not isinstance(row.target_month, date):
            raise BaselineError("invalid_date", "Observation months must be date objects")
        if row.predictor_month.day != 1 or row.target_month != add_months(row.predictor_month, 6):
            raise BaselineError("misaligned_target", "Each target month must equal predictor month + 6 months")
        if previous is not None and row.predictor_month <= previous:
            raise BaselineError("invalid_dates", "Predictor months must be unique and ascending")
        previous = row.predictor_month
        try:
            finite = np.isfinite(float(row.spread))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            raise BaselineError("missing_input", f"Missing/invalid spread at {row.predictor_month}")
        if row.target is None:
            if require_labels:
                raise BaselineError("missing_input", f"Missing training target at {row.predictor_month}")
        elif row.target not in (0, 1):
            raise BaselineError("invalid_target", "Training targets must be 0 or 1")


def load_economic(path: Path) -> list:
    """Read Step 2 output; validate its dates, finite values and spread identity.

    Missing future targets are permitted for prediction, but never for fitting.
    Missing calendar months are not filled: training counts actual eligible rows.
    """
    rows = []
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames != list(OUTPUT_COLUMNS):
                raise BaselineError("invalid_schema", "Expected the Step 2 economic CSV schema")
            for raw in reader:
                location = f"{path}:{reader.line_num}"
                if None in raw or any(v is None for v in raw.values()):
                    raise BaselineError("invalid_schema", f"{location}: malformed CSV row")
                long_yield = parse_value(raw["GS10_t"], "GS10", location)
                short_yield = parse_value(raw["TB3MS_t"], "TB3MS", location)
                spread = parse_value(raw["spread_t"], "spread", location)
                target = parse_value(raw["target_t"], "USREC", location)
                if any(v is None for v in (long_yield, short_yield, spread)):
                    raise BaselineError("missing_input", f"{location}: missing predictor")
                if spread != long_yield - short_yield:
                    raise BaselineError("invalid_spread", f"{location}: spread differs from GS10 - TB3MS")
                rows.append(Observation(
                    parse_month(raw["predictor_month"]), parse_month(raw["target_month"]),
                    float(spread), int(target) if target is not None else None,
                ))
    except (csv.Error, UnicodeError, ValidationError) as exc:
        raise BaselineError("invalid_input", str(exc)) from exc
    validate_observations(rows)
    return rows


def select_training(rows: Sequence[Observation], predictor_month: date,
                    config: BaselineConfig) -> list:
    """Select actual rows with i >= start and i+6 <= t-label_lag (default t-1)."""
    validate_observations(rows)
    cutoff = add_months(predictor_month, -config.training_target_lag_months)
    start = parse_month(config.training_start)
    training = [r for r in rows if r.predictor_month >= start and r.target_month <= cutoff]
    if len(training) < config.min_training_months:
        raise BaselineError(
            "insufficient_training",
            f"Need {config.min_training_months} eligible months; found {len(training)}",
            {"eligible_months": len(training), "latest_allowed_target_month": cutoff.isoformat()},
        )
    validate_observations(training, require_labels=True)
    return training


def first_eligible_forecast_month(rows: Sequence[Observation], config: BaselineConfig) -> date:
    """Find the first sufficiently populated window without fitting any model."""
    validate_observations(rows)
    candidates = [r for r in rows if r.predictor_month >= parse_month(config.training_start)]
    if len(candidates) < config.min_training_months:
        raise BaselineError("insufficient_training", "Not enough history for any forecast")
    threshold = add_months(candidates[config.min_training_months - 1].target_month,
                           config.training_target_lag_months)
    for row in rows:
        if row.predictor_month >= threshold:
            select_training(rows, row.predictor_month, config)
            return row.predictor_month
    raise BaselineError("insufficient_training", "No predictor month follows the initial training window")


def _finite_vector(values, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BaselineError("missing_input", f"{name} must contain finite numeric values") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise BaselineError("missing_input", f"{name} must be a nonempty finite one-dimensional vector")
    return vector


@dataclass
class ProbitFit:
    """A validated converged fit, its metadata, and spread-only prediction API."""
    report: dict
    _result: object = field(repr=False)

    def predict_probability(self, spread: Sequence[float]) -> np.ndarray:
        values = _finite_vector(spread, "Prediction spread")
        design = np.column_stack((np.ones(len(values)), values))
        with np.errstate(over="raise", invalid="raise"):
            try:
                probabilities = np.asarray(self._result.predict(design), dtype=float)
            except (ValueError, FloatingPointError) as exc:
                raise BaselineError("prediction_failure", str(exc)) from exc
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise BaselineError("prediction_failure", "Invalid Probit probabilities")
        return probabilities


def fit_probit(training: Sequence[Observation], config: BaselineConfig) -> ProbitFit:
    """Fit a supplied training sample. Use select_training for forecast eligibility.

    There is no implicit row dropping, scaling, penalization or fallback model.
    Failure codes and any optimizer diagnostics are attached to BaselineError.
    """
    validate_observations(training, require_labels=True)
    if len(training) < config.min_training_months:
        raise BaselineError("insufficient_training", f"Need at least {config.min_training_months} rows")
    x = _finite_vector([r.spread for r in training], "Training spread")
    y = _finite_vector([r.target for r in training], "Training target")
    if np.unique(y).size != 2:
        raise BaselineError("single_class", "Probit training requires both recession classes")
    if np.unique(x).size < 2:
        raise BaselineError("constant_predictor", "Spread is constant in the training sample")
    # With one continuous predictor and intercept, disjoint class ranges imply
    # complete separation; touching ranges imply quasi-complete separation.
    x0, x1 = x[y == 0], x[y == 1]
    if x0.max() <= x1.min() or x1.max() <= x0.min():
        raise BaselineError("separation", "Complete or quasi-complete separation in training spread")
    design = np.column_stack((np.ones(len(x)), x))
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition) or condition > config.max_condition_number:
        raise BaselineError("ill_conditioned", "Training design is numerically ill-conditioned")
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = Probit(y, design, missing="raise", check_rank=True)
            result = model.fit(start_params=np.zeros(2), method=config.optimizer,
                               maxiter=config.maxiter, tol=config.tolerance,
                               disp=False, full_output=True, warn_convergence=True)
    except PerfectSeparationError as exc:
        raise BaselineError("separation", str(exc)) from exc
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise BaselineError("optimization_failure", str(exc), {
            "exception_type": type(exc).__name__,
            "warnings": [str(w.message) for w in caught],
        }) from exc
    diagnostics = {
        "converged": bool(result.mle_retvals.get("converged", False)),
        "iterations": int(result.mle_retvals.get("iterations", 0)),
        "optimizer_warnflag": int(result.mle_retvals.get("warnflag", 0)),
        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
        "design_condition_number": condition,
    }
    if any(issubclass(w.category, PerfectSeparationWarning) for w in caught):
        raise BaselineError("separation", "statsmodels reported separation", diagnostics)
    if not diagnostics["converged"] or diagnostics["optimizer_warnflag"] != 0 or any(
        issubclass(w.category, ConvergenceWarning) for w in caught
    ):
        raise BaselineError("nonconvergence", "Probit optimizer did not converge", diagnostics)
    if any(issubclass(w.category, (RuntimeWarning, HessianInversionWarning)) for w in caught):
        raise BaselineError("numerical_failure", "Probit reported a numerical warning", diagnostics)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            params = np.asarray(result.params)
            standard_errors = np.asarray(result.bse)
            information = -np.asarray(model.hessian(params))
            score = np.asarray(model.score(params)) / len(training)
            loglike = float(result.llf)
            if not all(np.isfinite(a).all() for a in (params, standard_errors, information, score, loglike)):
                raise ValueError("Non-finite fitted parameters or diagnostics")
            eigenvalues = np.linalg.eigvalsh(information)
            information_condition = float(np.linalg.cond(information))
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise BaselineError("numerical_failure", str(exc), diagnostics) from exc
    diagnostics.update({
        "log_likelihood": loglike,
        "max_abs_average_score": float(np.max(np.abs(score))),
        "information_min_eigenvalue": float(eigenvalues.min()),
        "information_condition_number": information_condition,
        "separation_check": "class ranges overlap; no separation warning",
    })
    if eigenvalues.min() <= 0 or not np.isfinite(information_condition) or information_condition > config.max_condition_number:
        raise BaselineError("ill_conditioned", "Fitted information matrix is not well-conditioned positive definite", diagnostics)
    if diagnostics["max_abs_average_score"] > config.max_average_score:
        raise BaselineError("nonconvergence", "Likelihood score exceeds configured convergence threshold", diagnostics)
    report = {
        "model": "statsmodels.discrete.discrete_model.Probit",
        "formula": "Phi(intercept + spread_t_coefficient * spread_t)",
        "coefficients": {"intercept": float(params[0]), "spread_t": float(params[1])},
        "standard_errors": {"intercept": float(standard_errors[0]), "spread_t": float(standard_errors[1])},
        "standard_errors_note": "Model-based diagnostics only; no time-series inference claimed",
        "standardization": "none; spread is in percentage points",
        "training": {
            "sample_count": len(training),
            "predictor_start": training[0].predictor_month.isoformat(),
            "predictor_end": training[-1].predictor_month.isoformat(),
            "target_start": training[0].target_month.isoformat(),
            "target_end": training[-1].target_month.isoformat(),
            "class_counts": {str(k): int(np.sum(y == k)) for k in (0, 1)},
            "recession_share": float(y.mean()),
        },
        "diagnostics": diagnostics,
        "config": asdict(config),
    }
    return ProbitFit(report, result)


def forecast_one(rows: Sequence[Observation], predictor_month: date,
                 config: BaselineConfig) -> dict:
    """Fit one eligible window and predict using only the forecast month's spread."""
    training = select_training(rows, predictor_month, config)
    forecast_row = next((r for r in rows if r.predictor_month == predictor_month), None)
    if forecast_row is None:
        raise BaselineError("missing_forecast", f"No spread observation for {predictor_month}")
    fitted = fit_probit(training, config)
    probability = float(fitted.predict_probability([forecast_row.spread])[0])
    cutoff = add_months(predictor_month, config.forecast_cutoff_month_offset).replace(day=config.forecast_cutoff_day)
    return {
        "status": "retrospective single-forecast smoke test; not a forecasting performance result",
        "predictor_month": predictor_month.isoformat(),
        "forecast_cutoff": cutoff.isoformat(),
        "target_month": add_months(predictor_month, config.horizon_months).isoformat(),
        "latest_allowed_training_target_month": add_months(predictor_month, -config.training_target_lag_months).isoformat(),
        "spread_t": forecast_row.spread,
        "recession_probability": probability,
        "fit": fitted.report,
        "limitations": [
            "Final FRED observations and NBER classifications contain historical hindsight.",
            "The cutoff convention does not establish historical publication or label availability.",
            "No full backtest or aggregate forecast score has been calculated.",
            "Later music comparisons must rerun both models on compatible training windows and identical evaluation months.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/economic_monthly.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/baseline.json"))
    parser.add_argument("--predictor-month", default="1973-10-01")
    parser.add_argument("--output", type=Path, default=Path("reports/baseline_smoke.json"))
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {args.data.resolve(), args.config.resolve()} or args.output.suffix != ".json":
            raise BaselineError("invalid_output", "Use a .json output separate from data and configuration")
        config = BaselineConfig.load(args.config)
        rows = load_economic(args.data)
        report = forecast_one(rows, parse_month(args.predictor_month), config)
        report["first_eligible_forecast_month"] = first_eligible_forecast_month(rows, config).isoformat()
        report["inputs"] = {
            "economic_csv": {"path": str(args.data), "sha256": hashlib.sha256(args.data.read_bytes()).hexdigest()},
            "config": {"path": str(args.config), "sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()},
        }
        report["software"] = {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "statsmodels", "pandas")
        }}
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    except (BaselineError, OSError) as exc:
        failure = {"status": "failed", "error": str(exc)}
        if isinstance(exc, BaselineError):
            failure.update(code=exc.code, diagnostics=exc.diagnostics)
        parser.exit(1, json.dumps(failure, indent=2) + "\n")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
