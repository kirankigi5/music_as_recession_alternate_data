"""Uncertainty and extra metrics for paired Model A/B forecasts on a gappy calendar.

Forecasts are six months ahead and issued monthly, so neighbouring loss
differentials share most of their training rows and overlap in target windows.
Feature eligibility also leaves gaps of up to several years between forecast
months. The HAC variance and the block bootstrap below therefore measure
distance in calendar months, never in row order: two forecasts a decade apart
are not treated as neighbours just because nothing was issued in between.

Works from saved predictions only. No fitting, emotion inference or network.
This lives in its own module so the hashed files recorded in existing result
manifests (backtest.py, music_model.py, emotions.py, ...) stay unchanged.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
from scipy.stats import t as student_t

from .backtest import _vectors, brier_score, brier_skill_score, roc_auc


HORIZON = 6
SEED = 20261026
REPLICATIONS = 10_000
BLOCK_LENGTHS = (6, 12, 24)
HAC_LAGS = (5, 12, 24)
ACF_LAGS = (1, 3, 6, 12, 24)


def month_positions(months) -> np.ndarray:
    """Absolute month numbers for strictly increasing first-of-month dates."""
    values = []
    for month in months:
        month = month if isinstance(month, date) else date.fromisoformat(month)
        if month.day != 1:
            raise ValueError("Months must be first-of-month dates")
        values.append(month.year * 12 + month.month - 1)
    positions = np.asarray(values, dtype=int)
    if len(positions) == 0 or np.any(np.diff(positions) <= 0):
        raise ValueError("Months must be nonempty and strictly increasing")
    return positions


def _series(values, months):
    d = np.asarray(values, dtype=float)
    positions = month_positions(months)
    if d.ndim != 1 or d.shape != positions.shape or not np.isfinite(d).all():
        raise ValueError("Expected one finite value per month")
    return d, positions


def _centred_calendar(d, positions):
    # Unobserved months are zero after centring, so they add nothing to any
    # cross-product and the Bartlett estimate below cannot go negative.
    z = np.zeros(positions[-1] - positions[0] + 1)
    z[positions - positions[0]] = d - d.mean()
    return z


def average_precision(actual, probability) -> Optional[float]:
    """Step-wise area under the precision-recall curve; tied scores form one threshold."""
    y, p = _vectors(actual, probability)
    positives = int(y.sum())
    if not positives:
        return None
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    true_positives = np.cumsum(y)[last]
    precision = true_positives / (last + 1)
    recall = true_positives / positives
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def loss_differential(actual, model_a, model_b) -> np.ndarray:
    """Per-month Brier loss of B minus A. Positive means B was worse that month."""
    y, a = _vectors(actual, model_a)
    _, b = _vectors(actual, model_b)
    return (b - y) ** 2 - (a - y) ** 2


def autocorrelation(values, months, lags=ACF_LAGS) -> dict:
    """Autocorrelation at calendar lags, averaging only over pairs where both months exist.

    The pair count is returned with each value. With few pairs the estimate is
    noisy and can fall outside [-1, 1].
    """
    d, positions = _series(values, months)
    z = _centred_calendar(d, positions)
    present = np.zeros_like(z)
    present[positions - positions[0]] = 1
    variance = z @ z / len(d)
    result = {}
    for lag in lags:
        pairs = int(present[:-lag] @ present[lag:]) if 0 < lag < len(z) else 0
        value = None if not pairs or variance == 0 else float(z[:-lag] @ z[lag:] / pairs / variance)
        result[str(lag)] = dict(value=value, pairs=pairs)
    return result


def hac_variance_of_mean(values, months, max_lag: int) -> float:
    """Bartlett (Newey-West) variance of the sample mean on a calendar with gaps.

    Lag-k cross-products use only pairs of observed months exactly k months
    apart. With no gaps this is the usual Newey-West estimate without the
    small-sample correction.
    """
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    d, positions = _series(values, months)
    z = _centred_calendar(d, positions)
    total = z @ z
    for lag in range(1, min(max_lag, len(z) - 1) + 1):
        total += 2 * (1 - lag / (max_lag + 1)) * (z[:-lag] @ z[lag:])
    return float(total / len(d) ** 2)


def diebold_mariano(values, months, max_lag: int) -> dict:
    """Mean loss differential over its HAC standard error, referred to t(n-1)."""
    d, _ = _series(values, months)
    variance = hac_variance_of_mean(d, months, max_lag)
    result = dict(max_lag=max_lag, kernel="bartlett", mean=float(d.mean()),
                  standard_error=None, statistic=None, p_value_two_sided=None,
                  effective_observations=None)
    # Identical differentials leave only rounding noise in the variance.
    if variance > 0 and np.ptp(d) > 0:
        statistic = float(d.mean() / np.sqrt(variance))
        result.update(standard_error=float(np.sqrt(variance)), statistic=statistic,
                      p_value_two_sided=float(2 * student_t.sf(abs(statistic), len(d) - 1)),
                      # How many independent months would give the same precision.
                      effective_observations=float(d.var() / variance))
    return result


def _interval(means, level):
    lower, upper = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return dict(replications=int(len(means)), ci_lower=float(lower), ci_upper=float(upper),
                share_at_or_below_zero=float(np.mean(means <= 0)))


def iid_bootstrap(values, replications=REPLICATIONS, seed=SEED, level=.95) -> dict:
    """Rows resampled independently. Kept only to reproduce the interval first reported."""
    d = np.asarray(values, dtype=float)
    if d.ndim != 1 or len(d) == 0 or not np.isfinite(d).all():
        raise ValueError("Expected a nonempty finite one-dimensional array")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(replications, len(d)))].mean(axis=1)
    return dict(method="iid_rows", **_interval(means, level))


def block_bootstrap(values, months, block_length: int, replications=REPLICATIONS,
                    seed=SEED, level=.95) -> dict:
    """Circular block bootstrap of the mean, with blocks measured in calendar months.

    Each replicate joins blocks of `block_length` consecutive calendar months,
    drawn with replacement, until it covers the evaluation span once. A block
    that lands in an ineligible stretch carries fewer forecasts or none, so the
    replicate statistic is the sum of differentials over the number of forecasts
    drawn. Replicates that draw no forecast at all are dropped and counted.
    """
    d, positions = _series(values, months)
    span = int(positions[-1] - positions[0] + 1)
    if not 1 <= block_length <= span:
        raise ValueError("block_length must be between 1 and the calendar span")
    filled, present = np.zeros(span), np.zeros(span)
    filled[positions - positions[0]] = d
    present[positions - positions[0]] = 1
    # Prefix sums over two copies of the span let a block wrap past the end.
    sum_prefix = np.r_[0., np.cumsum(np.r_[filled, filled])]
    count_prefix = np.r_[0., np.cumsum(np.r_[present, present])]
    blocks = -(-span // block_length)
    lengths = np.full(blocks, block_length)
    lengths[-1] = span - block_length * (blocks - 1)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, span, size=(replications, blocks))
    ends = starts + lengths
    sums = (sum_prefix[ends] - sum_prefix[starts]).sum(axis=1)
    counts = (count_prefix[ends] - count_prefix[starts]).sum(axis=1)
    usable = counts > 0
    return dict(method="circular_calendar_blocks", block_length_months=block_length,
                calendar_span_months=span, empty_replicates_dropped=int((~usable).sum()),
                **_interval(sums[usable] / counts[usable], level))


def recession_episodes(economic_rows) -> list:
    """(first, last) month of each contiguous USREC=1 run in the economic table."""
    episodes, start, previous = [], None, None
    for row in sorted(economic_rows, key=lambda r: r["predictor_month"]):
        month = date.fromisoformat(row["predictor_month"])
        position = month.year * 12 + month.month
        recession = int(row["USREC_t"]) == 1
        if start is not None and (not recession or position != previous[1] + 1):
            episodes.append((start, previous[0]))
            start = None
        if recession and start is None:
            start = month
        previous = (month, position)
    if start is not None:
        episodes.append((start, previous[0]))
    return episodes


def paired_summary(rows, episodes, block_lengths=BLOCK_LENGTHS, hac_lags=HAC_LAGS,
                   replications=REPLICATIONS, seed=SEED, level=.95) -> dict:
    """All metrics and intervals for one saved predictions.csv (status=ok rows)."""
    paired = [r for r in rows if r["status"] == "ok"]
    if not paired:
        raise ValueError("No paired forecasts to summarise")
    months = [r["predictor_month"] for r in paired]
    y = [int(r["actual_label"]) for r in paired]
    a = [float(r["model_a_probability"]) for r in paired]
    b = [float(r["model_b_probability"]) for r in paired]
    # Same reference as the Model A backtest: recession share of the training rows.
    reference = [int(r["training_recession_count"]) / int(r["training_count"]) for r in paired]
    reference_bs = brier_score(y, reference)
    models = {}
    for name, p in (("model_a", a), ("model_b", b)):
        bs = brier_score(y, p)
        models[name] = dict(brier_score=bs,
                            brier_skill_vs_training_prevalence=brier_skill_score(bs, reference_bs),
                            roc_auc=roc_auc(y, p), pr_auc=average_precision(y, p))
    d = loss_differential(y, a, b)
    positive_targets = [date.fromisoformat(r["target_month"]) for r, label in zip(paired, y) if label]
    reached = [(s, e) for s, e in episodes if any(s <= t <= e for t in positive_targets)]
    positions = month_positions(months)
    return dict(
        forecasts=len(paired), first_predictor_month=months[0], last_predictor_month=months[-1],
        calendar_span_months=int(positions[-1] - positions[0] + 1),
        recession_target_months=sum(y),
        recession_episodes_reached=["%s to %s" % (s.isoformat()[:7], e.isoformat()[:7])
                                    for s, e in reached],
        pr_auc_no_skill_level=sum(y) / len(y),
        training_prevalence_brier_score=reference_bs,
        **models,
        loss_differential=dict(
            definition="Brier loss of Model B minus Model A per forecast month; positive = B worse",
            mean=float(d.mean()), months_b_worse=int((d > 0).sum()), months_b_better=int((d < 0).sum()),
            autocorrelation_by_calendar_lag=autocorrelation(d, months),
            iid_bootstrap=iid_bootstrap(d, replications, seed, level),
            block_bootstrap=[block_bootstrap(d, months, n, replications, seed, level)
                             for n in block_lengths],
            diebold_mariano=[diebold_mariano(d, months, lag) for lag in hac_lags]),
        settings=dict(horizon_months=HORIZON, confidence_level=level,
                      replications=replications, seed=seed))
