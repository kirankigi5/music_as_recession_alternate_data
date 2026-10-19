"""Regenerate every research figure from ALREADY-COMPUTED experiment outputs.

Reads only derived CSV/JSON under reports/model_comparison/. Performs no emotion
inference, no model fitting and no network access, so figures can be rebuilt in
seconds without repeating the 16,290-song run.

Usage:
    python scripts/make_research_figures.py
    python scripts/make_research_figures.py --out docs/figures/music_recession
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RUNS = [('70% (primary)', 'exploratory_primary_cov70', '#1f77b4'),
        ('75% sensitivity', 'exploratory_sensitivity_cov75', '#ff7f0e'),
        ('80% sensitivity', 'exploratory_sensitivity_cov80', '#2ca02c')]
EMOTIONS = ['joy', 'sadness', 'fear', 'anger', 'nervousness', 'optimism']
BASE = Path('reports/model_comparison')
CAPTION = ('Exploratory course analysis: automated screens only; recordings, language '
           'and release dates unverified.')


def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def load():
    data = {}
    for label, name, colour in RUNS:
        directory = BASE / name
        data[name] = dict(
            label=label, colour=colour,
            metrics=json.loads((directory / 'metrics.json').read_text()),
            predictions=[r for r in read_csv_rows(directory / 'predictions.csv')],
            calibration=read_csv_rows(directory / 'calibration.csv'),
            coverage=read_csv_rows(directory / 'monthly_coverage.csv'),
            features=read_csv_rows(directory / 'monthly_features_and_coverage.csv'))
    return data


def months(rows, key='predictor_month'):
    return [date.fromisoformat(r[key]) for r in rows]


def break_gaps(times, *series):
    """Insert NaN breaks where consecutive months are not adjacent.

    Eligible months are not contiguous, so joining them with a straight line
    would imply forecasts that were never issued. This keeps the plot honest.
    """
    out_times = [times[0]]
    out_series = [[s[0]] for s in series]
    for index in range(1, len(times)):
        previous, current = times[index - 1], times[index]
        gap = (current.year - previous.year) * 12 + (current.month - previous.month)
        if gap > 1:
            out_times.append(previous + (current - previous) / 2)
            for bucket in out_series:
                bucket.append(np.nan)
        out_times.append(current)
        for bucket, s in zip(out_series, series):
            bucket.append(s[index])
    return out_times, [np.array(b, dtype=float) for b in out_series]


def fig1_pipeline(out):
    """Serpentine 5x2 flow drawn in DATA coordinates so boxes keep their aspect."""
    stages = [
        ('1. Billboard Hot 100\nweekly charts\n1973-2023', '#dbe9f6'),
        ('2. Song identity\nmatching\n21,542 identities', '#dbe9f6'),
        ('3. OSF lyrics corpus\n19,096 candidates', '#dbe9f6'),
        ('4. Automated screening\n16,290 pass\n(risk + English)', '#dbe9f6'),
        ('5. GoEmotions\ninference per song\ncached, scored once', '#fde3cf'),
        ('6. Popularity weighting\nweight = 101 - rank\nper weekly appearance', '#fde3cf'),
        ('7. Six monthly\nemotion features\n(kept separate)', '#fde3cf'),
        ('8. Merge economics\nGS10 - TB3MS spread\nUSREC target at t+6', '#e3f3dc'),
        ('9. Expanding-window\nProbit, min 240\neligible observations', '#e3f3dc'),
        ('10. Paired evaluation\nBrier / ROC-AUC /\ncalibration', '#e3f3dc')]

    fig, ax = plt.subplots(figsize=(15, 6.8), layout='constrained')
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 6.0)
    ax.axis('off')
    width, height = 1.78, 1.45
    xs = [0.35 + 2.05 * i for i in range(5)]
    rows = [4.05, 1.35]

    centres = []
    for index, (text, colour) in enumerate(stages):
        row, column = divmod(index, 5)
        x = xs[column] if row == 0 else xs[4 - column]
        y = rows[row]
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                    boxstyle='round,pad=0.04,rounding_size=0.12',
                                    linewidth=1.3, edgecolor='#33475b',
                                    facecolor=colour, zorder=2))
        ax.text(x + width / 2, y + height / 2, text, ha='center', va='center',
                fontsize=8.6, zorder=3, linespacing=1.45)
        centres.append((x + width / 2, y + height / 2))

    for index in range(len(stages) - 1):
        (x0, y0), (x1, y1) = centres[index], centres[index + 1]
        if abs(y0 - y1) < 0.01:                      # horizontal hop
            forward = x1 > x0
            start = (x0 + (width / 2 + 0.06) * (1 if forward else -1), y0)
            end = (x1 - (width / 2 + 0.06) * (1 if forward else -1), y1)
        else:                                        # drop between rows
            start = (x0, y0 - height / 2 - 0.06)
            end = (x1, y1 + height / 2 + 0.06)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                     mutation_scale=17, linewidth=1.4,
                                     color='#33475b', zorder=1))

    ax.text(5.3, 5.78, 'Figure 1. Analysis pipeline: Billboard charts to paired '
            'recession forecasts', ha='center', fontsize=13.5)
    ax.text(5.3, 0.30, CAPTION, ha='center', fontsize=8.8, style='italic', color='#444')
    ax.text(5.3, 0.06, 'Blue = data assembly and screening,  orange = emotion feature '
            'construction,  green = econometric modelling',
            ha='center', fontsize=8.4, color='#555')
    fig.savefig(out / 'fig01_pipeline_architecture.png', dpi=170)
    plt.close(fig)


def fig2_decade_coverage(out, data):
    rows = data['exploratory_primary_cov70']['coverage']
    by_decade = defaultdict(list)
    eligible = defaultdict(int)
    for r in rows:
        decade = int(r['predictor_month'][:4]) // 10 * 10
        by_decade[decade].append(float(r['scored_weight_coverage']))
        eligible[decade] += 1 if r['feature_eligible'] == 'True' else 0
    decades = sorted(by_decade)
    labels = ['%ss' % d for d in decades]
    medians = [float(np.median(by_decade[d])) for d in decades]
    counts = [len(by_decade[d]) for d in decades]
    eligibles = [eligible[d] for d in decades]
    x = np.arange(len(decades))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5), layout='constrained')
    bars = ax1.bar(x, medians, color='#4878a8', edgecolor='#2b4a66')
    ax1.axhline(0.70, color='#c0392b', linestyle='--', linewidth=1.3,
                label='70% primary threshold')
    ax1.set(xticks=x, xticklabels=labels, ylim=(0, 1.0),
            ylabel='Median monthly coverage (share of chart popularity weight)',
            xlabel='Decade', title='Median monthly lyric coverage by decade')
    for bar, value in zip(bars, medians):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.015, '%.3f' % value,
                 ha='center', fontsize=9)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    width = 0.42
    ax2.bar(x - width / 2, counts, width, label='Calendar months in decade',
            color='#b0bec5', edgecolor='#78909c')
    ax2.bar(x + width / 2, eligibles, width, label='Feature-eligible at 70%',
            color='#2e7d32', edgecolor='#1b5e20')
    for i, (total, elig) in enumerate(zip(counts, eligibles)):
        ax2.text(i + width / 2, elig + 1.5, str(elig), ha='center', fontsize=9)
    ax2.set(xticks=x, xticklabels=labels, xlabel='Decade', ylabel='Number of months',
            title='Months surviving the 70% coverage gate, by decade')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    fig.suptitle('Figure 2. Lyric coverage declines sharply after the 1980s, and '
                 'month eligibility collapses with it', fontsize=12.5)
    fig.savefig(out / 'fig02_coverage_by_decade.png', dpi=170)
    plt.close(fig)


def fig3_monthly_coverage(out, data):
    rows = data['exploratory_primary_cov70']['coverage']
    times = months(rows)
    coverage = [float(r['scored_weight_coverage']) for r in rows]
    eligible = [r['feature_eligible'] == 'True' for r in rows]
    fig, ax = plt.subplots(figsize=(13.5, 5), layout='constrained')
    ax.plot(times, coverage, linewidth=1.0, color='#1f4e79',
            label='Monthly scored-lyric coverage')
    ax.scatter([t for t, e in zip(times, eligible) if e],
               [c for c, e in zip(coverage, eligible) if e],
               s=5, color='#2e7d32', zorder=3, label='Month eligible at 70%')
    for level, colour in ((0.70, '#c0392b'), (0.75, '#e67e22'), (0.80, '#8e44ad')):
        ax.axhline(level, linestyle='--', linewidth=1.1, color=colour,
                   label='%d%% threshold' % round(level * 100))
    ax.set(xlabel='Predictor month', ylim=(0, 1.0),
           ylabel='Share of total chart popularity weight with scored lyrics',
           title='Figure 3. Monthly popularity-weighted lyric coverage, 1973-2023')
    ax.legend(fontsize=8.6, ncol=3, loc='lower left')
    ax.grid(alpha=0.3)
    ax.text(0.5, -0.17, CAPTION + ' Denominator includes chart entries with no usable lyrics.',
            ha='center', fontsize=8.2, style='italic', color='#444', transform=ax.transAxes)
    fig.savefig(out / 'fig03_monthly_coverage_timeseries.png', dpi=170)
    plt.close(fig)


def fig4_brier(out, data):
    labels = [d['label'] for d in (data[n] for _, n, _ in RUNS)]
    a = [data[n]['metrics']['model_a']['brier_score'] for _, n, _ in RUNS]
    b = [data[n]['metrics']['model_b']['brier_score'] for _, n, _ in RUNS]
    n_f = [data[n]['metrics']['evaluated_months'] for _, n, _ in RUNS]
    x = np.arange(3)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.6), layout='constrained')
    r1 = ax.bar(x - width / 2, a, width, label='Model A (yield spread only)',
                color='#2e7d32', edgecolor='#1b5e20')
    r2 = ax.bar(x + width / 2, b, width, label='Model B (spread + 6 emotions)',
                color='#c0392b', edgecolor='#7b241c')
    for bars in (r1, r2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0015,
                    '%.4f' % bar.get_height(), ha='center', fontsize=9.3)
    for i, (av, bv) in enumerate(zip(a, b)):
        ax.annotate('B-A = +%.4f' % (bv - av), (i, max(av, bv) + 0.012),
                    ha='center', fontsize=9.4, color='#7b241c', weight='bold')
    ax.set(xticks=x, xticklabels=['%s\n%d paired forecasts' % (l, k)
                                  for l, k in zip(labels, n_f)],
           ylabel='Brier score (lower is better)', ylim=(0, max(b) * 1.30),
           xlabel='Predeclared monthly coverage threshold',
           title='Figure 4. Brier score, Model A vs Model B, at all three predeclared thresholds')
    ax.legend(fontsize=9.6)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.5, -0.20, 'Samples differ across thresholds and are NOT directly comparable to '
            'one another. ' + CAPTION, ha='center', fontsize=8.2, style='italic',
            color='#444', transform=ax.transAxes)
    fig.savefig(out / 'fig04_brier_comparison.png', dpi=170)
    plt.close(fig)


def fig5_auc(out, data):
    labels = [data[n]['label'] for _, n, _ in RUNS]
    a = [data[n]['metrics']['model_a']['roc_auc'] for _, n, _ in RUNS]
    b = [data[n]['metrics']['model_b']['roc_auc'] for _, n, _ in RUNS]
    rec = [data[n]['metrics']['recession_months'] for _, n, _ in RUNS]
    x = np.arange(3)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.6), layout='constrained')
    r1 = ax.bar(x - width / 2, a, width, label='Model A', color='#2e7d32',
                edgecolor='#1b5e20')
    r2 = ax.bar(x + width / 2, b, width, label='Model B', color='#c0392b',
                edgecolor='#7b241c')
    for bars in (r1, r2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    '%.4f' % bar.get_height(), ha='center', fontsize=9.3)
    ax.axhline(0.5, color='#555', linestyle='--', linewidth=1.3,
               label='0.50 = chance ordering')
    ax.set(xticks=x, xticklabels=['%s\n%d recession months' % (l, r)
                                  for l, r in zip(labels, rec)],
           ylabel='ROC-AUC (higher is better)', ylim=(0, 1.12),
           xlabel='Predeclared monthly coverage threshold',
           title='Figure 5. ROC-AUC, Model A vs Model B, at all three predeclared thresholds')
    ax.legend(fontsize=9.6, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.5, -0.20, 'AUC on 4-15 positive months is highly unstable; compare within a '
            'threshold, not across. ' + CAPTION, ha='center', fontsize=8.2,
            style='italic', color='#444', transform=ax.transAxes)
    fig.savefig(out / 'fig05_roc_auc_comparison.png', dpi=170)
    plt.close(fig)


def fig6_probabilities(out, data):
    rows = [r for r in data['exploratory_primary_cov70']['predictions']
            if r['status'] == 'ok']
    target = months(rows, 'target_month')
    predictor = months(rows, 'predictor_month')
    pa = [float(r['model_a_probability']) for r in rows]
    pb = [float(r['model_b_probability']) for r in rows]
    labels = [int(r['actual_label']) for r in rows]
    fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')
    shaded = False
    for t, y in zip(target, labels):
        if y == 1:
            ax.axvspan(t, date(t.year + (t.month == 12), t.month % 12 + 1, 1),
                       color='#8899aa', alpha=0.33, lw=0,
                       label=None if shaded else 'Actual NBER recession in TARGET month')
            shaded = True
    gapped, (pa_g, pb_g) = break_gaps(target, pa, pb)
    ax.plot(gapped, pa_g, linewidth=1.5, color='#2e7d32', label='Model A probability')
    ax.plot(gapped, pb_g, linewidth=1.5, color='#c0392b', label='Model B probability')
    ax.scatter(target, pa, s=7, color='#2e7d32', zorder=3)
    ax.scatter(target, pb, s=7, color='#c0392b', zorder=3)
    ax.set(ylim=(0, 1), ylabel='Predicted P(recession in target month)',
           xlabel='TARGET month  =  predictor month + 6 calendar months',
           title='Figure 6. Paired out-of-sample recession probabilities, plotted at the '
                 'target month they describe')
    ax.legend(fontsize=9.4, loc='upper left')
    ax.grid(alpha=0.3)
    note = ('Each marker is a forecast issued from predictor month t = target - 6, using only '
            'information available by the 15th of t+1. Lines BREAK where no eligible month '
            'exists, so no forecast is implied there.\nFirst predictor month %s -> first '
            'target %s;  last predictor %s -> last target %s.  %d paired forecasts, %d '
            'recession-positive targets. Shading marks recession target months PRESENT IN '
            'THE SAMPLE only;\nmost of the 2008-09 recession is absent because those months '
            'failed the 70%% coverage gate.'
            % (predictor[0], target[0], predictor[-1], target[-1], len(rows), sum(labels)))
    ax.text(0.5, -0.215, note, ha='center', fontsize=8.4, color='#333',
            transform=ax.transAxes)
    fig.savefig(out / 'fig06_probability_timeline.png', dpi=170)
    plt.close(fig)


def fig7_calibration(out, data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), layout='constrained')
    for ax, (label, name, _) in zip(axes, RUNS):
        rows = data[name]['calibration']
        for model, colour, marker in (('A', '#2e7d32', 'o'), ('B', '#c0392b', 's')):
            points = [(float(r['mean_predicted_probability']),
                       float(r['observed_recession_frequency']), int(r['count']))
                      for r in rows if r['model'] == model and int(r['count']) > 0]
            if not points:
                continue
            xs, ys, ns = zip(*points)
            ax.plot(xs, ys, marker=marker, color=colour, linewidth=1.3,
                    markersize=5, label='Model ' + model)
            ax.scatter(xs, ys, s=[12 + 2.6 * n for n in ns], color=colour, alpha=0.25)
        ax.plot([0, 1], [0, 1], linestyle='--', color='#555', linewidth=1.1,
                label='Perfect calibration')
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability',
               title='%s\n(%d forecasts)' % (label, data[name]['metrics']['evaluated_months']))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Observed recession frequency in bin')
    axes[0].legend(fontsize=8.8, loc='upper left')
    fig.suptitle('Figure 7. Calibration (reliability) curves; marker area is proportional '
                 'to the number of forecasts in each decile bin', fontsize=12.5)
    fig.savefig(out / 'fig07_calibration_comparison.png', dpi=170)
    plt.close(fig)


def fig8_emotions(out, data):
    rows = [r for r in data['exploratory_primary_cov70']['features']
            if r['emotion_status'] == 'scored']
    times = months(rows)
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), layout='constrained', sharex=True)
    for ax, emotion in zip(axes.ravel(), EMOTIONS):
        values = np.array([float(r[emotion]) for r in rows])
        ax.plot(times, values, linewidth=0.85, color='#37474f', alpha=0.65,
                label='Monthly value')
        if len(values) >= 25:
            window = 25
            smooth = np.convolve(values, np.ones(window) / window, mode='valid')
            ax.plot(times[window - 1:], smooth, linewidth=2.1, color='#c62828',
                    label='25-month moving average')
        ax.set_title('%s  (mean %.4f, sd %.4f)' % (emotion, values.mean(), values.std()),
                     fontsize=10.5)
        ax.grid(alpha=0.3)
        ax.set_ylabel('Weighted score')
    for ax in axes[-1]:
        ax.set_xlabel('Predictor month')
    axes[0][0].legend(fontsize=8.4)
    fig.suptitle('Figure 8. Monthly popularity-weighted GoEmotions features, 1973-2023 '
                 '(six separate predictors, never combined into one index)', fontsize=12.5)
    fig.savefig(out / 'fig08_emotion_feature_trends.png', dpi=170)
    plt.close(fig)


def fig9_threshold_attrition(out, data):
    labels, elig, paired, rec = [], [], [], []
    for label, name, _ in RUNS:
        m = data[name]['metrics']
        labels.append(label)
        elig.append(m['eligible_feature_months'])
        paired.append(m['evaluated_months'])
        rec.append(m['recession_months'])
    x = np.arange(3)
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.6), layout='constrained')
    sets = [(elig, 'Feature-eligible months', '#90a4ae', -width),
            (paired, 'Paired out-of-sample forecasts', '#1565c0', 0.0),
            (rec, 'Recession-positive target months', '#c62828', width)]
    for values, name, colour, offset in sets:
        bars = ax.bar(x + offset, values, width, label=name, color=colour,
                      edgecolor='#37474f')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                    str(int(bar.get_height())), ha='center', fontsize=9.2)
    ax.axhline(240, color='#6a1b9a', linestyle='--', linewidth=1.3,
               label='240-observation training minimum')
    ax.set(xticks=x, xticklabels=labels, yscale='log', ylim=(1, 900),
           ylabel='Count of months (log scale)',
           xlabel='Predeclared monthly coverage threshold',
           title='Figure 9. Sample attrition as the coverage threshold tightens')
    ax.legend(fontsize=9.2, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.5, -0.19, 'Tightening coverage raises evidential quality but shrinks the '
            'evaluation sample and the number of recession events. ' + CAPTION,
            ha='center', fontsize=8.2, style='italic', color='#444', transform=ax.transAxes)
    fig.savefig(out / 'fig09_threshold_attrition.png', dpi=170)
    plt.close(fig)


def fig10_error_analysis(out, data):
    rows = [r for r in data['exploratory_primary_cov70']['predictions']
            if r['status'] == 'ok']
    target = months(rows, 'target_month')
    y = np.array([int(r['actual_label']) for r in rows], dtype=float)
    pa = np.array([float(r['model_a_probability']) for r in rows])
    pb = np.array([float(r['model_b_probability']) for r in rows])
    diff = (pb - y) ** 2 - (pa - y) ** 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), layout='constrained',
                                   height_ratios=[2, 1.25], sharex=True)
    colours = ['#c0392b' if d > 0 else '#2e7d32' for d in diff]
    ax1.bar(target, diff, width=24, color=colours, edgecolor='none')
    ax1.axhline(0, color='#333', linewidth=1.0)
    shaded = False
    for t, label in zip(target, y):
        if label == 1:
            ax1.axvspan(t, date(t.year + (t.month == 12), t.month % 12 + 1, 1),
                        color='#8899aa', alpha=0.28, lw=0,
                        label=None if shaded else 'Recession target month')
            shaded = True
    ax1.set(ylabel='Per-month Brier difference (B - A)',
            title='Figure 10. Where Model B gained and lost, by target month')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.text(0.012, 0.93, 'red = Model B worse,  green = Model B better',
             transform=ax1.transAxes, fontsize=9.2)
    gapped, (cumulative,) = break_gaps(target, np.cumsum(diff))
    ax2.plot(gapped, cumulative, linewidth=1.9, color='#7b241c')
    ax2.axhline(0, color='#333', linewidth=1.0)
    ax2.set(xlabel='TARGET month (predictor month + 6)',
            ylabel='Cumulative Brier difference',
            title='Cumulative loss: a rising line means Model B is falling further behind')
    ax2.grid(alpha=0.3)
    worse = int((diff > 0).sum())
    ax2.text(0.012, 0.09, 'Model B worse in %d of %d months (%.1f%%); mean difference %+.6f'
             % (worse, len(diff), 100.0 * worse / len(diff), diff.mean()),
             transform=ax2.transAxes, fontsize=9.4)
    fig.savefig(out / 'fig10_error_analysis.png', dpi=170)
    plt.close(fig)
    return diff, y, pa, pb, target


def summary_json(out, data, diff, y):
    payload = dict(
        analysis='exploratory_course_policy',
        caveat=CAPTION,
        emotion_model=data['exploratory_primary_cov70']['metrics']['model_config']['model_id']
        if 'model_config' in data['exploratory_primary_cov70']['metrics'] else
        'SamLowe/roberta-base-go_emotions',
        songs_scored=data['exploratory_primary_cov70']['metrics']['scored_song_count'],
        runs=[])
    for label, name, _ in RUNS:
        m = data[name]['metrics']
        payload['runs'].append(dict(
            label=label, coverage_threshold=m['coverage_threshold'],
            analysis_months=m['analysis_months'],
            eligible_feature_months=m['eligible_feature_months'],
            paired_forecasts=m['evaluated_months'],
            recession_positive_months=m['recession_months'],
            fit_failures=len(m['fit_failures']),
            forecast_start=m['forecast_start'], forecast_end=m['forecast_end'],
            target_start=m['target_start'], target_end=m['target_end'],
            model_a_brier=m['model_a']['brier_score'],
            model_b_brier=m['model_b']['brier_score'],
            brier_difference_b_minus_a=m['brier_difference_b_minus_a'],
            model_a_roc_auc=m['model_a']['roc_auc'],
            model_b_roc_auc=m['model_b']['roc_auc']))
    payload['primary_error_analysis'] = dict(
        paired_forecasts=int(len(diff)),
        months_model_b_worse=int((diff > 0).sum()),
        months_model_b_better=int((diff < 0).sum()),
        mean_per_month_brier_difference=float(diff.mean()),
        recession_positive_months=int(y.sum()))
    (out / 'results_summary.json').write_text(json.dumps(payload, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('docs/figures/music_recession'))
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = load()
    fig1_pipeline(out)
    fig2_decade_coverage(out, data)
    fig3_monthly_coverage(out, data)
    fig4_brier(out, data)
    fig5_auc(out, data)
    fig6_probabilities(out, data)
    fig7_calibration(out, data)
    fig8_emotions(out, data)
    fig9_threshold_attrition(out, data)
    diff, y, pa, pb, target = fig10_error_analysis(out, data)
    summary_json(out, data, diff, y)

    print('wrote figures to', out)
    for path in sorted(out.iterdir()):
        print('  %-44s %7.1f KB' % (path.name, path.stat().st_size / 1024))

    # Error-analysis numbers quoted in the report.
    print('\n=== ERROR ANALYSIS (primary run) ===')
    worse, better = int((diff > 0).sum()), int((diff < 0).sum())
    print('months Model B worse : %d of %d (%.1f%%)' % (worse, len(diff),
                                                        100.0 * worse / len(diff)))
    print('months Model B better: %d of %d (%.1f%%)' % (better, len(diff),
                                                        100.0 * better / len(diff)))
    print('mean per-month diff  : %+.10f' % diff.mean())
    rec = y == 1
    print('mean diff, recession target months     : %+.6f (n=%d)'
          % (diff[rec].mean(), rec.sum()))
    print('mean diff, non-recession target months : %+.6f (n=%d)'
          % (diff[~rec].mean(), (~rec).sum()))
    print('worst 5 months for Model B:')
    order = np.argsort(-diff)[:5]
    for i in order:
        print('   target %s  label=%d  A=%.4f  B=%.4f  diff=%+.4f'
              % (target[i], int(y[i]), pa[i], pb[i], diff[i]))
    print('best 5 months for Model B:')
    for i in np.argsort(diff)[:5]:
        print('   target %s  label=%d  A=%.4f  B=%.4f  diff=%+.4f'
              % (target[i], int(y[i]), pa[i], pb[i], diff[i]))
    by_decade = defaultdict(list)
    for t, d in zip(target, diff):
        by_decade[t.year // 10 * 10].append(d)
    print('mean per-month difference by target decade:')
    for decade in sorted(by_decade):
        values = by_decade[decade]
        print('   %ss  n=%-4d mean=%+.6f  B worse in %d'
              % (decade, len(values), float(np.mean(values)),
                 sum(1 for v in values if v > 0)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
