"""Explain the baseline: continuous spread -> refit Probit -> saved probability.

Reads only the saved original baseline forecasts (reports/baseline_backtest/
forecasts.csv) and the prepared economic table. No fitting, no network.
Every plotted probability is a saved prediction; the drawn curves are each
evaluated from ONE origin's saved refit coefficients, because the expanding
window refits the model at every forecast month.

Usage:
    python scripts/make_spread_probability_figure.py                # writes docs/figures/yield_spread
    python scripts/make_spread_probability_figure.py --out /tmp/spread-figure-FRESH
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date
from pathlib import Path

FORECASTS = Path('reports/baseline_backtest/forecasts.csv')
ECONOMIC = Path('data/processed/economic_monthly.csv')
DEFAULT_OUT = Path('docs/figures/yield_spread')
EXAMPLES = ('1973-10-01', '2006-12-01', '2023-01-01')
CURVE_ORIGINS = ('1973-10-01', '1990-01-01', '2007-01-01', '2026-03-01')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
NONREC, REC, SHADE = '#2a78d6', '#eb6834', '#d9d8d2'


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def phi(x):
    return .5 * (1 + math.erf(x / math.sqrt(2)))


def month(value):
    return date.fromisoformat(value)


def worked_examples(root):
    forecasts = {r['predictor_month']: r for r in read_rows(root / FORECASTS)}
    economic = {r['predictor_month']: r for r in read_rows(root / ECONOMIC)}
    result = []
    for key in EXAMPLES:
        f, e = forecasts[key], economic[key]
        intercept, slope = float(f['intercept']), float(f['spread_coefficient'])
        linear = intercept + slope * float(e['spread_t'])
        result.append(dict(
            predictor_month=key, target_month=f['target_month'], forecast_cutoff=f['forecast_cutoff'],
            GS10_t=e['GS10_t'], TB3MS_t=e['TB3MS_t'], spread_t_pp=e['spread_t'],
            inversion_indicator=int(float(e['spread_t']) < 0),
            training_rows=int(f['training_sample_count']),
            training_predictor_window='%s to %s' % (f['training_predictor_start'], f['training_predictor_end']),
            latest_training_target=f['training_target_end'],
            intercept=intercept, spread_coefficient=slope, linear_index=linear,
            reconstructed_probability=phi(linear), saved_probability=float(f['predicted_probability']),
            actual_target_USREC=int(f['actual_label'])))
    return result


FIGURES = ('spread_vs_probability.png', 'refit_coefficients.png', 'probability_timeline.png')
NOTE = ('Retrospective: final FRED values and NBER dates. Training uses only targets up to t-1; '
        'predictor months 1953-04 to 1973-09 are the 240-month warm-up.')


def draw(root, out):
    """Write three separate PNGs, one per idea."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    rows = read_rows(root / FORECASTS)
    ok = [r for r in rows if r['status'] == 'ok']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.titlecolor': INK})
    out.mkdir(parents=True, exist_ok=True)

    def new_figure(size):
        fig, ax = plt.subplots(figsize=size, layout='constrained', facecolor='#fcfcfb')
        ax.set_facecolor('#fcfcfb')
        ax.grid(color=GRID, lw=.6)
        ax.set_axisbelow(True)
        return fig, ax

    def save(fig, name, note=NOTE):
        fig.text(.01, -.01, note, fontsize=8, color=MUTED, ha='left', va='top')
        fig.savefig(out / name, dpi=170, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

    # 1. Every saved forecast, plus curves from four individual refits.
    fig, ax = new_figure((9, 5.6))
    for label, colour, name in ((0, NONREC, 'No recession 6 months later'),
                                (1, REC, 'Recession 6 months later')):
        subset = [r for r in ok if int(r['actual_label']) == label]
        ax.scatter([float(r['spread_t']) for r in subset], [float(r['predicted_probability']) for r in subset],
                   s=18, color=colour, alpha=.75, edgecolor='#fcfcfb', linewidth=.4,
                   label='%s (n=%d)' % (name, len(subset)), zorder=3)
    by_month = {r['predictor_month']: r for r in ok}
    grid_x = [-3 + i * .05 for i in range(151)]
    curves = []
    for origin, style in zip(CURVE_ORIGINS, ('-', '--', '-.', ':')):
        r = by_month[origin]
        b0, b1 = float(r['intercept']), float(r['spread_coefficient'])
        curves += ax.plot(grid_x, [phi(b0 + b1 * x) for x in grid_x], ls=style, color=MUTED, lw=1.4,
                          zorder=2, label='Model as refit in %s' % origin[:7])
    ax.axvline(0, color=MUTED, lw=.8)
    ax.annotate('<- inverted curve (spread < 0)', (0, .015), xytext=(-4, 0), textcoords='offset points',
                fontsize=8.5, color=MUTED, ha='right', va='bottom')
    ax.set(xlim=(-3, 4.6), ylim=(0, 1), ylabel='Predicted recession probability',
           xlabel='Yield spread = 10-year minus 3-month Treasury yield (percentage points)',
           title='A larger inversion gives a higher recession probability (%d monthly forecasts)' % len(ok))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    points = ax.legend(handles=ax.collections[:2], loc='upper right', fontsize=8.5, frameon=False,
                       title='Saved forecasts', title_fontsize=8.5)
    ax.add_artist(points)
    ax.legend(handles=curves, loc='center right', bbox_to_anchor=(1, .56), fontsize=8.5, frameon=False,
              title='The curve shifts each time the model is refit', title_fontsize=8.5)
    save(fig, FIGURES[0])

    # 2. Coefficients change at every origin.
    fig, ax = new_figure((9, 4.4))
    origins = [month(r['predictor_month']) for r in ok]
    ax.plot(origins, [float(r['intercept']) for r in ok], color=NONREC, lw=2, label='Intercept (β0)')
    ax.plot(origins, [float(r['spread_coefficient']) for r in ok], color=REC, lw=2,
            label='Spread coefficient (β1)')
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set(title='The model is re-estimated every month, so its coefficients drift',
           xlabel='Forecast month', ylabel='Probit coefficient')
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc='lower right', fontsize=8.5, frameon=False)
    save(fig, FIGURES[1], 'Each point is a fresh fit on all data available before that forecast month. '
                          'A negative spread coefficient means a lower spread raises recession risk.')

    # 3. Saved probabilities over time against the binary outcome.
    fig, ax = new_figure((12, 4.6))
    targets = [month(r['target_month']) for r in ok]
    shaded = False
    for r, t in zip(ok, targets):
        if int(r['actual_label']) == 1:
            end = date(t.year + (t.month == 12), t.month % 12 + 1, 1)
            ax.axvspan(t, end, color=SHADE, lw=0, label=None if shaded else 'Actual recession month')
            shaded = True
    ax.plot(targets, [float(r['predicted_probability']) for r in ok], color=NONREC, lw=1.6,
            label='Predicted probability')
    ax.set(ylim=(0, 1), xlim=(targets[0], targets[-1]), ylabel='Recession probability',
           xlabel='Month being forecast (6 months after the spread is observed)',
           title='Predicted recession probability vs. actual recessions, %s to %s' % (
               ok[0]['target_month'][:7], ok[-1]['target_month'][:7]))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc='upper right', fontsize=8.5, frameon=False)
    save(fig, FIGURES[2], NOTE + ' No forecast months are missing from 1973-10 to 2026-03; '
                          'months from 2026-04 on have no outcome yet.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    examples = worked_examples(root)
    for e in examples:
        if abs(e['reconstructed_probability'] - e['saved_probability']) > 1e-12:
            raise SystemExit('Reconstruction mismatch at %s' % e['predictor_month'])
    draw(root, out)
    (out / 'worked_examples.json').write_text(json.dumps(examples, indent=2) + '\n', encoding='utf-8')
    for e in examples:
        print('%s: %s - %s = %s pp; Phi(%.5f %+.5f*%s) = %.6f (saved %.6f); target %s USREC=%d' % (
            e['predictor_month'][:7], e['GS10_t'], e['TB3MS_t'], e['spread_t_pp'], e['intercept'],
            e['spread_coefficient'], e['spread_t_pp'], e['reconstructed_probability'],
            e['saved_probability'], e['target_month'][:7], e['actual_target_USREC']))
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
