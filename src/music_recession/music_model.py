"""Evidence-gated, paired Model A/B retrospective experiment; no alternate estimators."""
import argparse
import importlib.metadata
import json
import platform
import warnings
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.special import ndtr
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

from . import music_preparation
from .baseline import (BaselineConfig, BaselineError, fit_probit, load_economic,
                       select_training, validate_observations)
from .backtest import brier_score, roc_auc, calibration_table, _diagnostic_json
from .economic import add_months
from .emotions import score_eligible, validate_config, validate_scores, software_versions
from .lyrics_audit import file_hash, read_csv, sha, write_csv
from .music_preparation import EMOTIONS

FEATURES = ['spread_t'] + EMOTIONS


def aggregate(months, contributions, scores):
    """Pool eligible weekly rank weights, retaining all observations in coverage."""
    pooled = {}
    seen = set()
    for c in contributions:
        key = (c['predictor_month'], c['song_id'])
        if key in seen:
            raise ValueError('Duplicate song/month contribution')
        seen.add(key)
        weight = int(c['nlp_ready_weight'])
        if not 0 <= weight <= int(c['known_weight']) <= int(c['total_weight']):
            raise ValueError('Invalid contribution weights')
        if not weight:
            continue
        score = scores.get(c['song_id'])
        if score is None or score['metadata']['source_reference_id'] != c['source_reference_id']:
            raise ValueError('Missing score or mismatched song/source identity')
        values = validate_scores(score['scores'])
        total, vector = pooled.get(c['predictor_month'], (0, np.zeros(6)))
        pooled[c['predictor_month']] = (total+weight, vector+weight*values)
    result = []
    for m in months:
        row = dict(m)
        weight, vector = pooled.get(m['predictor_month'], (0, np.zeros(6)))
        if weight != int(m['nlp_ready_weight']):
            raise ValueError('Monthly eligible weight does not reconcile')
        row['scored_weight'] = weight
        row['scored_weight_coverage'] = weight/int(m['total_weight']) if int(m['total_weight']) else 0
        row['feature_eligible'] = m['preparation_eligible'] in (True, 'True') and weight > 0
        row['emotion_status'] = 'scored' if weight else 'no_eligible_lyrics'
        row.update({e: float(vector[i]/weight) if weight else '' for i, e in enumerate(EMOTIONS)})
        result.append(row)
    return result


def fit_extended(training, features, config):
    """Seven-predictor Probit; training-only scaling and multivariate separation check."""
    try:
        return _fit_extended(training, features, config)
    except BaselineError:
        raise
    except (KeyError, ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise BaselineError('optimization_failure', str(exc)) from exc


def _fit_extended(training, features, config):
    validate_observations(training, require_labels=True)
    if len(training) < config.min_training_months:
        raise BaselineError('insufficient_training', 'Too few eligible training months')
    x = np.asarray([[r.spread]+[features[r.predictor_month][e] for e in EMOTIONS]
                    for r in training], dtype=float)
    y = np.asarray([r.target for r in training], dtype=float)
    if not np.isfinite(x).all():
        raise BaselineError('missing_input', 'Nonfinite training features')
    if np.any((x[:,1:] < 0) | (x[:,1:] > 1)):
        raise BaselineError('invalid_features', 'Emotion probabilities outside [0,1]')
    if len(np.unique(y)) != 2:
        raise BaselineError('single_class', 'Both training classes are required')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    if np.any(scale <= 1e-12):
        raise BaselineError('constant_predictor', 'Constant or numerically constant training feature')
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    condition = float(np.linalg.cond(design))
    if np.linalg.matrix_rank(design) < 8 or condition > config.max_condition_number:
        raise BaselineError('ill_conditioned', 'Rank-deficient or ill-conditioned Model B design')
    # Find any nonzero nonnegative signed margins with L1-bounded coefficients.
    # Positive optimum detects complete OR quasi-complete separation.
    signed = (2*y-1)[:, None]*design
    objective = signed.sum(axis=0)
    lp = linprog(np.r_[-objective, objective],
        A_ub=np.vstack([np.column_stack((-signed, signed)), np.ones((1, 16))]),
        b_ub=np.r_[np.zeros(len(x)), 1.], bounds=(0, None), method='highs')
    if not lp.success:
        raise BaselineError('separation_check_failure', 'Could not establish nonseparation')
    if -lp.fun > 1e-7:
        raise BaselineError('separation', 'Complete or quasi-complete multivariate separation')
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = Probit(y, design, missing='raise', check_rank=True)
            fitted = model.fit(start_params=np.zeros(8), method=config.optimizer,
                maxiter=config.maxiter, tol=config.tolerance, disp=False,
                full_output=True, warn_convergence=True)
    except PerfectSeparationError as exc:
        raise BaselineError('separation', str(exc)) from exc
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise BaselineError('optimization_failure', str(exc)) from exc
    diagnostics = dict(converged=bool(fitted.mle_retvals.get('converged')),
        iterations=int(fitted.mle_retvals.get('iterations', 0)),
        optimizer_warnflag=int(fitted.mle_retvals.get('warnflag', 0)),
        design_condition_number=condition,
        warnings=[{'category':w.category.__name__, 'message':str(w.message)} for w in caught])
    if any(issubclass(w.category, PerfectSeparationWarning) for w in caught):
        raise BaselineError('separation', 'Optimizer reported separation', diagnostics)
    if not diagnostics['converged'] or diagnostics['optimizer_warnflag']:
        raise BaselineError('nonconvergence', 'Model B did not converge', diagnostics)
    if caught:
        raise BaselineError('numerical_failure', 'Optimizer warning; no automatic fallback', diagnostics)
    params = np.asarray(fitted.params)
    information = -np.asarray(model.hessian(params))
    gradient = np.asarray(model.score(params))/len(x)
    if not all(np.isfinite(v).all() for v in (params, fitted.bse, information, gradient, fitted.llf)):
        raise BaselineError('numerical_failure', 'Nonfinite fit diagnostics', diagnostics)
    info_condition = float(np.linalg.cond(information))
    diagnostics.update(log_likelihood=float(fitted.llf),
        max_abs_average_score=float(np.abs(gradient).max()),
        information_condition_number=info_condition,
        information_min_eigenvalue=float(np.linalg.eigvalsh(information).min()),
        separation_check='bounded signed-margin linear program')
    if diagnostics['information_min_eigenvalue'] <= 0 or info_condition > config.max_condition_number:
        raise BaselineError('ill_conditioned', 'Invalid Model B information matrix', diagnostics)
    if diagnostics['max_abs_average_score'] > config.max_average_score:
        raise BaselineError('nonconvergence', 'Gradient exceeds tolerance', diagnostics)
    original_slopes = params[1:]/scale
    original_params = np.r_[params[0]-original_slopes@mean, original_slopes]
    report = dict(coefficients=dict(zip(['intercept']+FEATURES, original_params.tolist())),
        standardized_coefficients=dict(zip(['intercept']+FEATURES, params.tolist())),
        standardized_standard_errors=np.asarray(fitted.bse).tolist(),
        training_mean=dict(zip(FEATURES, mean.tolist())),
        training_scale=dict(zip(FEATURES, scale.tolist())), diagnostics=diagnostics)
    return dict(params=params, mean=mean, scale=scale, report=report)


def extended_probability(fit, spread, emotions):
    values = np.r_[spread, validate_scores(emotions)]
    if not np.isfinite(values).all():
        raise BaselineError('missing_input', 'Nonfinite forecast features')
    probability = float(ndtr(np.r_[1., (values-fit['mean'])/fit['scale']]@fit['params']))
    if not np.isfinite(probability):
        raise BaselineError('prediction_failure', 'Nonfinite Model B probability')
    return probability


def evaluate(economic, months, config):
    """Identical eligible predictor rows and labels for both expanding-window fits."""
    validate_observations(economic)
    econ = {r.predictor_month: r for r in economic}
    features = {}
    for m in months:
        t = date.fromisoformat(m['predictor_month'])
        if t in features or not date(1973,1,1) <= t <= date(2023,12,1):
            raise ValueError('Duplicate or out-of-study feature month')
        if t not in econ or m['target_month'] != add_months(t,6).isoformat():
            raise ValueError('Economic/feature calendar mismatch')
        if float(m['spread_t']) != econ[t].spread or int(m['target_t']) != econ[t].target:
            raise ValueError('Economic values or labels differ between models')
        features[t] = m
    ready = {t for t,m in features.items() if m['feature_eligible'] in (True, 'True')}
    for t in ready:
        validate_scores({e:features[t][e] for e in EMOTIONS})
    common_rows = [r for r in economic if r.predictor_month in ready]
    paired_config = replace(config, training_start='1973-01-01')
    records = []
    for t,m in sorted(features.items()):
        record = dict(predictor_month=t.isoformat(), target_month=m['target_month'],
            forecast_cutoff=add_months(t,1).replace(day=15).isoformat(), actual_label=econ[t].target,
            status='ineligible_features', error_code='', error_message='',
            model_a_probability=None, model_b_probability=None, training_count=0,
            training_start='', training_end='', training_target_end='',
            training_recession_count=None, training_months_sha256='',
            model_a_fit_json='', model_b_fit_json='', diagnostics_json='')
        if t in ready:
            try:
                training = select_training(common_rows, t, paired_config)
                record.update(training_count=len(training),
                    training_start=training[0].predictor_month.isoformat(),
                    training_end=training[-1].predictor_month.isoformat(),
                    training_target_end=training[-1].target_month.isoformat(),
                    training_recession_count=sum(r.target for r in training),
                    training_months_sha256=sha('\n'.join(r.predictor_month.isoformat() for r in training).encode()))
                a = fit_probit(training, paired_config)
                record['model_a_probability'] = float(a.predict_probability([econ[t].spread])[0])
                record['model_a_fit_json'] = _diagnostic_json(a.report)
                b = fit_extended(training, features, paired_config)
                record['model_b_probability'] = extended_probability(b, econ[t].spread,
                    {e:m[e] for e in EMOTIONS})
                record['model_b_fit_json'] = _diagnostic_json(b['report'])
                record['status'] = 'ok'
            except BaselineError as exc:
                record.update(status='warmup' if exc.code=='insufficient_training' else 'failed',
                    error_code=exc.code, error_message=str(exc), diagnostics_json=_diagnostic_json(exc.diagnostics))
        records.append(record)
    paired = [r for r in records if r['status']=='ok']
    metrics = dict(status='complete' if paired else 'blocked_no_common_forecasts',
        analysis_months=len(records), eligible_feature_months=len(ready), evaluated_months=len(paired),
        forecast_start=paired[0]['predictor_month'] if paired else None,
        forecast_end=paired[-1]['predictor_month'] if paired else None,
        target_start=paired[0]['target_month'] if paired else None,
        target_end=paired[-1]['target_month'] if paired else None,
        recession_months=sum(r['actual_label'] for r in paired),
        fit_failures=[r['predictor_month'] for r in records if r['status']=='failed'],
        model_a=None, model_b=None, brier_difference_b_minus_a=None,
        comparison='Both models refit on the same eligible training observations; metrics use paired successful forecasts only.',
        timing='Retrospective: cutoff 15th of t+1; target t+6; training targets <= t-1; at least 240 eligible months from January 1973.')
    if paired:
        actual = [r['actual_label'] for r in paired]
        for model in ('a','b'):
            p = [r['model_'+model+'_probability'] for r in paired]
            metrics['model_'+model] = dict(brier_score=brier_score(actual,p), roc_auc=roc_auc(actual,p),
                calibration=calibration_table(actual,p))
        metrics['brier_difference_b_minus_a'] = metrics['model_b']['brier_score']-metrics['model_a']['brier_score']
    if metrics['fit_failures']:
        metrics['status'] = 'incomplete_fit_failures'
    return records, metrics


def plots(months, records, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = [date.fromisoformat(m['predictor_month']) for m in months]
    fig, ax = plt.subplots(figsize=(12,4), layout='constrained')
    ax.plot(times, [float(m['provisional_weight_coverage']) for m in months], label='Provisional selected OSF text')
    ax.plot(times, [float(m['scored_weight_coverage']) for m in months], label='Eligible scored text')
    ax.set(title='Monthly Billboard popularity-weight coverage', ylabel='Share of all observed rank weight', ylim=(-.02,1.02))
    ax.legend()
    fig.savefig(output/'lyric_coverage.png', dpi=160)
    plt.close(fig)
    if not any(r['status']=='ok' for r in records):
        return  # Never draw fictional model predictions for an empty evaluation.
    fig, ax = plt.subplots(figsize=(13,4), layout='constrained')
    targets = [date.fromisoformat(r['target_month']) for r in records]
    for r,t in zip(records, targets):
        if r['actual_label']==1:
            ax.axvspan(t,add_months(t,1),color='gray',alpha=.2,lw=0)
    for model in ('a','b'):
        ax.plot(targets,[r['model_'+model+'_probability'] if r['status']=='ok' else np.nan
                        for r in records],label='Model '+model.upper())
    ax.set(title='Paired retrospective six-month-ahead recession predictions',
           xlabel='Target month; shading = actual recession',ylabel='Probability',ylim=(0,1))
    ax.legend()
    fig.savefig(output/'probability_comparison.png',dpi=180)
    plt.close(fig)


def run(root, output, cache):
    root, output, cache = root.resolve(), output.resolve(), cache.resolve()
    if output.exists():
        raise ValueError('Choose a fresh output directory')
    config_path = root/'config/music_model.json'
    config = json.loads(config_path.read_text())
    validate_config(config)
    prepared = output/'preparation'
    prep_summary = music_preparation.build(root, root/'reports/lyrics_validation/run_002', prepared)
    mapping = list(read_csv(prepared/'song_source_mapping.csv'))
    months = list(read_csv(prepared/'monthly_preparation.csv'))
    contributions = list(read_csv(prepared/'monthly_contributions.csv'))
    scores = score_eligible(root, mapping, config, cache)
    monthly = aggregate(months, contributions, scores)
    baseline = BaselineConfig.load(root/'config/baseline.json')
    records, metrics = evaluate(load_economic(root/'data/processed/economic_monthly.csv'), monthly, baseline)
    if not prep_summary['nlp_ready_identities']:
        metrics['status'] = 'blocked_no_policy_eligible_lyrics'
    metrics.update(scored_song_count=len(scores), preparation=prep_summary,
        model_config=config, missing_evidence=['recording identity', 'English language',
            'source-use permission', 'recording release evidence'] if not scores else [],
        inference_executed=any(not s['cache_hit'] for s in scores.values()),
        emotion_cache_hits=sum(s['cache_hit'] for s in scores.values()),
        limitations=['Final FRED/NBER values contain hindsight.',
            'Chart issue dates are only a retrospective publication convention.',
            'A modern pretrained Reddit emotion model was not available at historical forecast origins.',
            'Emotion classification and long-text pooling need lyric-domain validation; nervousness is a weak pretrained label.',
            'Missing lyrics are nonrandom; no significance claim follows from point estimates.'])
    write_csv(output/'monthly_features_and_coverage.csv',monthly)
    write_csv(output/'predictions.csv',records)
    calibration=[]
    for model in ('a','b'):
        if metrics['model_'+model]:
            calibration.extend(dict(model=model.upper(),**b) for b in metrics['model_'+model]['calibration'])
    write_csv(output/'calibration.csv',calibration, ['model','bin_lower','bin_upper','upper_inclusive',
        'count','mean_predicted_probability','observed_recession_frequency'])
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    plots(monthly, records, output)
    provenance = dict(config_sha256=file_hash(config_path),
        preparation_manifest_sha256=file_hash(prepared/'manifest.json'),
        software=dict(software_versions(), python=platform.python_version(),
            **{p:importlib.metadata.version(p) for p in ('scipy','statsmodels','matplotlib')}),
        code_sha256={name:file_hash(Path(__file__).with_name(name)) for name in
            ['music_model.py','emotions.py','music_preparation.py','baseline.py','backtest.py']},
        cache_keys={k:v['cache_key'] for k,v in scores.items()},
        outputs_sha256={p.name:file_hash(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return metrics


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,default=Path('data/processed/emotion_cache'))
    args=parser.parse_args(argv)
    try:
        result=run(args.root,args.output_dir,args.cache_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1,'Model comparison aborted: '+str(exc)+'\n')
    print(json.dumps({k:result[k] for k in ['status','scored_song_count','eligible_feature_months',
        'evaluated_months','forecast_start','forecast_end','model_a','model_b']},indent=2))
    return 0 if result['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
