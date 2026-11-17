"""Synthetic integration and failure tests; no model download or real lyric inference."""
import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr

from music_recession.baseline import BaselineConfig, BaselineError, Observation, fit_probit, select_training
from music_recession.economic import add_months
from music_recession.emotions import GoEmotions, cached_score, require_eligible, score_eligible, validate_config
from music_recession.lyrics_audit import sha, write_csv
from music_recession.music_model import aggregate, evaluate, extended_probability, fit_extended, plots
from music_recession.music_preparation import EMOTIONS

ROOT = Path(__file__).resolve().parents[1]
TEXT = 'Synthetic fixture text created for testing; it is not a song lyric.'


class EmotionTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'config/music_model.json').read_text())
        self.row=dict(song_id='fixture', source_reference_id='source', content_sha256=sha(TEXT.encode()),
            nlp_ready='True', recording_verified='True', language='en', source='osf',
            mapping_status='provisional_selected', exclusion_reasons='', readiness_blockers='',
            reviewer='Synthetic reviewer', recording_evidence='fixture evidence',
            language_evidence='fixture English', permission_evidence='fixture permission',
            release_date='1972-01-01', release_evidence='fixture date')

    def test_cache_scored_once_no_text_and_identity_isolation(self):
        scorer=Mock(return_value=dict(scores={e:.25 for e in EMOTIONS},token_count=12,chunk_count=1))
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)
            a=cached_score(TEXT,self.row,self.config,cache,scorer)
            b=cached_score(TEXT,self.row,self.config,cache,scorer)
            self.assertEqual(a['scores'],b['scores'])
            self.assertFalse(a['cache_hit'])
            self.assertTrue(b['cache_hit'])
            self.assertEqual(scorer.call_count,1)
            cached_score(TEXT,dict(self.row,source_reference_id='different'),self.config,cache,scorer)
            self.assertEqual(scorer.call_count,2)
            for p in cache.iterdir():
                self.assertNotIn(TEXT,p.read_text())
            path=cache/(a['cache_key']+'.json')
            invalid=json.loads(path.read_text()); invalid['scores']['joy']=2
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                cached_score(TEXT,self.row,self.config,cache,scorer)

    def test_gate_before_inference_and_content_integrity(self):
        scorer=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            for change in ({'nlp_ready':'False'},{'permission_evidence':''},{'language':'es'},
                           {'source':'LRCLIB'},{'recording_verified':'False'},
                           {'content_sha256':'different'}):
                with self.subTest(change=change),self.assertRaises(ValueError):
                    cached_score(TEXT,dict(self.row,**change),self.config,Path(tmp),scorer)
            self.assertFalse(scorer.called)
        self.assertEqual(score_eligible(ROOT,[dict(self.row,nlp_ready='False')],
                                       self.config,Path('unused'),scorer),{})
        self.assertFalse(scorer.called)
        with self.assertRaises(ValueError): validate_config(dict(self.config,revision='main'))

    def test_token_chunk_pooling_and_label_order_with_synthetic_backend(self):
        scorer=GoEmotions(dict(self.config,chunk_tokens=2),Path('unused'))
        labels=list(reversed(EMOTIONS))
        scorer.tokenizer=SimpleNamespace(encode=lambda *a,**k:[1,2,3],
            prepare_for_model=lambda chunk,**k:{'ids':chunk})
        def logits(ids):
            return SimpleNamespace(logits=np.array([[float(len(ids))]*6]))
        scorer.model=Mock(side_effect=logits)
        scorer.model.config=SimpleNamespace(id2label=dict(enumerate(labels)),problem_type='multi_label_classification')
        class Tensor:
            def __init__(self,array): self.array=array
            def __getitem__(self,i): return Tensor(self.array[i])
            def cpu(self): return self
            def numpy(self): return self.array
        scorer.torch=SimpleNamespace(inference_mode=nullcontext,
            sigmoid=lambda a:Tensor(1/(1+np.exp(-a))))
        result=scorer(TEXT)
        self.assertEqual(result['token_count'],3)
        self.assertEqual(result['chunk_count'],2)
        self.assertAlmostEqual(result['scores']['joy'],(2/(1+np.exp(-2))+1/(1+np.exp(-1)))/3)

    def test_local_locator_replay_and_cached_scoring_integration(self):
        scorer=Mock(return_value=dict(scores={e:.25 for e in EMOTIONS},token_count=12,chunk_count=1))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'data/raw/lyrics/fixture.csv'
            raw.parent.mkdir(parents=True)
            write_csv(raw,[dict(Date='1973-01-06',Rank='1',Artist='Fixture',
                               **{'Song Title':'Synthetic Song','lyrics_raw':TEXT})])
            row=dict(self.row,source_path='data/raw/lyrics/fixture.csv',record_number='1',
                     source_title='Synthetic Song',source_artist='Fixture',
                     source_chart_date='1973-01-06',source_chart_rank='1')
            first=score_eligible(root,[row],self.config,root/'cache',scorer)
            second=score_eligible(root,[row],self.config,root/'cache',scorer)
            self.assertEqual(first['fixture']['scores'],second['fixture']['scores'])
            self.assertEqual(scorer.call_count,1)
            with self.assertRaises(ValueError):
                score_eligible(root,[dict(row,source_title='Wrong version')],self.config,root/'cache',scorer)

    def test_aggregation_repeated_weeks_and_provenance(self):
        month=dict(predictor_month='1973-01-01',preparation_eligible='True',
                   nlp_ready_weight='150',total_weight='200')
        contributions=[dict(predictor_month='1973-01-01',song_id=s,source_reference_id=s,
                            nlp_ready_weight=w,known_weight=w,total_weight=w)
                       for s,w in [('one',100),('two',50)]]
        scores={s:dict(metadata={'source_reference_id':s}, scores={e:p for e in EMOTIONS})
                for s,p in [('one',.2),('two',.8)]}
        row=aggregate([month],contributions,scores)[0]
        self.assertAlmostEqual(row['joy'],.4)
        self.assertEqual(row['scored_weight_coverage'],.75)
        with self.assertRaises(ValueError): aggregate([month],contributions+contributions[:1],scores)
        with self.assertRaises(ValueError): aggregate([month],contributions,{'one':scores['one']})


class PairedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(20261018)
        x=rng.normal(size=250)
        emotions=rng.uniform(.05,.95,size=(250,6))
        y=(rng.uniform(size=250)<ndtr(-.5-.8*x+1.0*(emotions[:,0]-.5))).astype(int)
        cls.rows=[]; cls.months=[]
        for i in range(250):
            t=add_months(date(1973,1,1),i)
            cls.rows.append(Observation(t,add_months(t,6),float(x[i]),int(y[i])))
            cls.months.append(dict(predictor_month=t.isoformat(),target_month=add_months(t,6).isoformat(),
                spread_t=float(x[i]),target_t=int(y[i]),feature_eligible=True,
                **dict(zip(EMOTIONS,emotions[i].tolist()))))
        cls.features={r.predictor_month:m for r,m in zip(cls.rows,cls.months)}
        cls.config=replace(BaselineConfig(),training_start='1973-01-01')
        cls.records,cls.metrics=evaluate(cls.rows,cls.months,cls.config)

    def test_paired_calendar_training_and_baseline_reuse(self):
        ok=[r for r in self.records if r['status']=='ok']
        self.assertEqual(len(ok),4)
        first=ok[0]
        self.assertEqual(first['predictor_month'],'1993-07-01')
        self.assertEqual(first['target_month'],'1994-01-01')
        self.assertEqual(first['training_count'],240)
        self.assertEqual(first['training_end'],'1992-12-01')
        self.assertEqual(first['training_target_end'],'1993-06-01')
        a=fit_probit(self.rows[:240],self.config)
        self.assertEqual(first['model_a_probability'],float(a.predict_probability([self.rows[246].spread])[0]))
        b=json.loads(first['model_b_fit_json'])
        self.assertLess(b['coefficients']['spread_t'],0)
        self.assertEqual(self.metrics['evaluated_months'],4)
        for model in ('a','b'):
            self.assertEqual(sum(b['count'] for b in self.metrics['model_'+model]['calibration']),4)
            self.assertAlmostEqual(self.metrics['model_'+model]['brier_score'],
                sum((r['model_'+model+'_probability']-r['actual_label'])**2 for r in ok)/4)

    def test_future_data_and_current_target_do_not_change_first_fit(self):
        rows=[replace(r,target=1-r.target) if r.target_month>date(1993,6,1) else r for r in self.rows]
        months=[dict(m,target_t=r.target) for m,r in zip(self.months,rows)]
        for i in range(247,250):
            months[i]=dict(months[i],joy=.99)
        changed,_=evaluate(rows,months,self.config)
        for key in ('model_a_probability','model_b_probability','model_a_fit_json',
                    'model_b_fit_json','training_months_sha256'):
            self.assertEqual(self.records[246][key],changed[246][key])

    def test_training_only_scaling_and_reproducibility(self):
        a=fit_extended(self.rows[:240],self.features,self.config)
        b=fit_extended(self.rows[:240],self.features,self.config)
        self.assertEqual(a['report'],b['report'])
        np.testing.assert_allclose(a['mean'][0],np.mean([r.spread for r in self.rows[:240]]))
        p=extended_probability(a,self.rows[246].spread,{e:self.months[246][e] for e in EMOTIONS})
        self.assertTrue(0<=p<=1)

    def test_missing_feature_month_delays_warmup(self):
        months=[dict(m,feature_eligible=False) if i==0 else m for i,m in enumerate(self.months)]
        records,metrics=evaluate(self.rows,months,self.config)
        first=next(r for r in records if r['status']=='ok')
        self.assertEqual(first['predictor_month'],'1993-08-01')
        self.assertEqual(first['training_count'],240)
        self.assertEqual(first['training_start'],'1973-02-01')

    def test_invalid_data_separation_and_optimizer_failure(self):
        constant={t:dict(m,joy=.5) for t,m in self.features.items()}
        invalid={t:dict(m,joy=float('nan')) for t,m in self.features.items()}
        cases=[(self.rows[:240],constant,'constant_predictor'),
               (self.rows[:240],invalid,'missing_input'),
               ([replace(r,target=0) for r in self.rows[:240]],self.features,'single_class'),
               ([replace(r,target=int(r.spread>0)) for r in self.rows[:240]],self.features,'separation')]
        for rows,features,code in cases:
            with self.subTest(code=code),self.assertRaises(BaselineError) as exc:
                fit_extended(rows,features,self.config)
            self.assertEqual(exc.exception.code,code)
        with self.assertRaises(BaselineError) as exc:
            fit_extended(self.rows[:240],self.features,replace(self.config,maxiter=1))
        self.assertEqual(exc.exception.code,'nonconvergence')
        with self.assertRaises(ValueError):
            evaluate(self.rows,[dict(self.months[0],target_month='1973-08-01')]+self.months[1:],self.config)

    def test_failures_and_empty_corpus_never_produce_comparison_scores(self):
        with patch('music_recession.music_model.fit_extended',side_effect=BaselineError('separation','synthetic')):
            records,metrics=evaluate(self.rows,self.months,self.config)
        self.assertEqual(metrics['evaluated_months'],0)
        self.assertEqual(len(metrics['fit_failures']),4)
        self.assertIsNone(metrics['model_a'])
        self.assertIsNone(metrics['model_b'])
        empty=[dict(m,feature_eligible=False) for m in self.months]
        with patch('music_recession.music_model.fit_probit') as a,patch('music_recession.music_model.fit_extended') as b:
            records,metrics=evaluate(self.rows,empty,self.config)
        self.assertFalse(a.called or b.called)
        self.assertEqual(metrics['evaluated_months'],0)
        self.assertTrue(all(r['model_b_probability'] is None for r in records))

    def test_comparison_plot_uses_actual_paired_predictions(self):
        months=[dict(m,provisional_weight_coverage=.95,scored_weight_coverage=.91) for m in self.months]
        with tempfile.TemporaryDirectory() as temp:
            plots(months,self.records,Path(temp))
            self.assertGreater((Path(temp)/'probability_comparison.png').stat().st_size,100)


if __name__=='__main__':
    unittest.main()
