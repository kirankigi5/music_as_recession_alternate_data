"""Local, evidence-gated GoEmotions inference. No lyric text is written or sent remotely."""
import importlib.metadata
import json
import re
from pathlib import Path

import numpy as np

from .lyrics_audit import file_hash, read_csv, sha
from .music_preparation import EMOTIONS


def validate_config(config):
    fixed = dict(model_id='SamLowe/roberta-base-go_emotions',
        chunk_pooling='token_weighted_mean_sigmoid', device='cpu', emotions=EMOTIONS,
        model_b_standardization='training_only_population_std',
        analysis_start='1973-01-01', analysis_end='2023-12-01')
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError('Unsupported music-model configuration')
    if not re.fullmatch(r'[0-9a-f]{40}', config.get('revision', '')):
        raise ValueError('Pin a full model commit, never a floating branch')
    if type(config.get('chunk_tokens')) is not int or not 1 <= config['chunk_tokens'] <= 510:
        raise ValueError('Invalid token chunk length')


def validate_scores(scores):
    if set(scores) != set(EMOTIONS):
        raise ValueError('Expected exactly the six study emotions')
    values = np.asarray([scores[e] for e in EMOTIONS], dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Emotion scores must be finite probabilities')
    return values


def require_eligible(row):
    # CSV values are strings; bool('False') must never authorize inference.
    if (row.get('nlp_ready') not in (True, 'True') or
        row.get('recording_verified') not in (True, 'True') or
        row.get('language') != 'en' or row.get('source') != 'osf' or
        row.get('mapping_status') != 'provisional_selected' or
        row.get('exclusion_reasons') or row.get('readiness_blockers')):
        raise ValueError('Source is not eligible under the primary analysis policy')
    for field in ('song_id', 'source_reference_id', 'content_sha256', 'reviewer',
                  'recording_evidence', 'language_evidence', 'permission_evidence',
                  'release_date', 'release_evidence'):
        if not row.get(field, '').strip():
            raise ValueError('Missing eligibility evidence: '+field)


def software_versions():
    result = {}
    for name in ('numpy', 'torch', 'transformers', 'safetensors'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not_installed'
    return result


class GoEmotions:
    """Lazy CPU model; downloaded weights are pinned and text remains local."""
    def __init__(self, config, model_cache):
        validate_config(config)
        self.config, self.model_cache = config, model_cache
        self.model = self.tokenizer = self.torch = None

    def __call__(self, text):
        if self.model is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError('Eligible lyrics found; install the optional [music] dependencies') from exc
            torch.manual_seed(0)
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            common = dict(revision=self.config['revision'], cache_dir=str(self.model_cache),
                          trust_remote_code=False)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_id'], **common)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config['model_id'], use_safetensors=True, **common).cpu().eval()
            self.torch = torch
        labels = {label: int(i) for i, label in self.model.config.id2label.items()}
        if not set(EMOTIONS) <= set(labels) or self.model.config.problem_type != 'multi_label_classification':
            raise ValueError('Unexpected model labels or objective')
        tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not tokens:
            raise ValueError('Empty tokenized lyrics')
        size = self.config['chunk_tokens']
        weighted = np.zeros(len(EMOTIONS))
        chunks = 0
        for start in range(0, len(tokens), size):
            chunk = tokens[start:start+size]
            encoded = self.tokenizer.prepare_for_model(chunk, add_special_tokens=True,
                return_attention_mask=True, return_tensors='pt', truncation=False)
            with self.torch.inference_mode():
                probabilities = self.torch.sigmoid(self.model(**encoded).logits)[0].cpu().numpy()
            weighted += len(chunk)*probabilities[[labels[e] for e in EMOTIONS]]
            chunks += 1
        return dict(scores=dict(zip(EMOTIONS, (weighted/len(tokens)).tolist())),
                    token_count=len(tokens), chunk_count=chunks)


def cached_score(text, row, config, directory, scorer):
    require_eligible(row)
    validate_config(config)
    if sha(text.encode('utf-8')) != row['content_sha256']:
        raise ValueError('Lyric content differs from the reviewed hash')
    metadata = {k: row[k] for k in ('song_id', 'source_reference_id', 'content_sha256')}
    metadata.update(config=config, implementation_sha256=file_hash(Path(__file__)),
                    software=software_versions())
    key = sha(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode())
    path = directory/(key+'.json')
    if path.exists():
        result = json.loads(path.read_text())
        if result['metadata'] != metadata:
            raise ValueError('Stale or mismatched emotion cache')
        validate_scores(result['scores'])
        return dict(result, cache_hit=True)
    inferred = scorer(text)
    validate_scores(inferred['scores'])
    # Explicit fields only: never cache text or arbitrary model responses.
    result = dict(cache_key=key, metadata=metadata, scores=inferred['scores'],
                  token_count=int(inferred['token_count']), chunk_count=int(inferred['chunk_count']))
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)
    return dict(result, cache_hit=False)


def score_eligible(root, mapping, config, cache, scorer=None):
    """Stream only reviewed OSF locators; each selected song/source is scored once."""
    eligible = [r for r in mapping if r['nlp_ready'] in (True, 'True')]
    if not eligible:
        return {}  # Do not import torch or download a model for an empty corpus.
    grouped = {}
    for row in eligible:
        require_eligible(row)
        path = (root/row['source_path']).resolve()
        if (root/'data/raw/lyrics').resolve() not in path.parents:
            raise ValueError('Reviewed text must remain within the local raw lyrics directory')
        grouped.setdefault(path, {}).setdefault(int(row['record_number']), []).append(row)
    scorer = scorer or GoEmotions(config, root/'data/raw/models')
    results = {}
    for path, wanted in grouped.items():
        for number, raw in enumerate(read_csv(path), 1):
            for row in wanted.pop(number, []):
                if (raw['Song Title'], raw['Artist'], raw['Date'], raw['Rank']) != (
                    row['source_title'], row['source_artist'], row['source_chart_date'], row['source_chart_rank']):
                    raise ValueError('Reviewed source locator no longer matches')
                results[row['song_id']] = cached_score(raw['lyrics_raw'], row, config, cache, scorer)
        if wanted:
            raise ValueError('Reviewed source record is absent')
    return results
