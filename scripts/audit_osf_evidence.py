"""Audit OSF candidates without promoting automated evidence to approval.

Install langid==1.1.6 into data/raw/osf_source_review/python_deps first.
Raw text is read locally, never exported. No source-use or release approval is inferred.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date
import importlib.metadata
import json
from pathlib import Path
import re
import sys

from music_recession.lyrics_audit import candidate_key, file_hash, identity, read_csv, sha, write_csv

VERSION = re.compile(r'\b(remix|mix|live|acoustic|remaster(?:ed)?|version|edit|part|pt)\b', re.I)
INSTRUMENTAL = re.compile(r'^\s*[\[(]?\s*(instrumental|instrumental only|no lyrics|no vocals)\s*[\])]?\s*$',re.I)
BLOCKERS = ('source_use_decision_missing;recording_evidence_not_independent;'
            'language_automated_not_reviewed;release_date_not_verified')


def tally(values):
    """Counts with key order fixed by name, so the summary never depends on row order."""
    return dict(sorted(Counter(values).items()))


def screen_language(text, classify):
    """Whole text plus complete 100-word blocks; probabilities are uncalibrated."""
    cleaned=re.sub(r'\[[^\]]*\]', ' ', text)
    words=cleaned.split()
    lang, confidence=classify(cleaned)
    blocks=[]
    # A final short block is retained with the previous block, not discarded.
    for start in range(0,len(words),100):
        if len(words)-start<40 and blocks:
            blocks[-1]+=' '+' '.join(words[start:])
        else:
            blocks.append(' '.join(words[start:start+100]))
    results=[(language,float(probability),len(block.split()))
             for block in blocks for language,probability in [classify(block)]]
    non_en=sum(n for language,probability,n in results if language!='en' and probability>=.95)
    english=sum(n for language,probability,n in results if language=='en' and probability>=.95)
    fraction=english/len(words) if words else 0
    if len(words)<40:
        status='too_short_for_language_screen'
    elif lang!='en' and confidence>=.95:
        status='non_english_likely'
    elif non_en:
        status='mixed_or_non_english_block_review'
    elif lang=='en' and confidence>=.95 and fraction>=.9:
        status='english_automated_consistent'
    else:
        status='language_uncertain'
    return dict(language_detected=lang,language_confidence=float(confidence),language_status=status,
        language_word_count=len(words),english_high_confidence_word_fraction=fraction,
        non_english_high_confidence_words=non_en,
        block_language_counts=json.dumps(dict(Counter(x[0] for x in results)),sort_keys=True),
        language_verified=False)


def instrumental_status(text):
    if not text.strip() or text.strip().lower() in ('nan','none','null'):
        return 'no_text_cannot_determine_instrumental'
    if INSTRUMENTAL.fullmatch(text):
        return 'instrumental_marker_only'
    if re.search(r'\binstrumental\b',text,re.I):
        return 'instrumental_marker_with_other_text_review'
    words=re.findall(r'\b[^\W\d_]+\b',text.lower())
    if len(set(words))<8:
        return 'very_low_lexical_diversity_review'
    return 'no_instrumental_marker_detected'


def version_terms(title):
    # "Live Your Life" and "Part of Me" are titles, not version annotations.
    qualifiers=' '.join(re.findall(r'\([^)]*\)|\[[^]]*\]|\s[-–—]\s.*$',title))
    terms={m.lower() for m in VERSION.findall(qualifiers)}
    if re.search(r'\bremix\s*$',title,re.I):terms.add('remix')
    if re.search(r'\b(?:part|pt\.?)\s+(?:one|two|three|four|\d+|[ivx]+)\b',title,re.I):terms.add('part')
    return ';'.join(sorted(terms))


def require(path, description):
    """Fail with the missing input's identity instead of a bare OSError."""
    if not path.exists():
        raise ValueError('Missing required input ('+description+'): '+str(path))
    return path


def run(root, output, mapping_path):
    root=root.resolve();output=output.resolve()
    if output.exists():raise ValueError('Choose a fresh output directory')
    # Every input is addressed relative to --root so the audit is path independent.
    mapping_path=mapping_path if mapping_path.is_absolute() else (root/mapping_path)
    mapping_path=mapping_path.resolve()
    if root not in mapping_path.parents:
        raise ValueError('Mapping must live under --root so provenance stays relative')
    deps=root/'data/raw/osf_source_review/python_deps'
    require(deps,'isolated langid==1.1.6 install')
    sys.path.insert(0,str(deps))
    from langid.langid import LanguageIdentifier, model
    detector=LanguageIdentifier.from_modelstring(model,norm_probs=True)
    mapping=list(read_csv(require(mapping_path,'song source mapping')))
    selected={int(r['record_number']):r for r in mapping if r['mapping_status']=='provisional_selected'}
    if len(selected)!=sum(r['mapping_status']=='provisional_selected' for r in mapping):
        raise ValueError('Selected source locator collision')
    if not selected:raise ValueError('Mapping contains no provisional_selected records to audit')
    source=require(root/'data/raw/lyrics/osf_lyrics.csv','local OSF corpus')
    remote=json.loads(require(root/'reports/osf_source_review/files.json','OSF storage listing').read_text())
    published=[x for x in remote['data'] if x['attributes']['guid']=='3tcq2']
    if len(published)!=1:
        raise ValueError('Expected exactly one OSF file with guid 3tcq2; found '+str(len(published)))
    published=published[0]
    local_hash=file_hash(source)
    if local_hash!=published['attributes']['extra']['hashes']['sha256']:
        raise ValueError('Local corpus differs from the published OSF file hash')
    first_all={};first_study={};slots={};charts=[]
    for r in read_csv(require(root/'billboard_hot100_all.csv','Billboard chart file')):
        key=identity(r['song'],r['artist']);d=r['date']
        first_all[key]=min(first_all.get(key,d),d)
        if '1973-01-01'<=d<='2023-12-31':
            first_study[key]=min(first_study.get(key,d),d)
            slot=(d,int(r['this_week']))
            if slot in slots:raise ValueError('Duplicate chart slot')
            slots[slot]=key;charts.append(r)
    evidence=[];languages={};all_source=set();nonempty_source=set();unique_text=set();row_count=0
    for number, raw in enumerate(read_csv(source),1):
        row_count+=1
        source_key=(raw['Song Title'],raw['Artist']);all_source.add(source_key)
        text=raw['lyrics_raw']
        if text.strip().lower() not in ('','nan','none','null'):
            nonempty_source.add(source_key);unique_text.add(sha(text.encode()))
        if number not in selected:continue
        r=selected.pop(number)
        if (raw['Song Title'],raw['Artist'],raw['Date'],raw['Rank'],sha(text.encode())) != (
            r['source_title'],r['source_artist'],r['source_chart_date'],r['source_chart_rank'],r['content_sha256']):
            raise ValueError('Source replay failed: '+r['song_id'])
        key=identity(r['song'],r['artist'])
        metadata_equal=candidate_key(r['song'],r['artist'])==candidate_key(raw['Song Title'],raw['Artist'])
        chart_equal=slots.get((raw['Date'],int(raw['Rank'])))==key
        if r['content_sha256'] not in languages:
            languages[r['content_sha256']]=screen_language(text,detector.classify)
        language=languages[r['content_sha256']]
        instrumental=instrumental_status(text)
        terms=version_terms(r['song'])
        query_artist=raw['Artist'].split(' feat',1)[0].split(' Feat',1)[0].strip()
        credit_reduced=query_artist!=raw['Artist']
        technical=[]
        if not metadata_equal:technical.append('title_artist_mismatch')
        if not chart_equal:technical.append('chart_slot_mismatch')
        if language['language_status']!='english_automated_consistent':technical.append(language['language_status'])
        if instrumental!='no_instrumental_marker_detected':technical.append(instrumental)
        if terms:technical.append('version_specific_recording_review')
        if credit_reduced:technical.append('upstream_featured_credit_removed_review')
        evidence.append(dict(song_id=r['song_id'],song=r['song'],artist=r['artist'],
            source_reference_id=r['source_reference_id'],content_sha256=r['content_sha256'],
            source_record_number=number,source_file_sha256=local_hash,
            source_file_url=published['links']['html'],source_file_version=published['attributes']['current_version'],
            source_title=raw['Song Title'],source_artist=raw['Artist'],
            metadata_match=metadata_equal,chart_slot_match=chart_equal,
            metadata_evidence='OSF query metadata agrees with Billboard; returned lyric recording identity is not supplied',
            first_observed_chart_date=first_all[key],first_observed_study_date=first_study[key],
            selected_source_chart_date=raw['Date'],observed_by_source_chart=first_all[key]<=raw['Date'],
            recording_id='',recording_verified=False,release_date='',release_verified=False,
            release_evidence_status='chart_observation_is_not_recording_release_date',
            version_terms=terms,instrumental_status=instrumental,
            upstream_ovh_query_artist=query_artist,upstream_credit_query_reduced=credit_reduced,
            upstream_version_risk=bool({'remix','live'} & set(terms.split(';'))),
            **language,automated_screen_pass=not technical,
            technical_review_reasons=';'.join(technical),
            source_license='CC BY 4.0 for rights the licensor controls',
            permission_status='underlying_lyric_rights_unresolved_fair_use_assessment_not_adopted',
            permission_verified=False,nlp_eligible=False,blockers=BLOCKERS))
        if len(evidence)%1000==0:print('Audited',len(evidence),'selected records',flush=True)
    if selected:raise ValueError('Unresolved source locators')
    evidence.sort(key=lambda r:r['song_id'])
    by_key={identity(r['song'],r['artist']):r for r in evidence}
    if len(by_key)!=len(evidence):
        # Silently keeping one row per identity would understate weighted coverage.
        raise ValueError('Duplicate chart identity among selected candidates')
    monthly=defaultdict(Counter)
    for r in charts:
        month=r['date'][:7]+'-01';w=101-int(r['this_week']);c=monthly[month]
        c['chart_entries']+=1;c['all_chart_weight']+=w
        e=by_key.get(identity(r['song'],r['artist']))
        if e:
            c['selected_candidate_weight']+=w
            cutoff=date.fromisoformat(month)
            cutoff=date(cutoff.year+(cutoff.month==12),cutoff.month%12+1,15).isoformat()
            if e['selected_source_chart_date']<=cutoff:
                c['observable_selected_weight']+=w
                if e['language_status']=='english_automated_consistent':c['automated_english_weight']+=w
                if e['automated_screen_pass']:c['automated_screen_pass_weight']+=w
    months=[]
    for month,c in sorted(monthly.items()):
        row=dict(predictor_month=month,**{k:c[k] for k in ['chart_entries','all_chart_weight',
            'selected_candidate_weight','observable_selected_weight','automated_english_weight','automated_screen_pass_weight']},
            eligible_weight=0)
        for k in ['observable_selected','automated_english','automated_screen_pass','eligible']:
            row[k+'_weight_coverage']=row[k+'_weight']/c['all_chart_weight']
        months.append(row)
    totals={k:sum(m[k] for m in months) for k in ['all_chart_weight','observable_selected_weight',
        'automated_english_weight','automated_screen_pass_weight','eligible_weight']}
    dispositions=[]
    selected_by_id={r['song_id']:r for r in evidence}
    for r in mapping:
        e=selected_by_id.get(r['song_id'])
        dispositions.append(dict(song_id=r['song_id'],song=r['song'],artist=r['artist'],
            disposition='blocked_pending_evidence' if e and e['automated_screen_pass'] else 'excluded_or_technical_review',
            reasons=e['blockers'] if e and e['automated_screen_pass'] else
                (e['technical_review_reasons'] if e else r['exclusion_reasons']),nlp_eligible=False))
    summary=dict(source_sha256=local_hash,published_hash_matches=True,source_rows=row_count,
        source_distinct_title_artist=len(all_source),source_nonempty_text_identities=len(nonempty_source),
        source_unique_nonempty_text_hashes=len(unique_text),paper_reported_english_lyrics=20186,
        study_identities=len(mapping),previously_excluded=len(mapping)-len(evidence),selected_candidates=len(evidence),
        metadata_matches=sum(e['metadata_match'] for e in evidence),chart_slot_matches=sum(e['chart_slot_match'] for e in evidence),
        language_status_counts=tally(e['language_status'] for e in evidence),
        instrumental_status_counts=tally(e['instrumental_status'] for e in evidence),
        version_review_count=sum(bool(e['version_terms']) for e in evidence),
        upstream_remix_live_risk_count=sum(e['upstream_version_risk'] for e in evidence),
        upstream_credit_reduced_count=sum(e['upstream_credit_query_reduced'] for e in evidence),
        automated_screen_pass=sum(e['automated_screen_pass'] for e in evidence),
        disposition_counts=tally(r['disposition'] for r in dispositions),
        eligible_records=0,independently_verified_recordings=0,verified_release_dates=0,
        monthly_rows=len(months),**totals,
        blockers_overlapping={b:len(evidence) for b in BLOCKERS.split(';')},
        decision='Gate unchanged: automated evidence is not approval or independently verified recording identity')
    output.mkdir(parents=True)
    write_csv(output/'candidate_evidence.csv',evidence)
    write_csv(output/'all_song_dispositions.csv',dispositions)
    write_csv(output/'monthly_coverage.csv',months)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest=dict(inputs={str(p.relative_to(root)):file_hash(p) for p in [source,mapping_path.resolve(),
        root/'billboard_hot100_all.csv',root/'reports/osf_source_review/files.json',
        root/'config/music_preparation.json',root/'config/music_recording_reviews.csv']},
        code_sha256=file_hash(Path(__file__)),langid_version=importlib.metadata.version('langid'),
        langid_model_sha256=sha(model),
        language_policy='All 97 languages; whole text and every 100-word block; >=.95 confidence; English share >=.9 and no confident non-English block; <40 words uncertain',
        output_hashes={p.name:file_hash(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mapping',type=Path,default=Path('reports/model_comparison/primary_2026_10_18_final/preparation/song_source_mapping.csv'))
    args=parser.parse_args()
    run(args.root,args.output,args.mapping)
