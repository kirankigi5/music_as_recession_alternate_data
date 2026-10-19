#!/usr/bin/env python3
"""Recover LRCLIB lyric candidates for unmatched Billboard 1973-2023 tracks.

Only produces provisional candidates, never edits the master inventory.
Full retrieved texts remain in the Git-ignored data/raw/lyrics directory.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

csv.field_size_limit(sys.maxsize)
UA = "MusicRecessionLyricsResearch/0.1 (https://github.com/kirankigi5/music_as_recession_alternate_data)"
BASE = "https://lrclib.net/api/search"


def norm(s):
    s = unicodedata.normalize("NFKD", str(s or "").casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\b(featuring|feat\.?|ft\.?|with|and|x)\b|&", " and ", s)
    return " ".join(re.sub(r"[^\w]+", " ", s).split())


def similarity(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def chart_candidates(root):
    inv = {}
    with (root / "data/metadata/song_lyrics_inventory.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["lyrics_status"] == "not_collected":
                inv[(r["song"].strip().casefold(), r["artist"].strip().casefold())] = r
    weights = defaultdict(int)
    with (root / "billboard_hot100_all.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if "1973" <= r["date"][:4] <= "2023":
                key = (r["song"].strip().casefold(), r["artist"].strip().casefold())
                if key in inv:
                    weights[key] += 101 - int(r["this_week"])
    return sorted([(inv[k], w) for k, w in weights.items()], key=lambda x: -x[1])


def request_search(params):
    url = BASE + "?" + urlencode(params)
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urlopen(req, timeout=25) as response:
                data = json.load(response)
            return data if isinstance(data, list) else []
        except HTTPError as exc:
            if exc.code == 429:
                delay = exc.headers.get("Retry-After", "60")
                try:
                    delay = max(1.0, float(delay))
                except ValueError:
                    delay = 60.0
                print(f"  LRCLIB rate limit: waiting {delay:.0f}s", flush=True)
                time.sleep(delay)
                continue
            if exc.code in (404, 400):
                return []
            if exc.code in (500, 502, 503, 504) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except (URLError, TimeoutError) as exc:
            if attempt == 2:
                raise exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("Rate limit persisted after three attempts")


def lyric_content(record):
    return (record.get("plainLyrics") or record.get("syncedLyrics") or "").strip()


def evaluate(record, song, artist):
    ts = similarity(record.get("trackName"), song)
    ars = similarity(record.get("artistName"), artist)
    # Retain only plausible candidates; never automatically certify a recording.
    plausible = ts >= 0.72 and ars >= 0.42
    exact_metadata = ts >= 0.98 and ars >= 0.98
    flagged_version = bool(re.search(r"\b(remix|mix|live|acoustic|version|edit|bayside)\b", song, re.I))
    status = "metadata_match_provisional" if exact_metadata and not flagged_version else "recording_review_required"
    return (0.62 * ts + 0.38 * ars, plausible, status, ts, ars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="new songs to check in this run")
    parser.add_argument("--retry-misses", action="store_true", help="revisit prior unsuccessful searches")
    parser.add_argument("--retry-errors-only", action="store_true", help="retry previous errors only")
    args = parser.parse_args()
    root = Path.cwd()
    queue = chart_candidates(root)
    out_dir = root / "data/raw/lyrics/api_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ensure lyrics can never accidentally be committed.
    ignored = subprocess.run(["git", "check-ignore", "-q", str(out_dir / "example.json")], cwd=root).returncode == 0
    if not ignored:
        raise SystemExit("STOP: data/raw/lyrics/api_recovery is not Git-ignored")
    report = root / "reports/lyrics_api_recovery.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    fields = ["song", "billboard_artist", "weight", "status", "lrclib_id", "source_title",
              "source_artist", "album", "duration_seconds", "characters", "lyrics_sha256",
              "title_similarity", "artist_similarity", "local_json_path", "error"]
    processed = set()
    if report.exists():
        with report.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                key = (r["song"].strip().casefold(), r["billboard_artist"].strip().casefold())
                if args.retry_errors_only:
                    if r["status"] != "error":
                        processed.add(key)
                elif not args.retry_misses or r["status"] not in ("no_candidate", "error"):
                    processed.add(key)
    written = 0
    found = 0
    with report.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if report.stat().st_size == 0:
            writer.writeheader()
        for songrow, weight in queue:
            key = (songrow["song"].strip().casefold(), songrow["artist"].strip().casefold())
            if key in processed:
                continue
            song, artist = songrow["song"], songrow["artist"]
            output = dict.fromkeys(fields, "")
            output.update(song=song, billboard_artist=artist, weight=weight)
            try:
                records = request_search({"track_name": song, "artist_name": artist})
                time.sleep(3)
                ranked = sorted([(evaluate(r, song, artist), r) for r in records],
                                key=lambda x: x[0][0], reverse=True)
                valid = [(score, r) for score, r in ranked if score[1] and (lyric_content(r) or r.get("instrumental"))]
                if not valid:
                    # Broader title-only lookup can help credit variants and soundtrack qualifiers.
                    simplified = re.sub(r"\s*\([^)]*\)", "", song).strip() or song
                    records = request_search({"track_name": simplified})
                    time.sleep(3)
                    ranked = sorted([(evaluate(r, song, artist), r) for r in records],
                                    key=lambda x: x[0][0], reverse=True)
                    valid = [(score, r) for score, r in ranked if score[1] and (lyric_content(r) or r.get("instrumental"))]
                if valid:
                    score, best = valid[0]
                    content = lyric_content(best)
                    if best.get("instrumental") or not content:
                        status = "instrumental_review_required"
                    else:
                        status = score[2]
                    ident = hashlib.sha256((song + "\0" + artist).encode("utf-8")).hexdigest()[:20]
                    path = out_dir / (ident + ".json")
                    # Preserve exact provider metadata and raw candidate for manual review.
                    path.write_text(json.dumps({"billboard_song": song, "billboard_artist": artist,
                                                "provider": "LRCLIB", "record": best}, ensure_ascii=False, indent=2))
                    output.update(status=status, lrclib_id=best.get("id", ""),
                                  source_title=best.get("trackName", ""), source_artist=best.get("artistName", ""),
                                  album=best.get("albumName", ""), duration_seconds=best.get("duration", ""),
                                  characters=len(content), lyrics_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest() if content else "",
                                  title_similarity=round(score[3], 3), artist_similarity=round(score[4], 3),
                                  local_json_path=str(path.relative_to(root)))
                    found += 1
                else:
                    output["status"] = "no_candidate"
            except Exception as exc:
                output.update(status="error", error=f"{type(exc).__name__}: {exc}"[:200])
            writer.writerow(output)
            f.flush()
            written += 1
            print(f"{written}: {song} — {artist}: {output['status']}", flush=True)
            if written >= args.limit:
                break
    print(f"\nThis batch: {written} searched, {found} candidates. Results: {report.relative_to(root)}")
    print("Lyrics are LOCAL and PROVISIONAL. Inventory unchanged; check version and usage rights before analysis.")


if __name__ == "__main__":
    main()
