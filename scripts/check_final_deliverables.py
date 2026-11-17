"""Validate the final documents against saved evidence and rendered exports.

No numerical model tests, inference, fitting or network access are required.
Run in the separate document-export environment described in docs/final/README.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import urllib.parse
from pathlib import Path

import fitz
from pptx import Presentation

from build_final_deliverables import ROOT, SOURCE, read_facts


def normalize(text):
    return text.replace("−", "-").replace("–", "-").replace("\xa0", " ")


def check_links(root):
    paths = subprocess.check_output(
        ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=root).decode().split("\0")
    def clean(text):
        return re.sub(r"^```[^\n]*\n.*?^```[^\n]*$", "", text, flags=re.M | re.S)
    def anchors(path):
        seen, found = {}, set()
        for m in re.finditer(r"^#{1,6}\s+(.+?)\s*#*$", clean(path.read_text()), re.M):
            label = re.sub(r"<[^>]+>", "", m[1]).lower()
            slug = "".join(c for c in label if c.isalnum() or c in "_- ").replace(" ", "-")
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            found.add(slug + ("-" + str(n) if n else ""))
        return found
    count = fragments = 0
    errors = []
    for name in sorted(set(paths)):
        if not name.endswith(".md"):
            continue
        path = root / name
        for m in re.finditer(r"\]\(([^)]+)\)", clean(path.read_text())):
            target = m[1].strip().split(' "')[0].strip("<>")
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            part, _, fragment = target.partition("#")
            dest = (path.parent / urllib.parse.unquote(part)).resolve() if part else path
            count += 1
            if not dest.exists():
                errors.append([name, target, "missing file"])
            elif fragment and dest.suffix == ".md":
                fragments += 1
                if urllib.parse.unquote(fragment) not in anchors(dest):
                    errors.append([name, target, "missing anchor"])
    assert not errors, errors
    return dict(local_links=count, markdown_anchors=fragments, failures=errors,
                external_urls="Not exhaustively probed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, help="optional new JSON check record")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("--output must be a fresh path")
    bundle = args.bundle.resolve()
    specs = json.loads((SOURCE / "slides.json").read_text())
    manifest = json.loads((bundle / "build_manifest.json").read_text())
    facts = read_facts()
    report = normalize((SOURCE / "report.md").read_text())
    checks = 0
    for path, digest in manifest["evidence_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
        checks += 1
    for filename in ("report.md", "slides.json"):
        assert hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest() == manifest["source_sha256"][filename]
        checks += 1
    assert hashlib.sha256((ROOT / "scripts/build_final_deliverables.py").read_bytes()).hexdigest() == manifest["source_sha256"]["build_final_deliverables.py"]
    checks += 1
    for label, run in zip(("70% primary", "75% sensitivity", "80% sensitivity"), facts["paired"]["runs"]):
        a, b = run["model_a"], run["model_b"]
        row = "| %s | %d / %d / %d | %.4f | %.4f | %+.6f | %.4f | %.4f |" % (
            label, run["forecasts"], run["recession_target_months"], len(run["recession_episodes_reached"]),
            a["brier_score"], b["brier_score"], run["loss_differential"]["mean"], a["roc_auc"], b["roc_auc"])
        assert row in report, row
        row = "| %s | %.4f | %.4f | %.4f |" % (
            label, a["average_precision"], b["average_precision"], run["average_precision_prevalence_reference"])
        assert row in report, row
        intervals = " | ".join("[%.6f, %.6f]" % (v["ci_lower"], v["ci_upper"])
                               for v in run["loss_differential"]["block_bootstrap"])
        assert "| " + label[:3] + " | " + intervals + " |" in report
        pvalues = " | ".join("%.6f" % v["p_value_two_sided"] for v in run["loss_differential"]["diebold_mariano"])
        assert "| " + label[:3] + " | " + pvalues + " |" in report
        checks += 4
    primary = facts["verified"]["primary_checks"]
    reference = facts["verified"]["runs"]["exploratory_primary_cov70"]["training_rate_brier"]
    for value in ("%.7f" % reference, "%.7f" % primary["full_history_a_same_dates"]["brier_score"],
                  "5.75-7.625", "44 evaluated months", "7,490", "8,800", "21,542", "19,096", "16,290"):
        assert value in report, value
        checks += 1
    assert round(facts["median_coverage"][1], 3) == .924
    assert round(facts["median_coverage"][-1], 3) == .608
    for i, key in enumerate(("exploratory_primary_cov70", "exploratory_sensitivity_cov75", "exploratory_sensitivity_cov80")):
        run = facts["verified"]["runs"][key]
        coverage = (70, 75, 80)[i]
        row = "| %d%% %s | %d | %d | %d |" % (coverage, "primary" if i == 0 else "sensitivity",
               run["coverage_only_months"], run["feature_eligible_months"], run["forecasts"])
        assert row in report, row
        checks += 1
    with (ROOT / "data/processed/economic_monthly.csv").open(newline="") as f:
        economic = list(csv.DictReader(f))
    assert len(economic) == 876
    assert economic[0]["predictor_month"] == "1953-04-01"
    assert economic[-1]["predictor_month"] == "2026-03-01"
    assert facts["summary"]["songs_scored"] == 16290
    assert all(r["analysis_months"] == 612 for r in facts["summary"]["runs"])
    checks += 2
    # Verify the concrete first-origin example without fitting a model.
    with (ROOT / "docs/results/exploratory_primary_cov70/predictions.csv").open(newline="") as f:
        first = next(r for r in csv.DictReader(f) if r["status"] == "ok")
    assert first["predictor_month"] == "1993-11-01" and first["target_month"] == "1994-05-01"
    assert first["forecast_cutoff"] == "1993-12-15"
    checks += 1
    slides = specs["slides"]
    assert len(slides) == 16 and sum(s["seconds"] > 0 for s in slides) == 11
    assert sum(s["seconds"] for s in slides) == specs["planned_seconds"] == 660
    assert specs["slot_seconds"] == 750 and specs["qa_included"] is None
    deck = Presentation(bundle / "presentation.pptx")
    assert len(deck.slides) == len(slides)
    native_text_boxes = 0
    for i, (slide, spec) in enumerate(zip(deck.slides, slides)):
        assert spec["notes"] in slide.notes_slide.notes_text_frame.text, spec["id"]
        assert len(spec["notes"].split()) >= 40
        for evidence in spec["evidence"]:
            assert (ROOT / evidence).exists(), evidence
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= deck.slide_width + 2
            assert shape.top + shape.height <= deck.slide_height + 2
            if shape.has_text_frame and shape.text:
                native_text_boxes += 1
        checks += 1
    page_counts = {}
    text_spans = 0
    report_pdf_links = 0
    for filename, expected in (("presentation.pdf", 16), ("report.pdf", 7)):
        doc = fitz.open(bundle / filename)
        assert len(doc) == expected
        page_counts[filename] = len(doc)
        for i, page in enumerate(doc):
            text = page.get_text()
            if filename == "report.pdf":
                report_pdf_links += len(page.get_links())
            assert len(text.strip()) > 200 and "\x00" not in text and "\ufffd" not in text
            for block in page.get_text("dict")["blocks"]:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        r = fitz.Rect(span["bbox"])
                        assert r.x0 >= -1 and r.y0 >= -1 and r.x1 <= page.rect.width + 1 and r.y1 <= page.rect.height + 1, (filename, i, span)
                        text_spans += 1
            if filename == "report.pdf":
                assert ("Can Popular Music" if i == 0 else str(i + 1) + ". ") in text
            checks += 1
    assert report_pdf_links >= 30
    note_text = (bundle / "speaker_notes.md").read_text()
    for s in slides:
        assert s["notes"] in note_text
    assert "10:15-11:00" in normalize(note_text).replace("–", "-")
    for filename in ("primary_benchmarks.png", "primary_uncertainty.png"):
        with fitz.open(bundle / "figures" / filename) as image:
            assert image[0].rect.width > 500
    links = check_links(ROOT)
    result = dict(status="passed", check_groups=checks, pages=page_counts,
                  report_pdf_links=report_pdf_links,
                  native_editable_text_boxes=native_text_boxes, pdf_text_spans_within_page=text_spans,
                  main_slides=11, appendix_slides=5, planned_seconds=660,
                  narration_words=sum(len(s["notes"].split()) for s in slides if s["seconds"]),
                  links=links, numerical_suite="Not repeated: unchanged numerical/model/statistical files",
                  visual_review="Separate human/model review of rendered pages; not inferred by this script")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
