"""Integrity and calendar-alignment tests; fixtures live only in temp directories."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from music_recession.economic import (
    FRED_COLUMNS, GS10_COLUMNS, OUTPUT_COLUMNS, ValidationError,
    add_months, build_monthly, cross_check_gs10, load_monthly, prepare,
)


ROOT = Path(__file__).resolve().parents[1]


class EconomicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fred = self.root / "fred.csv"
        self.check = self.root / "gs10.csv"
        self.output = self.root / "processed" / "economic.csv"
        self.manifest = self.root / "docs" / "manifest.json"
        months = ["2020-09-01", "2020-10-01", "2020-11-01", "2020-12-01",
                  "2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"]
        labels = ["0", "1", "0", "0", "0", "0", "1", "0"]
        self.rows = [[m, y, "2.83", "0.95"] for m, y in zip(months, labels)]
        self.rows[1][2:] = ["3.05", "3.50"]
        self.write_sources()

    def write_csv(self, path, columns, rows, encoding="utf-8"):
        with path.open("w", newline="", encoding=encoding) as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)

    def write_sources(self):
        self.write_csv(self.fred, FRED_COLUMNS, self.rows)
        self.write_csv(self.check, GS10_COLUMNS, [[r[0], r[2]] for r in self.rows])

    def load(self):
        return load_monthly(self.fred, FRED_COLUMNS)

    def run_preparation(self):
        return prepare(self.fred, self.check, self.output, self.manifest)

    def test_calendar_arithmetic_across_year_and_leap_year(self):
        for source, expected in [
            (date(2020, 9, 1), date(2021, 3, 1)),
            (date(2019, 8, 1), date(2020, 2, 1)),
            (date(2020, 2, 1), date(2020, 8, 1)),
            (date(2020, 12, 1), date(2021, 6, 1)),
        ]:
            with self.subTest(source=source):
                self.assertEqual(add_months(source, 6), expected)
        with self.assertRaises(ValueError):
            add_months(date(2020, 1, 31), 6)

    def test_spread_and_future_status_are_exact(self):
        rows, audit = build_monthly(self.load())
        self.assertEqual(rows, [
            {"predictor_month": "2020-09-01", "target_month": "2021-03-01",
             "USREC_t": 0, "GS10_t": "2.83", "TB3MS_t": "0.95",
             "spread_t": "1.88", "target_t": 1},
            {"predictor_month": "2020-10-01", "target_month": "2021-04-01",
             "USREC_t": 1, "GS10_t": "3.05", "TB3MS_t": "3.50",
             "spread_t": "-0.45", "target_t": 0},
        ])
        self.assertEqual(audit["retained_rows"], 2)
        self.assertEqual(audit["exclusions"]["absent_target_month"]["count"], 6)

    def test_target_lookup_does_not_shift_rows_when_calendar_is_sparse(self):
        # Public ingestion rejects gaps. This separately tests that the join
        # itself remains date-based even with a manually constructed sparse map.
        source = self.load()
        del source.observations[date(2020, 11, 1)]
        rows, _ = build_monthly(source)
        self.assertEqual([r["target_month"] for r in rows], ["2021-03-01", "2021-04-01"])
        self.assertEqual([r["target_t"] for r in rows], [1, 0])

    def test_missing_predictors_excluded_without_imputation(self):
        for column in (2, 3):
            for marker in ("", ".", "NA", "N/A"):
                with self.subTest(column=column, marker=marker):
                    original = self.rows[0][column]
                    self.rows[0][column] = marker
                    self.write_sources()
                    rows, audit = build_monthly(self.load())
                    self.assertEqual([r["predictor_month"] for r in rows], ["2020-10-01"])
                    self.assertEqual(audit["exclusions"]["missing_predictors"]["count"], 1)
                    self.assertEqual(audit["imputed_values"], 0)
                    self.rows[0][column] = original

    def test_missing_future_label_excludes_matching_predictor(self):
        self.rows[6][1] = "."
        self.write_sources()
        rows, audit = build_monthly(self.load())
        self.assertEqual([r["predictor_month"] for r in rows], ["2020-10-01"])
        self.assertEqual(audit["exclusions"]["missing_target_value"]["count"], 1)

    def test_missing_current_usrec_is_not_a_missing_predictor(self):
        self.rows[0][1] = ""
        self.write_sources()
        rows, _ = build_monthly(self.load())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["USREC_t"], "")
        self.assertEqual(rows[0]["target_t"], 1)

    def test_missing_month_rejected(self):
        del self.rows[2]
        self.write_sources()
        with self.assertRaisesRegex(ValidationError, "missing monthly dates: 2020-11-01"):
            self.load()

    def test_identical_and_conflicting_duplicate_dates_rejected(self):
        for duplicate in (self.rows[0][:], ["2020-09-01", "1", "5", "4"]):
            with self.subTest(duplicate=duplicate):
                self.write_csv(self.fred, FRED_COLUMNS, self.rows + [duplicate])
                with self.assertRaisesRegex(ValidationError, "duplicate date"):
                    self.load()

    def test_unsorted_dates_rejected(self):
        self.write_csv(self.fred, FRED_COLUMNS, list(reversed(self.rows)))
        with self.assertRaisesRegex(ValidationError, "ascending order"):
            self.load()

    def test_invalid_dates_rejected(self):
        for value in ("2020-09-15", "2020-13-01", "2020-9-01", "", "2020-09-01T00:00:00"):
            with self.subTest(value=value):
                self.rows[0][0] = value
                self.write_sources()
                with self.assertRaises(ValidationError):
                    self.load()

    def test_invalid_yields_rejected(self):
        for column in (2, 3):
            for value in ("hello", "NaN", "Infinity", "-Inf"):
                with self.subTest(column=column, value=value):
                    original = self.rows[0][column]
                    self.rows[0][column] = value
                    self.write_sources()
                    with self.assertRaises(ValidationError):
                        self.load()
                    self.rows[0][column] = original

    def test_nonbinary_labels_rejected(self):
        for value in ("2", "-1", "0.5", "NaN", "true"):
            with self.subTest(value=value):
                self.rows[0][1] = value
                self.write_sources()
                with self.assertRaises(ValidationError):
                    self.load()

    def test_negative_yield_is_valid(self):
        self.rows[0][2] = "-0.25"
        self.write_sources()
        rows, _ = build_monthly(self.load())
        self.assertEqual(rows[0]["spread_t"], "-1.20")

    def test_schema_and_row_width_validation(self):
        for columns, rows in [
            (FRED_COLUMNS[:-1], [r[:-1] for r in self.rows]),
            (FRED_COLUMNS, [self.rows[0] + ["extra"]]),
            (FRED_COLUMNS, [self.rows[0][:-1]]),
            (("observation_date", "USREC", "GS10", "GS10"), self.rows),
        ]:
            with self.subTest(columns=columns, rows=rows):
                self.write_csv(self.fred, columns, rows)
                with self.assertRaises(ValidationError):
                    self.load()

    def test_empty_file_and_header_only_rejected(self):
        for content in ("", ",".join(FRED_COLUMNS) + "\n"):
            with self.subTest(content=content):
                self.fred.write_text(content)
                with self.assertRaises(ValidationError):
                    self.load()

    def test_utf8_bom_accepted(self):
        self.write_csv(self.fred, FRED_COLUMNS, self.rows, encoding="utf-8-sig")
        self.assertEqual(len(self.load().observations), 8)

    def test_gs10_cross_check_uses_numeric_equality(self):
        check_rows = [[r[0], r[2] + "0"] for r in self.rows]
        self.write_csv(self.check, GS10_COLUMNS, check_rows)
        result = cross_check_gs10(self.load(), load_monthly(self.check, GS10_COLUMNS))
        self.assertEqual(result["matched_months"], 8)
        self.assertEqual(result["mismatches"], 0)

    def test_gs10_mismatch_fails_without_writing_outputs(self):
        check_rows = [[r[0], r[2]] for r in self.rows]
        check_rows[0][1] = "2.84"
        self.write_csv(self.check, GS10_COLUMNS, check_rows)
        with self.assertRaisesRegex(ValidationError, "value mismatch: 2020-09-01"):
            self.run_preparation()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest.exists())

    def test_gs10_coverage_mismatch_rejected(self):
        self.write_csv(self.check, GS10_COLUMNS, [[r[0], r[2]] for r in self.rows[1:]])
        with self.assertRaisesRegex(ValidationError, "coverage mismatch"):
            self.run_preparation()

    def test_no_gs10_comparisons_rejected(self):
        self.write_csv(self.check, GS10_COLUMNS, [[r[0], ""] for r in self.rows])
        with self.assertRaisesRegex(ValidationError, "no non-missing values"):
            self.run_preparation()

    def test_no_usable_rows_rejected(self):
        self.rows = self.rows[:6]
        self.write_sources()
        with self.assertRaisesRegex(ValidationError, "No usable"):
            self.run_preparation()
        self.assertFalse(self.output.exists())

    def test_outputs_cannot_overwrite_sources(self):
        for protected in (self.fred, self.check):
            before = protected.read_bytes()
            with self.assertRaisesRegex(ValidationError, "overwrite source"):
                prepare(self.fred, self.check, protected, self.manifest)
            self.assertEqual(before, protected.read_bytes())
        with self.assertRaisesRegex(ValidationError, "different paths"):
            prepare(self.fred, self.check, self.output, self.output)

    def test_artifact_hashes_determinism_and_source_preservation(self):
        before = {p: p.read_bytes() for p in (self.fred, self.check)}
        report = self.run_preparation()
        artifacts = [self.output.read_bytes(), self.manifest.read_bytes()]
        self.run_preparation()
        self.assertEqual(artifacts, [self.output.read_bytes(), self.manifest.read_bytes()])
        self.assertEqual(report, json.loads(self.manifest.read_text()))
        self.assertEqual(report["output"]["sha256"], hashlib.sha256(artifacts[0]).hexdigest())
        for source in report["sources"]:
            path = Path(source["path"])
            self.assertEqual(path.read_bytes(), before[path])
            self.assertEqual(source["sha256"], hashlib.sha256(before[path]).hexdigest())

    def test_manifest_records_internal_missing_values_and_exclusions(self):
        self.rows[1][3] = "."
        self.write_sources()
        report = self.run_preparation()
        series = report["sources"][0]["series"]["TB3MS"]
        self.assertEqual(series["internal_missing_months"], ["2020-10-01"])
        self.assertEqual(series["missing_values"], 1)
        audit = report["preparation"]
        self.assertEqual(audit["retained_rows"], 1)
        self.assertEqual(audit["excluded_rows"], 7)
        self.assertEqual(sum(v["count"] for v in audit["exclusions"].values()), 7)

    def test_validation_failure_preserves_previous_artifacts(self):
        self.run_preparation()
        before = (self.output.read_bytes(), self.manifest.read_bytes())
        self.rows[0][1] = "2"
        self.write_sources()
        with self.assertRaises(ValidationError):
            self.run_preparation()
        self.assertEqual(before, (self.output.read_bytes(), self.manifest.read_bytes()))

    def test_cli_success_and_validation_failure_exit_codes(self):
        command = [sys.executable, "-m", "music_recession.economic",
                   "--fred", str(self.fred), "--gs10-check", str(self.check),
                   "--output", str(self.output), "--manifest", str(self.manifest)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Prepared 2 rows x 7 columns", result.stdout)
        self.rows[0][1] = "2"
        self.write_sources()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("USREC must be 0 or 1", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class RepositorySnapshotTests(unittest.TestCase):
    def test_actual_snapshots_and_every_derived_row(self):
        # Independently read raw CSVs; do not use the implementation's loader,
        # month arithmetic or spread function to derive expected output values.
        with (ROOT / "fredgraph.csv").open(encoding="utf-8-sig", newline="") as stream:
            raw = {r["observation_date"]: r for r in csv.DictReader(stream)}
        with (ROOT / "GS10.csv").open(encoding="utf-8-sig", newline="") as stream:
            independent_gs10 = {r["observation_date"]: r["GS10"] for r in csv.DictReader(stream)}
        with tempfile.TemporaryDirectory() as temp:
            output, manifest = Path(temp) / "economic.csv", Path(temp) / "manifest.json"
            report = prepare(ROOT / "fredgraph.csv", ROOT / "GS10.csv", output, manifest)
            with output.open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, list(OUTPUT_COLUMNS))
                rows = list(reader)
        self.assertEqual(len(rows), 876)
        self.assertEqual(rows[0]["predictor_month"], "1953-04-01")
        self.assertEqual(rows[-1]["predictor_month"], "2026-03-01")
        self.assertEqual(rows[0]["target_month"], "1953-10-01")
        self.assertEqual(rows[-1]["target_month"], "2026-09-01")
        self.assertEqual(len({r["predictor_month"] for r in rows}), 876)
        self.assertEqual(sum(int(r["target_t"]) for r in rows), 111)
        previous = None
        for row in rows:
            predictor, target = date.fromisoformat(row["predictor_month"]), date.fromisoformat(row["target_month"])
            self.assertEqual((target.year - predictor.year) * 12 + target.month - predictor.month, 6)
            if previous:
                self.assertEqual((predictor.year - previous.year) * 12 + predictor.month - previous.month, 1)
            previous = predictor
            source = raw[row["predictor_month"]]
            self.assertEqual(row["USREC_t"], source["USREC"])
            self.assertEqual(row["GS10_t"], source["GS10"])
            self.assertEqual(row["TB3MS_t"], source["TB3MS"])
            self.assertEqual(Decimal(row["spread_t"]), Decimal(source["GS10"]) - Decimal(source["TB3MS"]))
            self.assertEqual(row["target_t"], raw[row["target_month"]]["USREC"])
            self.assertEqual(Decimal(row["GS10_t"]), Decimal(independent_gs10[row["predictor_month"]]))
        self.assertEqual(report["cross_check_gs10"]["matched_months"], 882)
        self.assertEqual(report["preparation"]["exclusions"]["missing_predictors"]["count"], 1180)
        self.assertEqual(report["preparation"]["exclusions"]["absent_target_month"]["count"], 6)
        self.assertEqual(report["preparation"]["exclusions"]["missing_target_value"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
