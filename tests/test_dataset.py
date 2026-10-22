"""Step 4: build_training_rows (labels, maturity, censoring, report)."""
import copy
import json
import random
import unittest
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType

from _factories import decision_record, event_record, label_record

from shipment_risk._canonical import canonical_json
from shipment_risk._dataset import DECISION_AFTER_CUTOFF, LABEL_WINDOW_IMMATURE, build_dataset
from shipment_risk._features import feature_digest
from shipment_risk._io import read_jsonl, read_manifest
from shipment_risk._labels import HORIZON, LABEL_REPORTING_LAG, check_reporting_lag
from shipment_risk._time import format_timestamp, parse_timestamp
from shipment_risk.errors import ConflictError, InvalidArgumentError, InvalidRecordError
from shipment_risk.solution import build_training_rows

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
END = parse_timestamp("2026-01-10T00:00:00Z", "t")
US = timedelta(microseconds=1)


def at(text):
    return parse_timestamp(text, "t")


def readings(entity_id="ship-1", start_hour=9, end_hour=13, value=4.0):
    """Hourly readings, received 2 minutes after each measurement."""
    return [
        event_record(entity_id=entity_id, event_id=f"r-{h}", value=value,
                     event_at=f"2026-01-01T{h:02d}:00:00Z", received_at=f"2026-01-01T{h:02d}:02:00Z")
        for h in range(start_hour, end_hour + 1)
    ]


def incident(at_text, available_text="2026-01-02T12:00:00Z", incident_id="inc-1", entity_id="ship-1"):
    return label_record(incident_at=at_text, label_available_at=available_text,
                        incident_id=incident_id, entity_id=entity_id)


def build(events=None, labels=(), decisions=(), end=END):
    return build_dataset(readings() if events is None else events, labels, decisions, end)


def only_row(dataset):
    assert len(dataset.rows) == 1, dataset.rows
    return dataset.rows[0]


class LabelTests(unittest.TestCase):
    def test_positive_and_negative_decisions(self):
        ds = build(labels=[incident("2026-01-01T18:00:00Z")],
                   decisions=[decision_record(as_of="2026-01-01T11:00:00Z"),
                              decision_record(as_of="2026-01-01T14:00:00Z")])
        negative, positive = ds.rows
        self.assertEqual((negative["label"], positive["label"]), (0, 1))
        self.assertEqual([i["incident_id"] for i in positive["metadata"]["incidents"]], ["inc-1"])
        self.assertTrue(positive["metadata"]["eligible"])

    def test_window_boundaries(self):
        cases = [
            ("2026-01-01T14:00:00Z", "2026-01-01T14:00:00Z", 0),          # incident at as_of
            ("2026-01-01T12:00:00Z", "2026-01-01T18:00:00Z", 1),          # exactly +6 h
            ("2026-01-01T11:59:59.999999Z", "2026-01-01T18:00:00Z", 0),   # +6 h + 1 us
        ]
        for as_of, incident_at, expected in cases:
            with self.subTest(as_of=as_of, incident_at=incident_at):
                row = only_row(build(labels=[incident(incident_at)],
                                     decisions=[decision_record(as_of=as_of)]))
                self.assertEqual(row["label"], expected)

    def test_label_available_after_cutoff_is_unknown(self):
        # Mature by the rule, but this label broke the 30 h promise and
        # arrived after the export: from the export's point of view it does
        # not exist yet.
        end = at("2026-01-03T02:00:00Z")                        # = as_of + 36 h
        ds = build(labels=[incident("2026-01-01T18:00:00Z", "2026-01-03T06:00:00Z")],
                   decisions=[decision_record(as_of="2026-01-01T14:00:00Z")], end=end)
        self.assertEqual(only_row(ds)["label"], 0)
        self.assertEqual(ds.report["labels"]["available_after_cutoff"], 1)

    def test_label_slower_than_promised_is_used_and_counted(self):
        ds = build(labels=[incident("2026-01-01T18:00:00Z", "2026-01-03T01:00:00Z")],   # 31 h
                   decisions=[decision_record(as_of="2026-01-01T14:00:00Z")])
        self.assertEqual(only_row(ds)["label"], 1)
        self.assertEqual(ds.report["labels"]["beyond_reporting_lag"], 1)

    def test_shipment_with_several_incidents(self):
        ds = build(labels=[incident("2026-01-01T19:00:00Z", incident_id="inc-b"),
                           incident("2026-01-01T15:00:00Z", incident_id="inc-a")],
                   decisions=[decision_record(as_of="2026-01-01T14:00:00Z")])
        row = only_row(ds)
        self.assertEqual([i["incident_id"] for i in row["metadata"]["incidents"]], ["inc-a", "inc-b"])


class CensoringTests(unittest.TestCase):
    as_of = at("2026-01-01T14:00:00Z")

    def test_maturity_boundary_applies_to_positives_too(self):
        labels = [incident("2026-01-01T18:00:00Z", "2026-01-02T00:00:00Z")]
        decisions = [decision_record(as_of=format_timestamp(self.as_of))]
        boundary = self.as_of + HORIZON + LABEL_REPORTING_LAG

        mature = only_row(build(labels=labels, decisions=decisions, end=boundary))
        self.assertEqual((mature["label"], mature["metadata"]["eligible"]), (1, True))

        early = only_row(build(labels=labels, decisions=decisions, end=boundary - US))
        self.assertIsNone(early["label"])                       # even though the incident is known
        self.assertEqual(early["metadata"]["censor_reason"], LABEL_WINDOW_IMMATURE)
        self.assertEqual(early["metadata"]["incidents"], [])
        self.assertIsNotNone(early["features"])                 # features ARE known at as_of

    def test_decision_after_cutoff(self):
        ds = build(decisions=[decision_record(as_of="2026-01-11T00:00:00Z")])
        row = only_row(ds)
        self.assertEqual((row["features"], row["label"]), (None, None))
        self.assertEqual(row["metadata"]["censor_reason"], DECISION_AFTER_CUTOFF)
        self.assertEqual(ds.report["decisions"]["censored"][DECISION_AFTER_CUTOFF], 1)

    def test_events_received_after_cutoff_change_nothing(self):
        decisions = [decision_record(as_of="2026-01-01T14:00:00Z")]
        late = event_record(event_id="r-13", revision=2, value=30.0,
                            event_at="2026-01-01T13:00:00Z", received_at="2026-01-10T00:00:01Z")
        without = build(decisions=decisions)
        with_late = build(events=readings() + [late], decisions=decisions)
        self.assertEqual(with_late.rows, without.rows)
        self.assertEqual(with_late.report["events"]["received_after_cutoff"], 1)


class InputHandlingTests(unittest.TestCase):
    def test_duplicate_decisions_merge_even_with_other_offsets(self):
        ds = build(decisions=[decision_record(as_of="2026-01-01T14:00:00Z"),
                              decision_record(as_of="2026-01-01T19:30:00+05:30")])
        self.assertEqual(only_row(ds)["metadata"]["duplicate_decisions"], 1)
        self.assertEqual(ds.report["decisions"]["duplicates_merged"], 1)

    def test_event_conflict_keeps_the_first_delivered_version(self):
        first = event_record(event_id="r-x", event_at="2026-01-01T13:30:00Z",
                             received_at="2026-01-01T13:31:00Z", value=5.0)
        second = dict(first, value=9.0)
        decisions = [decision_record(as_of="2026-01-01T14:00:00Z")]
        ds = build(events=[first, second], decisions=decisions)
        self.assertEqual(only_row(ds)["features"]["latest_temp_c"], 5.0)
        self.assertEqual(ds.report["events"]["conflict"], 1)
        swapped = build(events=[second, first], decisions=decisions)       # same as the engine
        self.assertEqual(only_row(swapped)["features"]["latest_temp_c"], 9.0)

    def test_label_conflict_stops_the_build(self):
        a = incident("2026-01-01T18:00:00Z")
        with self.assertRaises(ConflictError) as ctx:
            build(labels=[a, dict(a, incident_at="2026-01-01T19:00:00Z")])
        self.assertEqual(ctx.exception.code, "label_conflict")

    def test_identical_duplicate_label_is_counted(self):
        a = incident("2026-01-01T18:00:00Z")
        ds = build(labels=[a, dict(a)], decisions=[decision_record()])
        self.assertEqual((ds.report["labels"]["duplicate"], ds.report["labels"]["used"]), (1, 1))

    def test_malformed_record_is_rejected_with_its_position(self):
        bad = event_record(received_at="2026-01-01T08:00:00")      # no offset
        with self.assertRaises(InvalidRecordError) as ctx:
            build(events=readings()[:1] + [bad])
        self.assertEqual(ctx.exception.code, "missing_utc_offset")
        self.assertIn("events[1]", ctx.exception.detail)

    def test_bad_observation_end(self):
        for value, code in ((END.replace(tzinfo=None), "naive_datetime"),
                            ("2026-01-10T00:00:00Z", "not_a_datetime")):
            with self.subTest(value=value):
                with self.assertRaises(InvalidArgumentError) as ctx:
                    build(end=value)
                self.assertEqual(ctx.exception.code, code)

    def test_non_iterables_are_rejected(self):
        for value in (None, "events", decision_record()):
            with self.subTest(value=value):
                with self.assertRaises(InvalidArgumentError) as ctx:
                    build_dataset(value, [], [], END)
                self.assertEqual(ctx.exception.code, "not_iterable")

    def test_generators_and_read_only_records_work_and_nothing_is_mutated(self):
        events = readings()
        labels = [incident("2026-01-01T18:00:00Z")]
        decisions = [decision_record(as_of="2026-01-01T14:00:00Z")]
        before = copy.deepcopy((events, labels, decisions))
        expected = build(events=events, labels=labels, decisions=decisions).rows
        from_generators = build_dataset((MappingProxyType(e) for e in events),
                                        (l for l in labels), iter(decisions), END).rows
        self.assertEqual(from_generators, expected)
        self.assertEqual((events, labels, decisions), before)

    def test_rows_are_sorted_json_safe_and_self_consistent(self):
        events = readings("ship-1") + readings("ship-2")
        decisions = [decision_record(entity_id="ship-2", as_of="2026-01-01T14:00:00Z"),
                     decision_record(entity_id="ship-1", as_of="2026-01-01T14:00:00Z"),
                     decision_record(entity_id="ship-1", as_of="2026-01-01T11:00:00Z")]
        rows = build(events=events, decisions=decisions).rows
        self.assertEqual([(r["as_of"], r["entity_id"]) for r in rows], [
            ("2026-01-01T11:00:00Z", "ship-1"),
            ("2026-01-01T14:00:00Z", "ship-1"),
            ("2026-01-01T14:00:00Z", "ship-2"),
        ])
        canonical_json(rows)
        for row in rows:
            self.assertEqual(feature_digest(row["features"]), row["metadata"]["feature_digest"])

    def test_shipment_without_events_gets_an_honest_row(self):
        row = only_row(build(decisions=[decision_record(entity_id="ship-9")]))
        self.assertEqual(row["features"]["reading_count"], 0)
        self.assertEqual(row["metadata"]["data_reasons"], ["no_readings_in_window"])
        self.assertEqual(row["label"], 0)

    def test_public_function_returns_the_same_rows(self):
        args = (readings(), [incident("2026-01-01T18:00:00Z")],
                [decision_record(as_of="2026-01-01T14:00:00Z")], END)
        self.assertEqual(build_training_rows(*args), build_dataset(*args).rows)


@unittest.skipUnless((DATA_DIR / "MANIFEST.json").exists(), "run tools/generate_dataset.py first")
class SampleDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = read_manifest(DATA_DIR / "MANIFEST.json")
        check_reporting_lag(cls.manifest.label_reporting_lag_hours)
        cls.events = read_jsonl(DATA_DIR / "events.jsonl")
        cls.labels = read_jsonl(DATA_DIR / "labels.jsonl")
        cls.decisions = read_jsonl(DATA_DIR / "decisions.jsonl")
        cls.dataset = build_dataset(cls.events, cls.labels, cls.decisions, cls.manifest.observation_end)

    def test_labels_and_eligibility_match_an_independent_brute_force(self):
        end = self.manifest.observation_end
        incidents = [(l["entity_id"], at(l["incident_at"]), at(l["label_available_at"])) for l in self.labels]
        unique = {(at(d["as_of"]), d["entity_id"]) for d in self.decisions}
        self.assertEqual(len(self.dataset.rows), len(unique))
        for row in self.dataset.rows:
            t = at(row["as_of"])
            eligible = t + timedelta(hours=36) <= end
            expected = None
            if eligible:
                expected = int(any(e == row["entity_id"] and t < i <= t + timedelta(hours=6) and a <= end
                                   for e, i, a in incidents))
            self.assertEqual((row["metadata"]["eligible"], row["label"]), (eligible, expected), msg=row["as_of"])

    def test_input_order_does_not_matter(self):
        shuffled = list(self.events)
        random.Random(0).shuffle(shuffled)
        again = build_dataset(shuffled, self.labels[::-1], self.decisions[::-1], self.manifest.observation_end)
        self.assertEqual(canonical_json(again.rows), canonical_json(self.dataset.rows))

    def test_report_reconciles_with_the_inputs(self):
        report = self.dataset.report
        events, decisions = report["events"], report["decisions"]
        self.assertEqual(events["input"], self.manifest.files["events.jsonl"])
        self.assertEqual(events["input"], sum(v for k, v in events.items() if k != "input"))
        self.assertEqual(decisions["input"], decisions["rows"] + decisions["duplicates_merged"])
        self.assertEqual(decisions["rows"], decisions["eligible"] + sum(decisions["censored"].values()))
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
