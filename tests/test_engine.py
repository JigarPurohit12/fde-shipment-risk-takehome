"""Engine prototype: README demonstrations, customer notes 3 and 4, parity, replay."""
import copy
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from _factories import event_record, synthetic_training_rows

from shipment_risk import solution
from shipment_risk._io import read_jsonl, read_manifest
from shipment_risk._model import ARTIFACT_FILE, load_artifact
from shipment_risk._time import parse_timestamp
from shipment_risk._training import train_model
from shipment_risk.errors import (ArtifactError, ConflictError, InvalidArgumentError,
                                  InvalidRecordError, RetentionError)
from shipment_risk.solution import RiskEngine, build_training_rows, train

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DECISION = parse_timestamp("2026-01-01T14:00:00Z", "t")


def at(text):
    return parse_timestamp(text, "t")


def reading(hour, entity_id="ship-1", value=4.0, revision=1, received=None):
    return event_record(entity_id=entity_id, event_id=f"r-{hour}", revision=revision, value=value,
                        event_at=f"2026-01-01T{hour:02d}:00:00Z",
                        received_at=received or f"2026-01-01T{hour:02d}:02:00Z")


class EngineCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.model_a, cls.model_b, cls.broken = root / "a", root / "b", root / "broken"
        train_model(synthetic_training_rows(120), cls.model_a)
        train_model(synthetic_training_rows(90), cls.model_b)
        cls.broken.mkdir()
        (cls.broken / ARTIFACT_FILE).write_text("{not json")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def engine(self, max_entities=10000):
        engine = RiskEngine(self.model_a, max_entities)
        for h in range(9, 14):
            engine.ingest(reading(h, value=4.0 + h / 10))
        return engine


class IngestAndScoreTests(EngineCase):
    def test_duplicate_delivery_returns_false(self):
        engine = self.engine()
        self.assertIs(engine.ingest(reading(13, value=4.0 + 13 / 10)), False)
        self.assertIs(engine.ingest(reading(14)), True)

    def test_correction_after_a_decision_does_not_change_it(self):
        engine = self.engine()
        before = engine.score("ship-1", DECISION).to_wire()
        self.assertIs(engine.ingest(reading(12, revision=2, value=30.0, received="2026-01-01T17:00:00Z")), True)
        self.assertEqual(engine.score("ship-1", DECISION).to_wire(), before)            # note 1
        self.assertEqual(engine.score("ship-1", at("2026-01-01T17:00:00Z")).payload["features"]["max_temp_c"], 30.0)

    def test_reversed_revision_order(self):
        engine = self.engine()
        engine.ingest(reading(8, revision=2, received="2026-01-01T08:10:00Z", value=1.9))
        self.assertIs(engine.ingest(reading(8, revision=1, received="2026-01-01T11:00:00Z", value=3.9)), False)

    def test_bad_input_is_rejected_and_not_mutated(self):
        engine = self.engine()
        record = reading(15)
        before = copy.deepcopy(record)
        engine.ingest(record)
        self.assertEqual(record, before)
        with self.assertRaises(ConflictError):
            engine.ingest(reading(15, value=9.0))
        with self.assertRaises(InvalidRecordError):
            engine.ingest(reading(16, received="2026-01-01T16:02:00"))         # no UTC offset
        with self.assertRaises(InvalidArgumentError):
            engine.score("ship-1", datetime(2026, 1, 1, 14))                     # naive datetime
        counters = engine.stats()["counters"]
        self.assertEqual((counters["conflict"], counters["rejected_invalid"]), (1, 1))

    def test_payload_contract_and_unknown_shipment(self):
        payload = self.engine().score("ship-1", DECISION).payload
        self.assertTrue(0.0 <= payload["probability"] <= 1.0)
        for name, kind in (("model_version", str), ("feature_digest", str), ("degraded", bool), ("reasons", list)):
            self.assertIsInstance(payload[name], kind)
        unknown = self.engine().score("never-seen", DECISION).payload
        self.assertTrue(unknown["degraded"])
        self.assertIn("unknown_shipment", unknown["reasons"])


class CustomerNoteTests(EngineCase):
    def test_note3_no_valid_model_means_no_engine(self):
        with self.assertRaises(ArtifactError):
            RiskEngine(self.broken)

    def test_note3_failed_reload_keeps_the_previous_model(self):
        engine = self.engine()
        before = engine.score("ship-1", DECISION).to_wire()
        self.assertIs(engine.reload_model(self.broken), False)
        self.assertEqual(engine.score("ship-1", DECISION).to_wire(), before)
        self.assertEqual(engine.stats()["last_reload_error"], "artifact_corrupt")
        self.assertIs(engine.reload_model(self.model_b), True)
        self.assertEqual(engine.score("ship-1", DECISION).payload["model_version"], load_artifact(self.model_b)[1])

    def test_note4_entity_bound_and_late_correction_after_eviction(self):
        engine = RiskEngine(self.model_a, max_entities=2)
        for entity in ("a", "b", "c"):                       # "a" is least recently ingested
            engine.ingest(reading(9, entity_id=entity))
        self.assertEqual(engine.stats()["resident_entities"], 2)
        with self.assertRaises(RetentionError) as ctx:
            engine.ingest(reading(9, entity_id="a", revision=2, received="2026-01-01T19:00:00Z"))
        self.assertEqual(ctx.exception.code, "entity_evicted")
        with self.assertRaises(RetentionError):
            engine.score("a", DECISION)

    def test_note4_per_shipment_history_cap(self):
        engine = RiskEngine(self.model_a)
        with mock.patch.object(solution, "MAX_REVISIONS_PER_SHIPMENT", 2):
            engine.ingest(reading(9))
            engine.ingest(reading(10))
            with self.assertRaises(RetentionError) as ctx:
                engine.ingest(reading(11))
        self.assertEqual(ctx.exception.code, "history_full")


class SnapshotAndConcurrencyTests(EngineCase):
    def test_snapshot_is_deterministic_and_restore_is_lossless(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.engine(max_entities=3)
            for entity in ("x", "y", "z"):                   # forces one eviction
                engine.ingest(reading(9, entity_id=entity))
            first, second, third = (Path(tmp) / n for n in ("1.json", "2.json", "3.json"))
            engine.snapshot(first)
            engine.snapshot(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            restored = RiskEngine.restore(self.model_a, first)
            restored.snapshot(third)
            self.assertEqual(third.read_bytes(), first.read_bytes())
            self.assertEqual(restored.stats(), engine.stats())
            self.assertEqual(restored.score("y", DECISION).to_wire(), engine.score("y", DECISION).to_wire())

    def test_scores_during_reloads_never_mix_two_models(self):
        engine = self.engine()
        expected = {}
        for directory in (self.model_a, self.model_b):
            engine.reload_model(directory)
            payload = engine.score("ship-1", DECISION).payload
            expected[payload["model_version"]] = payload["probability"]
        stop, seen = threading.Event(), []

        def scorer():
            while not stop.is_set():
                p = engine.score("ship-1", DECISION).payload
                seen.append((p["model_version"], p["probability"]))

        threads = [threading.Thread(target=scorer) for _ in range(3)]
        for t in threads:
            t.start()
        for i in range(30):
            engine.reload_model([self.model_a, self.model_b, self.broken][i % 3])
        stop.set()
        for t in threads:
            t.join()
        self.assertTrue(seen)
        self.assertTrue(all(probability == expected[version] for version, probability in seen))


@unittest.skipUnless((DATA_DIR / "MANIFEST.json").exists(), "run tools/generate_dataset.py first")
class SampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        manifest = read_manifest(DATA_DIR / "MANIFEST.json")
        cls.events = read_jsonl(DATA_DIR / "events.jsonl")
        cls.rows = build_training_rows(cls.events, read_jsonl(DATA_DIR / "labels.jsonl"),
                                       read_jsonl(DATA_DIR / "decisions.jsonl"), manifest.observation_end)
        train(cls.rows, cls.root / "model")
        cls.model, _ = load_artifact(cls.root / "model")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def replay(self, engine, events):
        for e in events:
            try:
                engine.ingest(e)
            except RetentionError:
                pass
        return engine

    def outcomes(self, engine):
        result = []
        for row in self.rows:
            try:
                result.append(engine.score(row["entity_id"], at(row["as_of"])).to_wire())
            except RetentionError as exc:
                result.append(exc.code)
        return result

    def test_online_scores_equal_offline_rows(self):
        engine = self.replay(RiskEngine(self.root / "model"), self.events)
        for row in self.rows:
            payload = engine.score(row["entity_id"], at(row["as_of"])).payload
            self.assertEqual(payload["feature_digest"], row["metadata"]["feature_digest"])
            self.assertEqual(payload["probability"], round(self.model.predict(row["features"]), 6) + 0.0)

    def test_snapshot_restore_replay_equals_uninterrupted_replay(self):
        half = len(self.events) // 2
        for max_entities in (10000, 20):
            with self.subTest(max_entities=max_entities):
                straight = self.replay(RiskEngine(self.root / "model", max_entities), self.events)
                first = self.replay(RiskEngine(self.root / "model", max_entities), self.events[:half])
                path = self.root / f"half-{max_entities}.json"
                first.snapshot(path)
                resumed = self.replay(RiskEngine.restore(self.root / "model", path), self.events[half:])
                self.assertEqual(self.outcomes(resumed), self.outcomes(straight))
                self.assertLessEqual(straight.stats()["resident_entities"], max_entities)


if __name__ == "__main__":
    unittest.main()
