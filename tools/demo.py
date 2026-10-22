#!/usr/bin/env python3
"""End-to-end demo in a fresh process:   python tools/demo.py

Uses only data/ and writes only demo_output/ (recreated on every run).
Nothing is hard-coded: shipments and records are picked from the data.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # works without `pip install -e .` once the dependencies are installed

from shipment_risk._dataset import build_dataset          # noqa: E402
from shipment_risk._io import read_jsonl, read_manifest    # noqa: E402
from shipment_risk._labels import check_reporting_lag      # noqa: E402
from shipment_risk._time import format_timestamp, parse_timestamp  # noqa: E402
from shipment_risk.errors import RetentionError            # noqa: E402
from shipment_risk.solution import RiskEngine, train       # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "demo_output"


def at(text):
    return parse_timestamp(text, "t")


def replay(engine, events):
    returns = []
    for e in events:
        try:
            returns.append(engine.ingest(e))
        except RetentionError as exc:
            returns.append(exc.code)
    return returns


def outcomes(engine, rows):
    result = []
    for row in rows:
        try:
            result.append(engine.score(row["entity_id"], at(row["as_of"])).to_wire())
        except RetentionError as exc:
            result.append(exc.code)
    return result


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()

    print("[1] Data")
    manifest = read_manifest(DATA / "MANIFEST.json")
    check_reporting_lag(manifest.label_reporting_lag_hours)
    events, labels, decisions = (read_jsonl(DATA / f) for f in ("events.jsonl", "labels.jsonl", "decisions.jsonl"))
    print(f"    {len(events)} events, {len(labels)} labels, {len(decisions)} decisions; "
          f"observation_end {format_timestamp(manifest.observation_end)}; lag {manifest.label_reporting_lag_hours} h")

    print("[2] Point-in-time training rows (immature label windows are censored, not dropped)")
    dataset = build_dataset(events, labels, decisions, manifest.observation_end)
    rows, report = dataset.rows, dataset.report
    print(f"    {json.dumps(report['decisions'], sort_keys=True)}")
    print(f"    events: {json.dumps(report['events'], sort_keys=True)}")
    for row in [r for r in rows if r["metadata"]["censor_reason"]][:3]:
        print(f"    censored: {row['entity_id']} {row['as_of']} {row['metadata']['censor_reason']} "
              f"(label mature at {row['metadata']['label_mature_at']})")

    print("[3] Train: purged temporal split, logistic regression vs baselines")
    metrics = train(rows, OUT / "model")
    split = metrics["split"]
    print(f"    split_time {split['split_time']}: train {split['train_rows']} ({split['train_positives']} pos), "
          f"test {split['test_rows']} ({split['test_positives']} pos), purged {split['purged_rows']}")
    for name, m in metrics["test_metrics"].items():
        print(f"    {name:22s} AP {m['average_precision']}  Brier {m['brier']}")
    print(f"    model_version {metrics['model_version']}")

    print("[4] Engine replay in delivery order + offline/online parity")
    engine = RiskEngine(OUT / "model")
    returns = replay(engine, events)
    parity = sum(engine.score(r["entity_id"], at(r["as_of"])).payload["feature_digest"]
                 == r["metadata"]["feature_digest"] for r in rows)
    print(f"    ingest True {returns.count(True)}, False {returns.count(False)}; "
          f"parity {parity}/{len(rows)} decisions; stats {json.dumps(engine.stats()['counters'], sort_keys=True)}")

    print("[5] Demonstrations")
    print(f"    duplicate delivery -> ingest returned {engine.ingest(events[0])}")

    correction = next((e for e in events if e["revision"] > 1), None)
    if correction is None:
        print("    (no correction in this data)")
    else:
        original = next(e for e in events if e["event_id"] == correction["event_id"]
                        and e["entity_id"] == correction["entity_id"] and e["revision"] == 1)
        decision = max((r for r in rows if r["entity_id"] == correction["entity_id"]
                        and at(r["as_of"]) < at(correction["received_at"])),
                       key=lambda r: r["as_of"], default=rows[0])
        late = RiskEngine(OUT / "model")
        replay(late, [e for e in events if e is not correction])
        before = late.score(decision["entity_id"], at(decision["as_of"])).to_wire()
        late.ingest(correction)
        after = late.score(decision["entity_id"], at(decision["as_of"])).to_wire()
        print(f"    correction {correction['event_id']} r{correction['revision']} received {correction['received_at']} "
              f"after decision {decision['as_of']}: decision unchanged = {before == after}")

        flavor_a = RiskEngine(OUT / "model")
        flavor_a.ingest(correction)
        print(f"    reversed delivery (rev 2 delivered first, rev 1 received earlier) -> rev 1 ingest returned "
              f"{flavor_a.ingest(original)}")
        flavor_b = RiskEngine(OUT / "model")
        flavor_b.ingest(correction)
        received_later = dict(original, received_at=format_timestamp(at(correction["received_at"]).replace(minute=59)))
        print(f"    reversed receipt (rev 1 received after rev 2) -> rev 1 ingest returned {flavor_b.ingest(received_later)}")

    bounded = RiskEngine(OUT / "model", max_entities=20)
    last_ingest = {}                     # shipment -> position of its last accepted record
    for position, e in enumerate(events):
        try:
            bounded.ingest(e)
            last_ingest[e["entity_id"]] = position
        except RetentionError:
            pass
    by_recency = sorted(last_ingest, key=last_ingest.get)       # oldest first; last 20 are resident
    stats = bounded.stats()
    print(f"    eviction (max_entities=20): resident {stats['resident_entities']}, "
          f"evicted {stats['counters'].get('evicted', 0)}, eviction list remembers {stats['evicted_remembered']}")
    latest_receipt = max(at(e["received_at"]) for e in events)
    for label, shipment in (("recently evicted", by_recency[-21]), ("evicted long ago", by_recency[0])):
        original_event = next(e for e in events if e["entity_id"] == shipment)
        late_fix = dict(original_event, revision=original_event["revision"] + 1, value=original_event["value"] - 1.0,
                        received_at=format_timestamp(latest_receipt.replace(minute=59)))
        try:
            bounded.ingest(late_fix)
            verdict = "ACCEPTED as a new shipment (known limitation: the eviction list is bounded)"
        except RetentionError as exc:
            verdict = f"rejected: {exc.code}"
        print(f"    late correction for {label} shipment {shipment} -> {verdict}")

    broken = OUT / "broken_model"
    broken.mkdir()
    (broken / "model.json").write_text("{not json")
    sample = rows[0]
    before = engine.score(sample["entity_id"], at(sample["as_of"])).to_wire()
    ok = engine.reload_model(broken)
    after = engine.score(sample["entity_id"], at(sample["as_of"])).to_wire()
    print(f"    failed reload -> returned {ok}, last_error {engine.stats()['last_reload_error']}, "
          f"previous model still serving identical scores = {before == after}")

    print("[6] Uninterrupted replay vs snapshot/restore replay")
    half = len(events) // 2
    summary = {}
    for max_entities in (10000, 20):
        straight = RiskEngine(OUT / "model", max_entities)
        replay(straight, events)
        first_half = RiskEngine(OUT / "model", max_entities)
        replay(first_half, events[:half])
        snap = OUT / f"snapshot_half_{max_entities}.json"
        first_half.snapshot(snap)
        resumed = RiskEngine.restore(OUT / "model", snap)
        replay(resumed, events[half:])
        straight.snapshot(OUT / f"final_straight_{max_entities}.json")
        resumed.snapshot(OUT / f"final_resumed_{max_entities}.json")
        same_bytes = (OUT / f"final_straight_{max_entities}.json").read_bytes() == \
            (OUT / f"final_resumed_{max_entities}.json").read_bytes()
        same_outcomes = outcomes(straight, rows) == outcomes(resumed, rows)
        summary[max_entities] = {"same_final_snapshot": same_bytes, "same_720_outcomes": same_outcomes}
        print(f"    max_entities={max_entities}: identical final snapshot {same_bytes}, "
              f"identical decision outcomes {same_outcomes}")

    (OUT / "summary.json").write_text(json.dumps({"metrics": metrics, "report": report, "replay": summary},
                                                 sort_keys=True, indent=1) + "\n")
    print(f"Done. Outputs in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
