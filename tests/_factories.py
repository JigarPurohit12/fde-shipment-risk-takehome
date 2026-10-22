"""Builders for valid records. Each test overrides only the field it is about.

The file name starts with "_" so neither pytest nor unittest collects it as
a test module. IDs are invented here on purpose: tests must not depend on
IDs from the generated sample.
"""
from datetime import timedelta

from shipment_risk._features import FEATURE_SCHEMA_VERSION, feature_digest
from shipment_risk._records import validate_event


def event_record(**overrides):
    record = {
        "carrier": "coast",
        "entity_id": "ship-1",
        "event_at": "2026-01-01T00:00:00Z",
        "event_id": "reading-1",
        "kind": "temperature_c",
        "received_at": "2026-01-01T00:02:00Z",
        "revision": 1,
        "value": 4.25,
    }
    record.update(overrides)
    return record


def label_record(**overrides):
    record = {
        "entity_id": "ship-1",
        "incident_at": "2026-01-01T18:00:00Z",
        "incident_id": "inc-1",
        "label_available_at": "2026-01-02T12:00:00Z",
    }
    record.update(overrides)
    return record


def decision_record(**overrides):
    record = {"as_of": "2026-01-01T14:00:00Z", "entity_id": "ship-1"}
    record.update(overrides)
    return record


def make_event(**overrides):
    """A validated Event (runs the real step-1 validator)."""
    return validate_event(event_record(**overrides))


def training_row(entity_id="ship-1", as_of="2026-01-01T14:00:00Z", latest=4.0, label=0,
                 carrier="coast", label_available_at=None, eligible=True):
    """A training row shaped like build_training_rows output (step 4)."""
    features = {
        "carrier": carrier,
        "hours_since_latest_reading": 0.5,
        "latest_temp_c": latest,
        "max_temp_c": latest,
        "mean_temp_c": latest - 0.5,
        "reading_count": 5,
        "trend_c_per_hour": (latest - 4.0) / 4.0,
    }
    incidents = []
    if label:
        incidents = [{"entity_id": entity_id, "incident_at": as_of, "incident_id": f"inc-{entity_id}",
                      "label_available_at": label_available_at or as_of}]
    return {
        "entity_id": entity_id,
        "as_of": as_of,
        "features": features,
        "label": label if eligible else None,
        "metadata": {"eligible": eligible, "feature_schema": FEATURE_SCHEMA_VERSION,
                     "feature_digest": feature_digest(features), "incidents": incidents if eligible else [],
                     "censor_reason": None if eligible else "label_window_immature"},
    }


def synthetic_training_rows(count=120):
    """One decision every 4 h from a new shipment; every 6th one warms up and is positive."""
    from shipment_risk._time import format_timestamp, parse_timestamp

    start = parse_timestamp("2026-01-01T00:00:00Z", "start")
    return [
        training_row(entity_id=f"ship-{i:03d}", as_of=format_timestamp(start + timedelta(hours=4 * i)),
                     latest=8.5 if i % 6 == 0 else 4.0 + (i % 5) * 0.2, label=int(i % 6 == 0),
                     carrier=["coast", "north", "central"][i % 3])
        for i in range(count)
    ]
