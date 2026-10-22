"""The ONE feature function, shared by training and live scoring.

    KnownReadings (step 2)  ->  compute_features()  ->  FeatureSet

Both build_training_rows (step 4) and RiskEngine.score (step 6) call
EventStore.known_at() and then compute_features(). Because there is only one
implementation, "offline and online features must agree" holds by
construction; the digest lets tests prove it.

Principles:
  * Features are facts about the readings known at as_of. A fact that does
    not exist (no readings yet) is None, never an invented number. How the
    model fills a None is the model's job (step 5), recorded in its artifact.
  * Only readings whose device time is inside the look-back window
    (as_of - LOOKBACK, as_of + clock tolerance] are used. This also makes a
    device clock stuck far in the past irrelevant.
  * Deliberately NO "hours since shipment start" or "decision hour" feature:
    in the synthetic data only one decision offset is ever positive, so such
    a feature would memorize the generator's schedule instead of learning
    from temperatures, and would break on data with other decision times.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

from ._canonical import sha256_digest
from ._store import KnownReadings

FEATURE_SCHEMA_VERSION = "features-v1"   # bump whenever a definition changes

LOOKBACK = timedelta(hours=6)            # same length as the prediction horizon
_HOUR = timedelta(hours=1)

# Data-quality reason codes (step 6 turns them into score reasons).
NO_READINGS_IN_WINDOW = "no_readings_in_window"
# Readings excluded by the store's clock policy keep their store reason code,
# e.g. "future_device_clock".

FEATURE_NAMES = (
    "carrier",                     # str | None  carrier on the latest usable reading
    "hours_since_latest_reading",  # float       0..6, capped at the window length
    "latest_temp_c",               # float | None
    "max_temp_c",                  # float | None
    "mean_temp_c",                 # float | None
    "reading_count",               # int         usable readings in the window
    "trend_c_per_hour",            # float | None least-squares slope, needs 2+ times
)


@dataclass(frozen=True)
class FeatureSet:
    values: dict          # feature name -> value; treat as read-only
    reasons: tuple        # sorted data-quality reason codes
    digest: str           # "sha256:..." of schema version + values


def compute_features(known: KnownReadings) -> FeatureSet:
    as_of = known.as_of
    window_start = as_of - LOOKBACK
    # known.readings is already sorted by (event_at, event_id) -> deterministic.
    window = [e for e in known.readings if e.event_at > window_start]

    temps = [e.value for e in window]
    if window:
        latest = window[-1]
        age = max(timedelta(0), as_of - latest.event_at)   # small clock skew -> 0
        hours_since_latest = min(age, LOOKBACK) / _HOUR
        latest_temp = latest.value
        max_temp = max(temps)
        mean_temp = math.fsum(temps) / len(temps)
    else:
        hours_since_latest = LOOKBACK / _HOUR              # "at least 6 hours"
        latest_temp = max_temp = mean_temp = None

    # Carrier describes the shipment, not the moment, so any usable reading
    # counts; the latest one wins if a shipment changes carrier.
    carrier = known.readings[-1].carrier if known.readings else None

    values = {
        "carrier": carrier,
        "hours_since_latest_reading": _clean(hours_since_latest),
        "latest_temp_c": _clean(latest_temp),
        "max_temp_c": _clean(max_temp),
        "mean_temp_c": _clean(mean_temp),
        "reading_count": len(window),
        "trend_c_per_hour": _clean(_trend_per_hour(window, as_of)),
    }

    reasons = {reason for reason, _count in known.excluded}
    if not window:
        reasons.add(NO_READINGS_IN_WINDOW)

    return FeatureSet(values=values, reasons=tuple(sorted(reasons)), digest=feature_digest(values))


def feature_digest(values: dict) -> str:
    """Fingerprint of the features AND their definition version."""
    return sha256_digest({"schema": FEATURE_SCHEMA_VERSION, "features": values})


def _trend_per_hour(window, as_of) -> float | None:
    """Least-squares slope of temperature over device time, in degrees C per hour."""
    xs = [(e.event_at - as_of) / _HOUR for e in window]
    if len(set(xs)) < 2:
        return None                                      # 0 or 1 distinct time points
    vs = [e.value for e in window]
    x_mean = math.fsum(xs) / len(xs)
    v_mean = math.fsum(vs) / len(vs)
    numerator = math.fsum((x - x_mean) * (v - v_mean) for x, v in zip(xs, vs))
    denominator = math.fsum((x - x_mean) ** 2 for x in xs)
    return numerator / denominator


def _clean(number: float | None) -> float | None:
    """Plain float with -0.0 turned into 0.0, so equal values serialize equally."""
    return None if number is None else float(number) + 0.0
