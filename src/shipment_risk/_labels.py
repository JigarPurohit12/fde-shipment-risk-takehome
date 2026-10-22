"""Label rules in ONE place (README "Supplied data and semantics").

  * A decision at t is positive iff an incident occurs in (t, t + HORIZON]
    and its label was available by the cutoff we are allowed to know.
  * A decision's label is trustworthy (mature) only if
        t + HORIZON + LABEL_REPORTING_LAG <= cutoff,
    for positives AND negatives alike.

Step 4 applies these with cutoff = observation_end. Step 5 applies the same
functions again with cutoff = the temporal split time.
A requirement change such as "horizon is now 4 hours" is a one-line edit here.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from ._records import Label
from .errors import InvalidRecordError

HORIZON = timedelta(hours=6)
LABEL_REPORTING_LAG = timedelta(hours=30)   # must match MANIFEST label_reporting_lag_hours


def label_mature_at(as_of: datetime) -> datetime:
    """The moment from which this decision's label can be trusted."""
    return as_of + HORIZON + LABEL_REPORTING_LAG


def is_mature(as_of: datetime, cutoff: datetime) -> bool:
    return label_mature_at(as_of) <= cutoff


def incidents_in_window(labels: Iterable[Label], as_of: datetime, known_by: datetime) -> list[Label]:
    """Incidents in (as_of, as_of + HORIZON] whose label was available by known_by.

    Open at as_of: an incident at exactly as_of is already happening, not a
    future risk. Closed at as_of + HORIZON: an incident at exactly six hours
    still counts. Sorted for deterministic output.
    """
    window_end = as_of + HORIZON
    found = [
        label for label in labels
        if as_of < label.incident_at <= window_end and label.label_available_at <= known_by
    ]
    return sorted(found, key=lambda label: (label.incident_at, label.incident_id))


def check_reporting_lag(manifest_lag_hours: int) -> None:
    """Refuse to build labels if the data promises a different reporting lag."""
    expected = LABEL_REPORTING_LAG / timedelta(hours=1)
    if manifest_lag_hours != expected:
        raise InvalidRecordError(
            "label_lag_mismatch",
            f"manifest label_reporting_lag_hours={manifest_lag_hours}, code assumes {expected:g}; "
            "update LABEL_REPORTING_LAG deliberately before training",
        )
