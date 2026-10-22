"""Error types that shipment_risk raises on purpose.

Every error carries:
  * code   - a short, stable, machine-readable string (for example
             "missing_utc_offset"). Codes are documented in DECISIONS.md and
             will be counted in RiskEngine.stats().
  * detail - a human-readable explanation for logs and people.

Input errors also inherit from ValueError, so a caller who only knows the
standard library (``except ValueError``) still catches them.
"""
from __future__ import annotations


class ShipmentRiskError(Exception):
    """Base class for every error this package raises deliberately."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class InvalidRecordError(ShipmentRiskError, ValueError):
    """An input record (event, label, decision, or JSONL line) is malformed."""


class InvalidArgumentError(ShipmentRiskError, ValueError):
    """A method argument is malformed, for example a naive datetime."""


class ConflictError(ShipmentRiskError, ValueError):
    """The same identity (entity_id, event_id, revision) arrived again with
    different content. The first version is kept; the new one is rejected."""


class ArtifactError(ShipmentRiskError, ValueError):
    """A model artifact is missing, corrupt, or incompatible with this code."""


class RetentionError(ShipmentRiskError, ValueError):
    """Outside the memory policy: the shipment was evicted, or its history is full."""


class SnapshotError(ShipmentRiskError, ValueError):
    """A snapshot file is missing, unreadable, or not a valid engine snapshot."""
