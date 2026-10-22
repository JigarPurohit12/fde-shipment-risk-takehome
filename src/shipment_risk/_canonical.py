"""Canonical JSON: one value -> exactly one byte string.

Same settings as contracts.Result.to_wire() (README "Canonical JSON must sort
object keys, preserve list order, reject NaN/infinity, and contain no
wall-clock metadata"):
  * sort_keys=True          object keys sorted at every nesting level
  * separators=(",", ":")   no optional whitespace
  * allow_nan=False         NaN / Infinity raise ValueError
  * lists keep their order  so callers must build lists deterministically

Used for: feature_digest (step 3), artifact version (step 5),
snapshot files and their checksums (step 7).
"""
from __future__ import annotations

import hashlib
import json


def canonical_json(value: object) -> bytes:
    """Serialize plain JSON data (dict/list/str/int/float/bool/None)."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_digest(value: object) -> str:
    """Fingerprint of a value's canonical JSON, e.g. 'sha256:3f2a...'."""
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()
