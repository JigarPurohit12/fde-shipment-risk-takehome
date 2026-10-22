"""Strict reading of the supplied files: JSONL records and MANIFEST.json.

This module only answers "is each line a well-formed JSON object?".
Whether that object is a valid event/label/decision is checked in _records,
because the public methods receive mappings directly, not files.

Python's json module is more permissive than JSON itself, so we tighten it:
  * NaN / Infinity / -Infinity are accepted by json.loads by default -> rejected
  * {"value": 1, "value": 2} silently keeps the last value by default -> rejected
Problems are reported with file name and line number.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ._time import parse_timestamp
from .errors import InvalidRecordError

SUPPORTED_MANIFEST_SCHEMA = 1


@dataclass(frozen=True)
class Manifest:
    observation_end: datetime        # aware UTC datetime
    label_reporting_lag_hours: int
    files: dict                      # file name -> row count, for sanity checks only


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Return one dict per line. Raises InvalidRecordError on any bad line.

    A missing file raises the normal FileNotFoundError: that is an
    environment problem, not a malformed record.
    """
    path = Path(path)
    rows: list[dict[str, object]] = []
    try:
        # Iterating a text file splits only on \n, \r\n and \r. (str.splitlines
        # would also split on U+2028 and similar characters that are legal
        # inside a JSON string, breaking a valid record in two.)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                rows.append(_parse_line(line, f"{path.name}:{line_number}"))
    except UnicodeDecodeError as exc:
        raise InvalidRecordError(
            "invalid_encoding", f"{path.name} is not valid UTF-8 ({exc.reason})"
        ) from None
    return rows


def read_manifest(path: str | Path) -> Manifest:
    """Read and validate MANIFEST.json (same strict JSON rules as JSONL)."""
    path = Path(path)
    data = _parse_line(path.read_text(encoding="utf-8"), path.name)
    version = data.get("schema_version")
    if type(version) is not int or version != SUPPORTED_MANIFEST_SCHEMA:   # True == 1 in Python
        raise InvalidRecordError(
            "unsupported_schema_version",
            f"{path.name}: schema_version={version!r}, expected {SUPPORTED_MANIFEST_SCHEMA}",
        )
    lag = data.get("label_reporting_lag_hours")
    if type(lag) is not int or lag < 1:
        raise InvalidRecordError(
            "invalid_label_lag", f"{path.name}: label_reporting_lag_hours={lag!r}"
        )
    files = data.get("files", {})
    if not isinstance(files, dict):
        raise InvalidRecordError("invalid_manifest", f"{path.name}: files must be an object")
    return Manifest(
        observation_end=parse_timestamp(data.get("observation_end"), "manifest.observation_end"),
        label_reporting_lag_hours=lag,
        files=dict(files),
    )


def parse_json_object(text: str, where: str) -> dict[str, object]:
    """Strictly parse one JSON object (same rules as a JSONL line)."""
    return _parse_line(text, where)


def write_atomic(path: str | Path, data: bytes) -> None:
    """Write a file so readers see either the old or the new content, never half.

    Write to a temporary file in the same directory, flush it to disk, then
    os.replace() it over the target (an atomic rename on the same filesystem).
    """
    path = Path(path)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _parse_line(line: str, where: str) -> dict[str, object]:
    if not line.strip():
        raise InvalidRecordError("blank_line", f"{where} is empty")
    try:
        value = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as exc:
        raise InvalidRecordError("invalid_json", f"{where}: {exc.msg}") from None
    except RecursionError:  # absurdly deep nesting like [[[[...]]]]
        raise InvalidRecordError("invalid_json", f"{where}: nesting is too deep") from None
    except InvalidRecordError as exc:  # raised by our hooks below; add location
        raise InvalidRecordError(exc.code, f"{where}: {exc.detail}") from None
    if not isinstance(value, dict):
        raise InvalidRecordError(
            "not_a_mapping",
            f"{where} is a JSON {type(value).__name__}, expected an object",
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidRecordError("duplicate_json_key", f"key {key!r} appears twice")
        result[key] = value
    return result


def _reject_non_finite_constant(name: str) -> object:
    raise InvalidRecordError("non_finite_value", f"{name} is not allowed in JSON input")
