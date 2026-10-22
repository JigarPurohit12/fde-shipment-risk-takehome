"""The single place where time enters and leaves the package.

Input  : strings like 2026-01-03T14:00:00Z or 2026-01-03T19:30:00+05:30
         (RFC 3339 profile of ISO 8601, explicit offset required).
Inside : aware datetime objects, always converted to UTC.
Output : strings like 2026-01-03T14:00:00Z (".ffffff" only if needed).

Because everything is converted to UTC on entry, two strings that name the
same instant with different offsets become equal values.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .errors import InvalidArgumentError, InvalidRecordError

UTC = timezone.utc

# Sanity bounds. They exist only so later arithmetic such as
# as_of + 6h + 30h can never overflow Python's datetime range (years 1-9999).
# A plausible-looking but wrong device clock (e.g. a sensor reset to 1970) is
# NOT rejected here; the clock policy in a later step decides what it means.
MIN_YEAR = 1900
MAX_YEAR = 9998

# Longest valid form is 32 characters (6 fractional digits + "+HH:MM").
MAX_TIMESTAMP_LENGTH = 40

# [0-9] instead of \d on purpose: in Python, \d also matches non-ASCII digits
# such as the Arabic-Indic "٢", and int("٢٠٢٦") happily returns 2026.
_TIMESTAMP_RE = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})"   # date
    r"T([0-9]{2}):([0-9]{2}):([0-9]{2})"  # time (uppercase T only)
    r"(?:\.([0-9]{1,6}))?"                # optional fraction, max microseconds
    r"(Z|[+-][0-9]{2}:[0-9]{2})?"         # offset (optional here so we can
)                                         # report "missing" separately)


def parse_timestamp(value: object, field: str) -> datetime:
    """Parse an input timestamp string and return an aware UTC datetime.

    Raises InvalidRecordError with code:
      invalid_timestamp       wrong type, wrong format, impossible date/time
      missing_utc_offset      well-formed but no offset (we never guess UTC)
      timestamp_out_of_range  year outside MIN_YEAR..MAX_YEAR
    """
    if not isinstance(value, str):
        raise InvalidRecordError(
            "invalid_timestamp",
            f"{field} must be a string, got {type(value).__name__}",
        )
    if len(value) > MAX_TIMESTAMP_LENGTH:
        raise InvalidRecordError("invalid_timestamp", f"{field} is too long")

    match = _TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise InvalidRecordError(
            "invalid_timestamp",
            f"{field}={value!r} is not YYYY-MM-DDTHH:MM:SS[.ffffff] plus Z or +HH:MM",
        )

    year, month, day, hour, minute, second, fraction, offset = match.groups()
    if offset is None:
        raise InvalidRecordError(
            "missing_utc_offset", f"{field}={value!r} has no UTC offset"
        )
    if not MIN_YEAR <= int(year) <= MAX_YEAR:
        raise InvalidRecordError(
            "timestamp_out_of_range",
            f"{field}={value!r} is outside years {MIN_YEAR}-{MAX_YEAR}",
        )

    # ".5" means 500000 microseconds, so pad on the right before converting.
    microsecond = int(fraction.ljust(6, "0")) if fraction else 0
    try:
        local = datetime(
            int(year), int(month), int(day),
            int(hour), int(minute), int(second), microsecond,
            tzinfo=_parse_offset(offset),
        )
    except ValueError as exc:  # month 13, Feb 30, hour 24, second 60, bad offset
        raise InvalidRecordError(
            "invalid_timestamp", f"{field}={value!r}: {exc}"
        ) from None
    return local.astimezone(UTC)


def _parse_offset(offset: str) -> timezone:
    """Turn "Z", "+05:30" or "-03:00" into a timezone object."""
    if offset == "Z":
        return UTC
    hours, minutes = int(offset[1:3]), int(offset[4:6])
    if hours > 23 or minutes > 59:
        raise ValueError(f"offset {offset} is out of range")
    sign = -1 if offset[0] == "-" else 1
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def require_aware_datetime(value: object, name: str) -> datetime:
    """Validate a datetime ARGUMENT (as_of, observation_end); return it in UTC.

    Raises InvalidArgumentError with code:
      not_a_datetime          anything that is not a datetime (str, date, None)
      naive_datetime          a datetime without timezone information
      timestamp_out_of_range  year outside MIN_YEAR..MAX_YEAR
    """
    utc = _to_utc(value, name)
    if not MIN_YEAR <= utc.year <= MAX_YEAR:
        raise InvalidArgumentError(
            "timestamp_out_of_range",
            f"{name}={value!r} is outside years {MIN_YEAR}-{MAX_YEAR}",
        )
    return utc


def format_timestamp(moment: datetime) -> str:
    """Format an aware datetime as canonical UTC text.

    Always YYYY-MM-DDTHH:MM:SSZ, plus .ffffff only when microseconds are not
    zero, so one instant always produces exactly one string. Built with
    f-strings rather than strftime, whose year padding differs by platform.
    """
    utc = _to_utc(moment, "timestamp")
    text = (
        f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}"
        f"T{utc.hour:02d}:{utc.minute:02d}:{utc.second:02d}"
    )
    if utc.microsecond:
        text += f".{utc.microsecond:06d}"
    return text + "Z"


def _to_utc(value: object, name: str) -> datetime:
    # isinstance(date_obj, datetime) is False, so plain dates are rejected too.
    if not isinstance(value, datetime):
        raise InvalidArgumentError(
            "not_a_datetime",
            f"{name} must be a datetime, got {type(value).__name__}",
        )
    # utcoffset() is None for naive datetimes AND for odd tzinfo objects that
    # cannot say their offset; both are rejected rather than assumed UTC.
    if value.utcoffset() is None:
        raise InvalidArgumentError("naive_datetime", f"{name} must be timezone-aware")
    try:
        return value.astimezone(UTC)
    except OverflowError:  # e.g. year 1 with a positive offset
        raise InvalidArgumentError(
            "timestamp_out_of_range", f"{name}={value!r} cannot be converted to UTC"
        ) from None
