"""Step 1: timestamp parsing, formatting, and datetime-argument checks."""
import unittest
from datetime import date, datetime, timedelta, timezone

from shipment_risk._time import (
    UTC,
    format_timestamp,
    parse_timestamp,
    require_aware_datetime,
)
from shipment_risk.errors import InvalidArgumentError, InvalidRecordError

PLUS_0530 = timezone(timedelta(hours=5, minutes=30))


class ParseTimestampTests(unittest.TestCase):
    def assert_code(self, text, code):
        with self.assertRaises(InvalidRecordError) as ctx:
            parse_timestamp(text, "t")
        self.assertEqual(ctx.exception.code, code)

    def test_equivalent_offsets_become_the_same_utc_instant(self):
        expected = datetime(2026, 1, 1, tzinfo=UTC)
        for text in (
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00-00:00",
            "2026-01-01T05:30:00+05:30",
            "2025-12-31T21:00:00-03:00",
        ):
            with self.subTest(text=text):
                parsed = parse_timestamp(text, "t")
                self.assertEqual(parsed, expected)
                self.assertEqual(parsed.utcoffset(), timedelta(0))

    def test_fractional_seconds_are_padded_not_truncated(self):
        self.assertEqual(parse_timestamp("2026-01-01T00:00:00.5Z", "t").microsecond, 500000)
        self.assertEqual(parse_timestamp("2026-01-01T00:00:00.123456Z", "t").microsecond, 123456)

    def test_missing_offset_is_rejected_never_assumed_utc(self):
        for text in ("2026-01-01T00:00:00", "2026-01-01T00:00:00.250"):
            with self.subTest(text=text):
                self.assert_code(text, "missing_utc_offset")

    def test_malformed_strings_are_rejected(self):
        for text in (
            "",
            "2026-01-01",                      # date only
            "2026-01-01 00:00:00Z",            # space instead of T
            "2026-01-01t00:00:00Z",            # lowercase t
            "2026-01-01T00:00:00z",            # lowercase z
            " 2026-01-01T00:00:00Z",           # leading space
            "2026-01-01T00:00:00Z ",           # trailing space
            "2026-13-01T00:00:00Z",            # month 13
            "2026-02-30T00:00:00Z",            # Feb 30
            "2026-01-01T24:00:00Z",            # hour 24
            "2026-01-01T00:00:60Z",            # leap second
            "2026-01-01T00:00:00.1234567Z",    # 7 fractional digits
            "2026-01-01T00:00:00+24:00",       # offset hours too big
            "2026-01-01T00:00:00+05:60",       # offset minutes too big
            "2026-01-01T00:00:00+0530",        # offset without colon
            "20260101T000000Z",                # ISO basic format
            "2026-W01-1T00:00:00Z",            # ISO week date
            "\u0662\u0660\u0662\u0666-01-01T00:00:00Z",  # Arabic-Indic digits
            "2026-01-01T00:00:00Z" + "0" * 30,  # too long
        ):
            with self.subTest(text=text):
                self.assert_code(text, "invalid_timestamp")

    def test_non_strings_are_rejected(self):
        for value in (None, 1767225600, 1.5, b"2026-01-01T00:00:00Z",
                      datetime(2026, 1, 1, tzinfo=UTC)):
            with self.subTest(value=value):
                self.assert_code(value, "invalid_timestamp")

    def test_years_outside_supported_range_are_rejected(self):
        for text in ("0000-01-01T00:00:00Z", "1899-12-31T23:59:59Z", "9999-01-01T00:00:00Z"):
            with self.subTest(text=text):
                self.assert_code(text, "timestamp_out_of_range")


class FormatTimestampTests(unittest.TestCase):
    def test_output_is_always_utc_with_z(self):
        self.assertEqual(
            format_timestamp(datetime(2026, 1, 1, 5, 30, tzinfo=PLUS_0530)),
            "2026-01-01T00:00:00Z",
        )

    def test_microseconds_only_when_non_zero_and_always_six_digits(self):
        self.assertEqual(format_timestamp(datetime(2026, 1, 1, tzinfo=UTC)), "2026-01-01T00:00:00Z")
        self.assertEqual(
            format_timestamp(datetime(2026, 1, 1, 0, 0, 0, 500000, tzinfo=UTC)),
            "2026-01-01T00:00:00.500000Z",
        )

    def test_round_trip_keeps_the_instant(self):
        for text in ("2026-01-03T14:00:00Z", "2026-01-03T19:30:00.25+05:30", "2026-01-03T11:00:00-03:00"):
            with self.subTest(text=text):
                parsed = parse_timestamp(text, "t")
                self.assertEqual(parse_timestamp(format_timestamp(parsed), "t"), parsed)

    def test_naive_or_non_datetime_is_rejected(self):
        with self.assertRaises(InvalidArgumentError) as ctx:
            format_timestamp(datetime(2026, 1, 1))
        self.assertEqual(ctx.exception.code, "naive_datetime")
        with self.assertRaises(InvalidArgumentError) as ctx:
            format_timestamp("2026-01-01T00:00:00Z")
        self.assertEqual(ctx.exception.code, "not_a_datetime")


class RequireAwareDatetimeTests(unittest.TestCase):
    def assert_code(self, value, code):
        with self.assertRaises(InvalidArgumentError) as ctx:
            require_aware_datetime(value, "as_of")
        self.assertEqual(ctx.exception.code, code)

    def test_aware_datetime_is_converted_to_utc(self):
        result = require_aware_datetime(datetime(2026, 1, 1, 5, 30, tzinfo=PLUS_0530), "as_of")
        self.assertEqual(result, datetime(2026, 1, 1, tzinfo=UTC))
        self.assertEqual(result.utcoffset(), timedelta(0))

    def test_naive_datetime_is_rejected(self):
        self.assert_code(datetime(2026, 1, 1), "naive_datetime")

    def test_non_datetimes_are_rejected(self):
        for value in (date(2026, 1, 1), "2026-01-01T00:00:00Z", None, 0):
            with self.subTest(value=value):
                self.assert_code(value, "not_a_datetime")

    def test_out_of_range_years_are_rejected(self):
        self.assert_code(datetime(1800, 1, 1, tzinfo=UTC), "timestamp_out_of_range")
        # Converting year 1 with a positive offset to UTC would overflow.
        self.assert_code(
            datetime(1, 1, 1, tzinfo=timezone(timedelta(hours=5))), "timestamp_out_of_range"
        )


if __name__ == "__main__":
    unittest.main()
