"""Step 1: strict JSONL reading."""
import tempfile
import unittest
from pathlib import Path

from shipment_risk._io import read_jsonl
from shipment_risk.errors import InvalidRecordError


class ReadJsonlTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "input.jsonl"

    def write_text(self, text):
        self.path.write_bytes(text.encode("utf-8"))

    def assert_code(self, code, line_number=None):
        with self.assertRaises(InvalidRecordError) as ctx:
            read_jsonl(self.path)
        self.assertEqual(ctx.exception.code, code)
        if line_number is not None:
            self.assertIn(f"input.jsonl:{line_number}", ctx.exception.detail)

    def test_reads_one_object_per_line_including_crlf(self):
        self.write_text('{"a":1}\r\n{"a":2}\n')
        self.assertEqual(read_jsonl(self.path), [{"a": 1}, {"a": 2}])

    def test_empty_file_gives_no_rows(self):
        self.write_text("")
        self.assertEqual(read_jsonl(self.path), [])

    def test_line_separator_inside_a_string_stays_one_record(self):
        self.write_text('{"note":"a\u2028b"}\n')
        self.assertEqual(read_jsonl(self.path), [{"note": "a\u2028b"}])

    def test_nan_and_infinity_are_rejected(self):
        for literal in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(literal=literal):
                self.write_text('{"a":1}\n{"value":' + literal + "}\n")
                self.assert_code("non_finite_value", line_number=2)

    def test_duplicate_keys_are_rejected(self):
        self.write_text('{"value":1,"value":2}\n')
        self.assert_code("duplicate_json_key", line_number=1)

    def test_blank_line_is_rejected(self):
        self.write_text('{"a":1}\n\n{"a":2}\n')
        self.assert_code("blank_line", line_number=2)

    def test_non_object_line_is_rejected(self):
        for text in ("[1,2]", '"text"', "5", "null"):
            with self.subTest(text=text):
                self.write_text(text + "\n")
                self.assert_code("not_a_mapping", line_number=1)

    def test_broken_json_is_rejected(self):
        self.write_text('{"a":1}\n{"a":\n')
        self.assert_code("invalid_json", line_number=2)

    def test_deep_nesting_is_rejected_not_crashing(self):
        self.write_text("[" * 100_000 + "]" * 100_000 + "\n")
        self.assert_code("invalid_json", line_number=1)

    def test_invalid_utf8_is_rejected(self):
        self.path.write_bytes(b'{"a":"\xff"}\n')
        self.assert_code("invalid_encoding")

    def test_missing_file_is_an_os_error_not_a_record_error(self):
        with self.assertRaises(FileNotFoundError):
            read_jsonl(Path(self._tmp.name) / "nope.jsonl")


if __name__ == "__main__":
    unittest.main()
