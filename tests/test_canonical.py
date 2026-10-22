"""Step 3: canonical JSON and digests."""
import hashlib
import unittest

from shipment_risk._canonical import canonical_json, sha256_digest
from shipment_risk.contracts import Result


class CanonicalJsonTests(unittest.TestCase):
    def test_keys_sorted_at_every_level_and_no_whitespace(self):
        self.assertEqual(
            canonical_json({"b": 1, "a": {"d": 2, "c": 3}}),
            b'{"a":{"c":3,"d":2},"b":1}',
        )

    def test_list_order_is_preserved(self):
        self.assertEqual(canonical_json({"x": [3, 1, 2]}), b'{"x":[3,1,2]}')

    def test_non_finite_numbers_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    canonical_json({"x": bad})

    def test_non_ascii_is_escaped_deterministically(self):
        self.assertEqual(canonical_json({"x": "\u00e9"}), b'{"x":"\\u00e9"}')

    def test_same_bytes_as_result_to_wire(self):
        payload = {"z": 1, "a": [2, 1], "m": None}
        wire = Result("ship-1", "2026-01-01T00:00:00Z", payload).to_wire()
        ours = canonical_json({"entity_id": "ship-1", "as_of": "2026-01-01T00:00:00Z", "payload": payload})
        self.assertEqual(ours, wire)

    def test_digest_is_sha256_of_canonical_bytes(self):
        value = {"b": [1, 2], "a": "x"}
        expected = "sha256:" + hashlib.sha256(b'{"a":"x","b":[1,2]}').hexdigest()
        self.assertEqual(sha256_digest(value), expected)
        self.assertEqual(sha256_digest({"a": "x", "b": [1, 2]}), expected)   # key order irrelevant


if __name__ == "__main__":
    unittest.main()
