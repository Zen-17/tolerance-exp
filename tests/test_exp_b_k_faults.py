"""CPU unit tests for experiment B K-element faults (no vLLM)."""

from __future__ import annotations

import unittest

import torch

from smallexp2.analyze_b import lifetime_harm
from smallexp2.k_faults import KFaultSpec, apply_k_element, bit_for_class


class TestKNumeric(unittest.TestCase):
    def test_numeric_writes_parent(self):
        cache = torch.zeros(2, 4, 16, 8, 128, dtype=torch.bfloat16)
        cache[0, 1, 3, 2, 7] = 2.0
        elem = cache[0, 1, 3, 2, 7]
        spec = KFaultSpec(kind="numeric", kv_head=2, dim=7, rel=0.1)
        rec = apply_k_element(elem, spec, torch.Generator().manual_seed(0))
        self.assertAlmostEqual(rec["old_value"], 2.0, places=3)
        self.assertAlmostEqual(abs(rec["intended_delta"]), 0.1 * (2.0 + 1e-8), places=4)
        self.assertAlmostEqual(
            abs(float(cache[0, 1, 3, 2, 7].item()) - 2.0),
            rec["abs_delta"],
            places=5,
        )
        self.assertFalse(rec.get("quantized_away"))
        self.assertNotEqual(int(rec["old_bits"]), int(rec["new_bits"]))

    def test_tiny_rel_records_stored_delta(self):
        t = torch.tensor([1.0], dtype=torch.bfloat16)
        spec = KFaultSpec(kind="numeric", kv_head=0, dim=0, rel=1e-6)
        rec = apply_k_element(t[0], spec, torch.Generator().manual_seed(0))
        stored = abs(float(t[0].item()) - rec["old_value"])
        self.assertAlmostEqual(rec["abs_delta"], stored, places=6)


class TestKBitflip(unittest.TestCase):
    def test_sign_bit_negates_one(self):
        t = torch.tensor([1.0], dtype=torch.bfloat16)
        spec = KFaultSpec(
            kind="bitflip", kv_head=0, dim=0, bit_class="sign",
            bit=bit_for_class("sign"),
        )
        rec = apply_k_element(t[0], spec, torch.Generator().manual_seed(0))
        self.assertAlmostEqual(float(t[0].item()), -1.0, places=4)
        self.assertEqual(rec["bit"], 15)
        self.assertEqual(rec["old_bits"] ^ rec["new_bits"], 1 << 15)

    def test_mantissa_changes_parent_cache(self):
        cache = torch.zeros(2, 2, 8, 4, 16, dtype=torch.bfloat16)
        cache[0, 0, 1, 1, 3] = 1.0
        spec = KFaultSpec(
            kind="bitflip", kv_head=1, dim=3, bit_class="mantissa",
            bit=bit_for_class("mantissa"),
        )
        apply_k_element(cache[0, 0, 1, 1, 3], spec, torch.Generator().manual_seed(1))
        self.assertNotAlmostEqual(float(cache[0, 0, 1, 1, 3].item()), 1.0, places=6)

    def test_unknown_kind(self):
        elem = torch.tensor(1.0, dtype=torch.bfloat16)
        spec = KFaultSpec(kind="nope", kv_head=0, dim=0)
        with self.assertRaises(ValueError):
            apply_k_element(elem, spec, torch.Generator().manual_seed(0))


class TestLifetimeHarm(unittest.TestCase):
    def test_delayed_full_harm(self):
        trial = {
            "window_metrics": {
                "1": {"harmful": False},
                "full": {"harmful": True},
            },
            "clean_ids": [1, 1],
            "fault_ids": [1, 2],
        }
        self.assertFalse(lifetime_harm(trial, 1))
        self.assertTrue(lifetime_harm(trial, "full"))

    def test_monotonic_windows(self):
        trial = {
            "window_metrics": {
                "1": {"harmful": True},
                "2": {"harmful": False},
                "16": {"harmful": False},
                "full": {"harmful": False},
            },
            "clean_ids": [1, 1, 1],
            "fault_ids": [1, 1, 1],
            "score_steps": [{"has_nan_inf": False}] * 3,
        }
        self.assertTrue(lifetime_harm(trial, 1))
        self.assertTrue(lifetime_harm(trial, 16))
        self.assertTrue(lifetime_harm(trial, "full"))

    def test_fallback_token_change(self):
        trial = {
            "clean_ids": [7, 8, 9],
            "fault_ids": [7, 0, 9],
            "score_steps": [
                {"has_nan_inf": False},
                {"has_nan_inf": False},
                {"has_nan_inf": False},
            ],
        }
        self.assertFalse(lifetime_harm(trial, 1))
        self.assertTrue(lifetime_harm(trial, 2))


if __name__ == "__main__":
    unittest.main()
