"""CPU tests for experiment C transfer aggregation (no vLLM)."""

from __future__ import annotations

import unittest

from smallexp2.analyze_c import (
    ek_bin,
    fail_s_window,
    identity_summary,
    trial_transfer_row,
)


class TestBins(unittest.TestCase):
    def test_ek_bin_edges(self):
        self.assertEqual(ek_bin(3e-6), "[1e-06,1e-05)")
        self.assertEqual(ek_bin(0.05), "[0.01,0.1)")
        self.assertEqual(ek_bin(float("inf")), "nonfinite")
        self.assertEqual(ek_bin(None), "nonfinite")


class TestFailWindow(unittest.TestCase):
    def test_exists_u(self):
        scores = [
            {"pass_s_recommended": True},
            {"pass_s_recommended": False},
            {"pass_s_recommended": True},
        ]
        self.assertFalse(fail_s_window(scores, 1))
        self.assertTrue(fail_s_window(scores, 2))
        self.assertTrue(fail_s_window(scores, "full"))


class TestTransferRow(unittest.TestCase):
    def test_future_amplified_and_gain(self):
        trial = {
            "kind": "fault",
            "prompt_id": "seq_0000",
            "layer": 18,
            "ctx": 64,
            "kv_head": 1,
            "k_kind": "numeric",
            "rel": 0.001,
            "abs_delta_k": 0.01,
            "score_steps": [
                {"max_abs_es_scaled": 0.001, "pass_s_recommended": True, "query_u": 0},
                {"max_abs_es_scaled": 0.004, "pass_s_recommended": False, "query_u": 1},
            ],
        }
        row = trial_transfer_row(trial)
        self.assertTrue(row["future_amplified"])
        self.assertFalse(row["fail_s_L1"])
        self.assertTrue(row["fail_s_L2"])
        self.assertAlmostEqual(row["gain_u0"], 0.1, places=5)
        self.assertEqual(row["ek_bin"], "[0.01,0.1)")


class TestIdentity(unittest.TestCase):
    def test_rel_err_summary(self):
        rows = [
            {"k_kind": "numeric", "rel_err": 0.01},
            {"k_kind": "numeric", "rel_err": 0.02},
            {"k_kind": "bitflip", "rel_err": 0.5},
        ]
        s = identity_summary(rows)
        self.assertEqual(s["n_numeric_finite"], 2)
        self.assertEqual(s["frac_rel_err_lt_5pct"]["numerator"], 2)


if __name__ == "__main__":
    unittest.main()
