"""CPU unit tests for experiment A score injection (no vLLM)."""

from __future__ import annotations

import unittest

import torch

from smallexp2.faults import SFaultSpec, inject_s, pass_s
from smallexp2.geometry import load_geometry
from smallexp2.metrics import first_divergence, is_harmful, top1_change_rate, wilson_ci


class TestGeometry(unittest.TestCase):
    def test_qwen3_8b_head_dim_from_config(self):
        geom = load_geometry("/opt/data/data/models/Qwen3-8B")
        self.assertEqual(geom.head_dim, 128)
        self.assertEqual(geom.num_layers, 36)
        self.assertEqual(geom.num_q_heads, 32)
        self.assertEqual(geom.num_kv_heads, 8)
        self.assertEqual(geom.selected_layers(), [4, 18, 31])


class TestInjectS(unittest.TestCase):
    def setUp(self):
        self.s = torch.tensor([[[1.0, 0.5, -0.2, 0.0]]], dtype=torch.float32)
        self.valid = torch.ones_like(self.s, dtype=torch.bool)
        self.rng = torch.Generator().manual_seed(0)

    def test_single_absolute(self):
        spec = SFaultSpec("single", 0.1, False, (0,))
        st, rec = inject_s(self.s, self.valid, spec, self.rng)
        self.assertEqual(rec["n_perturbed"], 1)
        self.assertAlmostEqual(float((st - self.s).abs().max()), 0.1, places=5)
        ok, _ = pass_s(self.s, st, self.valid, rtol=0.0, atol=0.05)
        self.assertFalse(ok)
        ok2, _ = pass_s(self.s, st, self.valid, rtol=0.0, atol=0.11)
        self.assertTrue(ok2)

    def test_sparse_count(self):
        s = torch.zeros(1, 2, 100)
        valid = torch.ones_like(s, dtype=torch.bool)
        spec = SFaultSpec("sparse", 1e-3, False, (0, 1), sparse_cap=8)
        st, rec = inject_s(s, valid, spec, torch.Generator().manual_seed(1))
        self.assertGreaterEqual(rec["n_perturbed"], 2)
        self.assertLessEqual(rec["n_perturbed"], 8)
        self.assertEqual(int((st != 0).sum().item()), rec["n_perturbed"])

    def test_top2_shrinks_gap(self):
        s = torch.tensor([[[4.0, 1.0, 0.0, -1.0]]])
        valid = torch.ones_like(s, dtype=torch.bool)
        spec = SFaultSpec("top2_gap", 0.5, False, (0,))
        st, rec = inject_s(s, valid, spec, torch.Generator().manual_seed(2))
        self.assertEqual(rec["n_perturbed"], 2)
        gap0 = 4.0 - 1.0
        gap1 = float(st[0, 0].max() - torch.topk(st[0, 0], 2).values[1])
        self.assertLess(gap1, gap0)

    def test_mask_excluded_from_pass(self):
        s = torch.zeros(1, 1, 3)
        st = s.clone()
        st[0, 0, 2] = 9.0
        valid = torch.tensor([[[True, True, False]]])
        ok, _ = pass_s(s, st, valid, rtol=0.0, atol=1e-6)
        self.assertTrue(ok)


class TestMetrics(unittest.TestCase):
    def test_wilson_and_top1(self):
        ci = wilson_ci(1, 10)
        self.assertEqual(ci["numerator"], 1)
        self.assertAlmostEqual(ci["estimate"], 0.1)
        self.assertLess(ci["ci95_low"], ci["estimate"])
        self.assertGreater(ci["ci95_high"], ci["estimate"])
        self.assertEqual(first_divergence([1, 2, 3], [1, 9, 3]), 1)
        self.assertIsNone(first_divergence([1, 2], [1, 2]))
        r = top1_change_rate([1, 2, 3, 4], [1, 2, 0, 4])
        self.assertEqual(r["numerator"], 1)
        self.assertTrue(is_harmful(0.002, 0.0, False))
        self.assertFalse(is_harmful(0.0001, 0.0001, False))
        self.assertTrue(is_harmful(0.0, 0.0, True))


if __name__ == "__main__":
    unittest.main()
