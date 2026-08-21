"""Model-quality and S-tolerance metrics for experiment A."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import torch


def wilson_ci(k: int, n: int, z: float = 1.96) -> dict:
    if n <= 0:
        return {"numerator": k, "denominator": n, "estimate": 0.0,
                "ci95_low": 0.0, "ci95_high": 0.0}
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return {
        "numerator": k,
        "denominator": n,
        "estimate": p,
        "ci95_low": max(0.0, center - half),
        "ci95_high": min(1.0, center + half),
    }


def first_divergence(clean: Sequence[int], fault: Sequence[int]) -> Optional[int]:
    for i, (a, b) in enumerate(zip(clean, fault)):
        if a != b:
            return i
    if len(clean) != len(fault):
        return min(len(clean), len(fault))
    return None


def top1_change_rate(clean: Sequence[int], fault: Sequence[int]) -> dict:
    n = min(len(clean), len(fault))
    k = sum(1 for i in range(n) if clean[i] != fault[i])
    k += abs(len(clean) - len(fault))
    denom = max(len(clean), len(fault), 1)
    out = wilson_ci(k, denom)
    out["n_compared"] = n
    return out


def token_nll_from_logits(logits: torch.Tensor, token_id: int) -> float:
    logp = torch.log_softmax(logits.float(), dim=-1)
    tid = max(0, min(token_id, logp.numel() - 1))
    return float(-logp[tid].item())


def sequence_nll(logits_steps: Sequence[torch.Tensor], token_ids: Sequence[int]) -> list[float]:
    n = min(len(logits_steps), len(token_ids))
    return [token_nll_from_logits(logits_steps[i], token_ids[i]) for i in range(n)]


def ppl_from_nll(nlls: Iterable[float]) -> float:
    vals = list(nlls)
    if not vals:
        return float("nan")
    return math.exp(sum(vals) / len(vals))


def logits_errors(clean: torch.Tensor, fault: torch.Tensor) -> dict:
    c = clean.float().reshape(-1)
    f = fault.float().reshape(-1)
    diff = f - c
    c_norm = float(c.norm().item())
    return {
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "rel_l2": float(diff.norm().item() / (c_norm + 1e-12)),
        "has_nan_inf": bool((~torch.isfinite(f)).any().item() or (~torch.isfinite(c)).any().item()),
    }


def tv_distance(p_clean: torch.Tensor, p_fault: torch.Tensor, valid: torch.Tensor) -> float:
    """Mean total variation over query/head; P is [Tq, H, Tk]."""
    pc = torch.where(valid, p_clean.float(), torch.zeros_like(p_clean, dtype=torch.float32))
    pf = torch.where(valid, p_fault.float(), torch.zeros_like(p_fault, dtype=torch.float32))
    tv = 0.5 * (pc - pf).abs().sum(dim=-1)
    return float(tv.mean().item())


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.float().reshape(-1)
    bb = b.float().reshape(-1)
    return float((bb - aa).norm().item() / (aa.norm().item() + 1e-12))


def paired_bootstrap_mean(values: Sequence[float], n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile CI of the mean."""
    if not values:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    t = torch.tensor(list(values), dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    n = t.numel()
    idx = torch.randint(n, (n_boot, n), generator=g)
    means = t[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=means.dtype))
    return {"mean": float(t.mean().item()), "ci95_low": float(lo.item()), "ci95_high": float(hi.item())}


def is_harmful(
    rel_ppl_rise: float,
    top1_rate: float,
    extra_nan_inf: bool,
    ppl_budget: float = 0.001,
    top1_budget: float = 0.001,
) -> bool:
    if extra_nan_inf:
        return True
    if math.isnan(rel_ppl_rise) or rel_ppl_rise > ppl_budget:
        return True
    if top1_rate > top1_budget:
        return True
    return False
