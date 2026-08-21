"""S-score perturbation morphologies for experiment A."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch

GAMMA_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
RTOL_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
ATOL_GRID = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
REQUIRED_TOLERANCE = (1e-5, 1e-6)  # rtol, atol


@dataclass
class SFaultSpec:
    mode: str  # single | sparse | top2_gap
    gamma: float
    relative: bool
    heads: tuple[int, ...]
    sparse_cap: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


def _valid_index_pairs(valid: torch.Tensor, heads: tuple[int, ...]) -> torch.Tensor:
    """Return (N, 3) int64 index tensor of valid (q, h, k) among selected heads."""
    tq, n_heads, tk = valid.shape
    head_mask = torch.zeros(n_heads, dtype=torch.bool, device=valid.device)
    head_mask[list(heads)] = True
    sel = valid & head_mask.view(1, n_heads, 1)
    idx = sel.nonzero(as_tuple=False)
    return idx


def inject_s(
    s: torch.Tensor,
    valid: torch.Tensor,
    spec: SFaultSpec,
    rng: torch.Generator,
) -> tuple[torch.Tensor, dict]:
    """Return S_tilde and a small injection record. s/valid: [Tq, H, Tk]."""
    s_tilde = s.clone()
    idx = _valid_index_pairs(valid, spec.heads)
    n_valid = int(idx.shape[0])
    rec = {
        "n_valid": n_valid,
        "n_perturbed": 0,
        "mode": spec.mode,
        "gamma": spec.gamma,
        "relative": spec.relative,
    }
    if n_valid == 0:
        return s_tilde, rec

    idx_cpu = idx.detach().to("cpu")
    if spec.mode == "single":
        pick = idx_cpu[int(torch.randint(n_valid, (1,), generator=rng).item())]
        _add_at(s_tilde, s, pick, spec, rng)
        rec["n_perturbed"] = 1
        rec["loc"] = [int(x) for x in pick.tolist()]
    elif spec.mode == "sparse":
        k = min(spec.sparse_cap, max(2, n_valid // 100), n_valid)
        perm = torch.randperm(n_valid, generator=rng)[:k]
        for row in idx_cpu[perm]:
            _add_at(s_tilde, s, row, spec, rng)
        rec["n_perturbed"] = k
    elif spec.mode == "top2_gap":
        rec.update(_shrink_top2_gap(s_tilde, s, valid, spec, rng))
    else:
        raise ValueError(f"unknown S fault mode: {spec.mode}")
    return s_tilde, rec


def _signed_delta(base: torch.Tensor, spec: SFaultSpec, rng: torch.Generator) -> float:
    sign = 1.0 if int(torch.randint(2, (1,), generator=rng, device="cpu").item()) == 0 else -1.0
    mag = spec.gamma * (float(base.abs().item()) if spec.relative else 1.0)
    return sign * mag


def _add_at(
    s_tilde: torch.Tensor,
    s: torch.Tensor,
    loc: torch.Tensor,
    spec: SFaultSpec,
    rng: torch.Generator,
) -> None:
    q, h, k = (int(x) for x in loc.tolist())
    s_tilde[q, h, k] = s[q, h, k] + _signed_delta(s[q, h, k], spec, rng)


def _shrink_top2_gap(
    s_tilde: torch.Tensor,
    s: torch.Tensor,
    valid: torch.Tensor,
    spec: SFaultSpec,
    rng: torch.Generator,
) -> dict:
    q = s.shape[0] - 1
    head_i = int(spec.heads[int(torch.randint(len(spec.heads), (1,), generator=rng, device="cpu").item())])
    row = s[q, head_i]
    mask = valid[q, head_i]
    if int(mask.sum().item()) < 2:
        return {"n_perturbed": 0, "head": head_i}
    neg_inf = torch.finfo(row.dtype).min
    masked = torch.where(mask, row, torch.full_like(row, neg_inf))
    topv, topi = torch.topk(masked, 2)
    gap = float((topv[0] - topv[1]).item())
    if gap <= 0.0:
        return {"n_perturbed": 0, "head": head_i, "gap": gap}
    step = spec.gamma * (abs(float(topv[0].item())) if spec.relative else 1.0)
    step = min(step, gap / 2.0)
    s_tilde[q, head_i, int(topi[0].item())] = s[q, head_i, int(topi[0].item())] - step
    s_tilde[q, head_i, int(topi[1].item())] = s[q, head_i, int(topi[1].item())] + step
    return {
        "n_perturbed": 2,
        "head": head_i,
        "gap": gap,
        "step": step,
        "top1": int(topi[0].item()),
        "top2": int(topi[1].item()),
    }


def pass_s(
    s_clean: torch.Tensor,
    s_tilde: torch.Tensor,
    valid: torch.Tensor,
    rtol: float,
    atol: float,
) -> tuple[bool, Optional[str]]:
    """All valid scores must satisfy |dS| <= atol + rtol |S|. NaN/Inf fail."""
    if valid.any() and not torch.isfinite(s_tilde[valid]).all():
        return False, "nan_inf"
    diff = (s_tilde - s_clean).abs()
    tol = atol + rtol * s_clean.abs()
    ok = torch.where(valid, diff <= tol, torch.ones_like(valid, dtype=torch.bool))
    return bool(ok.all().item()), None
