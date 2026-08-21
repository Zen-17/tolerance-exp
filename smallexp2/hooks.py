"""Reference attention + S injection + logits capture on vLLM 0.8.5 V1."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import torch

from smallexp2.faults import RTOL_GRID, ATOL_GRID, REQUIRED_TOLERANCE, SFaultSpec, inject_s, pass_s
from smallexp2.k_faults import KFaultSpec, apply_k_element
from smallexp2.metrics import rel_l2, tv_distance

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

_HOOK: Optional["ExpAHook"] = None
_ATTN_INSTALLED = False
_SAMPLER_INSTALLED = False


def layer_index(layer) -> Optional[int]:
    name = getattr(layer, "layer_name", None)
    if not name:
        return None
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


@dataclass
class ExpAHook:
    selected_layers: set[int] = field(default_factory=set)
    heads: tuple[int, ...] = ()
    use_reference: bool = True
    fault: Optional[SFaultSpec] = None
    rng: Optional[torch.Generator] = None
    score_records: list[dict] = field(default_factory=list)
    logits_steps: list[torch.Tensor] = field(default_factory=list)
    y_flash_ref_max_abs: list[float] = field(default_factory=list)
    compare_flash: bool = False
    inject_on_decode: bool = False
    original_attn: Optional[object] = None
    k_fault: Optional[KFaultSpec] = None
    k_injected: bool = False
    k_inject_record: Optional[dict] = None
    s_rtol: float = 1e-5
    s_atol: float = 1e-5

    def begin_run(self, fault: Optional[SFaultSpec] = None, seed: int = 0,
                  k_fault: Optional[KFaultSpec] = None) -> None:
        self.fault = fault
        self.k_fault = k_fault
        self.rng = torch.Generator().manual_seed(seed)
        self.score_records = []
        self.logits_steps = []
        self.y_flash_ref_max_abs = []
        self.k_injected = False
        self.k_inject_record = None

    def capture_logits(self, logits: torch.Tensor) -> None:
        row = logits[0] if logits.dim() == 2 else logits
        self.logits_steps.append(row.detach().float().cpu().clone())


def get_hook() -> ExpAHook:
    global _HOOK
    if _HOOK is None:
        _HOOK = ExpAHook()
    return _HOOK


def install() -> ExpAHook:
    """Patch FlashAttentionImpl.forward and Sampler.forward once."""
    global _ATTN_INSTALLED, _SAMPLER_INSTALLED
    hook = get_hook()

    if not _ATTN_INSTALLED:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

        original = FlashAttentionImpl.forward
        hook.original_attn = original

        def patched(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
            if output is None:
                return original(self, layer, query, key, value, kv_cache, attn_metadata, output)
            if attn_metadata is None:
                return original(self, layer, query, key, value, kv_cache, attn_metadata, output)
            idx = layer_index(layer)
            if (not hook.use_reference) or idx is None or idx not in hook.selected_layers:
                return original(self, layer, query, key, value, kv_cache, attn_metadata, output)
            if kv_cache is None or kv_cache.numel() == 0:
                return original(self, layer, query, key, value, kv_cache, attn_metadata, output)
            _reference_forward(self, layer, query, key, value, kv_cache,
                               attn_metadata, output, idx)
            return output

        FlashAttentionImpl.forward = patched  # type: ignore[method-assign]
        _ATTN_INSTALLED = True

    if not _SAMPLER_INSTALLED:
        from vllm.v1.sample.sampler import Sampler

        original_sampler = Sampler.forward

        def patched_sampler(self, logits, sampling_metadata):
            try:
                hook.capture_logits(logits)
            except Exception:
                pass
            return original_sampler(self, logits, sampling_metadata)

        Sampler.forward = patched_sampler  # type: ignore[method-assign]
        _SAMPLER_INSTALLED = True

    return hook


def _write_cache(impl, layer, key, value, kv_cache, attn_metadata):
    key_cache, value_cache = kv_cache.unbind(0)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        attn_metadata.slot_mapping,
        impl.kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )


def _gather_kv(kv_cache: torch.Tensor, block_table: torch.Tensor, seq_len: int):
    block_size = int(kv_cache.shape[2])
    n_blocks = (seq_len + block_size - 1) // block_size
    ids = block_table[0, :n_blocks].to(dtype=torch.long)
    k = kv_cache[0, ids].reshape(-1, kv_cache.shape[3], kv_cache.shape[4])[:seq_len]
    v = kv_cache[1, ids].reshape(-1, kv_cache.shape[3], kv_cache.shape[4])[:seq_len]
    return k, v


def _causal_valid(tq: int, tk: int, seq_len: int, device) -> torch.Tensor:
    q_pos = torch.arange(seq_len - tq, seq_len, device=device)
    k_pos = torch.arange(tk, device=device)
    return k_pos.view(1, 1, tk) <= q_pos.view(tq, 1, 1)


def _reference_forward(impl, layer, query, key, value, kv_cache, attn_metadata, output, layer_idx):
    hook = get_hook()
    num_actual = int(attn_metadata.num_actual_tokens)
    _write_cache(impl, layer, key, value, kv_cache, attn_metadata)

    if hook.k_fault is not None and (not hook.k_injected) and num_actual > 1:
        _inject_persistent_k(hook, kv_cache, attn_metadata, layer_idx)

    seq_len = int(attn_metadata.seq_lens[0].item())
    q = query[:num_actual]
    tq = q.shape[0]
    n_q_heads = q.shape[1]
    head_dim = q.shape[2]
    k_paged, v_paged = _gather_kv(kv_cache, attn_metadata.block_table, seq_len)
    n_kv = k_paged.shape[1]
    repeat = n_q_heads // n_kv
    v = v_paged.repeat_interleave(repeat, dim=1)

    scale = float(impl.scale)
    s_from_k = hook.k_injected and hook.k_inject_record is not None
    if s_from_k:
        rec_k = hook.k_inject_record
        k_clean_p = k_paged.clone()
        ti, kh, di = rec_k["token_index"], rec_k["kv_head"], rec_k["dim"]
        if 0 <= ti < k_clean_p.shape[0]:
            k_clean_p[ti, kh, di] = rec_k["old_value"]
        k = k_paged.repeat_interleave(repeat, dim=1)
        k_c = k_clean_p.repeat_interleave(repeat, dim=1)
        s = torch.einsum("qhd,khd->qhk", q.float(), k_c.float()) * scale
        s_work = torch.einsum("qhd,khd->qhk", q.float(), k.float()) * scale
    else:
        k = k_paged.repeat_interleave(repeat, dim=1)
        s = torch.einsum("qhd,khd->qhk", q.float(), k.float()) * scale
        s_work = s

    valid = _causal_valid(tq, seq_len, seq_len, s.device).expand_as(s)
    s = s.masked_fill(~valid, torch.finfo(s.dtype).min)
    s_work = s_work.masked_fill(~valid, torch.finfo(s.dtype).min)

    inj_rec: dict = {"n_perturbed": 0}
    inject_now = (
        (not s_from_k)
        and hook.fault is not None
        and hook.rng is not None
        and (tq > 1 or hook.inject_on_decode)
    )
    if inject_now:
        if tq > 1:
            s_last = s[-1:]
            v_last = valid[-1:]
            s_inj, inj_rec = inject_s(s_last, v_last, hook.fault, hook.rng)
            s_work = s.clone()
            s_work[-1:] = s_inj
        else:
            s_work, inj_rec = inject_s(s, valid, hook.fault, hook.rng)

    p_clean = torch.softmax(s, dim=-1)
    p_work = torch.softmax(s_work, dim=-1) if (inject_now or s_from_k) else p_clean
    y = torch.einsum("qhk,khd->qhd", p_work, v.float())
    output[:num_actual].copy_(y.to(dtype=output.dtype))

    if hook.compare_flash and hook.original_attn is not None:
        out_flash = torch.empty_like(output)
        hook.original_attn(
            impl, layer, query, key, value, kv_cache, attn_metadata, out_flash)
        hook.y_flash_ref_max_abs.append(
            float((out_flash[:num_actual].float() - y).abs().max().item())
        )

    record_query = s_from_k or tq > 1 or hook.inject_on_decode
    if record_query:
        s_cmp_clean = s[-1]
        s_cmp_work = s_work[-1]
        v_cmp = valid[-1]
        max_abs = float((s_cmp_work - s_cmp_clean).abs()[v_cmp].max().item()) if v_cmp.any() else 0.0
        denom = s_cmp_clean.abs()[v_cmp]
        rel = (s_cmp_work - s_cmp_clean).abs()[v_cmp] / (denom + 1e-12)
        max_rel = float(rel.max().item()) if rel.numel() else 0.0
        pass_map = {}
        tols = [(rt, at) for rt in RTOL_GRID for at in ATOL_GRID]
        if REQUIRED_TOLERANCE not in tols:
            tols.append(REQUIRED_TOLERANCE)
        for rt, at in tols:
            ok, reason = pass_s(s[-1:], s_work[-1:], valid[-1:], rt, at)
            pass_map[f"rtol_{rt:g}_atol_{at:g}"] = {"pass": ok, "reason": reason}
        rec_rt, rec_at = hook.s_rtol, hook.s_atol
        rec_pass, rec_reason = pass_s(s[-1:], s_work[-1:], valid[-1:], rec_rt, rec_at)
        finite_work = bool(torch.isfinite(s_work[valid]).all().item()) if valid.any() else True
        y_last_clean = torch.einsum("qhk,khd->qhd", p_clean[-1:], v.float())
        k2s = _k_to_s_identity(s_from_k, hook, q, s_cmp_clean, s_cmp_work, scale, repeat)
        hook.score_records.append({
            "layer": layer_idx,
            "tq": tq,
            "seq_len": seq_len,
            "query_u": len(hook.score_records),
            "max_abs_es_scaled": max_abs,
            "max_abs_es_raw": max_abs / scale if scale else max_abs,
            "max_rel_es": max_rel,
            "tv": tv_distance(p_clean[-1:], p_work[-1:], valid[-1:]),
            "y_rel_l2": rel_l2(y_last_clean, y[-1:]),
            "has_nan_inf": not finite_work,
            "pass_s": pass_map,
            "pass_s_recommended": rec_pass,
            "pass_s_recommended_reason": rec_reason,
            "inject": inj_rec,
            **k2s,
        })


def _k_to_s_identity(s_from_k, hook, q, s_cmp_clean, s_cmp_work, scale, repeat) -> dict:
    """E_S at the corrupted key column vs predicted |E_K| * |q_j| * scale."""
    if not s_from_k or hook.k_inject_record is None:
        return {}
    rec_k = hook.k_inject_record
    ti, kh, di = rec_k["token_index"], rec_k["kv_head"], rec_k["dim"]
    n_q = int(q.shape[1])
    n_dim = int(q.shape[2])
    h0 = int(kh) * int(repeat)
    h1 = min(h0 + int(repeat), n_q)
    q_abs = 0.0
    if 0 <= h0 < h1 <= n_q and 0 <= int(di) < n_dim:
        q_abs = float(q[-1, h0:h1, int(di)].float().abs().max().item())
    delta = abs(float(rec_k.get("abs_delta") or 0.0))
    pred = delta * q_abs * float(scale)
    es_at_p = 0.0
    if 0 <= int(ti) < s_cmp_work.shape[-1] and h0 < h1:
        es_at_p = float(
            (s_cmp_work[h0:h1, int(ti)] - s_cmp_clean[h0:h1, int(ti)]).abs().max().item()
        )
    return {
        "q_abs_at_dim": q_abs,
        "predicted_es_scaled": pred,
        "es_at_fault_key_scaled": es_at_p,
    }


def _inject_persistent_k(hook, kv_cache, attn_metadata, layer_idx: int) -> None:
    spec = hook.k_fault
    if spec is None or hook.rng is None:
        return
    n_kv = int(kv_cache.shape[3])
    n_dim = int(kv_cache.shape[4])
    if not 0 <= spec.kv_head < n_kv or not 0 <= spec.dim < n_dim:
        return
    num_actual = int(attn_metadata.num_actual_tokens)
    tok = spec.token_index if spec.token_index is not None else max(0, num_actual // 2)
    tok = max(0, min(tok, num_actual - 1))
    slot = int(attn_metadata.slot_mapping[tok].item())
    if slot < 0:
        return
    block_size = int(kv_cache.shape[2])
    block_id = slot // block_size
    off = slot % block_size
    elem = kv_cache[0, block_id, off, spec.kv_head, spec.dim]
    rec = apply_k_element(elem, spec, hook.rng)
    rec["token_index"] = tok
    rec["slot"] = slot
    rec["block_id"] = int(block_id)
    rec["token_offset"] = int(off)
    rec["layer"] = layer_idx
    hook.k_inject_record = rec
    hook.k_injected = True
