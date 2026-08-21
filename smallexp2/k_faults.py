"""Persistent single-element K faults: numeric delta and BF16 bit flips."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch

NUMERIC_RELS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
# BF16: bit15 sign, bits 7-14 exponent, bits 0-6 mantissa.
BF16_BITS = {
    "sign": 15,
    "exponent": 14,
    "mantissa": 0,
}


@dataclass
class KFaultSpec:
    kind: str  # numeric | bitflip
    kv_head: int
    dim: int
    token_index: Optional[int] = None  # None -> middle prompt token
    rel: Optional[float] = None
    bit_class: Optional[str] = None
    bit: Optional[int] = None
    eps: float = 1e-8

    def to_dict(self) -> dict:
        return asdict(self)


def bit_for_class(bit_class: str) -> int:
    if bit_class not in BF16_BITS:
        raise ValueError(f"unknown bit class {bit_class}")
    return BF16_BITS[bit_class]


def apply_k_element(
    elem: torch.Tensor,
    spec: KFaultSpec,
    rng: torch.Generator,
) -> dict:
    """In-place corrupt one BF16/FP16 K scalar. Returns injection record."""
    old_value = float(elem.item())
    old_bits = _bits_u16(elem)
    if spec.kind == "numeric":
        if spec.rel is None:
            raise ValueError("numeric fault needs rel")
        sign = 1.0 if int(torch.randint(2, (1,), generator=rng).item()) == 0 else -1.0
        delta = sign * float(spec.rel) * (abs(old_value) + spec.eps)
        new_value = old_value + delta
        elem.copy_(torch.tensor(new_value, dtype=elem.dtype, device=elem.device))
        stored = float(elem.item())
        actual = stored - old_value
        rec = {
            "kind": "numeric",
            "rel": spec.rel,
            "intended_delta": delta,
            "delta": actual,
            "abs_delta": abs(actual),
            "quantized_away": abs(actual) == 0.0,
        }
    elif spec.kind == "bitflip":
        bit = spec.bit if spec.bit is not None else bit_for_class(spec.bit_class or "mantissa")
        _xor_bit16(elem, bit)
        rec = {
            "kind": "bitflip",
            "bit_class": spec.bit_class,
            "bit": bit,
            "delta": float(elem.item()) - old_value,
            "abs_delta": abs(float(elem.item()) - old_value),
        }
    else:
        raise ValueError(f"unknown K fault kind {spec.kind}")
    rec.update({
        "old_value": old_value,
        "new_value": float(elem.item()),
        "old_bits": old_bits,
        "new_bits": _bits_u16(elem),
        "kv_head": spec.kv_head,
        "dim": spec.dim,
    })
    return rec


def _bits_u16(elem: torch.Tensor) -> int:
    return int(elem.reshape(1).view(torch.int16)[0].item()) & 0xFFFF


def _xor_bit16(elem: torch.Tensor, bit: int) -> None:
    if not 0 <= bit <= 15:
        raise ValueError(f"bit must be in [0, 15], got {bit}")
    # 0-d cache scalars: reshape(1) then int16 view writes through to paged K.
    view = elem.reshape(1).view(torch.int16)
    mask = 1 << bit
    mask_i16 = mask - 0x10000 if mask >= 0x8000 else mask
    view[0] = view[0] ^ mask_i16
