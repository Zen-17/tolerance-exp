"""Read model geometry from config; never hard-code head dim."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelGeometry:
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    torch_dtype: str
    early_layer: int
    middle_layer: int
    late_layer: int

    @property
    def scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def queries_per_kv(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    def selected_layers(self, override: list[int] | None = None) -> list[int]:
        if override:
            return list(override)
        return [self.early_layer, self.middle_layer, self.late_layer]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scale"] = self.scale
        d["queries_per_kv"] = self.queries_per_kv
        return d


def load_geometry(model_path: str | Path) -> ModelGeometry:
    cfg = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    n = int(cfg["num_hidden_layers"])
    num_q = int(cfg["num_attention_heads"])
    num_kv = int(cfg["num_key_value_heads"])
    hidden = int(cfg["hidden_size"])
    if "head_dim" in cfg and cfg["head_dim"] is not None:
        head_dim = int(cfg["head_dim"])
    else:
        head_dim = hidden // num_q
    return ModelGeometry(
        num_layers=n,
        num_q_heads=num_q,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        hidden_size=hidden,
        torch_dtype=str(cfg.get("torch_dtype", "bfloat16")),
        early_layer=max(0, n // 8),
        middle_layer=n // 2,
        late_layer=min(n - 1, n - 1 - n // 8),
    )
