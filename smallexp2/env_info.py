"""Record runtime fingerprint required by TASK_SPEC section 8."""

from __future__ import annotations

from pathlib import Path


def collect_environment(model_path: str | Path, seed: int) -> dict:
    import torch
    import vllm

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "vllm": getattr(vllm, "__version__", "unknown"),
        "vllm_file": getattr(vllm, "__file__", None),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "model_path": str(model_path),
        "weights_dtype": "bfloat16",
        "activation_dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "attention_backend": "FLASH_ATTN_VLLM_V1",
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "greedy": True,
        "seed": seed,
    }
