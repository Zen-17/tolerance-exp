#!/usr/bin/env python3
"""Experiment C (TASK_SPEC): K→S transfer from persistent K faults.

Does not invent new fault types. Analyzes Experiment B trials, then optionally
runs a small identity probe that records E_S vs E_K q / sqrt(d_h).

    source /opt/data/data/anaconda3/bin/activate vllm0.8.5
    cd /opt/data/data/tolerance-exp
    PYTHONUNBUFFERED=1 python run_exp_c.py --profile smoke --from-b results/expB_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = Path("/opt/data/data/models/Qwen3-8B")
DEFAULT_B = ROOT / "results" / "expB_smoke"
DEFAULT_S_REC = ROOT / "results" / "expA_smoke_spec" / "tables" / "recommendation.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dataset-path", type=Path, default=None)
    p.add_argument("--dataset-name", default="smallexp2_synthetic_lm_v1")
    p.add_argument("--profile", choices=["smoke", "phase1"], default="smoke")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--heads", type=int, nargs="*", default=None,
                   help="KV-head indices for the identity probe.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--from-b", type=Path, default=DEFAULT_B,
                   help="Experiment B result directory with raw/trials.jsonl.")
    p.add_argument("--s-rec", type=Path, default=DEFAULT_S_REC)
    p.add_argument("--skip-identity", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pack_identity(seq, ctx, layer, spec, fault) -> dict:
    scores = []
    for s in fault["scores"]:
        scores.append({
            "layer": s.get("layer"),
            "query_u": s.get("query_u"),
            "tq": s.get("tq"),
            "seq_len": s.get("seq_len"),
            "max_abs_es_scaled": s.get("max_abs_es_scaled"),
            "max_abs_es_raw": s.get("max_abs_es_raw"),
            "pass_s_recommended": s.get("pass_s_recommended"),
            "has_nan_inf": s.get("has_nan_inf"),
            "q_abs_at_dim": s.get("q_abs_at_dim"),
            "predicted_es_scaled": s.get("predicted_es_scaled"),
            "es_at_fault_key_scaled": s.get("es_at_fault_key_scaled"),
        })
    k_inj = fault.get("k_inject") or {}
    return {
        "kind": "fault",
        "prompt_id": seq["id"],
        "split": seq.get("split", "calibration"),
        "ctx": ctx,
        "layer": layer,
        "kv_head": spec.kv_head,
        "dim": spec.dim,
        "k_kind": spec.kind,
        "rel": spec.rel,
        "bit_class": spec.bit_class,
        "abs_delta_k": k_inj.get("abs_delta"),
        "k_inject": {k: v for k, v in k_inj.items() if k != "loc"},
        "score_steps": scores,
        "probe": True,
        "fault_seconds": fault["seconds"],
    }


def run_identity_probe(args, out_dir: Path, s_tol: dict) -> list[dict]:
    from run_exp_b import sample_kv_heads
    from smallexp2.data import load_lm_sequences
    from smallexp2.env_info import collect_environment
    from smallexp2.geometry import load_geometry
    from smallexp2.hooks import install
    from smallexp2.k_faults import BF16_BITS, KFaultSpec

    geom = load_geometry(args.model_path)
    layers = args.layers if args.layers else [geom.middle_layer]
    sequences, _ = load_lm_sequences(args.model_path, "smoke")
    sequences = [s for s in sequences if s["split"] == "calibration"][:4]
    layer = layers[0]
    kv_heads = (
        tuple(args.heads) if args.heads else
        sample_kv_heads(geom.num_kv_heads, 2, args.seed, layer)
    )
    kv_head = kv_heads[0]
    ctx = 64
    max_tokens = 8

    import vllm
    from vllm import LLM, SamplingParams

    hook = install()
    hook.inject_on_decode = False
    hook.use_reference = True
    hook.compare_flash = False
    hook.s_rtol = s_tol["s_rtol_scaled"]
    hook.s_atol = s_tol["s_atol_scaled"]
    hook.selected_layers = {layer}
    hook.heads = kv_heads

    print(
        f"expC identity probe | vLLM {vllm.__version__} | layer {layer} | "
        f"kv_head {kv_head} | {len(sequences)} seq ctx={ctx}",
        flush=True,
    )
    llm = LLM(
        model=str(args.model_path),
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )
    (out_dir / "environment.json").write_text(
        json.dumps(collect_environment(args.model_path, args.seed), indent=2),
        encoding="utf-8",
    )
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=args.seed,
        ignore_eos=True,
    )

    def generate(token_ids, k_spec, run_seed):
        hook.begin_run(fault=None, seed=run_seed, k_fault=k_spec)
        t0 = time.perf_counter()
        out = llm.generate(
            [{"prompt_token_ids": token_ids}],
            sampling,
            use_tqdm=False,
        )[0].outputs[0]
        return {
            "token_ids": list(out.token_ids),
            "scores": list(hook.score_records),
            "k_inject": hook.k_inject_record,
            "seconds": time.perf_counter() - t0,
        }

    specs_for = []
    for rel in (1e-4, 1e-3, 1e-2):
        specs_for.append(("numeric", rel, None))
    specs_for.append(("bitflip", None, "mantissa"))

    raw_path = out_dir / "raw" / "identity_trials.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    trials = []
    with raw_path.open("w", encoding="utf-8") as fh:
        for i, seq in enumerate(sequences):
            if len(seq["token_ids"]) < ctx + 1:
                continue
            dim = int(
                hashlib.md5(f"cprobe|{seq['id']}|{layer}".encode()).hexdigest()[:8], 16
            ) % geom.head_dim
            prompt = seq["token_ids"][:ctx]
            for kind, rel, bit_class in specs_for:
                spec = KFaultSpec(
                    kind=kind, kv_head=kv_head, dim=dim, rel=rel,
                    bit_class=bit_class,
                    bit=None if bit_class is None else BF16_BITS[bit_class],
                )
                run_seed = args.seed + 17 * i + (0 if rel is None else int(rel * 1e6))
                fault = generate(prompt, spec, run_seed)
                if fault.get("k_inject") is None:
                    print(f"[skip] no K inject {seq['id']} {kind}", flush=True)
                    continue
                rec = pack_identity(seq, ctx, layer, spec, fault)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                trials.append(rec)
                s0 = rec["score_steps"][0] if rec["score_steps"] else {}
                print(
                    f"[identity] {seq['id']} {kind} rel={rel} "
                    f"pred={s0.get('predicted_es_scaled')} "
                    f"meas={s0.get('es_at_fault_key_scaled')} "
                    f"({fault['seconds']:.1f}s)",
                    flush=True,
                )
                del fault
    return trials


def main() -> None:
    args = parse_args()
    from run_exp_b import load_s_tol
    from smallexp2.analyze_c import analyze_trials_c

    b_dir = args.from_b.resolve()
    trials_path = b_dir / "raw" / "trials.jsonl"
    if not trials_path.exists():
        raise FileNotFoundError(
            f"Experiment C needs B trials at {trials_path}. "
            "Run run_exp_b.py first or pass --from-b."
        )
    b_trials = [t for t in load_jsonl(trials_path) if t.get("split") != "test"]
    if not b_trials:
        raise RuntimeError(f"no calibration fault trials in {trials_path}")

    s_tol = load_s_tol(args.s_rec)
    b_cfg = {}
    b_cfg_path = b_dir / "config.json"
    if b_cfg_path.exists():
        b_cfg = json.loads(b_cfg_path.read_text(encoding="utf-8"))
        if "s_tolerance" in b_cfg:
            s_tol = b_cfg["s_tolerance"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output or (ROOT / "results" / f"expC_{args.profile}")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    config = {
        "profile": args.profile,
        "experiment": "C",
        "model_path": str(args.model_path),
        "dataset_name": args.dataset_name,
        "from_b": str(b_dir),
        "b_prompt_ids": b_cfg.get("prompt_ids"),
        "b_data": b_cfg.get("data"),
        "s_tolerance": s_tol,
        "skip_identity": bool(args.skip_identity),
        "created_utc": stamp,
        "run_split": "calibration",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "raw" / "b_source.json").write_text(
        json.dumps({"from_b": str(b_dir), "n_trials": len(b_trials)}, indent=2),
        encoding="utf-8",
    )
    env_b = b_dir / "environment.json"
    if env_b.exists() and not (out_dir / "environment.json").exists():
        copy2(env_b, out_dir / "environment.json")

    identity_trials: list[dict] = []
    id_path = out_dir / "raw" / "identity_trials.jsonl"
    if args.skip_identity:
        identity_trials = load_jsonl(id_path)
        print(f"expC analyze only | {len(b_trials)} B trials | out {out_dir}", flush=True)
    else:
        if args.resume and id_path.exists():
            identity_trials = load_jsonl(id_path)
            print(f"[resume] loaded {len(identity_trials)} identity trials", flush=True)
        else:
            identity_trials = run_identity_probe(args, out_dir, s_tol)

    rec = analyze_trials_c(
        b_trials, out_dir, s_tol,
        identity_trials=identity_trials,
        b_source=str(b_dir),
    )
    print("recommendation:", json.dumps({
        "n_trials": rec.get("n_trials"),
        "p_fail_s_full": rec.get("p_fail_s_full"),
        "future_amplified": rec.get("future_amplified"),
        "identity": rec.get("identity"),
    }), flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
