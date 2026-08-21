#!/usr/bin/env python3
"""Experiment A (TASK_SPEC): scaled-score perturbation and S tolerance.

Uses local Qwen3-8B at --model-path. Smoke: 16 sequences of >=256 tokens.

    source /opt/data/data/anaconda3/bin/activate vllm0.8.5
    cd /opt/data/data/tolerance-exp
    PYTHONUNBUFFERED=1 python run_exp_a.py --profile smoke
    PYTHONUNBUFFERED=1 python run_exp_a.py --profile phase1 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = Path("/opt/data/data/models/Qwen3-8B")
GAMMA_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dataset-path", type=Path, default=None)
    p.add_argument("--dataset-name", default="smallexp2_synthetic_lm_v1")
    p.add_argument("--profile", choices=["smoke", "phase1"], default="smoke")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--heads", type=int, nargs="*", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-consistency", action="store_true")
    p.add_argument("--split", choices=["calibration", "test", "all"], default=None,
                   help="Override which split to run. Test is eval-only; never for selection.")
    return p.parse_args()


def profile_cfg(profile: str) -> dict:
    if profile == "smoke":
        return {
            "gammas": list(GAMMA_GRID),
            "relatives": [False, True],
            "modes": ["single", "sparse", "top2_gap"],
            "max_tokens": 16,
            "n_heads": 2,
            "ctxs": [64, 256],
            "n_seeds": 1,
        }
    return {
        "gammas": list(GAMMA_GRID),
        "relatives": [False, True],
        "modes": ["single", "sparse", "top2_gap"],
        "max_tokens": 16,
        "n_heads": 2,
        "ctxs": [64, 256],
        "n_seeds": 3,
        "densify": False,
        "cycle_one_layer": True,
        "n_inject_cal": 84,
        "n_inject_test": 20,
    }


def trial_key(**kwargs) -> str:
    return json.dumps(kwargs, sort_keys=True, default=str)


def load_done_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if "key" in rec:
                    done.add(rec["key"])
    return done


def sample_heads(n_q_heads: int, n_take: int, seed: int, layer: int) -> tuple[int, ...]:
    rng = random.Random(seed * 1009 + layer)
    pool = list(range(n_q_heads))
    rng.shuffle(pool)
    return tuple(sorted(pool[:n_take]))


def midpoints(gammas: list[float]) -> list[float]:
    out = []
    for a, b in zip(gammas, gammas[1:]):
        out.append(math.sqrt(a * b))
    return out


def main() -> None:
    args = parse_args()
    cfg = profile_cfg(args.profile)
    if args.max_tokens is not None:
        cfg["max_tokens"] = args.max_tokens
    seeds = args.seeds or [args.seed + i for i in range(cfg["n_seeds"])]

    from smallexp2.analyze_a import analyze_trials, write_csv, write_test_check
    from smallexp2.data import LONG_CTX, SHORT_CTX, load_lm_sequences, subset_for_injection
    from smallexp2.env_info import collect_environment
    from smallexp2.faults import SFaultSpec
    from smallexp2.geometry import load_geometry
    from smallexp2.hooks import install
    from smallexp2.metrics import (
        first_divergence,
        is_harmful,
        logits_errors,
        ppl_from_nll,
        sequence_nll,
        token_nll_from_logits,
        top1_change_rate,
    )

    geom = load_geometry(args.model_path)
    layers = args.layers if args.layers else geom.selected_layers()
    sequences, data_meta = load_lm_sequences(args.model_path, args.profile)
    run_split = args.split or ("calibration" if args.profile == "phase1" else "calibration")
    if args.profile == "smoke":
        # TASK_SPEC smoke: 16 calibration sequences, 1 seed.
        sequences = [s for s in sequences if s["split"] == "calibration"][:16]
        run_split = "calibration"
    elif run_split != "all":
        sequences = [s for s in sequences if s["split"] == run_split]
    census_sequences = []
    if args.profile == "phase1":
        census_sequences = list(sequences)
        sequences = subset_for_injection(sequences, args.profile, run_split)
        if run_split != "calibration":
            census_sequences = []

    heads_by_layer = {}
    for layer in layers:
        heads_by_layer[layer] = (
            tuple(args.heads) if args.heads else
            sample_heads(geom.num_q_heads, cfg["n_heads"], args.seed, layer)
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output or (ROOT / "results" / f"expA_{args.profile}_{stamp}")
    out_dir = out_dir.resolve()
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trials_path = raw_dir / "trials.jsonl"

    config = {
        "profile": args.profile,
        "model_path": str(args.model_path),
        "dataset_name": args.dataset_name,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "data": data_meta,
        "geometry": geom.to_dict(),
        "layers": layers,
        "heads_by_layer": {str(k): list(v) for k, v in heads_by_layer.items()},
        "seeds": seeds,
        "max_tokens": cfg["max_tokens"],
        "modes": cfg["modes"],
        "gammas": cfg["gammas"],
        "relatives": cfg["relatives"],
        "ctxs": cfg["ctxs"],
        "run_split": run_split,
        "prompt_ids": [s["id"] for s in sequences],
        "census_prompt_ids": [s["id"] for s in census_sequences],
        "created_utc": stamp,
        "seq_len": LONG_CTX,
        "short_ctx": SHORT_CTX,
        "inject": "prefill_last_query_only",
    }
    cfg_path = out_dir / "config.json"
    if not (args.resume and cfg_path.exists()):
        cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    else:
        (out_dir / "config_last.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    import vllm
    from vllm import LLM, SamplingParams

    hook = install()
    hook.inject_on_decode = False
    hook.use_reference = True

    print(
        f"expA {args.profile} | vLLM {vllm.__version__} | layers {layers} | "
        f"heads {heads_by_layer} | d_h {geom.head_dim} | "
        f"{len(sequences)} seq x {LONG_CTX} tok | split={run_split} | out {out_dir}",
        flush=True,
    )

    llm = LLM(
        model=str(args.model_path),
        dtype="bfloat16",
        seed=seeds[0],
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )
    (out_dir / "environment.json").write_text(
        json.dumps(collect_environment(args.model_path, seeds[0]), indent=2),
        encoding="utf-8",
    )
    sampling = SamplingParams(
        max_tokens=cfg["max_tokens"],
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=seeds[0],
        ignore_eos=True,
    )

    def generate(token_ids: list[int], spec, run_seed: int) -> dict:
        hook.begin_run(spec, run_seed)
        t0 = time.perf_counter()
        out = llm.generate(
            [{"prompt_token_ids": token_ids}],
            sampling,
            use_tqdm=False,
        )[0].outputs[0]
        return {
            "token_ids": list(out.token_ids),
            "text": out.text,
            "logits": list(hook.logits_steps),
            "scores": list(hook.score_records),
            "y_flash_ref_max_abs": list(hook.y_flash_ref_max_abs),
            "seconds": time.perf_counter() - t0,
        }

    consistency = None
    cons_path = raw_dir / "consistency.json"
    seq0 = sequences[0]
    layer0 = layers[0]
    prompt0 = seq0["token_ids"][:LONG_CTX]
    if args.resume and cons_path.exists():
        consistency = json.loads(cons_path.read_text(encoding="utf-8"))
        print(f"[consistency] resume {cons_path}", flush=True)
    elif not args.skip_consistency:
        hook.selected_layers = {layer0}
        hook.heads = heads_by_layer[layer0]
        hook.use_reference = False
        hook.compare_flash = False
        flash = generate(prompt0, None, seeds[0])
        hook.use_reference = True
        hook.compare_flash = True
        ref = generate(prompt0, None, seeds[0])
        hook.compare_flash = False
        le = (logits_errors(flash["logits"][0], ref["logits"][0])
              if flash["logits"] and ref["logits"] else {})
        consistency = {
            "layer": layer0,
            "ctx": LONG_CTX,
            "first_divergence": first_divergence(flash["token_ids"], ref["token_ids"]),
            "top1": top1_change_rate(flash["token_ids"], ref["token_ids"]),
            "ppl_flash": ppl_from_nll(sequence_nll(flash["logits"], flash["token_ids"])),
            "ppl_ref": ppl_from_nll(sequence_nll(ref["logits"], ref["token_ids"])),
            "logits": le,
            "attn_output_max_abs": (max(ref["y_flash_ref_max_abs"])
                                    if ref["y_flash_ref_max_abs"] else None),
            "n_flash_tokens": len(flash["token_ids"]),
            "n_ref_tokens": len(ref["token_ids"]),
        }
        (raw_dir / "consistency.json").write_text(
            json.dumps(consistency, indent=2), encoding="utf-8")
        print(
            f"[consistency] div={consistency['first_divergence']} "
            f"top1={consistency['top1']['estimate']:.4f} "
            f"Y_max_abs={consistency['attn_output_max_abs']} "
            f"logits_max_abs={le.get('max_abs')}",
            flush=True,
        )
        del flash, ref

    hook.use_reference = True
    hook.compare_flash = False

    def layers_for_index(i: int) -> list[int]:
        # TASK_SPEC 6.2: sample early/middle/late, do not visit every layer on every sequence.
        return [layers[i % len(layers)]]

    specs = []
    for mode in cfg["modes"]:
        for gamma in cfg["gammas"]:
            for relative in cfg["relatives"]:
                specs.append((mode, gamma, relative))

    n_plan = 0
    for seed in seeds:
        for i, seq in enumerate(sequences):
            for layer in layers_for_index(i):
                for ctx in cfg["ctxs"]:
                    if len(seq["token_ids"]) >= ctx + 1:
                        n_plan += len(specs)
    n_clean_plan = 0
    for seed in seeds:
        for i, seq in enumerate(sequences):
            for ctx in cfg["ctxs"]:
                if len(seq["token_ids"]) < ctx + 1:
                    continue
                n_clean_plan += len(layers_for_index(i))
    n_census = len(census_sequences)
    print(
        f"planned cleans {n_clean_plan} + faults {n_plan} + census {n_census} "
        f"(~{(n_clean_plan + n_plan + n_census) * 0.8 / 3600:.1f} h at 0.8s/trial) "
        f"| inject {len(sequences)} seq | {len(seeds)} seed | split={run_split}",
        flush=True,
    )

    # ---- clean baselines per (seq, ctx, layer, seed)
    cleans: dict[tuple, dict] = {}
    for seed in seeds:
        for i, seq in enumerate(sequences):
            for ctx in cfg["ctxs"]:
                if len(seq["token_ids"]) < ctx + 1:
                    continue
                for layer in layers_for_index(i):
                    key = (seq["id"], ctx, layer, seed)
                    if key in cleans:
                        continue
                    hook.selected_layers = {layer}
                    hook.heads = heads_by_layer[layer]
                    prompt = seq["token_ids"][:ctx]
                    run = generate(prompt, None, seed)
                    true_next = seq["token_ids"][ctx]
                    nll_true = (token_nll_from_logits(run["logits"][0], true_next)
                                if run["logits"] else float("nan"))
                    nll_gen = sequence_nll(run["logits"], run["token_ids"])
                    cleans[key] = {
                        "token_ids": run["token_ids"],
                        "logits": run["logits"],
                        "nll_gen": nll_gen,
                        "ppl": ppl_from_nll(nll_gen),
                        "true_next": true_next,
                        "nll_true": nll_true,
                        "greedy_next": run["token_ids"][0] if run["token_ids"] else None,
                        "seconds": run["seconds"],
                        "prompt_tokens": ctx,
                    }
                    print(
                        f"[clean] {seq['id']} ctx={ctx} L{layer} seed={seed} "
                        f"ppl={cleans[key]['ppl']:.3f} ({run['seconds']:.1f}s)",
                        flush=True,
                    )

    census_rows = []
    inject_ids = {s["id"] for s in sequences}
    layer_c = layers[0]
    for seq in census_sequences:
        if seq["id"] in inject_ids:
            continue
        if len(seq["token_ids"]) < LONG_CTX + 1:
            continue
        hook.selected_layers = {layer_c}
        hook.heads = heads_by_layer[layer_c]
        run = generate(seq["token_ids"][:LONG_CTX], None, seeds[0])
        nll_gen = sequence_nll(run["logits"], run["token_ids"])
        census_rows.append({
            "prompt_id": seq["id"],
            "ctx": LONG_CTX,
            "layer": layer_c,
            "seed": seeds[0],
            "prompt_tokens": LONG_CTX,
            "gen_tokens": len(run["token_ids"]),
            "ppl": ppl_from_nll(nll_gen),
            "nll_true_next": (token_nll_from_logits(run["logits"][0], seq["token_ids"][LONG_CTX])
                              if run["logits"] else float("nan")),
            "seconds": run["seconds"],
        })
        print(
            f"[census] {seq['id']} ctx={LONG_CTX} L{layer_c} ppl={census_rows[-1]['ppl']:.3f} "
            f"({run['seconds']:.1f}s)",
            flush=True,
        )
        del run

    done = load_done_keys(trials_path) if args.resume else set()
    trials: list[dict] = []
    if args.resume and trials_path.exists():
        with trials_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("kind") == "fault":
                        trials.append(rec)

    mode_fh = "a" if args.resume else "w"
    n_done = len(done)
    with trials_path.open(mode_fh, encoding="utf-8") as fh:
        for seed in seeds:
            for i, seq in enumerate(sequences):
                for layer in layers_for_index(i):
                    heads = heads_by_layer[layer]
                    hook.selected_layers = {layer}
                    hook.heads = heads
                    for ctx in cfg["ctxs"]:
                        if len(seq["token_ids"]) < ctx + 1:
                            continue
                        clean = cleans[(seq["id"], ctx, layer, seed)]
                        prompt = seq["token_ids"][:ctx]
                        for mode, gamma, relative in specs:
                            spec = SFaultSpec(
                                mode=mode, gamma=gamma, relative=relative, heads=heads)
                            key = trial_key(
                                prompt=seq["id"], ctx=ctx, layer=layer, seed=seed,
                                mode=mode, gamma=gamma, relative=relative, heads=list(heads),
                            )
                            if key in done:
                                continue
                            run_seed = seed + int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10007
                            fault = generate(prompt, spec, run_seed)
                            rec = pack_trial(
                                seq, ctx, layer, heads, seed, spec, clean, fault,
                                run_split, first_divergence, is_harmful, logits_errors,
                                ppl_from_nll, sequence_nll, token_nll_from_logits,
                                top1_change_rate,
                            )
                            rec["key"] = key
                            fh.write(json.dumps(rec) + "\n")
                            fh.flush()
                            trials.append(rec)
                            n_done += 1
                            print(
                                f"[fault] {n_done}/{n_plan} {seq['id']} ctx={ctx} L{layer} {mode} "
                                f"g={gamma:g} rel={relative} harmful={rec['harmful']} "
                                f"div={rec['first_divergence']} "
                                f"dPPL={rec['rel_ppl_rise']:.4g} ({fault['seconds']:.1f}s)",
                                flush=True,
                            )
                            del fault

        # Local densify: midpoints where quality starts changing.
        if args.profile == "phase1" and run_split != "test" and cfg.get("densify"):
            extra = densify_gammas(trials, cfg["gammas"])
            if extra:
                print(f"[densify] extra gammas {extra}", flush=True)
            for seed in seeds:
                for i, seq in enumerate(sequences):
                    for layer in layers_for_index(i):
                        heads = heads_by_layer[layer]
                        hook.selected_layers = {layer}
                        hook.heads = heads
                        for ctx in cfg["ctxs"]:
                            if len(seq["token_ids"]) < ctx + 1:
                                continue
                            clean = cleans[(seq["id"], ctx, layer, seed)]
                            prompt = seq["token_ids"][:ctx]
                            for mode in cfg["modes"]:
                                for gamma in extra:
                                    for relative in cfg["relatives"]:
                                        spec = SFaultSpec(
                                            mode=mode, gamma=gamma, relative=relative,
                                            heads=heads)
                                        key = trial_key(
                                            prompt=seq["id"], ctx=ctx, layer=layer,
                                            seed=seed, mode=mode, gamma=gamma,
                                            relative=relative, heads=list(heads),
                                        )
                                        if key in done:
                                            continue
                                        run_seed = seed + int(
                                            hashlib.md5(key.encode()).hexdigest()[:8], 16
                                        ) % 10007
                                        fault = generate(prompt, spec, run_seed)
                                        rec = pack_trial(
                                            seq, ctx, layer, heads, seed, spec, clean, fault,
                                            run_split, first_divergence, is_harmful,
                                            logits_errors, ppl_from_nll, sequence_nll,
                                            token_nll_from_logits, top1_change_rate,
                                        )
                                        rec["key"] = key
                                        rec["densified"] = True
                                        fh.write(json.dumps(rec) + "\n")
                                        fh.flush()
                                        trials.append(rec)
                                        del fault

    baseline_rows = []
    for (pid, ctx, layer, seed), c in cleans.items():
        baseline_rows.append({
            "prompt_id": pid,
            "ctx": ctx,
            "layer": layer,
            "seed": seed,
            "prompt_tokens": c["prompt_tokens"],
            "gen_tokens": len(c["token_ids"]),
            "ppl": c["ppl"],
            "nll_true_next": c["nll_true"],
            "seconds": c["seconds"],
        })
    baseline_rows.extend(census_rows)
    write_csv(
        out_dir / "tables" / "baseline_metrics.csv",
        baseline_rows,
        ["prompt_id", "ctx", "layer", "seed", "prompt_tokens", "gen_tokens",
         "ppl", "nll_true_next", "seconds"],
    )
    unique_tok = {}
    for (pid, ctx, layer, seed), c in cleans.items():
        unique_tok[pid] = max(unique_tok.get(pid, 0), int(c["prompt_tokens"]))
    for row in census_rows:
        unique_tok[row["prompt_id"]] = max(
            unique_tok.get(row["prompt_id"], 0), int(row["prompt_tokens"]))
    n_clean_tokens = sum(unique_tok.values())

    # Never select tolerances on test.
    select_trials = [t for t in trials if t.get("split") != "test"]
    rec_path = out_dir / "tables" / "recommendation.json"
    if select_trials:
        rec = analyze_trials(
            select_trials, out_dir, geom.head_dim, consistency=consistency,
            split_name="calibration",
        )
    elif rec_path.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
    else:
        raise RuntimeError("no calibration trials to select S tolerances")
    test_trials = [t for t in trials if t.get("split") == "test"]
    if test_trials:
        tchk = write_test_check(test_trials, rec, out_dir)
        print("test_check:", json.dumps({
            "n_trials": tchk["n_trials"],
            "harmful_rate": tchk["harmful_rate"],
            "balanced_harmful_pass": (tchk.get("balanced") or {}).get("harmful_pass"),
        }), flush=True)
    print("recommendation:", json.dumps({
        "strict": [rec["strict_s_rtol_scaled"], rec["strict_s_atol_scaled"]],
        "balanced": [rec["balanced_s_rtol_scaled"], rec["balanced_s_atol_scaled"]],
        "result_rtol": rec["result_rtol"],
        "result_atol": rec["result_atol"],
        "n_select_trials": len(select_trials),
        "n_test_trials": len(test_trials),
        "n_clean_tokens": n_clean_tokens,
    }), flush=True)
    print(f"wrote {out_dir}", flush=True)


def densify_gammas(trials: list[dict], coarse: list[float]) -> list[float]:
    """Add log-midpoints where |mean rel PPL| or top-1 starts to move."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for t in trials:
        if t.get("kind") != "fault":
            continue
        buckets[float(t["gamma"])].append(t)
    extra = []
    ordered = sorted(coarse)
    for a, b in zip(ordered, ordered[1:]):
        ga = buckets.get(a, [])
        gb = buckets.get(b, [])
        if not ga or not gb:
            continue
        pa = sum(t["rel_ppl_rise"] for t in ga) / len(ga)
        pb = sum(t["rel_ppl_rise"] for t in gb) / len(gb)
        ta = sum(t["top1"]["estimate"] for t in ga) / len(ga)
        tb = sum(t["top1"]["estimate"] for t in gb) / len(gb)
        if abs(pb - pa) > 0.0005 or abs(tb - ta) > 0.0005:
            extra.append(math.sqrt(a * b))
    return extra


def pack_trial(
    seq, ctx, layer, heads, seed, spec, clean, fault, split,
    first_divergence, is_harmful, logits_errors, ppl_from_nll,
    sequence_nll, token_nll_from_logits, top1_change_rate,
) -> dict:
    clean_ids = clean["token_ids"]
    fault_ids = fault["token_ids"]
    div = first_divergence(clean_ids, fault_ids)
    top1 = top1_change_rate(clean_ids, fault_ids)
    n_tf = min(len(clean_ids), len(fault["logits"]), len(clean["logits"]))
    if div is not None:
        n_tf = min(n_tf, div + 1)
    if n_tf >= 1:
        ppl_c = ppl_from_nll(sequence_nll(clean["logits"][:n_tf], clean_ids[:n_tf]))
        ppl_f = ppl_from_nll(sequence_nll(fault["logits"][:n_tf], clean_ids[:n_tf]))
        rel_ppl = (ppl_f - ppl_c) / ppl_c if ppl_c else float("nan")
    else:
        ppl_c, ppl_f, rel_ppl = clean["ppl"], float("nan"), float("nan")
    logit_errs = [
        logits_errors(clean["logits"][i], fault["logits"][i]) for i in range(n_tf)
    ]
    extra_nan = any(s.get("has_nan_inf") for s in fault["scores"]) or any(
        e["has_nan_inf"] for e in logit_errs)
    nll_true_f = (token_nll_from_logits(fault["logits"][0], clean["true_next"])
                  if fault["logits"] else float("nan"))
    scores = []
    for s in fault["scores"]:
        inj = dict(s.get("inject") or {})
        inj.pop("loc", None)
        scores.append({
            "layer": s.get("layer"),
            "tq": s.get("tq"),
            "seq_len": s.get("seq_len"),
            "max_abs_es_scaled": s.get("max_abs_es_scaled"),
            "max_abs_es_raw": s.get("max_abs_es_raw"),
            "max_rel_es": s.get("max_rel_es"),
            "tv": s.get("tv"),
            "y_rel_l2": s.get("y_rel_l2"),
            "has_nan_inf": s.get("has_nan_inf"),
            "pass_s": s.get("pass_s"),
            "inject": inj,
        })
    mean_tv = sum(s.get("tv", 0.0) for s in scores) / len(scores) if scores else 0.0
    mean_y = sum(s.get("y_rel_l2", 0.0) for s in scores) / len(scores) if scores else 0.0
    first_changed = bool(clean_ids and fault_ids and clean_ids[0] != fault_ids[0])
    harmful = is_harmful(
        rel_ppl if not math.isnan(rel_ppl) else 0.0,
        float(top1["estimate"]),
        extra_nan,
    )
    return {
        "kind": "fault",
        "prompt_id": seq["id"],
        "split": seq.get("split", split),
        "ctx": ctx,
        "layer": layer,
        "n_selected_layers": 1,
        "heads": list(heads),
        "seed": seed,
        "mode": spec.mode,
        "gamma": spec.gamma,
        "relative": spec.relative,
        "clean_ids": clean_ids,
        "fault_ids": fault_ids,
        "true_next": clean["true_next"],
        "nll_true_clean": clean["nll_true"],
        "nll_true_fault": nll_true_f,
        "first_divergence": div,
        "first_token_changed": first_changed,
        "top1": top1,
        "ppl_clean": ppl_c,
        "ppl_fault_paired": ppl_f,
        "rel_ppl_rise": rel_ppl,
        "logits_max_abs": max((e["max_abs"] for e in logit_errs), default=0.0),
        "logits_rel_l2": (sum(e["rel_l2"] for e in logit_errs) / len(logit_errs)
                          if logit_errs else 0.0),
        "mean_tv": mean_tv,
        "mean_y_rel_l2": mean_y,
        "extra_nan_inf": extra_nan,
        "harmful": harmful,
        "score_steps": scores,
        "fault_seconds": fault["seconds"],
    }


if __name__ == "__main__":
    main()
