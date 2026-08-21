"""Aggregate experiment B lifetime risk and K→S transfer (experiment C)."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from smallexp2.metrics import wilson_ci

WINDOWS = (1, 2, 16, "full")
_WINDOW_RANK = {1: 0, 2: 1, 16: 2, "full": 3, "1": 0, "2": 1, "16": 2}


def _harm_at_u(trial: dict, u: int) -> bool:
    scores = trial.get("score_steps") or []
    clean = trial.get("clean_ids") or []
    fault = trial.get("fault_ids") or []
    nan = False
    if u < len(scores):
        nan = bool(scores[u].get("has_nan_inf"))
    changed = u < len(clean) and u < len(fault) and clean[u] != fault[u]
    return bool(nan or changed)


def lifetime_harm(trial: dict, l) -> bool:
    """Monotonic H_K^(L): any earlier window or any of the first L queries."""
    n = len(trial.get("score_steps") or [])
    if n == 0:
        n = min(len(trial.get("clean_ids") or []), len(trial.get("fault_ids") or []))
    cap = n if l == "full" else min(int(l), n)
    if any(_harm_at_u(trial, u) for u in range(cap)):
        return True
    wm = trial.get("window_metrics") or {}
    target = _WINDOW_RANK.get(l, 3)
    for w in WINDOWS:
        if _WINDOW_RANK[w] <= target and bool((wm.get(str(w)) or {}).get("harmful")):
            return True
    return False


def k_tolerance_table(trials: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        if t.get("kind") != "fault":
            continue
        key = (
            t.get("k_kind"),
            t.get("rel"),
            t.get("bit_class"),
            t.get("layer"),
        )
        buckets[key].append(t)
        buckets[(t.get("k_kind"), t.get("rel"), t.get("bit_class"), "all")].append(t)
    rows = []
    for (kind, rel, bit_class, layer), group in sorted(
            buckets.items(), key=lambda kv: (
                str(kv[0][0]), str(kv[0][1]), str(kv[0][2]), str(kv[0][3]))):
        row: dict[str, Any] = {
            "k_kind": kind,
            "rel": rel,
            "bit_class": bit_class,
            "layer": layer,
            "n_trials": len(group),
        }
        for w in WINDOWS:
            h = sum(1 for t in group if lifetime_harm(t, w))
            row[f"H_L{w}"] = wilson_ci(h, len(group))
        delayed = sum(
            1 for t in group if (not lifetime_harm(t, 1)) and lifetime_harm(t, "full")
        )
        row["delayed_harm"] = wilson_ci(delayed, len(group))
        rows.append(row)
    return rows


def k_to_s_rows(trials: list[dict]) -> list[dict]:
    out = []
    for t in trials:
        if t.get("kind") != "fault":
            continue
        scores = t.get("score_steps") or []
        if not scores:
            continue
        e0 = float(scores[0].get("max_abs_es_scaled") or 0.0)
        future = [float(s.get("max_abs_es_scaled") or 0.0) for s in scores[1:]]
        max_future = max(future) if future else 0.0
        amp = bool(future) and max_future > e0 + 1e-12
        fail_L = {}
        for w in WINDOWS:
            cap = len(scores) if w == "full" else min(int(w), len(scores))
            fail_L[str(w)] = any(not s.get("pass_s_recommended", True) for s in scores[:cap])
        out.append({
            "prompt_id": t.get("prompt_id"),
            "layer": t.get("layer"),
            "ctx": t.get("ctx"),
            "k_kind": t.get("k_kind"),
            "rel": t.get("rel"),
            "bit_class": t.get("bit_class"),
            "kv_head": t.get("kv_head"),
            "abs_delta_k": t.get("abs_delta_k"),
            "max_es_u0": e0,
            "max_es_future": max_future,
            "future_amplified": amp,
            "n_queries": len(scores),
            "fail_s_L1": fail_L["1"],
            "fail_s_L2": fail_L["2"],
            "fail_s_L16": fail_L["16"],
            "fail_s_full": fail_L["full"],
        })
    return out


def transfer_summary(ks_rows: list[dict]) -> dict:
    n = len(ks_rows)
    amp_n = sum(1 for r in ks_rows if r["future_amplified"])
    by_bit = defaultdict(list)
    by_layer = defaultdict(list)
    for r in ks_rows:
        if r.get("k_kind") == "bitflip":
            by_bit[r.get("bit_class")].append(r)
        if r.get("layer") is not None:
            by_layer[r.get("layer")].append(r)

    def fail_full(group):
        if not group:
            return wilson_ci(0, 0)
        k = sum(1 for r in group if r["fail_s_full"])
        return wilson_ci(k, len(group))

    def mean_es(group):
        vals = [r["max_es_future"] for r in group
                if isinstance(r.get("max_es_future"), (int, float))
                and math.isfinite(r["max_es_future"])]
        n_inf = sum(1 for r in group
                    if isinstance(r.get("max_es_future"), (int, float))
                    and math.isinf(r["max_es_future"]))
        n_nan = sum(1 for r in group
                    if isinstance(r.get("max_es_future"), (int, float))
                    and math.isnan(r["max_es_future"]))
        return {
            "mean_max_es_future_finite": (sum(vals) / len(vals) if vals else None),
            "n_finite": len(vals),
            "n_inf": n_inf,
            "n_nan": n_nan,
        }

    return {
        "n": n,
        "future_amplified": wilson_ci(amp_n, n),
        "p_fail_s_full": fail_full(ks_rows),
        "by_bit_class": {str(k): {
            "n": len(v),
            "p_fail_s_full": fail_full(v),
            **mean_es(v),
        } for k, v in by_bit.items()},
        "by_layer": {str(k): {
            "n": len(v),
            "p_fail_s_full": fail_full(v),
            **mean_es(v),
        } for k, v in by_layer.items()},
    }


def flatten_h(ci: dict, prefix: str) -> dict:
    return {
        f"{prefix}": ci["estimate"],
        f"{prefix}_k": ci["numerator"],
        f"{prefix}_n": ci["denominator"],
        f"{prefix}_ci95_low": ci["ci95_low"],
        f"{prefix}_ci95_high": ci["ci95_high"],
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def try_figures(out_dir: Path, tol_rows: list[dict], ks_rows: list[dict]) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return written
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    num = [r for r in tol_rows if r.get("k_kind") == "numeric"
           and r.get("rel") is not None and r.get("layer") == "all"]
    if not num:
        num = [r for r in tol_rows if r.get("k_kind") == "numeric" and r.get("rel") is not None]
    if num:
        fig, ax = plt.subplots(figsize=(6, 4))
        rels = sorted({float(r["rel"]) for r in num})
        for w, marker in ((1, "o"), (16, "s"), ("full", "^")):
            ys = []
            for rel in rels:
                grp = [r for r in num if float(r["rel"]) == rel]
                if not grp:
                    ys.append(float("nan"))
                    continue
                ys.append(sum(r[f"H_L{w}"]["estimate"] for r in grp) / len(grp))
            ax.plot(rels, ys, marker=marker, label=f"L={w}")
        ax.set_xscale("log")
        ax.set_xlabel(r"$|\delta_K|/(|K|+\epsilon)$")
        ax.set_ylabel(r"$P(H_K^{(L)}=1)$")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = fig_dir / f"k_lifetime_harm.{ext}"
            fig.savefig(p, dpi=300)
            written.append(str(p))
        plt.close(fig)

    scatter = [r for r in ks_rows if r.get("abs_delta_k") is not None]
    if scatter:
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [math.log10(float(r["abs_delta_k"]) + 1e-12) for r in scatter]
        ys = [math.log10(r["max_es_future"] + 1e-12) for r in scatter]
        ax.scatter(xs, ys, s=12, alpha=0.5)
        ax.set_xlabel(r"$\log_{10}|E_K|$")
        ax.set_ylabel(r"$\log_{10}\max_{u>0}|E_S|$")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = fig_dir / f"k_to_s_scatter.{ext}"
            fig.savefig(p, dpi=300)
            written.append(str(p))
        plt.close(fig)

        bits = sorted({r["bit_class"] for r in ks_rows if r.get("k_kind") == "bitflip" and r.get("bit_class")})
        if bits:
            fig, ax = plt.subplots(figsize=(5, 4))
            means, labels = [], []
            for b in bits:
                grp = [r for r in ks_rows if r.get("bit_class") == b]
                means.append(sum(r["fail_s_full"] for r in grp) / max(len(grp), 1))
                labels.append(b)
            ax.bar(labels, means)
            ax.set_ylabel(r"$P(\exists u:\neg Pass_S)$")
            ax.set_title("bit-class K→S")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                p = fig_dir / f"k_bitclass.{ext}"
                fig.savefig(p, dpi=300)
                written.append(str(p))
            plt.close(fig)

        layers = sorted({r["layer"] for r in ks_rows if r.get("layer") is not None})
        if layers:
            fig, ax = plt.subplots(figsize=(5, 4))
            means = []
            for layer in layers:
                grp = [r for r in ks_rows if r.get("layer") == layer]
                means.append(sum(r["fail_s_full"] for r in grp) / max(len(grp), 1))
            ax.bar([str(x) for x in layers], means)
            ax.set_xlabel("layer")
            ax.set_ylabel(r"$P(\exists u:\neg Pass_S)$")
            ax.set_title("early/middle/late")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                p = fig_dir / f"k_layer.{ext}"
                fig.savefig(p, dpi=300)
                written.append(str(p))
            plt.close(fig)
    return written


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def delayed_and_lifetime(trials: list[dict]) -> dict:
    n = len(trials)
    out: dict[str, Any] = {"n": n, "windows": ["1", "2", "16", "full"]}
    for w in WINDOWS:
        h = sum(1 for t in trials if lifetime_harm(t, w))
        out[f"H_L{w}"] = wilson_ci(h, n)
    delayed = sum(1 for t in trials if (not lifetime_harm(t, 1)) and lifetime_harm(t, "full"))
    out["delayed_harm"] = wilson_ci(delayed, n)
    rels = []
    for t in trials:
        v = t.get("rel_ppl_rise")
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            rels.append(float(v))
    out["rel_ppl_p95"] = _percentile(rels, 0.95)
    out["rel_ppl_p99"] = _percentile(rels, 0.99)
    out["rel_ppl_worst"] = max(rels) if rels else float("nan")
    return out


def analyze_trials_b(
    trials: list[dict],
    out_dir: Path,
    s_tol: dict,
    consistency: Optional[dict] = None,
) -> dict:
    fault = [t for t in trials if t.get("kind") == "fault"]
    tol_rows = k_tolerance_table(fault)
    ks_rows = k_to_s_rows(fault)
    summary = transfer_summary(ks_rows)
    life = delayed_and_lifetime(fault)

    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    flat_tol = []
    for r in tol_rows:
        item = {
            "k_kind": r["k_kind"], "rel": r["rel"], "bit_class": r["bit_class"],
            "layer": r["layer"], "n_trials": r["n_trials"],
        }
        for w in WINDOWS:
            item.update(flatten_h(r[f"H_L{w}"], f"H_L{w}"))
        item.update(flatten_h(r["delayed_harm"], "delayed_harm"))
        flat_tol.append(item)
    fields = list(flat_tol[0].keys()) if flat_tol else ["k_kind"]
    write_csv(tables / "k_tolerance.csv", flat_tol, fields)
    ks_fields = list(ks_rows[0].keys()) if ks_rows else ["prompt_id"]
    write_csv(tables / "k_to_s_transfer.csv", ks_rows, ks_fields)

    rec = {
        "single_k_rtol": None,
        "single_k_atol": None,
        "use_risk_curves": True,
        "reason": "No evidence for a single K rtol/atol; use lifetime risk curves.",
        "s_tolerance_used": s_tol,
        "sparsity": "single_K_element_per_sample",
        "lifetime": life,
        "k_to_s": summary,
    }
    (tables / "recommendation.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    figs = try_figures(out_dir, tol_rows, ks_rows)
    rec["figures"] = figs
    (tables / "recommendation.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    lines = [
        "# Experiment B report (K lifetime + K→S)",
        "",
        "## K layer",
        "",
        "No single K_rtol/K_atol. Use lifetime risk curves in k_tolerance.csv.",
        f"Sparsity: one K element per sample. S-tol used: {s_tol}.",
        "",
        f"- H_K L=1: {life['H_L1']['estimate']:.4f} "
        f"[{life['H_L1']['ci95_low']:.4f}, {life['H_L1']['ci95_high']:.4f}]",
        f"- H_K L=16: {life['H_L16']['estimate']:.4f} "
        f"[{life['H_L16']['ci95_low']:.4f}, {life['H_L16']['ci95_high']:.4f}]",
        f"- H_K full: {life['H_Lfull']['estimate']:.4f} "
        f"[{life['H_Lfull']['ci95_low']:.4f}, {life['H_Lfull']['ci95_high']:.4f}]",
        f"- delayed harm (current q OK, later q harmful): {life['delayed_harm']['estimate']:.4f}",
        f"- rel PPL p95/p99/worst: {life['rel_ppl_p95']:.4g} / {life['rel_ppl_p99']:.4g} / {life['rel_ppl_worst']:.4g}",
        "",
        "## K→S",
        "",
        f"- future-q amplified: {summary['future_amplified']['estimate']:.4f} "
        f"[{summary['future_amplified']['ci95_low']:.4f}, {summary['future_amplified']['ci95_high']:.4f}]",
        f"- P(exists u: not Pass_S) full lifetime: {summary['p_fail_s_full']['estimate']:.4f}",
        f"- by bit class: {json.dumps(summary['by_bit_class'])}",
        f"- by layer: {json.dumps(summary['by_layer'])}",
        "",
        "## Consistency",
        "",
        json.dumps(consistency or {}, indent=2),
        "",
        "## Limits",
        "",
        "Does not study K-block size, BER, or ABFT. Test split was not used for selection.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return rec


def write_test_check_b(trials: list[dict], out_dir: Path) -> dict:
    """Held-out test risk curves. Never used to choose a K tolerance."""
    fault = [t for t in trials if t.get("kind") == "fault"]
    ks_rows = k_to_s_rows(fault)
    out = {
        "n_trials": len(fault),
        "used_for_selection": False,
        "lifetime": delayed_and_lifetime(fault),
        "k_to_s": transfer_summary(ks_rows),
    }
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "test_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
