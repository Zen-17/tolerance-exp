"""Aggregate experiment A trials into tables, recommendation, and figures."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from smallexp2.faults import ATOL_GRID, REQUIRED_TOLERANCE, RTOL_GRID
from smallexp2.metrics import paired_bootstrap_mean, wilson_ci


def _pass_key(rtol: float, atol: float) -> str:
    return f"rtol_{rtol:g}_atol_{atol:g}"


def group_score_steps(records: list[dict], n_layers: int) -> list[list[dict]]:
    if n_layers <= 0 or not records:
        return []
    steps = []
    for i in range(0, len(records), n_layers):
        chunk = records[i:i + n_layers]
        if len(chunk) == n_layers:
            steps.append(chunk)
    return steps


def step_pass(chunk: list[dict], key: str) -> bool:
    return all(rec.get("pass_s", {}).get(key, {}).get("pass", False) for rec in chunk)


def collect_step_labels(trial: dict) -> list[dict]:
    """One row per query step: harmful if greedy token differs or NaN/Inf."""
    n_layers = max(1, int(trial.get("n_selected_layers", 1)))
    steps = group_score_steps(trial.get("score_steps", []), n_layers)
    clean = trial.get("clean_ids", [])
    fault = trial.get("fault_ids", [])
    n = min(len(steps), len(clean), len(fault))
    rows = []
    for i in range(n):
        nan = any(rec.get("has_nan_inf") for rec in steps[i])
        rows.append({
            "harmful": bool(trial.get("harmful") or clean[i] != fault[i] or nan),
            "nan": nan,
            "pass_s": steps[i],
            "max_abs_es": max(
                rec.get("max_abs_es_scaled", rec.get("max_abs_es", 0.0))
                for rec in steps[i]
            ),
            "tv": max(rec.get("tv", 0.0) for rec in steps[i]),
            "y_rel_l2": max(rec.get("y_rel_l2", 0.0) for rec in steps[i]),
        })
    return rows


def tolerance_table(trials: list[dict]) -> list[dict]:
    tols = [(rt, at) for rt in RTOL_GRID for at in ATOL_GRID]
    if REQUIRED_TOLERANCE not in tols:
        tols.append(REQUIRED_TOLERANCE)
    rows = []
    for rt, at in tols:
        key = _pass_key(rt, at)
        n_harm = n_harm_pass = n_ok = n_ok_reject = 0
        for trial in trials:
            if trial.get("kind") != "fault":
                continue
            for step in collect_step_labels(trial):
                passed = step_pass(step["pass_s"], key)
                if step["harmful"]:
                    n_harm += 1
                    n_harm_pass += int(passed)
                else:
                    n_ok += 1
                    n_ok_reject += int(not passed)
        hp = wilson_ci(n_harm_pass, n_harm)
        br = wilson_ci(n_ok_reject, n_ok)
        rows.append({
            "rtol_scaled": rt,
            "atol_scaled": at,
            "required": (rt, at) == REQUIRED_TOLERANCE,
            "harmful_pass": hp,
            "benign_reject": br,
        })
    return rows


def pick_recommendation(tol_rows: list[dict], head_dim: int) -> dict:
    def hp(row: dict) -> float:
        return float(row["harmful_pass"]["estimate"])

    def br(row: dict) -> float:
        n = int(row["benign_reject"]["denominator"])
        if n == 0:
            return 1.0
        return float(row["benign_reject"]["estimate"])

    strict = min(tol_rows, key=lambda r: (hp(r), r["rtol_scaled"], r["atol_scaled"]))
    candidates = [r for r in tol_rows if hp(r) <= 0.05]
    if not candidates:
        candidates = tol_rows
    # Prefer lower false-reject; if no benign samples exist, keep the required
    # pair when it is admissible, otherwise the loosest pair with min hp.
    if all(int(r["benign_reject"]["denominator"]) == 0 for r in candidates):
        required = [r for r in candidates if r.get("required")]
        balanced = required[0] if required else max(
            candidates, key=lambda r: (r["rtol_scaled"], r["atol_scaled"]))
    else:
        balanced = min(
            candidates,
            key=lambda r: (
                br(r),
                hp(r),
                not r.get("required", False),
                r["rtol_scaled"],
                r["atol_scaled"],
            ),
        )

    def pack(row: dict, prefix: str) -> dict:
        return {
            f"{prefix}_s_rtol_scaled": row["rtol_scaled"],
            f"{prefix}_s_atol_scaled": row["atol_scaled"],
            f"{prefix}_harmful_pass": row["harmful_pass"],
            f"{prefix}_benign_reject": row["benign_reject"],
        }

    rec = {}
    rec.update(pack(strict, "strict"))
    rec.update(pack(balanced, "balanced"))
    rec["recommended_s_rtol_raw"] = balanced["rtol_scaled"]
    rec["recommended_s_atol_raw"] = math.sqrt(head_dim) * balanced["atol_scaled"]
    rec["result_rtol"] = rec["recommended_s_rtol_raw"]
    rec["result_atol"] = rec["recommended_s_atol_raw"]
    rec["head_dim"] = head_dim
    rec["note"] = (
        "raw atol = sqrt(d_h) * scaled atol; raw rtol = scaled rtol. "
        "Smoke/phase-1 values are calibration only; test split is not used for selection."
    )
    return rec


def condition_curves(trials: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for trial in trials:
        if trial.get("kind") != "fault":
            continue
        key = (trial["mode"], bool(trial["relative"]), float(trial["gamma"]))
        buckets[key].append(trial)
    rows = []
    for (mode, relative, gamma), group in sorted(buckets.items()):
        rel_ppl = [float(t["rel_ppl_rise"]) for t in group if not math.isnan(t["rel_ppl_rise"])]
        top1 = [float(t["top1"]["estimate"]) for t in group]
        harmful_n = sum(1 for t in group if t["harmful"])
        rows.append({
            "mode": mode,
            "relative": relative,
            "gamma": gamma,
            "n_trials": len(group),
            "rel_ppl_rise": paired_bootstrap_mean(rel_ppl),
            "top1_change": paired_bootstrap_mean(top1),
            "harmful": wilson_ci(harmful_n, len(group)),
            "mean_tv": sum(t["mean_tv"] for t in group) / len(group),
            "mean_y_rel_l2": sum(t["mean_y_rel_l2"] for t in group) / len(group),
            "p95_rel_ppl": _percentile([t["rel_ppl_rise"] for t in group], 0.95),
            "p99_rel_ppl": _percentile([t["rel_ppl_rise"] for t in group], 0.99),
        })
    return rows


def _percentile(vals: list[float], q: float) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    if not clean:
        return float("nan")
    clean.sort()
    i = min(len(clean) - 1, max(0, int(math.ceil(q * len(clean)) - 1)))
    return clean[i]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def flatten_tol_row(row: dict) -> dict:
    hp, br = row["harmful_pass"], row["benign_reject"]
    return {
        "rtol_scaled": row["rtol_scaled"],
        "atol_scaled": row["atol_scaled"],
        "required": row["required"],
        "harmful_pass_k": hp["numerator"],
        "harmful_pass_n": hp["denominator"],
        "harmful_pass": hp["estimate"],
        "harmful_pass_ci95_low": hp["ci95_low"],
        "harmful_pass_ci95_high": hp["ci95_high"],
        "benign_reject_k": br["numerator"],
        "benign_reject_n": br["denominator"],
        "benign_reject": br["estimate"],
        "benign_reject_ci95_low": br["ci95_low"],
        "benign_reject_ci95_high": br["ci95_high"],
    }


def flatten_curve_row(row: dict) -> dict:
    rp, t1, h = row["rel_ppl_rise"], row["top1_change"], row["harmful"]
    return {
        "mode": row["mode"],
        "relative": row["relative"],
        "gamma": row["gamma"],
        "n_trials": row["n_trials"],
        "rel_ppl_rise": rp["mean"],
        "rel_ppl_ci95_low": rp["ci95_low"],
        "rel_ppl_ci95_high": rp["ci95_high"],
        "top1_change": t1["mean"],
        "top1_ci95_low": t1["ci95_low"],
        "top1_ci95_high": t1["ci95_high"],
        "harmful": h["estimate"],
        "harmful_k": h["numerator"],
        "harmful_n": h["denominator"],
        "mean_tv": row["mean_tv"],
        "mean_y_rel_l2": row["mean_y_rel_l2"],
        "p95_rel_ppl": row["p95_rel_ppl"],
        "p99_rel_ppl": row["p99_rel_ppl"],
    }


def try_figures(out_dir: Path, curves: list[dict], tol_rows: list[dict]) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return written

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode in sorted({r["mode"] for r in curves}):
        for rel, ls in ((False, "-"), (True, "--")):
            sub = [r for r in curves if r["mode"] == mode and r["relative"] is rel]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: r["gamma"])
            xs = [r["gamma"] for r in sub]
            label = f"{mode} {'rel' if rel else 'abs'}"
            axes[0].plot(xs, [r["rel_ppl_rise"]["mean"] for r in sub],
                         marker="o", linestyle=ls, label=label)
            axes[1].plot(xs, [r["top1_change"]["mean"] for r in sub],
                         marker="o", linestyle=ls, label=label)
    for ax, title in zip(axes, ["relative PPL rise", "top-1 change rate"]):
        ax.set_xscale("log")
        ax.set_xlabel("gamma")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fig_dir / f"s_error_vs_quality.{ext}"
        fig.savefig(p, dpi=300)
        written.append(str(p))
    plt.close(fig)

    rtols = list(RTOL_GRID)
    atols = list(ATOL_GRID)
    grid = np.full((len(rtols), len(atols)), float("nan"))
    lookup = {(r["rtol_scaled"], r["atol_scaled"]): r["harmful_pass"]["estimate"]
              for r in tol_rows}
    for i, rt in enumerate(rtols):
        for j, at in enumerate(atols):
            grid[i, j] = lookup.get((rt, at), float("nan"))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, origin="lower", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(atols)), [f"{a:g}" for a in atols], rotation=45)
    ax.set_yticks(range(len(rtols)), [f"{r:g}" for r in rtols])
    ax.set_xlabel("atol_scaled")
    ax.set_ylabel("rtol_scaled")
    ax.set_title("R_harmful_pass")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fig_dir / f"harmful_pass_heatmap.{ext}"
        fig.savefig(p, dpi=300)
        written.append(str(p))
    plt.close(fig)
    return written


def write_report(
    path: Path,
    rec: dict,
    consistency: Optional[dict],
    n_trials: int,
    limitations: str,
) -> None:
    lines = [
        "# Experiment A report",
        "",
        "## S-layer recommendation",
        "",
        f"- strict scaled (rtol, atol) = ({rec['strict_s_rtol_scaled']}, {rec['strict_s_atol_scaled']})",
        f"- balanced scaled (rtol, atol) = ({rec['balanced_s_rtol_scaled']}, {rec['balanced_s_atol_scaled']})",
        f"- recommended raw rtol = {rec['recommended_s_rtol_raw']}",
        f"- recommended raw atol = {rec['recommended_s_atol_raw']}",
        f"- result_rtol = {rec['result_rtol']}",
        f"- result_atol = {rec['result_atol']}",
        "",
        f"strict harmful-pass = {rec['strict_harmful_pass']['estimate']:.4f} "
        f"[{rec['strict_harmful_pass']['ci95_low']:.4f}, {rec['strict_harmful_pass']['ci95_high']:.4f}]",
        f"balanced benign-reject = {rec['balanced_benign_reject']['estimate']:.4f} "
        f"[{rec['balanced_benign_reject']['ci95_low']:.4f}, {rec['balanced_benign_reject']['ci95_high']:.4f}]",
        "",
        f"Trials used for selection: {n_trials} (calibration/smoke only).",
        "",
        "## Flash vs reference consistency",
        "",
        json.dumps(consistency or {}, ensure_ascii=False, indent=2),
        "",
        "## Limits",
        "",
        limitations,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_trials(
    trials: list[dict],
    out_dir: Path,
    head_dim: int,
    consistency: Optional[dict] = None,
    split_name: str = "calibration",
) -> dict[str, Any]:
    fault_trials = [t for t in trials if t.get("kind") == "fault"]
    curves = condition_curves(fault_trials)
    tols = tolerance_table(fault_trials)
    rec = pick_recommendation(tols, head_dim)

    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(
        tables / "s_tolerance.csv",
        [flatten_tol_row(r) for r in tols],
        list(flatten_tol_row(tols[0]).keys()) if tols else ["rtol_scaled"],
    )
    curve_flat = [flatten_curve_row(r) for r in curves]
    write_csv(
        tables / "s_quality_curves.csv",
        curve_flat,
        list(curve_flat[0].keys()) if curve_flat else ["mode"],
    )
    (tables / "recommendation.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8")
    figs = try_figures(out_dir, curves, tols)
    write_report(
        out_dir / "REPORT.md",
        rec,
        consistency,
        n_trials=len(fault_trials),
        limitations=(
            f"Parameter choice used the {split_name} split only. "
            "This experiment does not study K-block size, BER, or ABFT. "
            "Tolerances were not chosen on the test split."
        ),
    )
    rec["figures"] = figs
    return rec


def write_test_check(trials: list[dict], rec: dict, out_dir: Path) -> dict:
    """Evaluate calibration-chosen tols on held-out test trials. Never selects."""
    fault = [t for t in trials if t.get("kind") == "fault"]
    n_harm_t = sum(1 for t in fault if t.get("harmful"))
    out: dict[str, Any] = {
        "n_trials": len(fault),
        "harmful_rate": wilson_ci(n_harm_t, len(fault)),
        "used_for_selection": False,
    }
    pairs = [
        ("strict", rec.get("strict_s_rtol_scaled"), rec.get("strict_s_atol_scaled"),
         rec.get("strict_harmful_pass")),
        ("balanced", rec.get("balanced_s_rtol_scaled"), rec.get("balanced_s_atol_scaled"),
         rec.get("balanced_harmful_pass")),
    ]
    for name, rt, at, cal in pairs:
        if rt is None or at is None:
            continue
        key = _pass_key(float(rt), float(at))
        n_harm = n_hp = n_ok = n_br = 0
        for trial in fault:
            for step in collect_step_labels(trial):
                passed = step_pass(step["pass_s"], key)
                if step["harmful"]:
                    n_harm += 1
                    n_hp += int(passed)
                else:
                    n_ok += 1
                    n_br += int(not passed)
        out[name] = {
            "rtol_scaled": rt,
            "atol_scaled": at,
            "harmful_pass": wilson_ci(n_hp, n_harm),
            "benign_reject": wilson_ci(n_br, n_ok),
            "calibration_harmful_pass": cal,
        }
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "test_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
