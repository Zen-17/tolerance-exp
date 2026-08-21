"""Experiment C: K→S transfer statistics from persistent K faults."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from smallexp2.metrics import wilson_ci

WINDOWS = (1, 2, 16, "full")
EK_EDGES = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, float("inf"))
EPS = 1e-12


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def ek_bin(abs_ek: Optional[float]) -> str:
    if not finite(abs_ek):
        return "nonfinite"
    x = float(abs_ek)
    lo = 0.0
    for hi in EK_EDGES:
        if x < hi:
            return f"[{lo:g},{hi:g})"
        lo = hi
    return f"[{lo:g},inf)"


def fail_s_window(scores: list[dict], l) -> bool:
    if not scores:
        return False
    cap = len(scores) if l == "full" else min(int(l), len(scores))
    return any(not s.get("pass_s_recommended", True) for s in scores[:cap])


def trial_transfer_row(t: dict) -> Optional[dict]:
    if t.get("kind") != "fault":
        return None
    scores = t.get("score_steps") or []
    if not scores:
        return None
    e0 = scores[0].get("max_abs_es_scaled")
    e0f = float(e0) if finite(e0) else float("nan")
    future = [s.get("max_abs_es_scaled") for s in scores[1:]]
    future_f = [float(x) for x in future if finite(x)]
    max_future = max(future_f) if future_f else 0.0
    amp = bool(future_f) and finite(e0f) and max_future > e0f + 1e-12
    abs_ek = t.get("abs_delta_k")
    gains = []
    if finite(abs_ek):
        for s in scores:
            es = s.get("max_abs_es_scaled")
            if finite(es):
                gains.append(float(es) / (float(abs_ek) + EPS))
    row = {
        "prompt_id": t.get("prompt_id"),
        "layer": t.get("layer"),
        "ctx": t.get("ctx"),
        "kv_head": t.get("kv_head"),
        "k_kind": t.get("k_kind"),
        "rel": t.get("rel"),
        "bit_class": t.get("bit_class"),
        "abs_delta_k": abs_ek if finite(abs_ek) else None,
        "ek_bin": ek_bin(abs_ek if finite(abs_ek) else None),
        "n_queries": len(scores),
        "max_es_u0": e0f if finite(e0f) else None,
        "max_es_future": max_future,
        "max_es_lifetime": max(([e0f] if finite(e0f) else []) + future_f) if (finite(e0f) or future_f) else None,
        "future_amplified": amp,
        "gain_u0": (e0f / (float(abs_ek) + EPS)) if finite(e0f) and finite(abs_ek) else None,
        "gain_p50": _percentile(gains, 0.5) if gains else None,
        "n_es_nonfinite": sum(1 for s in scores if not finite(s.get("max_abs_es_scaled"))),
    }
    for w in WINDOWS:
        row[f"fail_s_L{w}"] = fail_s_window(scores, w)
    return row


def query_rows(trials: list[dict]) -> list[dict]:
    out = []
    for t in trials:
        if t.get("kind") != "fault":
            continue
        abs_ek = t.get("abs_delta_k")
        for s in t.get("score_steps") or []:
            u = s.get("query_u")
            if u is None:
                u = len(out)
            es = s.get("max_abs_es_scaled")
            out.append({
                "prompt_id": t.get("prompt_id"),
                "layer": t.get("layer"),
                "ctx": t.get("ctx"),
                "kv_head": t.get("kv_head"),
                "k_kind": t.get("k_kind"),
                "bit_class": t.get("bit_class"),
                "query_u": u,
                "seq_len": s.get("seq_len"),
                "abs_delta_k": abs_ek if finite(abs_ek) else None,
                "max_abs_es_scaled": float(es) if finite(es) else None,
                "pass_s_recommended": bool(s.get("pass_s_recommended", True)),
                "q_abs_at_dim": s.get("q_abs_at_dim"),
                "predicted_es_scaled": s.get("predicted_es_scaled"),
                "es_at_fault_key_scaled": s.get("es_at_fault_key_scaled"),
            })
    return out


def group_fail_table(rows: list[dict], key_fn, extra_keys: list[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r)
    out = []
    for key, group in sorted(buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        item: dict[str, Any] = {}
        for name, val in zip(extra_keys, key):
            item[name] = val
        item["n_trials"] = len(group)
        for w in WINDOWS:
            k = sum(1 for r in group if r.get(f"fail_s_L{w}"))
            item.update(_flat_ci(wilson_ci(k, len(group)), f"fail_s_L{w}"))
        amp = sum(1 for r in group if r.get("future_amplified"))
        item.update(_flat_ci(wilson_ci(amp, len(group)), "future_amplified"))
        es_f = [r["max_es_future"] for r in group if finite(r.get("max_es_future"))]
        item["max_es_future_p50"] = _percentile(es_f, 0.5) if es_f else None
        item["max_es_future_p90"] = _percentile(es_f, 0.9) if es_f else None
        gains = [r["gain_p50"] for r in group if finite(r.get("gain_p50"))]
        item["gain_p50"] = _percentile(gains, 0.5) if gains else None
        item["n_es_nonfinite"] = sum(int(r.get("n_es_nonfinite") or 0) for r in group)
        out.append(item)
    return out


def _flat_ci(ci: dict, prefix: str) -> dict:
    return {
        f"{prefix}": ci["estimate"],
        f"{prefix}_k": ci["numerator"],
        f"{prefix}_n": ci["denominator"],
        f"{prefix}_ci95_low": ci["ci95_low"],
        f"{prefix}_ci95_high": ci["ci95_high"],
    }


def query_summary(qrows: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in qrows:
        buckets[(r.get("query_u"), r.get("ctx"))].append(r)
    out = []
    for (u, ctx), group in sorted(buckets.items(), key=lambda kv: (kv[0][0] if kv[0][0] is not None else -1, kv[0][1] or 0)):
        es = [r["max_abs_es_scaled"] for r in group if finite(r.get("max_abs_es_scaled"))]
        fail = sum(1 for r in group if not r.get("pass_s_recommended", True))
        out.append({
            "query_u": u,
            "ctx": ctx,
            "n": len(group),
            "max_es_p50": _percentile(es, 0.5) if es else None,
            "max_es_p90": _percentile(es, 0.9) if es else None,
            **_flat_ci(wilson_ci(fail, len(group)), "fail_s"),
        })
    return out


def identity_rows(trials: list[dict]) -> list[dict]:
    out = []
    for t in trials:
        if t.get("kind") != "fault":
            continue
        abs_ek = t.get("abs_delta_k")
        for s in t.get("score_steps") or []:
            pred = s.get("predicted_es_scaled")
            meas = s.get("es_at_fault_key_scaled")
            if pred is None and meas is None:
                continue
            rel_err = None
            if finite(pred) and finite(meas):
                rel_err = abs(float(meas) - float(pred)) / (abs(float(pred)) + EPS)
            out.append({
                "prompt_id": t.get("prompt_id"),
                "layer": t.get("layer"),
                "ctx": t.get("ctx"),
                "query_u": s.get("query_u"),
                "k_kind": t.get("k_kind"),
                "rel": t.get("rel"),
                "bit_class": t.get("bit_class"),
                "abs_delta_k": abs_ek if finite(abs_ek) else None,
                "q_abs_at_dim": s.get("q_abs_at_dim"),
                "predicted_es_scaled": float(pred) if finite(pred) else None,
                "es_at_fault_key_scaled": float(meas) if finite(meas) else None,
                "max_abs_es_scaled": (
                    float(s["max_abs_es_scaled"]) if finite(s.get("max_abs_es_scaled")) else None
                ),
                "rel_err": rel_err,
            })
    return out


def identity_summary(rows: list[dict]) -> dict:
    numeric = [r for r in rows if r.get("k_kind") == "numeric" and finite(r.get("rel_err"))]
    errs = [float(r["rel_err"]) for r in numeric]
    close = sum(1 for e in errs if e < 0.05)
    return {
        "n_identity_records": len(rows),
        "n_numeric_finite": len(numeric),
        "rel_err_p50": _percentile(errs, 0.5) if errs else None,
        "rel_err_p90": _percentile(errs, 0.9) if errs else None,
        "frac_rel_err_lt_5pct": (wilson_ci(close, len(errs)) if errs else wilson_ci(0, 0)),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def json_safe(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def try_figures(out_dir: Path, rows: list[dict], qrows: list[dict], id_rows: list[dict]) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return written
    try:
        return _draw_figures(out_dir, rows, qrows, id_rows, plt)
    except Exception as exc:
        print(f"[figures] skipped: {exc}", flush=True)
        return written


def _draw_figures(out_dir: Path, rows: list[dict], qrows: list[dict], id_rows: list[dict], plt) -> list[str]:
    written: list[str] = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    scatter = [r for r in rows if finite(r.get("abs_delta_k")) and finite(r.get("max_es_future"))]
    if scatter:
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [math.log10(float(r["abs_delta_k"]) + EPS) for r in scatter]
        ys = [math.log10(float(r["max_es_future"]) + EPS) for r in scatter]
        ax.scatter(xs, ys, s=10, alpha=0.4, label="trials")
        bin_pts = defaultdict(list)
        for r in scatter:
            bin_pts[r["ek_bin"]].append((float(r["abs_delta_k"]), float(r["max_es_future"])))
        ordered = []
        for b, pts in bin_pts.items():
            ordered.append((min(p[0] for p in pts), b, pts))
        ordered.sort()
        bx, p50, p90 = [], [], []
        for _, _, pts in ordered:
            eks = [p[0] for p in pts]
            ess = [p[1] for p in pts]
            bx.append(math.log10(sum(eks) / len(eks) + EPS))
            p50.append(math.log10(_percentile(ess, 0.5) + EPS))
            p90.append(math.log10(_percentile(ess, 0.9) + EPS))
        if bx:
            ax.plot(bx, p50, "o-", color="C1", label="p50")
            ax.plot(bx, p90, "s--", color="C3", label="p90")
        ax.set_xlabel(r"$\log_{10}|E_K|$")
        ax.set_ylabel(r"$\log_{10}\max_{u>0}|E_S|$")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = fig_dir / f"k_to_s_scatter.{ext}"
            fig.savefig(p, dpi=300)
            written.append(str(p))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        labels, y1, y16 = [], [], []
        by_bin = group_fail_table(rows, lambda r: (r["ek_bin"],), ["ek_bin"])
        for item in by_bin:
            if item["ek_bin"] == "nonfinite":
                continue
            labels.append(item["ek_bin"])
            y1.append(item["fail_s_L1"])
            y16.append(item["fail_s_L16"])
        ax.plot(range(len(labels)), y1, "o-", label="L=1")
        ax.plot(range(len(labels)), y16, "s-", label="L=16")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("P(fail Pass_S | |E_K|, L)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = fig_dir / f"k_to_s_pass_by_ek.{ext}"
            fig.savefig(p, dpi=300)
            written.append(str(p))
        plt.close(fig)

    bits = sorted({r["bit_class"] for r in rows if r.get("k_kind") == "bitflip" and r.get("bit_class")})
    if bits:
        fig, ax = plt.subplots(figsize=(5, 4))
        means = []
        for b in bits:
            grp = [r for r in rows if r.get("bit_class") == b]
            means.append(sum(bool(r["fail_s_Lfull"]) for r in grp) / max(len(grp), 1))
        ax.bar(bits, means)
        ax.set_ylabel("P(fail Pass_S)")
        ax.set_title("bit-class K→S")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = fig_dir / f"k_bitclass.{ext}"
            fig.savefig(p, dpi=300)
            written.append(str(p))
        plt.close(fig)

    layers = sorted({r["layer"] for r in rows if r.get("layer") is not None})
    if layers:
        fig, ax = plt.subplots(figsize=(5, 4))
        means = []
        for layer in layers:
            grp = [r for r in rows if r.get("layer") == layer]
            means.append(sum(bool(r["fail_s_Lfull"]) for r in grp) / max(len(grp), 1))
        ax.bar([str(x) for x in layers], means)
        ax.set_xlabel("layer")
        ax.set_ylabel("P(fail Pass_S)")
        ax.set_title("early/middle/late")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = fig_dir / f"k_layer.{ext}"
            fig.savefig(p, dpi=300)
            written.append(str(p))
        plt.close(fig)

    if qrows:
        fig, ax = plt.subplots(figsize=(6, 4))
        for ctx, marker in ((64, "o"), (256, "s")):
            xs, ys = [], []
            for item in query_summary([r for r in qrows if r.get("ctx") == ctx]):
                if item["max_es_p50"] is None:
                    continue
                xs.append(item["query_u"])
                ys.append(item["max_es_p50"])
            if xs:
                ax.plot(xs, ys, marker=marker, label=f"ctx={ctx}")
        ax.set_xlabel("query u after K inject")
        ax.set_ylabel(r"median $\max|E_S|$")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = fig_dir / f"k_to_s_by_query.{ext}"
            fig.savefig(p, dpi=300)
            written.append(str(p))
        plt.close(fig)

    id_num = [r for r in id_rows if finite(r.get("predicted_es_scaled")) and finite(r.get("es_at_fault_key_scaled"))]
    if id_num:
        fig, ax = plt.subplots(figsize=(5, 5))
        xs = [math.log10(abs(r["predicted_es_scaled"]) + EPS) for r in id_num]
        ys = [math.log10(abs(r["es_at_fault_key_scaled"]) + EPS) for r in id_num]
        ax.scatter(xs, ys, s=12, alpha=0.5)
        lim = [min(xs + ys), max(xs + ys)]
        ax.plot(lim, lim, "k--", linewidth=1)
        ax.set_xlabel(r"$\log_{10}$ predicted $|E_K q|/\sqrt{d_h}$")
        ax.set_ylabel(r"$\log_{10}$ measured $|E_S|$ at key $p$")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = fig_dir / f"k_to_s_identity.{ext}"
            fig.savefig(p, dpi=300)
            written.append(str(p))
        plt.close(fig)
    return written


def analyze_trials_c(
    trials: list[dict],
    out_dir: Path,
    s_tol: dict,
    identity_trials: Optional[list[dict]] = None,
    b_source: Optional[str] = None,
) -> dict:
    rows = [r for t in trials if (r := trial_transfer_row(t)) is not None]
    qrows = query_rows(trials)
    id_src = identity_trials if identity_trials else trials
    id_rows = identity_rows(id_src)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    write_csv(tables / "k_to_s_transfer.csv", rows)
    write_csv(
        tables / "k_to_s_by_ek_bin.csv",
        group_fail_table(rows, lambda r: (r["ek_bin"], r.get("k_kind")), ["ek_bin", "k_kind"]),
    )
    write_csv(tables / "k_to_s_by_query.csv", query_summary(qrows))
    write_csv(
        tables / "k_to_s_by_layer.csv",
        group_fail_table(rows, lambda r: (r.get("layer"),), ["layer"]),
    )
    write_csv(
        tables / "k_to_s_by_head.csv",
        group_fail_table(rows, lambda r: (r.get("kv_head"), r.get("layer")), ["kv_head", "layer"]),
    )
    write_csv(
        tables / "k_to_s_by_bitclass.csv",
        group_fail_table(
            [r for r in rows if r.get("k_kind") == "bitflip"],
            lambda r: (r.get("bit_class"),),
            ["bit_class"],
        ),
    )
    write_csv(tables / "identity_check.csv", id_rows)

    n = len(rows)
    amp = sum(1 for r in rows if r.get("future_amplified"))
    fail_full = sum(1 for r in rows if r.get("fail_s_Lfull"))
    fail_1 = sum(1 for r in rows if r.get("fail_s_L1"))
    ident = identity_summary(id_rows)
    figs = try_figures(out_dir, rows, qrows, id_rows)

    rec = {
        "experiment": "C",
        "b_source": b_source,
        "s_tolerance_used": s_tol,
        "n_trials": n,
        "identity": ident,
        "future_amplified": wilson_ci(amp, n),
        "p_fail_s_L1": wilson_ci(fail_1, n),
        "p_fail_s_full": wilson_ci(fail_full, n),
        "by_ek_bin": group_fail_table(rows, lambda r: (r["ek_bin"],), ["ek_bin"]),
        "by_bit_class": group_fail_table(
            [r for r in rows if r.get("k_kind") == "bitflip"],
            lambda r: (r.get("bit_class"),),
            ["bit_class"],
        ),
        "by_layer": group_fail_table(rows, lambda r: (r.get("layer"),), ["layer"]),
        "by_ctx": group_fail_table(rows, lambda r: (r.get("ctx"),), ["ctx"]),
        "figures": figs,
        "note": "Transfer is E_S = E_K q / sqrt(d_h) on the corrupted key column and GQA heads. No single K rtol.",
    }
    rec = json_safe(rec)
    (tables / "recommendation.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    def _fmt(ci):
        return f"{ci['estimate']:.4f} [{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]"

    ident_line = (
        f"numeric finite n={ident['n_numeric_finite']}, "
        f"rel_err p50={ident['rel_err_p50']}, p90={ident['rel_err_p90']}"
        if ident["n_numeric_finite"]
        else "no identity records (run probe or a B collect with hook identity fields)"
    )
    lines = [
        "# Experiment C report (K→S transfer)",
        "",
        "## Mapping",
        "",
        f"S-tol used: {s_tol}. Source B: {b_source}. n={n}.",
        "Single-element E_K maps to E_S = E_K q / sqrt(d_h) on the corrupted key column.",
        "",
        f"- P(fail Pass_S) L=1: {_fmt(rec['p_fail_s_L1'])}",
        f"- P(fail Pass_S) full: {_fmt(rec['p_fail_s_full'])}",
        f"- future-q amplified: {_fmt(rec['future_amplified'])}",
        "",
        "## Identity",
        "",
        ident_line,
        "",
        "## Bit class / layer / ctx",
        "",
        json.dumps({
            "bit": rec["by_bit_class"],
            "layer": rec["by_layer"],
            "ctx": rec["by_ctx"],
        }, indent=2),
        "",
        "## Limits",
        "",
        "Does not study K-block size, BER, or ABFT. Test split was not used for selection.",
        "Numeric deltas smaller than the BF16 ulp of the stored K element quantize to 0; identity uses the stored delta.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return rec
