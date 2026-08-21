"""Reproducible language-modeling sequences for experiment A.

Dataset: smallexp2_synthetic_lm_v1
Split: calibration (first 80%) / test (last 20%). Smoke uses the first 16
calibration sequences. Sequences are packed to seq_len tokens plus one
held-out next token. Do not choose tolerances on the test split.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

DATASET_NAME = "smallexp2_synthetic_lm_v1"
DATASET_VERSION = "1.0"
SEQ_LEN = 256
SHORT_CTX = 64
LONG_CTX = 256

# TASK_SPEC §6.3 minima (not a full factorial):
# 196 cal sequences * 256 tokens = 50,176 unique clean prompt tokens.
# Inject on 84 cal sequences, 1 layer (cycled over early/middle/late),
# both ctx, 3 seeds => 84*2*3 = 504 positions per condition (>= 500).
PHASE1_N_CAL = 196
PHASE1_N_TEST = 20
PHASE1_N_INJECT_CAL = 84
PHASE1_N_INJECT_TEST = 20

_SEED_PARAGRAPHS = [
    "Self-attention compares a query vector with stored keys and mixes the matching values. "
    "Scores are divided by the square root of the head dimension before softmax so that "
    "large inner products do not saturate the distribution. A causal mask sets future "
    "positions to negative infinity so that those scores never enter the probability vector.",
    "Rivers collect rainfall from hills, carry sediment through valleys, and empty into seas. "
    "Seasonal snowmelt can raise water levels for weeks. Cities that pave too much land "
    "increase runoff and reduce groundwater recharge, which later makes summer droughts worse.",
    "A public library is a shared room for reading, study, and quiet work. Story hour in the "
    "morning is noisy on purpose. In the afternoon students occupy tables with exams in mind. "
    "Rules about silence exist so that many different tasks can happen in one building.",
    "Bread dough becomes strong when gluten strands align during kneading. Yeast produces gas "
    "that stretches those strands. If the dough over-proofs, the structure collapses in the oven. "
    "Cooling after baking lets steam finish the crumb, so slicing too early makes the loaf gummy.",
    "Photosynthesis splits water and captures light energy, then the Calvin cycle fixes carbon "
    "dioxide into sugars. Forests store carbon in wood, but planting trees cannot cancel "
    "unlimited fossil emissions. Teachers often ask students to separate the light reactions "
    "from the sugar-building steps.",
    "Trains move many people along a fixed path. Buses fill gaps where rails do not go. "
    "Bicycles are cheap over short distances if the weather is mild and the roads are safe. "
    "A student living eight kilometers from campus may mix modes during a single week.",
    "Numerical errors in pre-softmax scores can be tiny and still change the greedy token when "
    "two leading scores are close. A large error on a score that softmax already treats as "
    "negligible may leave the output unchanged. Reliability work therefore cares about both "
    "magnitude and location of the error.",
    "Climate is not the same as weather. Weather is the state of the atmosphere this afternoon. "
    "Climate is the distribution of those states over decades. A warmer climate can still have "
    "cold days, but the frequency of extremes can shift.",
]


def _article(i: int) -> str:
    base = _SEED_PARAGRAPHS[i % len(_SEED_PARAGRAPHS)]
    return (
        f"Article {i:04d}. Section one. {base} "
        f"Section two. For record {i}, the running index is {i}, the remainder class is "
        f"{i % 17}, and the paired example is article {(i * 3 + 5) % 200}. "
        f"Section three. The same facts are restated with the numbers {i}, {i + 1}, and "
        f"{i + 2} so that token sequences stay unique while remaining grammatical. "
        f"Section four. Closing sentence for article {i:04d}: the paragraph is complete."
    )


def corpus_texts(n_articles: int = 400) -> list[str]:
    return [_article(i) for i in range(n_articles)]


def data_hash_ids(sequences: list[list[int]]) -> str:
    h = hashlib.sha256()
    for seq in sequences:
        h.update(bytes(str(seq[:8]), "utf-8"))
        h.update(b"|")
        h.update(str(len(seq)).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def pack_sequences(
    tokenizer,
    n_sequences: int,
    seq_len: int = SEQ_LEN,
    n_articles: int = 800,
) -> list[dict]:
    """Return sequences of seq_len prompt tokens plus 1 target token."""
    ids: list[int] = []
    eos = tokenizer.eos_token_id
    for text in corpus_texts(n_articles):
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if eos is not None:
            ids.append(int(eos))
    need = seq_len + 1
    out: list[dict] = []
    start = 0
    while len(out) < n_sequences and start + need <= len(ids):
        chunk = ids[start:start + need]
        out.append({
            "id": f"seq_{len(out):04d}",
            "token_ids": chunk,
            "prompt_len": seq_len,
        })
        start += seq_len
    if len(out) < n_sequences:
        raise RuntimeError(
            f"corpus too short: built {len(out)} sequences, need {n_sequences}"
        )
    return out


def assign_split(sequences: list[dict], test_frac: float = 0.2) -> list[dict]:
    n_test = max(1, int(round(len(sequences) * test_frac)))
    n_cal = len(sequences) - n_test
    for i, seq in enumerate(sequences):
        seq["split"] = "calibration" if i < n_cal else "test"
    return sequences


def load_lm_sequences(
    model_path: str | Path,
    profile: Literal["smoke", "phase1"],
    seq_len: int = SEQ_LEN,
) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True)
    if profile == "smoke":
        n_cal, n_test = 16, 4
    else:
        n_cal, n_test = PHASE1_N_CAL, PHASE1_N_TEST
    packed = pack_sequences(tokenizer, n_cal + n_test, seq_len=seq_len)
    packed = assign_split(packed, test_frac=n_test / (n_cal + n_test))
    meta = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "tokenizer_name": getattr(tokenizer, "name_or_path", str(model_path)),
        "preprocessing": f"pack_nonoverlap_{seq_len}_plus_1_target_no_specials",
        "seq_len": seq_len,
        "short_ctx": SHORT_CTX,
        "long_ctx": LONG_CTX,
        "n_sequences": len(packed),
        "split_counts": {
            "calibration": sum(s["split"] == "calibration" for s in packed),
            "test": sum(s["split"] == "test" for s in packed),
        },
        "data_hash": data_hash_ids([s["token_ids"] for s in packed]),
        "phase1_inject_cal": PHASE1_N_INJECT_CAL,
        "phase1_inject_test": PHASE1_N_INJECT_TEST,
    }
    return packed, meta


def subset_for_injection(sequences: list[dict], profile: str, split: str) -> list[dict]:
    """Fault-injection subset. Census/clean-token sequences stay in the full split."""
    if profile != "phase1":
        return list(sequences)
    n = PHASE1_N_INJECT_TEST if split == "test" else PHASE1_N_INJECT_CAL
    return list(sequences[:n])
