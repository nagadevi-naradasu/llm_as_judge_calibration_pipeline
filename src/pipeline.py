"""Shared pipeline helpers (paths, split loading, judge construction).

Factor used by ``evaluate.py``, ``src/bias_probes.py`` and
``scripts/induce_leakage.py`` so all three build the judge from exactly the same
rubric + train few-shot examples.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.judge import LLMJudge, build_system_prompt  # noqa: E402

LABELS_PATH = os.path.join(REPO_ROOT, "data", "labels.csv")
SPLITS_DIR = os.path.join(REPO_ROOT, "data", "splits")
RUBRIC_PATH = os.path.join(REPO_ROOT, "data", "rubric.md")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def ensure_splits() -> None:
    """Regenerate split files if they are missing (deterministic)."""
    if all(os.path.exists(os.path.join(SPLITS_DIR, f"{s}.csv"))
           for s in ("train", "dev", "test")):
        return
    from src.split_data import main as split_main

    old_argv = sys.argv
    try:
        sys.argv = ["split_data.py", "--input", LABELS_PATH, "--output-dir", SPLITS_DIR]
        split_main()
    finally:
        sys.argv = old_argv


def load_split(split: str) -> pd.DataFrame:
    ensure_splits()
    path = os.path.join(SPLITS_DIR, f"{split}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"split file not found: {path}")
    return pd.read_csv(path)


def train_few_shot() -> List[Dict]:
    train = load_split("train")
    return [
        {"output": r["output"], "label": int(r["label"]), "critique": r["critique"]}
        for _, r in train.iterrows()
    ]


def select_few_shot(examples: List[Dict], k: int) -> List[Dict]:
    """Deterministically pick a balanced subset of at most ``k`` few-shots.

    Embedding all train examples makes the prompt huge and slow (especially for
    local models) with little accuracy gain; 4-8 balanced examples suffice.
    ``k <= 0`` means "use every example".
    """
    if k <= 0 or len(examples) <= k:
        return list(examples)
    passes = [e for e in examples if int(e["label"]) == 1]
    fails = [e for e in examples if int(e["label"]) == 0]
    n_pass = min(len(passes), (k + 1) // 2)
    n_fail = min(len(fails), k - n_pass)
    picked = passes[:n_pass] + fails[:n_fail]
    # top up if one class was short
    if len(picked) < k:
        extra = [e for e in examples if e not in picked]
        picked += extra[: k - len(picked)]
    return picked


def build_judge(
    model: str = "gpt-4o-mini",
    mode: str = "auto",
    budget: float = 5.0,
    seed: int = 42,
    extra_few_shot: Optional[List[Dict]] = None,
) -> LLMJudge:
    """Build a judge whose prompt is the rubric + train few-shot (+ extras)."""
    rubric = read_text(RUBRIC_PATH)
    k = int(os.getenv("JUDGE_FEWSHOT", "0"))
    few_shot = select_few_shot(train_few_shot(), k)
    if extra_few_shot:
        few_shot = few_shot + list(extra_few_shot)
    system_prompt = build_system_prompt(rubric, few_shot)
    return LLMJudge(
        model_name=model,
        system_prompt=system_prompt,
        few_shot_examples=few_shot,
        budget_usd=budget,
        mode=mode,
        seed=seed,
    )
