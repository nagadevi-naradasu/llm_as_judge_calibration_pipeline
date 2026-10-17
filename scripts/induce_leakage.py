#!/usr/bin/env python
"""Induced-failure experiment: prove that dev-data leakage inflates metrics.

The classic accident: an engineer is debugging the judge on the dev split, so
they paste the *troublesome dev items* (with their labels) into the few-shot
prompt to see what the model does - and then forgets to remove them. The judge
is now graded on items whose answers are literally in its prompt, so dev metrics
look great while test performance is unchanged.

This script reproduces that mistake in a controlled way:

1. Run the clean judge over the dev split -> ``baseline_dev_kappa``.
2. Take the 10 dev items the clean judge got **wrong** and inject them into the
   few-shot prompt (``leaked_items_count = 10``).
3. Re-run dev -> ``contaminated_dev_kappa``.
4. Emit ``results/induced_failure.json`` and assert the inflation is real.

Run::

    python scripts/induce_leakage.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.pipeline import RESULTS_DIR, build_judge, load_split  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.features import WEIGHTS, extract_features, hash_noise  # noqa: E402


def _predict(judge, outputs) -> list:
    return [1 if judge.evaluate(t)["pass"] else 0 for t in outputs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leak", type=int, default=10, dest="n_leak")
    ap.add_argument("--output", default=os.path.join(RESULTS_DIR, "induced_failure.json"))
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    ap.add_argument("--mode", default=os.getenv("JUDGE_MODE", "auto"))
    ap.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    ap.add_argument("--budget", type=float,
                    default=float(os.getenv("JUDGE_BUDGET_USD", "5.0")))
    args = ap.parse_args(argv)

    dev = load_split("dev").sort_values("item_id").reset_index(drop=True)
    y_true = dev["label"].astype(int).tolist()

    # 1) clean run ---------------------------------------------------------
    clean = build_judge(model=args.model, mode=args.mode, budget=args.budget, seed=args.seed)
    clean_pred = _predict(clean, dev["output"].tolist())
    baseline_kappa = float(cohen_kappa_score(y_true, clean_pred))
    baseline = compute_metrics(y_true, clean_pred, n_iterations=1000, seed=args.seed)

    # 2) choose the items to leak: the dev items closest to the judge's decision
    #    boundary. This is the realistic "paste the borderline cases I'm confused
    #    about into the prompt while debugging" accident, and it mixes correct and
    #    incorrect clean predictions so the result is not degenerate (kappa != 1).
    margins = [
        abs(extract_features(t)["score"] + hash_noise(t) - WEIGHTS["threshold"])
        for t in dev["output"]
    ]
    leaked = dev.assign(_margin=margins).sort_values("_margin").head(args.n_leak)
    leaked_examples = [
        {"output": r["output"], "label": int(r["label"]), "critique": r["critique"]}
        for _, r in leaked.iterrows()
    ]

    # 3) contaminated run ---------------------------------------------------
    dirty = build_judge(model=args.model, mode=args.mode, budget=args.budget,
                        seed=args.seed, extra_few_shot=leaked_examples)
    dirty_pred = _predict(dirty, dev["output"].tolist())
    contaminated_kappa = float(cohen_kappa_score(y_true, dirty_pred))
    contaminated = compute_metrics(y_true, dirty_pred, n_iterations=1000, seed=args.seed)

    # 4) artifact -----------------------------------------------------------
    result = {
        "baseline_dev_kappa": round(baseline_kappa, 4),
        "contaminated_dev_kappa": round(contaminated_kappa, 4),
        "leaked_items_count": int(args.n_leak),
        "baseline_dev_agreement": round(baseline["agreement"], 4),
        "contaminated_dev_agreement": round(contaminated["agreement"], 4),
        "baseline_dev_tpr": round(baseline["tpr_mean"], 4),
        "contaminated_dev_tpr": round(contaminated["tpr_mean"], 4),
        "leaked_item_ids": leaked["item_id"].tolist(),
        "seed": args.seed,
        "model": args.model,
        "judge_mode": clean.mode,
        "explanation": (
            "Induced failure: the 10 dev items closest to the judge's decision "
            "boundary (the borderline cases an engineer pastes into the prompt to "
            "debug) were added to the few-shot examples *with their labels*. The "
            "judge now recalls those labels from the prompt, so dev kappa rises "
            "even though the prompt generalises no better. This number is invalid "
            "for release gating; test is the only untouched split."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps({k: result[k] for k in (
        "baseline_dev_kappa", "contaminated_dev_kappa", "leaked_items_count")}, indent=2))
    print(f"wrote {args.output}", file=sys.stderr)

    assert result["contaminated_dev_kappa"] > result["baseline_dev_kappa"], (
        "leakage did not inflate dev kappa; the experiment is not demonstrating "
        "the failure mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
