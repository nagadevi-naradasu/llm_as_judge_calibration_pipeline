#!/usr/bin/env python
"""Calibrated LLM-as-a-judge pipeline - main CLI.

Usage
-----
    python evaluate.py --split test            # frozen-test evaluation
    python evaluate.py --split dev             # iteration evaluation
    python evaluate.py --test-judge "..."      # print the judge JSON contract

Artifacts
---------
* ``results/metrics.csv``   - TPR/TNR/agreement/kappa with bootstrap CIs
* ``results/judge_test.csv``- per-item judge predictions for the frozen test run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from src.pipeline import RESULTS_DIR, build_judge, load_split  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402

METRICS_COLUMNS = [
    "split", "iteration", "model", "judge_mode", "n", "n_fail",
    "tpr_mean", "tpr_ci_low", "tpr_ci_high",
    "tnr_mean", "tnr_ci_low", "tnr_ci_high",
    "agreement", "agreement_ci_low", "agreement_ci_high",
    "kappa_mean", "kappa_ci_low", "kappa_ci_high",
    "tp_fail_caught", "fn_fail_missed", "tn_pass_kept", "fp_pass_called_fail",
]

JUDGE_TEST_COLUMNS = ["item_id", "human_label", "judge_label", "judge_critique"]


def run_judge_over_split(judge, df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for i, row in df.iterrows():
        result = judge.evaluate(row["output"])
        records.append({
            "item_id": row["item_id"],
            "human_label": int(row["label"]),
            "judge_label": 1 if result["pass"] else 0,
            "judge_critique": result["critique"],
        })
        if (i + 1) % 25 == 0:
            print(f"  judged {i + 1}/{len(df)} (spend=${judge.spend:.4f})", file=sys.stderr)
    return pd.DataFrame(records, columns=JUDGE_TEST_COLUMNS)


def upsert_metrics(row: dict, path: str) -> pd.DataFrame:
    """Replace any existing row for the same split, keep the others, stable order."""
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[existing["split"] != row["split"]]
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        combined = pd.DataFrame([row])
    combined = combined.reindex(columns=METRICS_COLUMNS)
    order = {"dev": 0, "test": 1}
    combined["_o"] = combined["split"].map(order).fillna(2)
    combined = combined.sort_values(by=["_o", "iteration"]).drop(columns="_o")
    combined = combined.reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def cmd_test_judge(judge, text: str) -> int:
    result = judge.evaluate(text)
    # The verifier parses stdout as JSON: emit exactly the contract, nothing else.
    print(json.dumps({"critique": result["critique"], "pass": bool(result["pass"])}, indent=2))
    return 0


def cmd_evaluate(args) -> int:
    df = load_split(args.split)
    judge = build_judge(model=args.model, mode=args.mode, budget=args.budget, seed=args.seed)
    print(f"[{args.split}] judging {len(df)} items with {args.model} "
          f"(mode={judge.mode})", file=sys.stderr)

    preds = run_judge_over_split(judge, df)
    metrics = compute_metrics(
        preds["human_label"].to_numpy(),
        preds["judge_label"].to_numpy(),
        n_iterations=args.iterations,
        seed=args.seed,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    row = {"split": args.split, "iteration": args.iteration,
           "model": args.model, "judge_mode": judge.mode, **metrics}
    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    upsert_metrics(row, metrics_path)

    if args.split == "test":
        preds.to_csv(os.path.join(RESULTS_DIR, "judge_test.csv"), index=False)

    print(f"[{args.split}] TPR={metrics['tpr_mean']:.3f} "
          f"[{metrics['tpr_ci_low']:.3f},{metrics['tpr_ci_high']:.3f}]  "
          f"TNR={metrics['tnr_mean']:.3f} "
          f"[{metrics['tnr_ci_low']:.3f},{metrics['tnr_ci_high']:.3f}]  "
          f"agree={metrics['agreement']:.3f}  kappa={metrics['kappa_mean']:.3f}  "
          f"(always-pass agreement would be {(preds['human_label'] == 1).mean():.3f})",
          file=sys.stderr)
    print(f"[{args.split}] wrote {metrics_path}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Calibrated LLM-as-a-judge evaluation pipeline.",
    )
    ap.add_argument("--split", choices=["dev", "test"], default="test",
                    help="which frozen split to evaluate (dev=40%%, test=45%%)")
    ap.add_argument("--test-judge", metavar="TEXT",
                    help="grade TEXT once and print the judge JSON contract to stdout")
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    ap.add_argument("--mode", default=os.getenv("JUDGE_MODE", "auto"),
                    choices=["auto", "api", "mock"])
    ap.add_argument("--iterations", type=int,
                    default=int(os.getenv("BOOTSTRAP_ITERATIONS", "1000")),
                    help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    ap.add_argument("--iteration", type=int, default=1,
                    help="prompt iteration index recorded in metrics.csv")
    ap.add_argument("--budget", type=float,
                    default=float(os.getenv("JUDGE_BUDGET_USD", "5.0")))
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.test_judge is not None:
        judge = build_judge(model=args.model, mode=args.mode,
                            budget=args.budget, seed=args.seed)
        return cmd_test_judge(judge, args.test_judge)

    return cmd_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
