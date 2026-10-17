"""Bias probes: position bias (pairwise) and verbosity bias (padding).

Position probe
    Build N pairs (one human-PASS, one human-FAIL) and ask the judge which is
    better **in both orders**. If the preferred *item* changes when the order is
    swapped, the verdict tracked position rather than content.
    ``position_flip_rate = mean(chosen_item(AB) != chosen_item(BA))``.

Verbosity probe
    Take N pass and N fail summaries and append 2-3 sentences of fluent but
    irrelevant padding (no new anchors, no cause, no hedging). Re-judge and
    report the pass rate per class before/after. A robust judge is flat; a
    verbosity-biased judge rewards the longer text.

Run::

    python src/bias_probes.py --output results/bias.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.judge import build_pairwise_system_prompt  # noqa: E402
from src.pipeline import RESULTS_DIR, build_judge, read_text, RUBRIC_PATH  # noqa: E402

# Fluent, completely irrelevant padding: no anchors, no cause, no hedging, no
# meta-commentary, so a rubric-faithful judge should not change its verdict.
PADDINGS = [
    "The board is refreshed each morning and these notes are archived for the weekly sync. The reporting workflow has matured a lot this quarter and keeps improving.",
    "This entry was reviewed during the regular triage rotation and the team agreed to keep it visible for now. The surrounding process documentation was also updated recently.",
    "For context, the on-call rota changed this month and the handover checklist was revised. The wider engineering org has been standardising how these write-ups are stored.",
    "A short note on process: the dashboard now refreshes every fifteen minutes and the weekly digest links to the same source. Nobody has raised concerns about the format so far.",
]

BIAS_COLUMNS = [
    "probe_type", "class", "sample_size",
    "position_flip_rate", "pass_rate_before", "pass_rate_after",
]


def _load_labels() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(REPO_ROOT, "data", "labels.csv"))
    df["label"] = df["label"].astype(int)
    return df


def _pick(df: pd.DataFrame, label: int, n: int, seed: int) -> pd.DataFrame:
    pool = df[df["label"] == label].sort_values("item_id").reset_index(drop=True)
    if len(pool) < n:
        raise SystemExit(f"need {n} label={label} items, have {len(pool)}")
    return pool.sample(n=n, random_state=seed).sort_values("item_id").reset_index(drop=True)


def run_position_probe(judge, df: pd.DataFrame, n_pairs: int, seed: int) -> dict:
    passes = _pick(df, 1, n_pairs, seed)
    fails = _pick(df, 0, n_pairs, seed)
    system = build_pairwise_system_prompt(read_text(RUBRIC_PATH))

    flips = 0
    for i in range(n_pairs):
        p, f = passes.iloc[i]["output"], fails.iloc[i]["output"]
        ab = judge.compare(p, f, system)          # A=pass, B=fail
        chosen_ab = "pass" if ab["preferred"] == "A" else "fail"
        ba = judge.compare(f, p, system)          # A=fail, B=pass
        chosen_ba = "fail" if ba["preferred"] == "A" else "pass"
        if chosen_ab != chosen_ba:
            flips += 1
    return {
        "probe_type": "position",
        "class": "all",
        "sample_size": n_pairs,
        "position_flip_rate": round(flips / n_pairs, 4),
        "pass_rate_before": "",
        "pass_rate_after": "",
    }


def run_verbosity_probe(judge, df: pd.DataFrame, n_pass: int, n_fail: int, seed: int) -> list:
    passes = _pick(df, 1, n_pass, seed)
    fails = _pick(df, 0, n_fail, seed + 1)

    def padded(text: str, idx: int) -> str:
        return text.rstrip() + " " + PADDINGS[idx % len(PADDINGS)]

    rows = []
    per_class = {"pass": passes, "fail": fails}
    overall_before, overall_after = [], []
    for cls, frame in per_class.items():
        before = [1 if judge.evaluate(t)["pass"] else 0 for t in frame["output"]]
        after = [1 if judge.evaluate(padded(t, i))["pass"] else 0
                 for i, t in enumerate(frame["output"])]
        overall_before += before
        overall_after += after
        rows.append({
            "probe_type": "verbosity",
            "class": cls,
            "sample_size": len(frame),
            "position_flip_rate": "",
            "pass_rate_before": round(sum(before) / len(before), 4),
            "pass_rate_after": round(sum(after) / len(after), 4),
        })
    rows.append({
        "probe_type": "verbosity",
        "class": "all",
        "sample_size": len(overall_before),
        "position_flip_rate": "",
        "pass_rate_before": round(sum(overall_before) / len(overall_before), 4),
        "pass_rate_after": round(sum(overall_after) / len(overall_after), 4),
    })
    # stable order: all, pass, fail
    order = {"all": 0, "pass": 1, "fail": 2}
    rows.sort(key=lambda r: order[r["class"]])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=os.path.join(RESULTS_DIR, "bias.csv"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--n-pass", type=int, default=50)
    ap.add_argument("--n-fail", type=int, default=50)
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    ap.add_argument("--mode", default=os.getenv("JUDGE_MODE", "auto"))
    ap.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    ap.add_argument("--budget", type=float,
                    default=float(os.getenv("JUDGE_BUDGET_USD", "5.0")))
    args = ap.parse_args(argv)

    df = _load_labels()
    judge = build_judge(model=args.model, mode=args.mode, budget=args.budget, seed=args.seed)

    print(f"position probe: {args.pairs} pairs x2 orders ...", file=sys.stderr)
    pos_row = run_position_probe(judge, df, args.pairs, args.seed)
    print(f"verbosity probe: {args.n_pass} pass + {args.n_fail} fail ...", file=sys.stderr)
    verb_rows = run_verbosity_probe(judge, df, args.n_pass, args.n_fail, args.seed)

    out = pd.DataFrame([pos_row] + verb_rows, columns=BIAS_COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}\n{out.to_string(index=False)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
