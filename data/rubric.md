# Rubric - GitHub Issue Summary Judge (v1.0)

## Task under evaluation

The model under test receives a raw GitHub issue body (title, environment,
reproduction steps, stack trace, discussion) and must produce a **single
short summary paragraph** for a triage board. The summary is consumed by an
on-call engineer who has not read the original issue.

## What counts as PASS

A summary is **PASS** only if **all** of the following hold:

| # | Criterion | Precision required |
|---|-----------|--------------------|
| R1 | **Concrete anchor** - cites at least one verifiable identifier copied from the issue: an issue/PR reference (`#1234`), a source file path (`api/gateway.py`), an error string (`ETIMEDOUT`, `NullPointerException`), or a release/version tag (`v2.3.1`). | one anchor is enough |
| R2 | **Mechanism or root cause** - states what actually goes wrong (a cause/effect statement: *because*, *caused by*, *due to*, *when X then Y*), not only the visible symptom. | one causal clause |
| R3 | **Brevity** - 20-70 words. Triage summaries are skimmed, not read. | hard bound |
| R4 | **No speculation** - no hedging or invented facts. Forbidden: *might*, *possibly*, *I think*, *seems like*, *probably*, *not sure*, *some kind of*, *maybe*. | zero occurrences |
| R5 | **No meta-commentary** - the summary describes the defect, it does not instruct the reader (*we should investigate*, *needs more info*, *someone should look*, *TODO*). | zero occurrences |

A summary that satisfies R1-R5 is actionable: the reader can jump straight to
the file, reproduce the failure, and understand the trigger.

## What counts as FAIL

It is **FAIL** if *any* criterion above is violated. The failure taxonomy used
during annotation (one primary class per item, recorded in the critique):

- `vague` - no anchor and no mechanism ("there is a problem with retries").
- `symptom_only` - anchored but only restates the symptom, no cause.
- `hedged` - otherwise complete but full of speculation / uncertainty markers.
- `waffle` - >70 words of fluent filler; dilutes the signal, hides the defect.
- `meta` - tells the reader to investigate instead of describing the defect.

## Scoring rules for the judge

1. The judge writes a 1-3 sentence **critique that cites the rubric IDs**
   (`R1`...`R5`) it is relying on, **before** emitting any verdict
   (chain-of-thought first, boolean second).
2. The verdict is a strict boolean. `pass=true` requires all of R1-R5 to hold.
3. Positive class for error detection is **FAIL (label 0)**: TPR below answers
   *"what share of real failures did the judge catch?"*
4. The judge never sees items from the frozen test split during prompt
   iteration (see `README.md` and `data/splits/`).

## Versioning

- `v0` - zero-shot, no rubric IDs. Dev kappa 0.31.
- `v1` - this rubric + 3 few-shot train examples. Frozen with git tag
  `judge-prompt-v1` before the single test run (see `results/metrics.csv`).
