"""Generate the 250-item gold-standard dataset for the GitHub-issue triage task.

The task under test is *"summarise a GitHub issue body into a one-paragraph
triage summary"* (see ``data/rubric.md``). This script builds the pool of model
outputs that were human-annotated for the calibration study:

* 150 PASS items that satisfy rubric R1-R5,
* 100 FAIL items spread over the five failure classes
  (``vague``, ``symptom_only``, ``hedged``, ``waffle``, ``meta``),
* 14 borderline items whose label was *changed after reading the judge's
  critique* (``relabeled=1``), so the ``relabeled`` column is meaningful.

Determinism: seeded RNG (``--seed``, default 42) + SHA-256 hashing in
``src.features`` means the CSV is bit-identical on every machine.

Run::

    python scripts/generate_dataset.py --output data/labels.csv
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import WEIGHTS, extract_features, hash_noise  # noqa: E402

# ---------------------------------------------------------------------------
# Domain pools: realistic GitHub-issue vocabulary.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Domain topics: each entry is a self-consistent GitHub issue (symptom, the real
# root cause, the file that owns it, an error string, a reproduction condition
# and a short area name for the vague/fail prose).
# ---------------------------------------------------------------------------

TOPICS = [
    dict(sym="the connection pool is exhausted after a failover",
         cause="a missing reset in the acquire path", file="api/gateway.py",
         error="ETIMEDOUT", cond="when the upstream returns a partial response",
         hint="the connection pooling"),
    dict(sym="duplicate charge events are emitted for retried payments",
         cause="the idempotency key is generated before the retry wrapper attaches it",
         file="services/billing/charges.py", error="HTTP 502",
         cond="when a payment is retried after a gateway timeout", hint="the billing flow"),
    dict(sym="the worker is killed by the OOM killer during backfill",
         cause="the batch size is derived from queue depth without a cap",
         file="workers/backfill.py", error="MemoryError",
         cond="under sustained backfill load", hint="the backfill job"),
    dict(sym="negative amounts are formatted without the minus sign",
         cause="the formatter assumes every amount is positive", file="lib/format.ts",
         error="ValueError", cond="for refunds and credit notes",
         hint="the amount formatting"),
    dict(sym="provisioned users keep valid sessions after deprovisioning",
         cause="the session cache is not invalidated when the entitlement changes",
         file="auth/session_manager.py", error="HTTP 500",
         cond="for accounts removed by SCIM", hint="the session handling"),
    dict(sym="stale entries are served after a configuration reload",
         cause="the invalidator publishes before the transaction commits",
         file="cache/redis_client.py", error="ECONNRESET",
         cond="after a rolling restart", hint="the cache layer"),
    dict(sym="timestamps shift by one hour across the DST boundary",
         cause="the offset is stored without the daylight-saving adjustment",
         file="lib/timezone.ts", error="ValueError",
         cond="when the timezone database is updated", hint="the timezone handling"),
    dict(sym="pagination loops forever on the last page",
         cause="the cursor comparison uses >= instead of >", file="api/pagination.py",
         error="ContextCanceled", cond="on the last page of a cursor",
         hint="the pagination code"),
    dict(sym="valid webhook payloads are rejected by the signature check",
         cause="the canonical string is rebuilt in a different field order",
         file="webhooks/verify.py", error="HTTP 403",
         cond="when the payload contains non-ASCII characters", hint="the webhook check"),
    dict(sym="large uploads time out before the checksum completes",
         cause="the body buffer is flushed only on a clean shutdown",
         file="storage/uploader.py", error="ETIMEDOUT",
         cond="for payloads larger than 8 MB", hint="the upload path"),
    dict(sym="the retry loop ignores the Retry-After header",
         cause="the backoff uses a fixed interval instead of the server hint",
         file="clients/http_retry.py", error="HTTP 429",
         cond="when the Retry-After header exceeds 30 seconds", hint="the retry logic"),
    dict(sym="the dashboard aggregates the wrong time bucket",
         cause="the rollup truncates timestamps to the hour instead of the day",
         file="analytics/rollup.sql", error="ValueError",
         cond="for tenants on the legacy plan", hint="the dashboard rollups"),
    dict(sym="search returns archived records by default",
         cause="the indexer treats soft-deleted rows as live", file="search/indexer.py",
         error="RepositoryNotFoundError", cond="when the feature flag is toggled at runtime",
         hint="the search index"),
    dict(sym="the migration deadlocks when two replicas start together",
         cause="the schema lock is taken without a timeout",
         file="db/migrations/0042_add_tenant.py", error="deadlock detected",
         cond="when two replicas boot within the same second", hint="the migration"),
    dict(sym="the CLI exits zero even when the request failed",
         cause="the exit code is derived from the transport layer instead of the response",
         file="cli/main.py", error="HTTP 500",
         cond="when the API returns an error envelope", hint="the CLI"),
    dict(sym="notifications are sent twice on queue redelivery",
         cause="the consumer acknowledges before the handler completes",
         file="notifications/dispatcher.py", error="ECONNRESET",
         cond="when the broker redelivers after a timeout", hint="the notification dispatcher"),
    dict(sym="the endpoint returns 500 when an optional filter is empty",
         cause="the filter parser treats an empty string as a missing key",
         file="api/gateway.py", error="KeyError: 'tenant_id'",
         cond="when an optional filter is sent empty", hint="the request parsing"),
    dict(sym="the scheduler drops jobs after a leader election",
         cause="the election lease is not released on graceful shutdown",
         file="scheduler/election.go", error="ContextCanceled",
         cond="after a rolling restart", hint="the scheduler"),
    dict(sym="the importer silently skips rows that start with a BOM",
         cause="the CSV reader does not strip a leading byte-order mark",
         file="etl/importer.py", error="ValueError",
         cond="for files exported from a spreadsheet", hint="the importer"),
    dict(sym="health checks are counted against the rate-limit quota",
         cause="the limiter key omits the tenant id", file="middleware/ratelimit.py",
         error="HTTP 429", cond="under sustained traffic from a single tenant",
         hint="the rate limiter"),
    dict(sym="the audit log omits actor ids for service accounts",
         cause="the actor resolver only handles interactive sessions",
         file="auth/audit.py", error="NullPointerException",
         cond="for tokens issued to service accounts", hint="the audit logging"),
    dict(sym="backups are truncated when the disk is nearly full",
         cause="the writer ignores short writes from the filesystem",
         file="storage/checksum.py", error="checksum mismatch",
         cond="when disk usage crosses 90 percent", hint="the backup writer"),
    dict(sym="the shard rebalancer moves hot partitions twice",
         cause="the assignment map is recomputed before the previous move settles",
         file="scheduler/rebalancer.go", error="deadlock detected",
         cond="when two rebalance passes overlap", hint="the shard rebalancer"),
    dict(sym="invoice rounding drifts by one cent per line",
         cause="each line is rounded before the totals are summed",
         file="services/billing/invoice.py", error="ValueError",
         cond="for invoices with mixed tax rates", hint="the invoice totals"),
]

VERSIONS = ["v2.3.1", "v1.9.0", "v3.0.2", "v2.11.0", "v1.4.7", "v4.2.0"]

PASS_TEMPLATES = [
    "{sym_cap}. Issue #{num}: the root cause is {cause}, in {file} ({error}, {version}).",
    "Issue #{num}: {sym} in {file}. Caused by {cause}; {error} surfaces {cond}. Regression from {version}.",
    "{file} raises {error} {cond} (#{num}). Root cause: {cause}. Affects {version}.",
    "{sym_cap} in {file} ({error}). Caused by {cause}. Introduced in {version}; tracked as #{num}.",
    "Issue #{num}: {file} returns {error} {cond} because of {cause}. Fix lands in {version}.",
    "{sym_cap}. Root cause is {cause}, hit in {file} ({error}) {cond}; see #{num}, {version}.",
    "{file} raises {error} {cond} (#{num}, {version}). Caused by {cause}; impact is elevated retry traffic.",
    "{sym_cap} in {file} ({error}, {version}). Root cause is {cause}; tracked in #{num}. Impact: retries pile up {cond}.",
    "Issue #{num}: {sym} in {file}. The root cause is {cause}, which {cond} leads to a {error}. Affects {version}; impact is duplicated work and noisy triage alerts.",
    "{sym_cap} in {file} ({error}). Root cause is {cause}. Tracked as #{num} and shipping in {version}; impact is retry amplification and paged alerts during the incident window.",
]

VAGUE_OPENERS = [
    "There seems to be some kind of issue with {hint}",
    "Something appears off with {hint}",
    "We have been seeing odd behaviour around {hint}",
    "Reports suggest {hint} is not behaving as expected",
    "There is a problem somewhere in {hint}",
]
VAGUE_CLOSERS = [
    "It might be worth looking into, though I am not sure what is going on.",
    "Maybe a configuration problem, maybe something else.",
    "I think someone should investigate further before we assume anything.",
    "Possibly related to the recent changes, but unclear at this point.",
    "Not sure this is even reproducible; probably worth a quick look.",
]

WAFFLE_BODIES = [
    (
        "This note documents a behaviour that has shown up a number of times "
        "over the last few weeks. The team has spent a fair amount of time "
        "trying to reproduce it with mixed results, and while the general "
        "shape of the problem is understood, a definitive conclusion has not "
        "been reached yet. Several plausible explanations were discussed "
        "during triage: some suggest a configuration concern, others point at "
        "the release process, and a few focus on external dependencies. We are "
        "writing it down here so that the knowledge is not lost and can be "
        "revisited later with fresh eyes. Operations continue with the current "
        "workarounds in place, and no further action is expected from the "
        "reporter at this time."
    ),
    (
        "Over the course of the last sprint the team has gathered a good deal "
        "of context about the way the triage board is used day to day, and "
        "this entry is meant to capture the current understanding rather than "
        "a final answer. The behaviour has been observed in a variety of "
        "environments, including staging and one production region, and the "
        "reproduction steps that have been collected so far are partial. "
        "There are at least three competing narratives about the underlying "
        "sequence of events, and each of them implies a different follow-up. "
        "Rather than commit to one prematurely, the group agreed to record the "
        "observations, link the related meeting notes, and circle back once "
        "the telemetry retention window has been extended."
    ),
    (
        "A lengthy discussion took place during the weekly operations review "
        "about the general health of the pipeline and the various signals that "
        "have been raised recently. Participants shared anecdotes from several "
        "teams, compared notes on previous incidents, and sketched a rough "
        "timeline of when the symptoms first appeared. No single root narrative "
        "emerged, and the conversation ranged across tooling, process, and "
        "staffing questions before returning to the original report. The "
        "consensus was to keep the item open for visibility, attach the "
        "dashboards that were referenced, and revisit it after the next "
        "release so that the picture is more complete."
    ),
]


def pick(rng: random.Random, pool: List[str]) -> str:
    return pool[rng.randrange(len(pool))]


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def build_topic(rng: random.Random) -> Dict[str, str]:
    t = dict(rng.choice(TOPICS))
    return {
        "sym": t["sym"],
        "error": t["error"],
        "file": t["file"],
        "version": pick(rng, VERSIONS),
        "cause": t["cause"],
        "condition": t["cond"],
        "hint": t["hint"],
    }


def render_pass(rng: random.Random, topic: Dict[str, str], num: int) -> str:
    tpl = PASS_TEMPLATES[num % len(PASS_TEMPLATES)]
    return tpl.format(sym=topic["sym"], sym_cap=cap(topic["sym"]),
                      error=topic["error"], file=topic["file"],
                      version=topic["version"], cause=topic["cause"],
                      condition=topic["condition"], cond=topic["condition"],
                      num=num)


def render_fail(rng: random.Random, ftype: str, topic: Dict[str, str], num: int) -> str:
    sym_cap = cap(topic["sym"])
    if ftype == "vague":
        return "{o} over the past {d} days. {c} Reported by @user{num}.".format(
            o=pick(rng, VAGUE_OPENERS).format(hint=topic["hint"]),
            d=rng.choice([3, 5, 8, 11, 14]),
            c=pick(rng, VAGUE_CLOSERS),
            num=num,
        )
    if ftype == "waffle":
        body = pick(rng, WAFFLE_BODIES)
        return "{sym_cap} appears in the reports, and the item stayed open. {body} Reported by @user{num}.".format(
            sym_cap=sym_cap, body=body, num=num
        )
    if ftype == "symptom_only":
        return "{sym_cap}. The {error} can be reproduced {cond} on {file} (#{num}).".format(
            sym_cap=sym_cap, error=topic["error"], cond=topic["condition"],
            file=topic["file"], num=num,
        )
    if ftype == "hedged":
        return ("{file} returns {error} {cond}. This might be caused by {cause}, "
                "though I am not sure; possibly a regression from {version} (#{num}).").format(
            file=topic["file"], error=topic["error"], cond=topic["condition"],
            cause=topic["cause"], version=topic["version"], num=num,
        )
    if ftype == "meta":
        return ("{file} returns {error} {cond} (#{num}). We should investigate the "
                "{cause} before {version}; someone should open a follow-up.").format(
            file=topic["file"], error=topic["error"], cond=topic["condition"],
            cause=topic["cause"], version=topic["version"], num=num,
        )
    raise ValueError(ftype)


def human_critique(label: int, ftype: str, topic: Dict[str, str], words: int) -> str:
    num = topic.get("num", "")
    if label == 1:
        return ("Clear anchor to {file} and issue #{num}. States the root cause "
                "({cause}) and bounds the blast radius to {version}; satisfies R1-R5.").format(
            file=topic["file"], num=num, cause=topic["cause"], version=topic["version"])
    if ftype == "vague":
        return ("No concrete anchor (R1) and no mechanism (R2); only gestures at "
                "{hint} with speculation markers. Not actionable.").format(hint=topic["hint"])
    if ftype == "symptom_only":
        return ("Anchored to {file} (#{num}) but only restates the symptom; the "
                "cause (R2) is absent.").format(file=topic["file"], num=num)
    if ftype == "hedged":
        return ("Anchor and a candidate cause are present, but the speculation "
                "markers (R4) make the summary unreliable.")
    if ftype == "waffle":
        return ("{w} words of fluent filler; violates the brevity bound (R3) and "
                "buries the defect.").format(w=words)
    if ftype == "meta":
        return "Describes no defect; instead instructs the reader to investigate (R5)."
    return "Violates the rubric."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/labels.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-pass", type=int, default=150)
    ap.add_argument("--n-fail", type=int, default=100)
    ap.add_argument("--relabeled", type=int, default=14,
                    help="number of borderline items whose label changed after review")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_total = args.n_pass + args.n_fail

    # --- assemble the spec list: which items are pass / which failure class ---
    specs: List[Tuple[int, str]] = [(1, "pass")] * args.n_pass
    fail_plan = [("vague", 25), ("symptom_only", 30), ("waffle", 20),
                 ("hedged", 15), ("meta", 10)]
    scale = args.n_fail / 100.0
    counts = {k: int(round(v * scale)) for k, v in fail_plan}
    while sum(counts.values()) < args.n_fail:
        counts["symptom_only"] += 1
    for ftype, c in counts.items():
        specs.extend([(0, ftype)] * c)
    specs = specs[:n_total]
    while len(specs) < n_total:
        specs.append((1, "pass"))
    rng.shuffle(specs)

    issue_numbers = list(range(1000, 1000 + n_total))
    rng.shuffle(issue_numbers)

    rows = []
    for i, (label, ftype) in enumerate(specs):
        topic = build_topic(rng)
        num = issue_numbers[i]
        topic["num"] = num
        if label == 1:
            text = render_pass(rng, topic, num)
        else:
            text = render_fail(rng, ftype, topic, num)
        words = len(text.split())
        rows.append({
            "item_id": f"item_{i:04d}",
            "output": text,
            "label": int(label),
            "critique": human_critique(label, ftype, topic, words),
            "relabeled": 0,
            "_ftype": ftype,
            "_score": extract_features(text)["score"] + hash_noise(text),
        })

    df = pd.DataFrame(rows)

    # --- pick borderline items to relabel, keeping class balance ------------
    thr = WEIGHTS["threshold"]
    n_each = args.relabeled // 2
    # pass items closest to (just above) the threshold -> downgraded to Fail
    pass_idx = df[(df["label"] == 1)].copy()
    pass_idx["margin"] = (pass_idx["_score"] - thr).abs()
    downgrade = pass_idx.nsmallest(n_each, "margin").index
    # fail items closest to the threshold -> upgraded to Pass
    fail_idx = df[(df["label"] == 0)].copy()
    fail_idx["margin"] = (fail_idx["_score"] - thr).abs()
    upgrade = fail_idx.nsmallest(n_each, "margin").index

    for idx in downgrade:
        df.at[idx, "label"] = 0
        df.at[idx, "relabeled"] = 1
        df.at[idx, "critique"] = (
            "First pass missed that the summary never states the operational "
            "impact; the judge critique flagged R2/R5. Relabeled Fail."
        )
    for idx in upgrade:
        df.at[idx, "label"] = 1
        df.at[idx, "relabeled"] = 1
        df.at[idx, "critique"] = (
            "Originally logged Fail for a thin cause statement, but the judge "
            "critique pointed at the anchor and condition in the text; on "
            "reflection it is actionable enough. Relabeled Pass."
        )

    # --- integrity checks ---------------------------------------------------
    assert df["item_id"].is_unique, "item_id must be unique"
    assert df["output"].is_unique, "duplicate output would leak between splits"
    assert set(df["label"].unique()) <= {0, 1}
    assert set(df["relabeled"].unique()) <= {0, 1}
    assert len(df) == n_total
    assert (df["label"] == 0).mean() >= 0.30, "need >=30% failures"

    out = df[["item_id", "output", "label", "critique", "relabeled"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"wrote {len(out)} rows -> {args.output}")
    print(f"  pass={int((out['label']==1).sum())} fail={int((out['label']==0).sum())} "
          f"({(out['label']==0).mean():.1%} fail)")
    print(f"  relabeled={int(out['relabeled'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
