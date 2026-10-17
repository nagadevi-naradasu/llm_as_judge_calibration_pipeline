"""Deterministic surface-feature extraction for GitHub-issue summaries.

These features encode the rubric in ``data/rubric.md`` (R1-R5) as regex/logic
checks over the *summary text alone*. They are shared by:

* the offline reference judge (:mod:`src.judge`, ``JUDGE_MODE=mock``), which
  turns them into a scored, noisy verdict, and
* the dataset generator (``scripts/generate_dataset.py``), which uses them to
  place items near/away from the decision threshold so the calibration metrics
  are realistic rather than degenerate.

Every function here is pure and deterministic: identical text always yields an
identical result, so the whole pipeline is bit-reproducible.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict

# --- Regexes ---------------------------------------------------------------

# R1: a concrete, verifiable anchor copied from the issue.
ANCHOR_RE = re.compile(
    r"(#\d+)"                                   # issue / PR reference
    r"|([\w./-]+\.(?:py|ts|tsx|go|sql|js|yaml|yml|json))"   # source file path
    r"|(\bv\d+\.\d+\.\d+\b)"                    # release tag
    r"|(\b[A-Z][A-Z0-9_]{4,}\b)"                # error string e.g. ETIMEDOUT
    r"|(\b\w+(?:Error|Exception)\b)"            # e.g. NullPointerException
    r"|(\bHTTP\s?\d{3}\b)"                      # HTTP status
)

# R2: an explicit mechanism / root-cause statement.
MECHANISM_MARKERS = (
    "caused by",
    "cause is",
    "root cause",
    "because",
    "due to",
    "triggered by",
    "as a result of",
)

# R4: speculation / hedging.
HEDGE_MARKERS = (
    "might",
    "possibly",
    "i think",
    "i'm not sure",
    "seems like",
    "seems",
    "probably",
    "maybe",
    "some kind of",
    "sort of",
    "perhaps",
    "unclear",
    "not sure",
)

# R5: meta-commentary that tells the reader what to do instead of describing.
META_MARKERS = (
    "should investigate",
    "needs more info",
    "someone should",
    "needs investigation",
    "todo",
    "look into",
    "follow up",
    "dig into",
    "we should",
    "needs to be looked",
)

BREVITY_MAX_WORDS = 70

# Weight applied to the ORIGINAL investigation's error-detection lambda. The
# final reference scorer's coefficients are below; see README for the rationale.
WEIGHTS = {
    "base": 0.20,
    "anchor": 0.30,
    "mechanism": 0.20,
    "brevity": 0.12,
    "hedge_penalty": 0.30,
    "meta_penalty": 0.30,
    "verbosity_bonus": 0.30,
    "threshold": 0.78,
}

# Noise half-width (uniform, symmetric) injected by the reference judge to model
# the irreducible disagreement of any text-only grader with the human expert.
NOISE_HALF_WIDTH = 0.18


def word_count(text: str) -> int:
    return len(text.split())


def count_markers(text: str, markers) -> int:
    low = text.lower()
    return sum(low.count(m) for m in markers)


def extract_features(text: str) -> Dict[str, object]:
    """Return the rubric features for one summary (all deterministic)."""
    words = word_count(text)
    hedge = count_markers(text, HEDGE_MARKERS)
    meta = count_markers(text, META_MARKERS)
    feats = {
        "words": words,
        "anchor": bool(ANCHOR_RE.search(text)),
        "mechanism": any(m in text.lower() for m in MECHANISM_MARKERS),
        "brevity": words <= BREVITY_MAX_WORDS,
        "hedge": hedge,
        "meta": meta,
    }
    feats["score"] = rule_score(feats)
    return feats


def rule_score(feats: Dict[str, object]) -> float:
    """Weighted rubric score in roughly [-0.5, 1.1] before noise."""
    w = WEIGHTS
    s = (
        w["base"]
        + w["anchor"] * float(bool(feats["anchor"]))
        + w["mechanism"] * float(bool(feats["mechanism"]))
        + w["brevity"] * float(bool(feats["brevity"]))
        - w["hedge_penalty"] * min(int(feats["hedge"]), 2)
        - w["meta_penalty"] * min(int(feats["meta"]), 2)
        + w["verbosity_bonus"] * min(int(feats["words"]) / 90.0, 1.0)
    )
    return round(s, 6)


def hash_noise(text: str, salt: str = "") -> float:
    """Deterministic uniform noise in [-NOISE_HALF_WIDTH, +NOISE_HALF_WIDTH]."""
    digest = hashlib.sha256((salt + "\x00" + text).encode("utf-8")).hexdigest()
    unit = int(digest[:16], 16) / float(16 ** 16)  # [0, 1)
    return (unit * 2.0 - 1.0) * NOISE_HALF_WIDTH
