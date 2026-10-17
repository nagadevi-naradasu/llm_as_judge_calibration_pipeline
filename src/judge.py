"""LLM judge: build a rubric prompt, call the model, return strict JSON.

Design goals
------------
* **Structured output.** ``evaluate`` always returns
  ``{"critique": str, "pass": bool}``. The system prompt demands JSON and the
  hosted path asks the provider for ``response_format={"type": "json_object"}``;
  a regex fallback extracts the first ``{...}`` block if the model misbehaves.
* **Chain-of-thought first.** The prompt forces the model to write the critique
  *before* the verdict, because ordering the fields this way measurably improves
  classifier accuracy on the dev split (see ``results/metrics.csv``).
* **Budget safety.** ``self.spend`` accumulates real (hosted) or nominal (mock)
  dollars and ``JudgeBudgetExceeded`` is raised once the ceiling is crossed.
* **Two backends, one interface.** ``JUDGE_MODE=api`` uses an OpenAI-compatible
  endpoint; ``JUDGE_MODE=mock`` uses a deterministic, offline reference judge so
  the whole pipeline is reproducible with no key and no network. ``auto`` picks
  the API when ``OPENAI_API_KEY`` is present, else mock.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional

try:  # optional at import time so mock mode works without the dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from src.features import HEDGE_MARKERS, META_MARKERS, WEIGHTS, extract_features, hash_noise

# USD per 1M tokens (gpt-4o-mini class). Override via env if you switch models.
INPUT_PRICE_PER_MTOK = float(os.getenv("JUDGE_INPUT_PRICE_PER_MTOK", "0.15"))
OUTPUT_PRICE_PER_MTOK = float(os.getenv("JUDGE_OUTPUT_PRICE_PER_MTOK", "0.60"))
MOCK_COST_PER_CALL = float(os.getenv("JUDGE_MOCK_COST_PER_CALL", "0.0"))


class JudgeBudgetExceeded(RuntimeError):
    """Raised when the running spend exceeds the configured budget ceiling."""


JSON_CONTRACT = (
    "Return strictly a JSON object, no markdown fence, no prose outside it:\n"
    '{"critique": "<1-3 sentences citing the rubric ids you relied on>", '
    '"pass": <true|false>}\n'
    'Write "critique" FIRST, decide "pass" SECOND.'
)


def build_system_prompt(rubric_text: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
    """Assemble the judge system prompt from the rubric + optional few-shots."""
    parts = [
        "You are a strict, consistent evaluator (an LLM-as-a-judge) for a "
        "GitHub-issue triage pipeline. You receive one summary paragraph and "
        "must decide whether it passes the rubric below.",
        "",
        "=== RUBRIC ===",
        rubric_text.strip(),
        "",
        "=== LABEL CONVENTION ===",
        "pass=true means the summary satisfies every rubric rule R1-R5. "
        "pass=false means it violates at least one rule.",
        "",
        "=== OUTPUT CONTRACT ===",
        JSON_CONTRACT,
    ]
    if few_shot_examples:
        parts += ["", "=== FEW-SHOT EXAMPLES (from the TRAIN split only) ==="]
        for ex in few_shot_examples:
            parts.append(
                f"---\nSUMMARY: {ex['output']}\nEXPERT LABEL: "
                f"{'PASS' if int(ex['label']) == 1 else 'FAIL'}\n"
                f"EXPERT CRITIQUE: {ex.get('critique', '')}"
            )
        parts.append(
            "Do not quote the examples back; apply the same standard to new items."
        )
    return "\n".join(parts)


def build_pairwise_system_prompt(rubric_text: str) -> str:
    return "\n".join([
        "You are a strict evaluator comparing two candidate triage summaries.",
        "Decide which summary better satisfies the rubric. Judge CONTENT only; "
        "ignore the ORDER in which the summaries are presented.",
        "",
        "=== RUBRIC ===",
        rubric_text.strip(),
        "",
        "=== OUTPUT CONTRACT ===",
        "Return strictly JSON: "
        '{"critique": "<1-2 sentences>", "preferred": "A" | "B"}',
    ])


class LLMJudge:
    """Judge wrapper around a hosted OpenAI-compatible model or a mock backend."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        system_prompt: str = "",
        few_shot_examples: Optional[List[Dict]] = None,
        budget_usd: float = 5.0,
        mode: str = "auto",
        seed: int = 42,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.few_shot_examples = list(few_shot_examples or [])
        self.budget_usd = float(budget_usd)
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.spend = 0.0
        self.calls = 0
        self.mode = self._resolve_mode(mode)
        # When a hosted call fails (no credits, bad key, network), don't crash
        # the pipeline: warn once and fall back to the deterministic mock so
        # `docker compose up` still exits 0 and produces artifacts. Set
        # JUDGE_API_FALLBACK=0 to fail loudly instead.
        self.api_fallback = os.getenv("JUDGE_API_FALLBACK", "1").lower() not in {"0", "false", "no"}
        self.api_error: Optional[str] = None
        self._client = None

    # -- backend selection ---------------------------------------------------
    @staticmethod
    def _resolve_mode(mode: str) -> str:
        mode = (mode or os.getenv("JUDGE_MODE", "auto")).lower()
        if mode not in {"auto", "api", "mock"}:
            raise ValueError(f"unknown judge mode: {mode}")
        if mode == "auto":
            return "api" if os.getenv("OPENAI_API_KEY") else "mock"
        return mode

    @property
    def client(self):
        if self._client is None:
            if OpenAI is None:  # pragma: no cover
                raise RuntimeError("openai package is not installed")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("JUDGE_MODE=api but OPENAI_API_KEY is not set")
            self._client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                timeout=float(os.getenv("JUDGE_TIMEOUT", "120")),
                max_retries=int(os.getenv("JUDGE_MAX_RETRIES", "1")),
            )
        return self._client

    # -- budget --------------------------------------------------------------
    def _charge(self, cost: float) -> None:
        self.spend += float(cost)
        if self.spend > self.budget_usd:
            raise JudgeBudgetExceeded(
                f"judge spend ${self.spend:.4f} exceeded budget ${self.budget_usd:.2f}"
            )

    # -- snapshot of few-shot examples used (for the leakage experiment) -----
    def few_shot_outputs(self) -> set:
        return {ex["output"] for ex in self.few_shot_examples}

    # -- public API ----------------------------------------------------------
    def _fallback_to_mock(self, exc: Exception) -> None:
        """Record the hosted failure and permanently switch to the mock backend."""
        if not self.api_fallback:
            raise exc
        self.api_error = f"{type(exc).__name__}: {exc}"
        if self.mode != "mock":
            print(
                "[judge] WARNING: hosted judge unavailable "
                f"({self.api_error}); falling back to the deterministic mock "
                "backend for the rest of this run.",
                file=sys.stderr,
            )
        self.mode = "mock"

    def evaluate(self, generated_text: str) -> dict:
        """Grade one summary. Always returns {"critique": str, "pass": bool}."""
        self.calls += 1
        if self.mode == "api":
            try:
                raw = self._call_api(generated_text)
                result = self._parse(raw)
                if result is None:
                    raw = self._call_api(generated_text, force_plain=True)
                    result = self._parse(raw) or {
                        "critique": f"Unparseable judge output; treated as FAIL: {raw[:200]}",
                        "pass": False,
                    }
                return result
            except JudgeBudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                self._fallback_to_mock(exc)
        return self._mock_evaluate(generated_text)

    def compare(self, text_a: str, text_b: str, system_prompt: str) -> dict:
        """Pairwise preference. Returns {"critique": str, "preferred": "A"|"B"}."""
        self.calls += 1
        if self.mode == "api":
            try:
                user = (
                    f"Candidate A:\n{text_a}\n\nCandidate B:\n{text_b}\n\n"
                    "Which is better? Return JSON with critique and preferred."
                )
                raw = self._call_api(user, system_prompt=system_prompt, pairwise=True)
                parsed = self._parse(raw, pairwise=True)
                if parsed is None:
                    return {"critique": raw[:200], "preferred": "A"}
                return parsed
            except JudgeBudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                self._fallback_to_mock(exc)
        return self._mock_compare(text_a, text_b)

    # -- hosted backend ------------------------------------------------------
    def _call_api(self, user_text: str, system_prompt: Optional[str] = None,
                  force_plain: bool = False, pairwise: bool = False) -> str:
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(os.getenv("JUDGE_MAX_TOKENS", "400")),
        }
        if not force_plain:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            cost = (
                getattr(usage, "prompt_tokens", 0) / 1e6 * INPUT_PRICE_PER_MTOK
                + getattr(usage, "completion_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_MTOK
            )
            self._charge(cost)
        return resp.choices[0].message.content or ""

    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        if not raw:
            return None
        text = raw.strip()
        # strip a markdown fence if the model added one
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    def _parse(self, raw: str, pairwise: bool = False) -> Optional[dict]:
        obj = self._extract_json(raw)
        if obj is None:
            return None
        if pairwise:
            pref = str(obj.get("preferred", "A")).strip().upper()[:1]
            if pref not in {"A", "B"}:
                pref = "A"
            return {"critique": str(obj.get("critique", "")), "preferred": pref}
        verdict = obj.get("pass", obj.get("passed", obj.get("verdict")))
        if isinstance(verdict, str):
            verdict = verdict.strip().lower() in {"true", "pass", "yes", "1"}
        if verdict is None:
            return None
        return {"critique": str(obj.get("critique", "")), "pass": bool(verdict)}

    # -- deterministic offline reference judge -------------------------------
    def _mock_evaluate(self, generated_text: str) -> dict:
        self._charge(MOCK_COST_PER_CALL)

        # Contamination hook: if the text is one of the few-shot examples, the
        # judge "remembers" the expert label. This is exactly the mechanism that
        # makes dev-data leakage inflate metrics (see scripts/induce_leakage.py).
        for ex in self.few_shot_examples:
            if ex["output"] == generated_text:
                return {
                    "critique": (
                        "This item matches a few-shot example verbatim, so the "
                        "expected label is available in-context; returning it."
                    ),
                    "pass": bool(int(ex["label"]) == 1),
                }

        feats = extract_features(generated_text)
        score = feats["score"] + hash_noise(generated_text)
        passed = score >= WEIGHTS["threshold"]
        violated = self._violations(generated_text, feats)
        if passed:
            critique = (
                "Satisfies R1-R5: it carries a concrete anchor, states the root "
                "cause, stays within the brevity bound and avoids speculation or "
                "meta-commentary."
            )
        else:
            critique = "Fails " + ", ".join(violated) + "; the summary is not actionable as written."
        return {"critique": critique, "pass": bool(passed)}

    @staticmethod
    def _violations(text: str, feats: Dict) -> List[str]:
        low = text.lower()
        out: List[str] = []
        if not feats["anchor"]:
            out.append("R1 (no concrete anchor)")
        if not feats["mechanism"]:
            out.append("R2 (no stated cause)")
        if not feats["brevity"]:
            out.append("R3 (too long, >70 words)")
        if any(m in low for m in HEDGE_MARKERS):
            out.append("R4 (speculation/hedging)")
        if any(m in low for m in META_MARKERS):
            out.append("R5 (meta-commentary)")
        return out or ["R2 (weak cause statement)"]

    def _mock_compare(self, text_a: str, text_b: str) -> dict:
        self._charge(MOCK_COST_PER_CALL)
        # Content score + a fixed position bonus for whichever candidate is
        # shown first. The bonus models the well-documented first-position bias
        # and is what the position-flip probe measures.
        first_bonus = 0.6
        noise_a = hash_noise(text_a, salt="pos")
        noise_b = hash_noise(text_b, salt="pos")
        score_a = extract_features(text_a)["score"] + first_bonus + noise_a
        score_b = extract_features(text_b)["score"] + noise_b
        preferred = "A" if score_a >= score_b else "B"
        critique = (
            "Candidate A reads slightly stronger on the rubric." if preferred == "A"
            else "Candidate B reads slightly stronger on the rubric."
        )
        return {"critique": critique, "preferred": preferred}
