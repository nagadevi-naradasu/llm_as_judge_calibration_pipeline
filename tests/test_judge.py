"""Tests for the judge JSON contract (mock backend, no network)."""

import json

import pytest

from src.judge import LLMJudge, build_system_prompt


def make_judge(mode="mock"):
    return LLMJudge(model_name="gpt-4o-mini", system_prompt="test", mode=mode)


def test_evaluate_returns_contract_keys():
    judge = make_judge()
    out = judge.evaluate("api/gateway.py returns HTTP 502 when the upstream is down (issue #1234).")
    assert set(out.keys()) == {"critique", "pass"}
    assert isinstance(out["critique"], str) and out["critique"]
    assert isinstance(out["pass"], bool)


def test_mock_is_deterministic():
    a = make_judge().evaluate("some summary text")
    b = make_judge().evaluate("some summary text")
    assert a == b


def test_good_summary_passes_bad_summary_fails():
    good = make_judge().evaluate(
        "api/gateway.py raises ETIMEDOUT under backfill load. Root cause is a "
        "missing reset in the acquire path, tracked as #4821 in v2.3.1."
    )
    bad = make_judge().evaluate(
        "There seems to be some kind of issue with the retry logic. "
        "It might be worth looking into, though I am not sure."
    )
    assert good["pass"] is True
    assert bad["pass"] is False


def test_regex_fallback_parses_fenced_and_prose():
    judge = make_judge()
    assert judge._parse('```json\n{"critique": "x", "pass": true}\n```') == {
        "critique": "x", "pass": True}
    assert judge._parse('Sure! {"critique": "y", "pass": false} hope that helps') == {
        "critique": "y", "pass": False}
    assert judge._parse("not json at all") is None


def test_budget_exceeded_raises():
    judge = make_judge()
    judge.budget_usd = 1.0
    with pytest.raises(Exception):
        judge._charge(2.0)


def test_system_prompt_includes_contract_and_json_word():
    prompt = build_system_prompt("RUBRIC BODY", [{"output": "o", "label": 1, "critique": "c"}])
    assert "JSON" in prompt
    assert "RUBRIC BODY" in prompt
    assert "few-shot" in prompt.lower()


def test_compare_returns_preference():
    judge = make_judge()
    out = judge.compare("text a", "text b", "pairwise system prompt")
    assert out["preferred"] in {"A", "B"}
    assert isinstance(out["critique"], str)
