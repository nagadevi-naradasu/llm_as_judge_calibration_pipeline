"""Contract tests for the generated result artifacts (skip if not built yet)."""

import json
import os

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")

METRICS_COLUMNS = {
    "split", "tpr_mean", "tpr_ci_low", "tpr_ci_high", "tnr_mean", "tnr_ci_low",
    "tnr_ci_high", "agreement", "kappa_mean", "kappa_ci_low", "kappa_ci_high",
}


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not generated yet")
    return path


def test_metrics_csv_contract():
    path = _require(os.path.join(RESULTS, "metrics.csv"))
    df = pd.read_csv(path)
    assert METRICS_COLUMNS <= set(df.columns)
    assert "test" in df["split"].values
    test = df[df["split"] == "test"].iloc[0]
    for c in METRICS_COLUMNS - {"split"}:
        assert float(test[c]) == float(test[c])  # not NaN
    assert test["tpr_ci_low"] <= test["tpr_mean"] <= test["tpr_ci_high"]
    assert test["tnr_ci_low"] <= test["tnr_mean"] <= test["tnr_ci_high"]


def test_judge_test_csv_contract():
    path = _require(os.path.join(RESULTS, "judge_test.csv"))
    df = pd.read_csv(path)
    assert {"item_id", "human_label", "judge_label", "judge_critique"} <= set(df.columns)
    labels = pd.read_csv(os.path.join(REPO_ROOT, "data", "labels.csv"))
    joined = df.merge(labels[["item_id", "label"]], on="item_id", how="left")
    assert joined["label"].notna().all()
    assert (joined["human_label"] == joined["label"]).all()


def test_bias_csv_contract():
    path = _require(os.path.join(RESULTS, "bias.csv"))
    df = pd.read_csv(path)
    assert {"probe_type", "sample_size", "position_flip_rate",
            "pass_rate_before", "pass_rate_after"} <= set(df.columns)
    pos = df[df["probe_type"] == "position"]
    assert len(pos) >= 1 and pos["position_flip_rate"].notna().all()
    verb = df[df["probe_type"] == "verbosity"]
    assert len(verb) >= 1
    assert verb["pass_rate_before"].notna().all()
    assert verb["pass_rate_after"].notna().all()


def test_induced_failure_json_contract():
    path = _require(os.path.join(RESULTS, "induced_failure.json"))
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for key in ("baseline_dev_kappa", "contaminated_dev_kappa", "leaked_items_count"):
        assert key in data
    assert data["contaminated_dev_kappa"] > data["baseline_dev_kappa"]
    assert data["leaked_items_count"] == 10
