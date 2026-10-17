"""Tests for the dataset contract and the deterministic splitter."""

import os

import pandas as pd
import pytest

from src.split_data import stratified_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(REPO_ROOT, "data", "labels.csv")


@pytest.fixture(scope="module")
def labels():
    df = pd.read_csv(LABELS)
    return df


def test_labels_schema(labels):
    assert list(labels.columns) == ["item_id", "output", "label", "critique", "relabeled"]


def test_labels_content(labels):
    assert len(labels) >= 250
    assert labels["item_id"].is_unique
    assert set(labels["label"].unique()) <= {0, 1}
    assert set(labels["relabeled"].unique()) <= {0, 1}
    assert (labels["label"] == 0).mean() >= 0.30, "need at least 30% failures"
    assert labels["critique"].astype(str).str.len().min() > 0


def test_split_ratios_and_disjoint(labels):
    splits = stratified_split(labels, seed=42)
    n = len(labels)
    sizes = {k: len(v) for k, v in splits.items()}
    assert abs(sizes["train"] - 0.15 * n) <= 1
    assert abs(sizes["dev"] - 0.40 * n) <= 1
    assert abs(sizes["test"] - 0.45 * n) <= 1
    assert sum(sizes.values()) == n
    ids = [set(v["item_id"]) for v in splits.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_split_is_deterministic(labels):
    a = stratified_split(labels, seed=42)["test"]["item_id"].tolist()
    b = stratified_split(labels, seed=42)["test"]["item_id"].tolist()
    assert a == b


def test_split_preserves_columns(labels):
    splits = stratified_split(labels, seed=42)
    for part in splits.values():
        assert list(part.columns) == list(labels.columns)


def test_committed_split_files_match_ratio(labels):
    for name, frac in (("train", 0.15), ("dev", 0.40), ("test", 0.45)):
        path = os.path.join(REPO_ROOT, "data", "splits", f"{name}.csv")
        if not os.path.exists(path):
            pytest.skip(f"{name}.csv not generated yet")
        part = pd.read_csv(path)
        assert abs(len(part) - frac * len(labels)) <= 1
