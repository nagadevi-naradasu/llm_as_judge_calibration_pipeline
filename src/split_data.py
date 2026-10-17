"""Deterministic, stratified Train / Dev / Test splitter.

The judge prompt is *built* from Train, *iterated* on Dev, and *frozen* before
Test is touched once. Because prompt tuning is overfitting, the split must be
stable across machines and runs: this script uses a seeded RNG and a
largest-remainder allocation so the row counts are exactly the target ratios.

Run::

    python src/split_data.py --input data/labels.csv --output-dir data/splits/

Outputs ``train.csv`` (15%), ``dev.csv`` (40%), ``test.csv`` (45%).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

TRAIN_FRAC, DEV_FRAC, TEST_FRAC = 0.15, 0.40, 0.45


def allocate(n: int, seed: int = 0) -> Dict[str, List[int]]:
    """Return shuffled index lists of sizes matching the split fractions.

    Uses largest-remainder so the three sizes sum to ``n`` exactly, and is
    stratified per class by the caller invoking it once per class.
    """
    rng = np.random.default_rng(seed)
    fracs = {"train": TRAIN_FRAC, "dev": DEV_FRAC, "test": TEST_FRAC}
    raw = {k: n * f for k, f in fracs.items()}
    sizes = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = n - sum(sizes.values())
    # give the leftover rows to the largest fractional parts (ties -> train,
    # then dev, then test) so the result is deterministic.
    order = sorted(fracs, key=lambda k: (-(raw[k] - sizes[k]), k))
    for k in order[:remainder]:
        sizes[k] += 1

    idx = rng.permutation(n)
    out: Dict[str, List[int]] = {}
    cursor = 0
    for k in ("train", "dev", "test"):
        out[k] = list(idx[cursor:cursor + sizes[k]])
        cursor += sizes[k]
    return out


def stratified_split(df: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    parts: Dict[str, List[int]] = {"train": [], "dev": [], "test": []}
    for cls, group in df.groupby("label", sort=True):
        alloc = allocate(len(group), seed=seed + int(cls))
        positions = group.index.to_numpy()
        for k, local in alloc.items():
            parts[k].extend(positions[local].tolist())
    rng = np.random.default_rng(seed)
    return {
        k: df.loc[rng.permutation(sorted(v))].reset_index(drop=True)
        for k, v in parts.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/labels.csv")
    ap.add_argument("--output-dir", default="data/splits/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    for col in ("item_id", "output", "label", "critique", "relabeled"):
        if col not in df.columns:
            raise SystemExit(f"missing required column: {col}")

    splits = stratified_split(df, args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    total = len(df)
    for name in ("train", "dev", "test"):
        part = splits[name]
        path = os.path.join(args.output_dir, f"{name}.csv")
        part.to_csv(path, index=False)
        fails = int((part["label"] == 0).sum())
        print(f"{name:5s}: {len(part):3d} rows ({len(part)/total:5.1%}), "
              f"{fails} fail / {len(part)-fails} pass -> {path}")

    # guardrails: disjoint, exhaustive, schema-preserving
    ids = [set(s["item_id"]) for s in splits.values()]
    assert sum(len(s) for s in ids) == total
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    print("ok: splits are disjoint and exhaustive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
