# inputs/

Fixed inputs for the calibration harness. Everything here is versioned and
hashed where the experiment requires reproducibility; nothing in this folder is
generated at evaluation time.

| Input | Role |
|-------|------|
| `../data/labels.csv` | The frozen gold set (n=250). Produced once by `scripts/generate_dataset.py` (seed 42) and then hand-labelled; treated as read-only afterwards. |
| `../data/rubric.md` | The rubric text embedded verbatim into the judge system prompt. Frozen as `judge-prompt-v1` before the test run. |
| `../data/splits/{train,dev,test}.csv` | Deterministic 15/40/45 split (seed 42). |

Regenerate the gold set only with:

```bash
python scripts/generate_dataset.py --output data/labels.csv
python src/split_data.py --input data/labels.csv --output-dir data/splits/
```
