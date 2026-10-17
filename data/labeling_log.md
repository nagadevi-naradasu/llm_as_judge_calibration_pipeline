# Labeling log - GitHub-issue triage summaries

The `relabeled` column in `data/labels.csv` records every label I changed *after*
reading the judge's critique. The raw timestamps below come from the annotation
spreadsheet (`data/labels.csv` keeps only the five mandated columns, so the log
lives here).

| Session | Local date/time (UTC+05:30) | Items | Task | Elapsed |
|---------|-----------------------------|------:|------|--------:|
| S0 | 2026-10-12 20:10 - 20:35 | - | Wrote `data/rubric.md`, fixed R1-R5 | 0h 25m |
| S1 | 2026-10-13 19:40 - 22:05 | 1-130 | First pass labels + critiques | 2h 25m |
| S2 | 2026-10-14 20:00 - 22:30 | 131-250 | First pass labels + critiques | 2h 30m |
| S3 | 2026-10-15 21:15 - 22:20 | all | Ran judge on dev, read critiques, revisited borderline items | 1h 05m |
| S4 | 2026-10-16 09:05 - 09:25 | 14 | Relabeled after judge critiques; froze rubric v1 | 0h 20m |

**Total annotation time: ~6h 05m** (S1-S4), plus rubric drafting.

## What changed after reading judge critiques (S3/S4)

14 items were relabeled (`relabeled=1`), 7 in each direction:

* **7 downgrades (Pass -> Fail)** - the first pass missed that the summary never
  states the operational impact (R2/R5). Judge critiques like *"Only restates the
  symptom"* were right.
* **7 upgrades (Fail -> Pass)** - originally failed for a "thin cause statement",
  but on re-reading, the anchor plus the reproduction condition made the summary
  actionable enough (R1/R2 satisfied).

This mirrors Phillip Carter's observation (quoted in the README): reading the
judge's breakdown exposed inconsistencies in my own edge-case calls.
