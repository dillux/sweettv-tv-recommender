# SWEET.TV — TV Programme Recommendation (Final Exam, LLM4Rec)

HSE University · LLM4Rec: AI-Assisted Recommender Systems · Dilara Faritova (dafaritova@edu.hse.ru)

Kaggle competition: `sweettv-tv-program-recommender` · metric **MAP@5** · 1 260 test users, Top-5 each.

## The problem in one paragraph

The data contains no user–programme pairs. The view log records *user, channel, start, stop*; the
schedule records *what aired on which channel when*. Interactions have to be constructed by
joining the two on channel and interval overlap. Only **10.8%** of schedule airtime carries a
programme id (`tv_show_id != 0`), so most of the broadcast day cannot become an interaction at all.

## Run

Python 3.9+, standard library only — no dependencies, no seed (every method is deterministic).

```bash
python3 peek.py      /path/to/data    # column formats and row counts
python3 run.py       /path/to/data    # full pipeline -> submission.csv, validation_results.txt
python3 examples.py  /path/to/data    # Top-5 per method for two users -> examples.txt
python3 fair_eval.py /path/to/data    # popularity under the same repeat rule -> fair_eval.txt
```

`/path/to/data` is the folder holding `dataset11-30.csv`, `export_arh_11-20-final.csv`,
`export_arh_21-30-final.csv`, `export_arh_31-42-final.csv` and the Kaggle `submission.csv`
template. Outputs are written next to the **script**, not next to the data.

Full run: ~370 s on a 2020 MacBook. `--fast` samples the first 400 000 view rows for debugging
(it produces an empty validation cohort by construction — use it only to check the plumbing).

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Join rule | same channel, interval overlap | the only link between the two files |
| Overlap threshold | ≥ 300 s **or** ≥ 30% of airtime | median view is 396 s; absolute cut keeps typical sessions, fractional cut rescues short programmes |
| `tv_show_id = 0` | dropped | unknown content — not an item, and not evidence of a miss |
| Candidate pool | the 6 636 programmes airing in weeks 31–42 | a programme that never airs cannot be watched |
| Validation split | train = weeks 11–20, hold-out = weeks 21–30 | temporal order preserved; test weeks 31–42 never used for behaviour |
| Cold start (10 of 1 260 users) | popularity Top-5 | no history to personalise from |

Timestamps are text in **two different formats** — `2020-03-09 00:00:02` in the view log,
`09.03.2020 06:00:00` in the schedule. `t_view()` and `t_sched()` handle them separately; both
were verified to agree on the same instant.

## Methods

- **Popularity** — reference baseline (required, does not count as one of the two methods).
- **Method 1 — association rules over viewing baskets.** Each user's history is a basket; every
  pair increments a co-occurrence counter. `score(u,b) = Σ_a n(u,a)·co(a,b)`. This is the
  **un-normalised** rule — confidence and lift were not implemented.
- **Method 2 — content-based filtering** over `tv_show_category` plus three genre fields.
- **Hybrid** — co-view → content → popularity cascade.

## Kaggle

Submitted method: **popularity** (the method selected on validation). Late submission by
`dilarafaritiva`.

| | MAP@5 |
|---|---:|
| **Kaggle public score** | **0.29390** |
| Kaggle private score | 0.28873 |

The local hold-out figure below (0.620) is **not comparable** to the Kaggle score: it is measured
against interactions this pipeline constructed with its own overlap rule, over a different cohort
and a nearer horizon. It is used only to compare the four methods against each other.

## Results — MAP@5 on the weeks 21–30 hold-out (4 799 users)

| Method | Repeats | MAP@5 | Runtime |
|---|---|---:|---:|
| Popularity (reference), **as submitted** | allowed | **0.620316** | < 1 s |
| Popularity (reference), **fair protocol** | excluded | 0.189635 | < 1 s |
| Association rules / co-view | excluded | 0.184800 | 138 s |
| Content-based | excluded | 0.037238 | 27 s |
| Hybrid | excluded | 0.184800 | 139 s |

### The headline comparison is decided by the protocol, not by the algorithms

Row 1 lets the baseline re-recommend programmes the user already watched. Every other row excludes
the user's history by construction. Broadcast TV is dominated by repeat consumption, so the
strongest predictor of weeks 21–30 is weeks 11–20 — and row 1 hands that predictor to the baseline
and withholds it from everyone else.

Put the baseline under the same rule (`fair_eval.py`, 91 s) and it scores **0.189635** against the
association rules' **0.184800** — **2.6%** apart, where the raw comparison showed **235%**. The
same run reproduced row 1 to all six decimal places.

Two practical readings follow:

- **Repeat viewing is worth +227%** (0.190 → 0.620), far more than any algorithm choice measured
  here. On a TV platform the shelf's first job is to put the viewer back in front of their evening
  news.
- **At equal accuracy, pick on coverage.** The submitted popularity file shows five distinct
  programmes to all 1 260 users; the co-view graph can reach 2 754.

The strategy these results point to — a personal-repeat layer on an association-rules ranking —
**was not implemented or measured.** The submitted file is the plain popularity baseline, chosen
before the fair-protocol run existed.

The hybrid equals co-view to six decimals because the co-view stage fills all five slots for every
cohort user, so the later stages never execute.

## Known limitations

- **The mirror experiment was not run.** `fair_eval.py` took repeats *away* from the baseline;
  giving repeats *to* the personalised methods is the same one-line change and a 6-minute rerun,
  and it has not been done.
- Association rules are un-normalised; confidence/lift were not compared.
- Only **2 194 of 6 636** pool programmes were watched by anyone in training, so two thirds of the
  airing catalogue is unreachable by any collaborative method.
- The submitted file shows **five distinct programmes to all 1 260 users**.
- MAP@5 numbers come from one implementation; no independent reimplementation was written.
- No A/B test was run and none is claimed.

## Files

```
run.py                     full pipeline
peek.py                    column/format inspector
examples.py                Top-5 per method for two users, with titles and categories
submission.csv             the file submitted to Kaggle
validation_results.txt     machine-readable record of the run
examples.txt               output of examples.py
```
