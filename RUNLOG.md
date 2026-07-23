# RUNLOG

Metric: **mean response delay (ms) @ ≤5% interrupted turns**, from the
unmodified official `score.py`. Lower is better.

Numbers come in three kinds and I keep them strictly apart:

- **OOF** — 5-fold `GroupKFold` grouped by `turn_id`, so no turn is split
  across train/test. This is the honest estimate and the headline.
- **en→hi / hi→en** — trained on one language, scored on the other. The hidden
  test is *mostly Hindi*, so this is the closest proxy to the real grade.
- **in-sample** — fit and predict on the same rows, as the starter's `train.py`
  does. Reported once at the end for the shipped `predictions.csv`, and
  labelled as such. With 100 turns per language and a 5% budget that is only
  5 turns, in-sample numbers are badly optimistic and I do not tune on them.

---

## Run 0 — silence-only baseline

| | english | hindi |
|---|---|---|
| delay | **1600 ms** | **850 ms** |
| AUC | 0.514 | 0.501 |

`p_eot = 1.0` everywhere, so the agent is a pure silence timer.

Two things fell out of this that shaped everything after:

English **degenerates** — no (threshold, delay) pair meets the 5% budget, so
the sweep falls back to never-firing and eats the 1.6 s timeout on every turn.

Hindi is a far tighter target than the 1600 ms quoted in the brief: **no Hindi
hold pause exceeds 1.5 s**, so a plain 850 ms timer already interrupts only 5%
of turns. Since the hidden set is mostly Hindi, **850 ms is the number that
actually has to be beaten.**

## Run 1 — 17 causal prosodic features + L2 logistic regression

| | english | hindi |
|---|---|---|
| OOF delay | **1325 ms** | **850 ms** |
| OOF AUC | 0.644 | 0.679 |
| transfer | hi→en 1472 ms (0.584) | en→hi 850 ms (0.643) |

F0 slope/level/range against the speaker's own prefix, energy decay, final
voiced-run lengthening, voiced-onset rate, spectral tilt, plus causal context
(`pause_start`, `pause_index`, current speech-run duration). C=0.1 picked on
pooled OOF AUC — deliberately *not* on the delay metric, which at a 5-turn
budget is far too noisy to tune against.

English 1600 → 1325. **Hindi does not move**, and its operating point is
`threshold=0.05` — the sweep is firing on everything, so the model is adding
nothing over the timer. AUC 0.679 says the ranking has real signal, so the
signal must be landing in the wrong place.

## Run 2 — diagnosing *where* the signal lands  (`analyze.py`)

Built a diagnostic around a property of the scorer: a false cutoff is charged
only when `p ≥ threshold AND delay < dur`, so **a hold shorter than the action
delay is free no matter how badly it is ranked**. Overall AUC is therefore the
wrong health metric. Measuring AUC of EOT against only holds longer than D:

| lang | D=0.0 | D=0.3 | D=0.5 | D=0.8 | D=1.0 |
|---|---|---|---|---|---|
| english | 0.644 | 0.626 | 0.609 | 0.617 | **0.513** |
| hindi | 0.679 | 0.692 | 0.628 | 0.627 | — |

Confirmed: the model is **at chance on the long holds that decide the score**
while looking respectable overall. It was separating the free ones.

The worst errors were also structured, not random — `hi__036` fires on pauses
#2, #4 *and* #5; `en__091` pauses 3.0 s only 0.6 s into the turn. These are
habitual slow talkers, and I had no feature describing how *this* speaker had
already been pausing.

## Run 3 — two changes at once (a mistake), then unpicking them

Changed the F0 tracker and added speaker pause-habit features together.
Result got **worse**: English OOF 1325 → 1405, en→hi AUC 0.643 → 0.585. Since
two things moved, the run was uninformative — so I isolated them.

**The F0 change was the culprit, and the finding is counter-intuitive.** The
starter's autocorrelation (which I had inherited) is the *biased* estimator: it
tapers with lag and so leans toward high F0. Correcting it to unbiased made the
Hz histogram look more physical (English p10 152 → 103 Hz) but made the pitch
features *less* discriminative. Adding proper octave resolution (shortest lag
reaching 85% of the peak) did not recover it either:

| `f0_last_z`, \|AUC−0.5\| | biased (original) | unbiased | octave-resolved |
|---|---|---|---|
| | **0.109** | 0.079 | 0.089 |

**Reverted to the biased estimator on evidence.** Best explanation: the taper
suppresses spurious long-lag peaks in exactly the low-energy frames before a
pause where these features are read, and since every pitch feature is
speaker-relative (slope, z-score against the talker's own median), a consistent
bias cancels out. Kept the speaker pause-habit features — measured separately,
they are among the better discriminators *on the dangerous long holds*
(`prior_sil_mean` AUC 0.415 there vs 0.520 overall).

## Run 4 — make the training loss match what the scorer charges

Since short holds are free and long holds are not, plain log-loss is optimising
the wrong thing. Each hold is now weighted by the fraction of the scorer's own
delay grid at which it would actually be dangerous (floor 0.15 so short holds
still shape calibration); EOT rows keep weight 1. This is a training-time
**cost**, not a feature — `dur` never reaches `predict.py`.

| variant | OOF en | OOF hi | en→hi | hi→en |
|---|---|---|---|---|
| LR, no cost weights | 1320 / 0.661 | 850 / 0.657 | 850 / 0.604 | 1465 / 0.562 |
| **LR + cost weights** | **1305** / 0.658 | 850 / 0.640 | 850 / 0.588 | 1432 / 0.555 |
| HGB + cost weights | 1232 / 0.625 | 857 / 0.705 | 865 / 0.628 | 1270 / 0.593 |
| LR, acoustic only (no context) | 1350 / 0.604 | 850 / 0.645 | 850 / 0.604 | 1362 / 0.557 |

Cost weights trade a little overall AUC for a better delay — which is the
trade they were designed to make. The acoustic-only ablation is worse
everywhere, so the causal context features are earning their place rather than
memorising "turns are about 11 s".

**Gradient boosting was rejected despite winning three of four columns.** Its
edge on the actual metric is nil where it counts (Hindi 857 vs 850), and
shipping it means a `joblib` pickle whose load can fail outright on a different
scikit-learn version on the graders' machine. Logistic regression ships as four
plain arrays in an `.npz` and infers as a dot product — a real AUC gain would
justify that risk, 7 ms in the wrong direction does not.

## Run 5 — spread the scores across the scorer's threshold grid

Hindi kept selecting `threshold=0.05`, the lowest grid value. That is a
*calibration* symptom, not a ranking one: `score.py` sweeps a fixed grid
(0.05…0.95) and only cares which side of a grid line a score falls on, so
scores bunched in a narrow band leave the sweep with almost no usable
operating points. Added a monotone map through the training-score CDF onto
[0.06, 0.97] — frozen at training time, so it is a fixed per-pause function,
not a transform fitted on the evaluation batch.

Being monotone it cannot change AUC; it exists purely to give the sweep
resolution. The lower bound sits just above the lowest threshold, which
**guarantees "fire on everything" stays reachable — so the model can never
score worse than the silence-timer baseline.**

| | english | hindi |
|---|---|---|
| OOF delay | **1290 ms** | 850 ms |
| OOF AUC | 0.660 | 0.642 |

## Final — shipped model

Pooled English+Hindi, language-blind, C=0.1, cost-weighted, calibrated.
(C is re-selected by the sweep on every run; it moved back to 0.1 once the
Run 3 F0 change was reverted. The value in `model.npz` is authoritative.)
`predictions_english.csv` / `predictions_hindi.csv` regenerated via
`predict.py` from the frozen `model.npz`.

| | english | hindi |
|---|---|---|
| **OOF (honest)** | **1290 ms** (AUC 0.660) | **850 ms** (AUC 0.642) |
| in-sample (shipped csv) | 1152 ms (AUC 0.720) | 850 ms (AUC 0.696) |
| baseline | 1600 ms | 850 ms |

English **1600 → 1290 ms OOF, a 19% cut in response delay**. Hindi holds at
850 ms: the calibration floor guarantees it never regresses below baseline, but
the model does not yet beat the timer there — see NOTES.md for why and what I
would do next.
