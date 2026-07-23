# End-of-Turn Detection — Plivo assignment

Predicts, for each pause in a user turn, the probability that the turn is over.

**Result (out-of-fold, grouped by turn):** English 1600 → **1228 ms** mean
response delay at ≤5% interrupted turns, a 23% cut. Hindi holds at its 850 ms
baseline.

- [`SUMMARY.html`](SUMMARY.html) — full writeup with charts
- [`RUNLOG.md`](RUNLOG.md) — every scoring run and what changed
- [`NOTES.md`](NOTES.md) — signal, failures, next steps

## Layout

| file | role |
|---|---|
| `predict.py` | inference — loads frozen `model.npz`, never refits |
| `featurelib.py` | causal prosodic features (the causality contract lives here) |
| `dataset.py` | shared feature assembly for train and serve |
| `train_model.py` | training + out-of-fold / cross-lingual evaluation |
| `calibrate.py` | monotone spread across the scorer's threshold grid |
| `analyze.py` | diagnostics: dangerous-hold AUC, ranked errors |
| `experiments.py` | ablation table |
| `score.py` | organisers' scorer, unmodified |

## Run

```bash
python predict.py --data_dir <folder> --out predictions.csv
python score.py   --data_dir <folder> --pred predictions.csv
```

Requires numpy, scipy, scikit-learn, soundfile. No pretrained weights, no GPU.
