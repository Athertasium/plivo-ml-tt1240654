"""Inference: score every pause in a folder this model has never seen.

    python predict.py --data_dir <folder> --out predictions.csv

Loads the frozen model from model.npz -- it does not refit. Inference is a
plain standardise-then-logistic dot product in numpy, so the shipped artefact
is four small arrays rather than a pickle that could break across
scikit-learn versions.

CAUSALITY
---------
Two independent guards, both cheap to audit:

  * Column level -- `dataset.read_pause_rows` parses only
    (turn_id, audio_file, pause_index, pause_start). `pause_end` and `label`
    are never read on this path. The assertion below fails loudly if anyone
    ever widens that whitelist.
  * Sample level -- `featurelib.n_causal_frames` keeps only frames lying
    entirely before `pause_start`, and every normalisation statistic is
    computed inside that prefix.

Both matter on this data, where `pause_end` and total file length are each
near-perfect predictors that a live agent could not possibly have.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

import calibrate
import dataset
import featurelib as fl

FORBIDDEN = ("pause_end", "label")


def load_model(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    names = [str(n) for n in z["feature_names"]]
    if names != fl.FEATURE_NAMES:
        raise SystemExit(
            "model.npz was trained on a different feature set:\n"
            f"  model    : {names}\n  featurelib: {fl.FEATURE_NAMES}\n"
            "Re-run train_model.py.")
    return {"mean": z["mean"], "scale": z["scale"], "coef": z["coef"],
            "intercept": float(z["intercept"][0]),
            "cal_knots": z["cal_knots"]}


def predict_proba(m: dict, X: np.ndarray) -> np.ndarray:
    z = (X - m["mean"]) / np.where(m["scale"] == 0, 1.0, m["scale"])
    raw = 1.0 / (1.0 + np.exp(-(z @ m["coef"] + m["intercept"])))
    return calibrate.apply_spread(m["cal_knots"], raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="model.npz")
    args = ap.parse_args()

    assert not set(FORBIDDEN) & set(dataset.WHITELIST_INFERENCE), \
        "causality violation: inference must not read future-facing columns"

    m = load_model(args.model)
    rows = dataset.read_pause_rows(args.data_dir, with_label=False)
    X = dataset.build_matrix(args.data_dir, rows)
    p = predict_proba(m, X)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for r, pi in zip(rows, p):
            w.writerow([r["turn_id"], r["pause_index"], f"{pi:.6f}"])
    print(f"wrote {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
