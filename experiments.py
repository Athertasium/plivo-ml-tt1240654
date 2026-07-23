"""Ablation table. Every row is OOF (grouped by turn) + cross-lingual transfer.

    python experiments.py --data_dirs <en> <hi>
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import dataset
import featurelib as fl
from train_model import (N_FOLDS, cost_weights, fit_full, make_model,
                         official_score, oof_predict, transfer)


def hgb():
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=7,
        min_samples_leaf=25, l2_regularization=1.0, random_state=0)


def oof_hgb(X, y, g, w):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, g):
        m = hgb().fit(X[tr], y[tr], sample_weight=None if w is None else w[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    args = ap.parse_args()

    S = {}
    for d in args.data_dirs:
        name = os.path.basename(os.path.normpath(d))
        rows, X, y, g = dataset.load_split(d)
        S[name] = {"dir": d, "rows": rows, "X": X, "y": y, "groups": g,
                   "dur": dataset.read_train_durations(d)}
    names = list(S)
    X = np.vstack([s["X"] for s in S.values()])
    y = np.concatenate([s["y"] for s in S.values()])
    g = np.concatenate([s["groups"] for s in S.values()])
    dur = np.concatenate([s["dur"] for s in S.values()])
    spans, cur = {}, 0
    for n_, s in S.items():
        spans[n_] = slice(cur, cur + len(s["rows"]))
        cur += len(s["rows"])

    W = cost_weights(y, dur)

    def report(tag, p_oof, tr_fn):
        cells = []
        for n_ in names:
            r = official_score(S[n_]["dir"], S[n_]["rows"], p_oof[spans[n_]])
            cells.append(f"{r['latency']*1000:6.0f}ms/{r['auc']:.3f}")
        tcells = []
        for src, dst in ((names[0], names[1]), (names[1], names[0])):
            p = tr_fn(src, dst)
            r = official_score(S[dst]["dir"], S[dst]["rows"], p)
            tcells.append(f"{r['latency']*1000:6.0f}ms/{r['auc']:.3f}")
        print(f"  {tag:<26}" + "".join(f"{c:>16}" for c in cells)
              + "  |" + "".join(f"{c:>16}" for c in tcells))

    hdr = "".join(f"{('OOF ' + n)[:14]:>16}" for n in names)
    thd = "".join(f"{(a[:2] + '->' + b[:2]):>16}"
                  for a, b in ((names[0], names[1]), (names[1], names[0])))
    print(f"  {'variant':<26}{hdr}  |{thd}")
    print("  " + "-" * (26 + 16 * 4 + 3))

    # 1. logistic regression, no scorer-matched weights
    report("LR  (no cost weights)",
           oof_predict(X, y, g, 0.1, None),
           lambda s, d: transfer(S[s]["X"], S[s]["y"], S[d]["X"], 0.1, None))

    # 2. logistic regression + scorer-matched weights
    report("LR  + cost weights",
           oof_predict(X, y, g, 0.1, W),
           lambda s, d: transfer(S[s]["X"], S[s]["y"], S[d]["X"], 0.1,
                                 W[spans[s]]))

    # 3. gradient boosting challenger
    def tr_hgb(s, d):
        m = hgb().fit(S[s]["X"], S[s]["y"], sample_weight=W[spans[s]])
        return m.predict_proba(S[d]["X"])[:, 1]

    report("HGB + cost weights", oof_hgb(X, y, g, W), tr_hgb)

    # 4. drop the turn-position context features (do they really generalise?)
    ctx = {"log_pause_start", "n_prior_pauses", "prior_sil_rate",
           "prior_sil_mean", "prior_sil_max"}
    keep = [i for i, n_ in enumerate(fl.FEATURE_NAMES) if n_ not in ctx]
    Xk = X[:, keep]
    Sk = {n_: S[n_]["X"][:, keep] for n_ in names}
    report("LR  acoustic only (no ctx)",
           oof_predict(Xk, y, g, 0.1, W),
           lambda s, d: transfer(Sk[s], S[s]["y"], Sk[d], 0.1, W[spans[s]]))


if __name__ == "__main__":
    main()
