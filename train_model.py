"""Train the EOT classifier and report HONEST scores.

Why this file exists in the shape it does: the starter's train.py refits on all
data and then predicts on those same rows, so its printed number is in-sample
and meaningless. With 100 turns per language and a metric whose 5% budget is
just 5 turns, in-sample scores are wildly optimistic.

So every headline number here is one of:
  * OOF   -- 5-fold GroupKFold out-of-fold, grouped by turn_id so no turn is
             ever split across train/test.
  * en->hi / hi->en -- trained on one language, scored on the other. The
             hidden test set is "mostly Hindi", so cross-lingual transfer is
             the closest proxy we have to the real grade.

Scoring always goes through the official score.py, never a reimplementation.

    python train_model.py --data_dirs ../eot_data/english ../eot_data/hindi
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import calibrate
import dataset
import featurelib as fl
import score as official

C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
N_FOLDS = 5


def make_model(C: float) -> LogisticRegression:
    return LogisticRegression(
        C=C, max_iter=5000, class_weight="balanced", solver="lbfgs")


def cost_weights(y: np.ndarray, dur: np.ndarray,
                 floor: float = 0.15) -> np.ndarray:
    """Weight each HOLD by how much the scorer could actually charge for it.

    The scorer only records a false cutoff when `p >= threshold AND
    delay < dur`, so a hold shorter than the action delay is free however
    badly we rank it. Plain log-loss disagrees: it spends capacity separating
    those free short holds, which is exactly the failure the diagnostics
    showed (overall AUC 0.64 but chance-level 0.51 on the long holds that
    decide the score).

    So each hold is weighted by the fraction of the scorer's own delay grid at
    which it would be dangerous, with a floor so short holds still shape
    calibration. EOT rows keep weight 1 -- every one of them contributes
    latency. Class imbalance stays with `class_weight="balanced"`; these
    weights only redistribute mass *within* the hold class.

    `dur` comes from training labels. It is a cost, not a feature -- nothing
    here is available to, or needed by, `predict.py`.
    """
    w = np.ones(len(y), dtype=np.float64)
    hold = y == 0
    if not hold.any():
        return w
    frac = np.array([np.mean(official.DELAYS < d) for d in dur])
    wh = floor + (1.0 - floor) * frac[hold]
    w[hold] = wh / wh.mean()   # mean-1 within holds, so balance is preserved
    return w


def official_score(data_dir: str, rows: list[dict], p: np.ndarray) -> dict:
    """Run the untouched official scorer on in-memory predictions."""
    fd, tmp = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["turn_id", "pause_index", "p_eot"])
            for r, pi in zip(rows, p):
                w.writerow([r["turn_id"], r["pause_index"], f"{pi:.6f}"])
        return official.score(os.path.join(data_dir, "labels.csv"), tmp)
    finally:
        os.unlink(tmp)


def oof_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                C: float, w: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold probabilities, folds grouped by turn."""
    p = np.zeros(len(y), dtype=np.float64)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = make_model(C).fit(sc.transform(X[tr]), y[tr],
                                sample_weight=None if w is None else w[tr])
        raw_te = clf.predict_proba(sc.transform(X[te]))[:, 1]
        # knots come from this fold's TRAINING scores only -- the held-out
        # turns never inform their own calibration
        knots = calibrate.fit_spread(clf.predict_proba(sc.transform(X[tr]))[:, 1])
        p[te] = calibrate.apply_spread(knots, raw_te)
    return p


def fit_full(X: np.ndarray, y: np.ndarray, C: float,
             w: np.ndarray | None = None):
    sc = StandardScaler().fit(X)
    clf = make_model(C).fit(sc.transform(X), y, sample_weight=w)
    return sc, clf


def transfer(Xtr, ytr, Xte, C, w=None):
    sc, clf = fit_full(Xtr, ytr, C, w)
    knots = calibrate.fit_spread(clf.predict_proba(sc.transform(Xtr))[:, 1])
    return calibrate.apply_spread(
        knots, clf.predict_proba(sc.transform(Xte))[:, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--out", default="model.npz")
    ap.add_argument("--C", type=float, default=None,
                    help="skip the sweep and use this C")
    ap.add_argument("--no_cost_weights", action="store_true",
                    help="ablate the scorer-matched training weights")
    ap.add_argument("--weight_floor", type=float, default=0.15)
    args = ap.parse_args()

    splits = {}
    for d in args.data_dirs:
        name = os.path.basename(os.path.normpath(d))
        rows, X, y, g = dataset.load_split(d)
        splits[name] = {"dir": d, "rows": rows, "X": X, "y": y, "groups": g,
                        "dur": dataset.read_train_durations(d)}
        print(f"[data] {name}: {len(rows)} pauses, "
              f"{len(set(g))} turns, {y.sum()} eot")

    X = np.vstack([s["X"] for s in splits.values()])
    y = np.concatenate([s["y"] for s in splits.values()])
    groups = np.concatenate([s["groups"] for s in splits.values()])

    # row span of each language inside the pooled matrices
    spans, cur = {}, 0
    for name, s in splits.items():
        spans[name] = slice(cur, cur + len(s["rows"]))
        cur += len(s["rows"])

    def per_lang(p: np.ndarray) -> dict[str, dict]:
        return {name: official_score(s["dir"], s["rows"], p[spans[name]])
                for name, s in splits.items()}

    dur = np.concatenate([s["dur"] for s in splits.values()])
    W = None if args.no_cost_weights else cost_weights(y, dur,
                                                       args.weight_floor)
    print(f"[weights] {'OFF (ablation)' if W is None else 'scorer-matched'}")

    # ---- pick C on pooled OOF AUC (never on the delay metric directly:
    #      the 5-turn budget makes that far too noisy to tune against) ----
    if args.C is None:
        best_C, best_auc = None, -1.0
        print("\n[sweep] pooled OOF AUC by C")
        for C in C_GRID:
            p = oof_predict(X, y, groups, C, W)
            aucs = [r["auc"] for r in per_lang(p).values()]
            mean_auc = float(np.mean(aucs))
            print(f"   C={C:<5} mean OOF AUC={mean_auc:.4f}  "
                  f"({', '.join(f'{a:.3f}' for a in aucs)})")
            if mean_auc > best_auc:
                best_C, best_auc = C, mean_auc
    else:
        best_C = args.C
    print(f"\n[model] C = {best_C}")

    # ---- honest per-language OOF scores ----
    p_oof = oof_predict(X, y, groups, best_C, W)
    print("\n=== OUT-OF-FOLD (5-fold, grouped by turn) ===")
    for name, r in per_lang(p_oof).items():
        print(f"  {name:<8} delay={r['latency']*1000:7.0f} ms   "
              f"cut={r['cutoff']*100:4.1f}%   AUC={r['auc']:.3f}   "
              f"(thr={r['threshold']}, delay={r['delay']*1000:.0f} ms)")

    # ---- cross-lingual transfer ----
    names = list(splits)
    if len(names) == 2:
        a, b = names
        print("\n=== CROSS-LINGUAL TRANSFER ===")
        for src, dst in ((a, b), (b, a)):
            p = transfer(splits[src]["X"], splits[src]["y"],
                         splits[dst]["X"], best_C,
                         None if W is None else W[spans[src]])
            r = official_score(splits[dst]["dir"], splits[dst]["rows"], p)
            print(f"  {src[:2]}->{dst[:2]}   delay={r['latency']*1000:7.0f} ms"
                  f"   cut={r['cutoff']*100:4.1f}%   AUC={r['auc']:.3f}")

    # ---- final model on everything, saved as plain arrays ----
    sc, clf = fit_full(X, y, best_C, W)
    knots = calibrate.fit_spread(clf.predict_proba(sc.transform(X))[:, 1])
    np.savez(args.out,
             mean=sc.mean_.astype(np.float64),
             scale=sc.scale_.astype(np.float64),
             coef=clf.coef_[0].astype(np.float64),
             intercept=np.array([clf.intercept_[0]], dtype=np.float64),
             cal_knots=knots,
             feature_names=np.array(fl.FEATURE_NAMES),
             C=np.array([best_C]))
    print(f"\n[saved] {args.out}")

    order = np.argsort(-np.abs(clf.coef_[0]))
    print("\n=== STANDARDISED COEFFICIENTS (+ => predicts EOT) ===")
    for i in order:
        print(f"  {fl.FEATURE_NAMES[i]:<18} {clf.coef_[0][i]:+.3f}")


if __name__ == "__main__":
    main()
