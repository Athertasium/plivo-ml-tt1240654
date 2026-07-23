"""Diagnostics. NOT part of the inference path.

This script reads `pause_end` (hence hold durations) because the *analysis*
question is "which holds can actually cause a false cutoff". That is a
property of the scorer, not a feature: nothing computed here is fed back into
featurelib. `predict.py` still never reads that column.

The scorer only charges a false cutoff when `p >= threshold AND delay < dur`.
So a hold shorter than the action delay is free no matter how badly we score
it, and overall AUC is the wrong health metric -- what matters is how well
EOT is separated from *long* holds. That is what this measures.

    python analyze.py --data_dirs <en> <hi>
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

import dataset
import featurelib as fl
from train_model import oof_predict


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    # Average ranks for ties -- discrete features (n_prior_pauses,
    # pause_index) have large tie blocks, and argsort would break them by
    # array position, making their AUC depend on row order.
    ranks = rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def durations(data_dir: str) -> np.ndarray:
    """Hold/eot pause durations, for diagnostics only."""
    out = []
    with open(os.path.join(data_dir, "labels.csv")) as f:
        for r in csv.DictReader(f):
            out.append(float(r["pause_end"]) - float(r["pause_start"]))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    packs = {}
    for d in args.data_dirs:
        name = os.path.basename(os.path.normpath(d))
        rows, X, y, g = dataset.load_split(d)
        packs[name] = {"dir": d, "rows": rows, "X": X, "y": y,
                       "groups": g, "dur": durations(d)}

    X = np.vstack([p["X"] for p in packs.values()])
    y = np.concatenate([p["y"] for p in packs.values()])
    g = np.concatenate([p["groups"] for p in packs.values()])
    p_oof = oof_predict(X, y, g, args.C)

    spans, cur = {}, 0
    for name, p in packs.items():
        spans[name] = slice(cur, cur + len(p["rows"]))
        cur += len(p["rows"])

    # ---- F0 tracker sanity: voiced F0 should sit in a human range ----
    print("=== F0 tracker sanity ===")
    for name, pk in packs.items():
        d = pk["dir"]
        r0 = pk["rows"][0]
        x, sr = fl.load_wav(os.path.join(d, r0["audio_file"]))
        fd = fl.analyse_file(x, sr)
        v = fd.f0[fd.f0 > 0]
        print(f"  {name}: voiced {len(v)}/{len(fd.f0)} frames "
              f"({len(v)/max(1,len(fd.f0)):.2f})  "
              f"F0 p10/50/90 = {np.percentile(v,10):.0f}/"
              f"{np.percentile(v,50):.0f}/{np.percentile(v,90):.0f} Hz")

    # ---- the metric that actually matters ----
    print("\n=== AUC: EOT vs holds longer than D (only these can cost) ===")
    print(f"{'lang':<9}{'D=0.0':>11}{'D=0.3':>11}{'D=0.5':>11}"
          f"{'D=0.8':>11}{'D=1.0':>11}")
    for name, pk in packs.items():
        p = p_oof[spans[name]]
        yy, dd = pk["y"], pk["dur"]
        cells = []
        for D in (0.0, 0.3, 0.5, 0.8, 1.0):
            keep = (yy == 1) | ((yy == 0) & (dd > D))
            n_neg = int(((yy == 0) & (dd > D)).sum())
            cells.append(f"{auc(yy[keep], p[keep]):.3f}({n_neg})")
        print(f"  {name:<7}" + "".join(f"{c:>11}" for c in cells))

    # ---- per-feature discriminative power on the dangerous subset ----
    print("\n=== per-feature AUC (pooled) ===")
    dur_all = np.concatenate([p["dur"] for p in packs.values()])
    danger = (y == 1) | ((y == 0) & (dur_all > 0.5))
    print(f"{'feature':<20}{'all holds':>11}{'long holds':>12}")
    stats = []
    for i, nm in enumerate(fl.FEATURE_NAMES):
        a_all = auc(y, X[:, i])
        a_lng = auc(y[danger], X[danger, i])
        stats.append((abs(a_lng - 0.5), nm, a_all, a_lng))
    for _, nm, a_all, a_lng in sorted(stats, reverse=True):
        print(f"  {nm:<18}{a_all:>11.3f}{a_lng:>12.3f}")

    # ---- worst errors, for listening ----
    print("\n=== WORST ERRORS (listen to these) ===")
    for name, pk in packs.items():
        p = p_oof[spans[name]]
        rows, yy, dd = pk["rows"], pk["y"], pk["dur"]
        print(f"\n--- {name}: LONG HOLDS scored high (cause cutoffs) ---")
        cand = [i for i in range(len(rows)) if yy[i] == 0 and dd[i] > 0.5]
        for i in sorted(cand, key=lambda i: -p[i])[:args.top]:
            print(f"  p={p[i]:.3f}  {rows[i]['turn_id']} "
                  f"pause#{rows[i]['pause_index']}  "
                  f"start={rows[i]['pause_start']:.2f}s  dur={dd[i]:.2f}s")
        print(f"--- {name}: EOTs scored low (cause 1.6 s timeouts) ---")
        cand = [i for i in range(len(rows)) if yy[i] == 1]
        for i in sorted(cand, key=lambda i: p[i])[:args.top]:
            print(f"  p={p[i]:.3f}  {rows[i]['turn_id']} "
                  f"pause#{rows[i]['pause_index']}  "
                  f"start={rows[i]['pause_start']:.2f}s")


if __name__ == "__main__":
    main()
