"""Dataset assembly shared by training and inference.

Both `train_model.py` and `predict.py` build their feature matrix through
`build_matrix()`, so the training and serving feature paths cannot drift apart.

`read_pause_rows()` whitelists the columns it parses. `pause_end` and `label`
are never read on the inference path -- see `WHITELIST_INFERENCE`.
"""
from __future__ import annotations

import csv
import os

import numpy as np

import featurelib as fl

WHITELIST_INFERENCE = ("turn_id", "audio_file", "pause_index", "pause_start")


def read_pause_rows(data_dir: str, with_label: bool = False) -> list[dict]:
    """Parse labels.csv, taking only the causally-available columns.

    `pause_end` is deliberately not parsed: at the instant a pause begins a
    live agent cannot know when (or whether) it will end.
    """
    path = os.path.join(data_dir, "labels.csv")
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            item = {k: r[k] for k in WHITELIST_INFERENCE}
            item["pause_index"] = int(item["pause_index"])
            item["pause_start"] = float(item["pause_start"])
            if with_label:
                item["label"] = r["label"]
            rows.append(item)
    return rows


def build_matrix(data_dir: str, rows: list[dict]) -> np.ndarray:
    """Feature matrix for `rows`. Each wav is analysed once and reused."""
    cache: dict[str, fl.FrameData] = {}
    X = np.zeros((len(rows), fl.N_FEATURES), dtype=np.float32)
    for i, r in enumerate(rows):
        path = os.path.join(data_dir, r["audio_file"])
        fd = cache.get(path)
        if fd is None:
            x, sr = fl.load_wav(path)
            if sr != 16000:
                raise ValueError(f"expected 16 kHz, got {sr} for {path}")
            fd = cache[path] = fl.analyse_file(x, sr)
        X[i] = fl.extract(fd, r["pause_start"], r["pause_index"])
    return X


def read_train_durations(data_dir: str) -> np.ndarray:
    """Pause durations. TRAINING/DIAGNOSTICS ONLY -- never a feature.

    This is the one place `pause_end` is parsed. It is used to weight training
    samples by how much the scorer would actually charge for getting them
    wrong (see `cost_weights` in train_model.py). A training-time cost is not
    an input: nothing derived from this reaches `predict.py`, which cannot see
    the column at all.
    """
    out = []
    with open(os.path.join(data_dir, "labels.csv")) as f:
        for r in csv.DictReader(f):
            out.append(float(r["pause_end"]) - float(r["pause_start"]))
    return np.array(out, dtype=np.float64)


def load_split(data_dir: str, with_label: bool = True):
    """Convenience: rows, X, y, groups for one language folder."""
    rows = read_pause_rows(data_dir, with_label=with_label)
    X = build_matrix(data_dir, rows)
    y = (np.array([r["label"] for r in rows]) == "eot").astype(np.int64) \
        if with_label else None
    groups = np.array([r["turn_id"] for r in rows])
    return rows, X, y, groups
