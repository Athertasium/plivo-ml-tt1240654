"""Smoke test: does predict.py really run on a folder it has never seen?

Builds a temp folder holding a subset of turns whose labels.csv has had BOTH
`pause_end` and `label` removed, then runs predict.py against it. If inference
secretly depended on either column this fails loudly instead of silently
scoring well on data that still contains them.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile

SRC = sys.argv[1] if len(sys.argv) > 1 else "../eot_handout/eot_data/eot_data/hindi"
N_TURNS = 12

tmp = tempfile.mkdtemp(prefix="unseen_")
os.makedirs(os.path.join(tmp, "audio"))

rows = list(csv.DictReader(open(os.path.join(SRC, "labels.csv"))))
keep = sorted({r["turn_id"] for r in rows})[:N_TURNS]
subset = [r for r in rows if r["turn_id"] in keep]

with open(os.path.join(tmp, "labels.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["turn_id", "audio_file",
                                      "pause_index", "pause_start"])
    w.writeheader()
    for r in subset:
        w.writerow({k: r[k] for k in w.fieldnames})
        src = os.path.join(SRC, r["audio_file"])
        dst = os.path.join(tmp, r["audio_file"])
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

out = os.path.join(tmp, "pred.csv")
res = subprocess.run([sys.executable, "predict.py", "--data_dir", tmp,
                      "--out", out], capture_output=True, text=True)
print(res.stdout.strip() or res.stderr.strip())
if res.returncode != 0:
    sys.exit("FAIL: predict.py errored on a folder without pause_end/label")

got = list(csv.DictReader(open(out)))
assert len(got) == len(subset), f"expected {len(subset)} rows, got {len(got)}"
ps = [float(g["p_eot"]) for g in got]
assert all(0.0 <= p <= 1.0 for p in ps), "p_eot outside [0,1]"
print(f"PASS: {len(got)} predictions on {N_TURNS} unseen turns, "
      f"no pause_end/label present, p range "
      f"[{min(ps):.3f}, {max(ps):.3f}]")
shutil.rmtree(tmp)
