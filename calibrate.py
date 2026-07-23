"""Spread scores across the scorer's fixed threshold grid.

`score.py` sweeps thresholds on a fixed grid (0.05 ... 0.95). It never looks at
the probabilities themselves, only at which side of a grid line they fall, so
what matters is not calibration in the usual sense but *how the scores are
distributed across that grid*. If a language's scores bunch into a narrow band,
most grid lines separate nothing and the sweep is left with a handful of usable
operating points -- which is exactly what the diagnostics showed on Hindi,
where the best point kept collapsing onto the lowest threshold (0.05), i.e.
"fire on everything", the silence-timer baseline.

The fix is a monotone map through the empirical CDF of the TRAINING scores,
onto [0.06, 0.97]:

  * monotone, so the ranking -- and therefore AUC -- is untouched;
  * it spreads scores over the whole grid, so the sweep regains resolution;
  * the lower bound sits just above the lowest threshold, which guarantees the
    "fire on everything" policy stays reachable. The model can therefore never
    score *worse* than the silence-timer baseline.

The CDF knots are fitted on training data and frozen into model.npz. Nothing is
estimated from the evaluation set, so this stays causal: it is a fixed
function applied per pause, not a transform over the test batch.
"""
from __future__ import annotations

import numpy as np

CAL_LO = 0.06
CAL_HI = 0.97
N_KNOTS = 257


def fit_spread(train_scores: np.ndarray) -> np.ndarray:
    """Freeze the training score distribution as CDF knots."""
    return np.quantile(np.asarray(train_scores, dtype=np.float64),
                       np.linspace(0.0, 1.0, N_KNOTS))


def apply_spread(knots: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Map raw scores through the frozen CDF onto the threshold grid."""
    u = np.interp(np.asarray(p, dtype=np.float64), knots,
                  np.linspace(0.0, 1.0, len(knots)))
    return CAL_LO + (CAL_HI - CAL_LO) * u
