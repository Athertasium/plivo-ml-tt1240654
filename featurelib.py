"""Causal prosodic features for end-of-turn detection.

CAUSALITY CONTRACT
------------------
For a pause at time `pause_start`, every feature is computed from
`audio[0 : pause_start]` only. Two rules enforce this:

1. Frame-level arrays (energy / F0 / periodicity / spectral tilt) are computed
   ONCE per file for speed, but every per-pause call slices them to
   `n_causal_frames`
   -- the number of frames lying *entirely* before `pause_start`. A frame is
   kept only if `i*hop + frame_len <= pause_start*sr`, so no frame ever peeks
   across the boundary.
2. Every normalisation statistic (noise floor, median pitch, mean voiced-run
   length, ...) is computed from that same prefix slice. No whole-file
   statistics are used anywhere -- those would smuggle future audio into the
   feature via the normaliser.

Deliberately NOT used, though both are in labels.csv and both are near-perfect
predictors on this data:
  * `pause_end`  -> pause duration. Future information (we cannot know how
    long a pause will last at the moment it begins). On this data hold pauses
    average 0.62 s vs 1.34 s for eot in English, so it leaks badly.
  * total file length -> "is there audio after this pause". Every eot pause in
    both languages ends within 0.9 ms of EOF, making this a 100% accurate and
    100% useless predictor.
`predict.py` never even reads those columns; see the assertion there.

Frames are 40 ms / 10 ms hop throughout. One unified framing keeps every
per-frame array index-aligned, which removes a whole class of off-by-one bugs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf

FRAME_MS = 40
HOP_MS = 10
F0_MIN = 60.0
F0_MAX = 400.0
VOICING_THRESH = 0.30
OCTAVE_TOL = 0.85

# analysis windows measured backwards from the pause, in frames (10 ms each)
W_SHORT = 20    # 200 ms
W_MID = 30      # 300 ms
W_LONG = 50     # 500 ms
W_CTX = 150     # 1.5 s


def load_wav(path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def frame_signal(x: np.ndarray, sr: int) -> np.ndarray:
    """Frames of the whole signal, shape (n_frames, frame_len)."""
    fl = int(sr * FRAME_MS / 1000)
    hp = int(sr * HOP_MS / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def n_causal_frames(pause_start: float, sr: int, n_total: int) -> int:
    """How many frames lie entirely before `pause_start`.

    Frame i spans samples [i*hop, i*hop + fl). It is causal iff
    i*hop + fl <= pause_start*sr, i.e. i <= (pause_start*sr - fl)/hop.
    """
    fl = int(sr * FRAME_MS / 1000)
    hp = int(sr * HOP_MS / 1000)
    limit = int(pause_start * sr)
    if limit < fl:
        return 0
    return int(min(n_total, (limit - fl) // hp + 1))


@dataclass(frozen=True)
class FrameData:
    """Per-frame acoustic arrays for one file. All arrays share length n."""
    sr: int
    rms_db: np.ndarray     # short-time energy, dB
    f0: np.ndarray         # Hz, 0.0 where unvoiced
    tilt_db: np.ndarray    # low-band / high-band energy ratio, dB
    vstr: np.ndarray       # autocorrelation peak ratio in [0,1]: periodicity

    def __len__(self) -> int:
        return len(self.rms_db)


def analyse_file(x: np.ndarray, sr: int) -> FrameData:
    """Frame-level analysis of a whole file (sliced causally by callers).

    A single FFT per frame yields the power spectrum, which is reused for both
    the autocorrelation pitch track (via Wiener-Khinchin) and spectral tilt.
    This replaces the starter's O(n^2) `np.correlate` per frame and makes the
    feature pass fast enough to iterate on.
    """
    fr = frame_signal(x, sr)
    n, fl = fr.shape
    if n == 0:
        z = np.zeros(0, dtype=np.float32)
        return FrameData(sr=sr, rms_db=z, f0=z.copy(), tilt_db=z.copy(),
                         vstr=z.copy())

    rms = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)
    rms_db = (20 * np.log10(rms + 1e-12)).astype(np.float32)

    centred = fr - fr.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * fl)))
    spec = np.fft.rfft(centred, n=nfft, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2)

    # --- pitch: autocorrelation = IFFT of power spectrum ---
    # This is the BIASED autocorrelation (no 1/(fl-lag) correction), kept
    # deliberately. It tapers with lag and so leans towards higher F0. Two
    # "principled" alternatives were implemented and measured, and both made
    # the pitch FEATURES less discriminative even though their Hz histograms
    # looked more physical -- see RUNLOG Run 2. Plausible reason: the taper
    # suppresses spurious long-lag peaks in the low-energy frames right before
    # a pause, which is exactly where these features are read, and the derived
    # features are all speaker-relative (slope, z-score against the talker's
    # own median), so a consistent bias cancels out anyway.
    ac = np.fft.irfft(power, n=nfft, axis=1)[:, :fl]
    ac0 = ac[:, 0].copy()
    lo = int(sr / F0_MAX)
    hi = min(int(sr / F0_MIN), fl - 1)
    f0 = np.zeros(n, dtype=np.float32)
    vstr = np.zeros(n, dtype=np.float32)
    if hi > lo:
        band = ac[:, lo:hi]
        lag = lo + np.argmax(band, axis=1)
        peak = ac[np.arange(n), lag]
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(ac0 > 1e-12, peak / ac0, 0.0)
        loud = np.max(np.abs(fr), axis=1) >= 1e-4
        ok = loud & (norm >= VOICING_THRESH) & (lag > 0)
        f0[ok] = (sr / lag[ok]).astype(np.float32)
        # `norm` is how periodic the frame is; above it is spent as a yes/no
        # voicing gate and then discarded. Keep the continuous value too --
        # it is a different signal from f0 (simply absent once the gate fails)
        # and from tilt_db (spectral balance, not periodicity).
        # Silent frames get 0 rather than the meaningless ratio of noise.
        #
        # NB the fitted sign is the opposite of the creak/devoicing hypothesis
        # this was added on: `vstr_slope` enters at +0.133, so periodicity
        # RISES into a turn-final pause. Best reading is that it is picking up
        # final lengthening -- a turn ends on a sustained, cleanly voiced
        # vowel, whereas a mid-turn hesitation tends to be cut off on a
        # consonant or trail into breath. Recorded as measured; the creak
        # story is not what the data shows.
        vstr = np.clip(np.nan_to_num(np.where(loud, norm, 0.0)),
                       0.0, 1.0).astype(np.float32)

    # --- spectral tilt: 80-1000 Hz vs 1000-4000 Hz ---
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    lo_band = (freqs >= 80) & (freqs < 1000)
    hi_band = (freqs >= 1000) & (freqs < 4000)
    e_lo = power[:, lo_band].sum(axis=1) + 1e-12
    e_hi = power[:, hi_band].sum(axis=1) + 1e-12
    tilt_db = (10 * np.log10(e_lo / e_hi)).astype(np.float32)

    return FrameData(sr=sr, rms_db=rms_db, f0=f0, tilt_db=tilt_db, vstr=vstr)


# ---------------------------------------------------------------- helpers

def _slope(y: np.ndarray) -> float:
    """Least-squares slope per frame-step, scaled to units/second."""
    if len(y) < 3:
        return 0.0
    t = np.arange(len(y), dtype=np.float64)
    t -= t.mean()
    denom = (t * t).sum()
    if denom <= 0:
        return 0.0
    return float((t * (y - y.mean())).sum() / denom * (1000.0 / HOP_MS))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end_exclusive) index pairs."""
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def _safe(v: float) -> float:
    return float(v) if np.isfinite(v) else 0.0


FEATURE_NAMES = [
    # --- pitch (statements fall; continuations stay level or rise) ---
    "f0_slope_end",       # log-F0 slope over the final voiced stretch
    "f0_last_z",          # final pitch vs speaker's own prefix pitch median
    "f0_range_end",       # pitch movement in the last 500 ms
    "voiced_frac_end",    # voiced fraction in the last 500 ms
    # --- energy (turn ends trail off) ---
    "en_slope_end",       # energy slope into the pause
    "en_last_rel",        # final energy vs speaker's own speech median
    "en_drop_ratio",      # last 200 ms vs preceding 300 ms
    # --- rhythm / final lengthening ---
    "final_run_dur",      # duration of the last voiced run
    "final_run_rel",      # that vs mean voiced-run length in the prefix
    "rate_end",           # voiced-onset rate in the last 1.5 s
    "rate_rel",           # rate_end vs rate over the whole prefix
    # --- voice quality (creak/breathiness at turn end) ---
    "tilt_end",           # spectral tilt in the last 200 ms
    "tilt_end_rel",       # that vs prefix speech median
    # --- causal turn context (a live agent knows all of these) ---
    "log_pause_start",    # how long the user has been talking
    "n_prior_pauses",     # pauses already seen in this turn (== pause_index)
    "cur_seg_dur",        # length of the speech run just uttered
    "speech_frac",        # fraction of the prefix that was speech
    # --- this speaker's own pausing habit, measured from the prefix audio.
    # A talker who has already produced several long mid-turn silences makes
    # any given silence weaker evidence of a finished turn. Derived from the
    # energy gate, not from earlier rows' pause_end, so it stays self-contained
    # and needs no annotation at inference time.
    "prior_sil_rate",     # prior within-turn silences per second
    "prior_sil_mean",     # their mean duration
    "prior_sil_max",      # their longest
    # --- periodicity (continuous, from the autocorrelation peak the voicing
    # gate throws away). Appended rather than filed under "voice quality"
    # above on purpose: `extract` writes by position, so appending keeps every
    # existing index stable and makes this a strictly additive change.
    # Only the SLOPE survived. Level versions (mean periodicity in the last
    # 200 ms, and that same mean minus the speaker's own speech median) were
    # both implemented and measured: 0.449 / 0.454 AUC on the dangerous long
    # holds, i.e. chance, and carrying them cost 30 ms of English OOF delay.
    # The rate of change is the cue; the absolute level is dominated by voice
    # and channel, and the speaker-median reference did not remove that.
    "vstr_slope",         # periodicity slope into the pause (see sign note)
]
N_FEATURES = len(FEATURE_NAMES)


def extract(fd: FrameData, pause_start: float, pause_index: int) -> np.ndarray:
    """Feature vector for one pause, from prefix audio only.

    Besides the audio prefix this uses just `pause_start` and `pause_index`,
    both of which a live agent knows the instant the pause begins
    (`pause_index` is simply how many pauses have already occurred). The
    duration of the current speech segment is recovered from the energy gate
    rather than from the previous row's `pause_end`, so no column carrying
    future information is ever touched.
    """
    n = n_causal_frames(pause_start, fd.sr, len(fd))
    f = np.zeros(N_FEATURES, dtype=np.float32)
    if n < 10:
        return f

    rms = fd.rms_db[:n].astype(np.float64)
    f0 = fd.f0[:n].astype(np.float64)
    tilt = fd.tilt_db[:n].astype(np.float64)
    vstr = fd.vstr[:n].astype(np.float64)

    # speech/silence gate, adapted to this speaker+channel from the prefix only
    floor = np.percentile(rms, 10)
    peak = np.percentile(rms, 95)
    thresh = floor + 0.35 * max(peak - floor, 1e-6)
    speech = rms > thresh
    voiced = f0 > 0

    end = slice(max(0, n - W_SHORT), n)
    mid = slice(max(0, n - W_MID), n)
    lng = slice(max(0, n - W_LONG), n)
    ctx = slice(max(0, n - W_CTX), n)

    # ---- pitch ----
    lf0 = np.log(np.maximum(f0, 1e-6))
    v_idx = np.flatnonzero(voiced)
    if len(v_idx) >= 3:
        prefix_med = np.median(lf0[v_idx])
        prefix_std = np.std(lf0[v_idx]) + 1e-6
        # final voiced stretch: voiced frames within the last 300 ms
        tail_v = v_idx[v_idx >= n - W_MID]
        if len(tail_v) >= 3:
            f[0] = _safe(_slope(lf0[tail_v]))
        last_v = v_idx[-3:]
        f[1] = _safe((lf0[last_v].mean() - prefix_med) / prefix_std)
        long_v = v_idx[v_idx >= n - W_LONG]
        if len(long_v) >= 2:
            f[2] = _safe(lf0[long_v].max() - lf0[long_v].min())
    f[3] = _safe(voiced[lng].mean())

    # ---- energy ----
    f[4] = _safe(_slope(rms[mid]))
    if speech.any():
        f[5] = _safe(rms[end].mean() - np.median(rms[speech]))
    prev = rms[max(0, n - W_MID - W_SHORT):max(0, n - W_SHORT)]
    if len(prev) >= 3:
        f[6] = _safe(rms[end].mean() - prev.mean())

    # ---- rhythm / final lengthening ----
    v_runs = _runs(voiced)
    if v_runs:
        lens = np.array([b - a for a, b in v_runs], dtype=np.float64)
        f[7] = _safe(lens[-1] * HOP_MS / 1000.0)
        f[8] = _safe(lens[-1] / (lens.mean() + 1e-6))
    ctx_onsets = sum(1 for a, _ in v_runs if a >= n - W_CTX)
    ctx_dur = min(n, W_CTX) * HOP_MS / 1000.0
    f[9] = _safe(ctx_onsets / max(ctx_dur, 1e-6))
    total_dur = n * HOP_MS / 1000.0
    rate_all = len(v_runs) / max(total_dur, 1e-6)
    f[10] = _safe(f[9] / (rate_all + 1e-6))

    # ---- voice quality ----
    f[11] = _safe(tilt[end].mean())
    if speech.any():
        f[12] = _safe(tilt[end].mean() - np.median(tilt[speech]))

    # ---- causal turn context ----
    f[13] = _safe(np.log1p(pause_start))
    f[14] = _safe(pause_index)
    s_runs = _runs(speech)
    f[15] = _safe(min((s_runs[-1][1] - s_runs[-1][0]) * HOP_MS / 1000.0, 30.0)
                  if s_runs else 0.0)
    f[16] = _safe(speech.mean())

    # ---- this speaker's pausing habit, from the prefix only ----
    # Keep runs that start after speech began (a > 0, so leading silence is
    # not a "pause") and that end before the prefix does (b < n, so the
    # ramp-down into the *current* pause is not counted as a prior one).
    # 10 frames = the 100 ms floor the annotation itself uses.
    sil_runs = [(a, b) for a, b in _runs(~speech)
                if a > 0 and b < n and (b - a) >= 10]
    if sil_runs:
        durs = np.array([(b - a) * HOP_MS / 1000.0 for a, b in sil_runs])
        f[17] = _safe(len(durs) / max(total_dur, 1e-6))
        f[18] = _safe(durs.mean())
        f[19] = _safe(durs.max())

    # ---- periodicity into the pause ----
    # A slope, not a level: it is already speaker-relative by construction,
    # which the measured level variants were not (see FEATURE_NAMES).
    f[20] = _safe(_slope(vstr[mid]))

    return f
