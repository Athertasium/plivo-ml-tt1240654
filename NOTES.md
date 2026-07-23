# NOTES

**Signal —** the model reads 21 causal prosodic features from the audio strictly
before each pause: terminal F0 slope and final pitch as a z-score against the
talker's own prefix distribution (statements fall to the bottom of a speaker's
range, continuations stay level), energy decay into the pause, final-syllable
lengthening, voiced-onset rate, spectral tilt, and — the one that came out of
error analysis — how *this* talker has already been pausing earlier in the same
turn, recovered from the energy gate rather than from annotations; plus the
slope of frame periodicity into the pause, recovered from the autocorrelation
peak the voicing gate was otherwise discarding. Two
near-perfect predictors are deliberately refused as causality violations:
pause duration (needs `pause_end`, which the agent cannot know when the pause
begins) and total file length (every eot pause here ends within 0.9 ms of EOF).

**Where it fails —** it is weakest exactly where it costs: AUC against holds
longer than 1.0 s was at chance (0.513) before I reweighted training toward
them, and it is still the binding constraint. Hindi does not beat its 850 ms
silence-timer baseline at all. Sweeping the scorer's whole grid rather than
reading its argmin locates the gap precisely: beating 850 ms needs
`D·f + 1.6·(1−f) < 0.85`, so at an 800 ms delay ≥94 of 100 turn-ends must fire
(we get 65%) but at a 500 ms delay only 68% must (we get 62%). **The real
shortfall is about six turn-ends at the 500 ms operating point, not thirty at
the 800 ms one.** That makes it a precision-at-the-top problem rather than a
ranking one, which is why gradient boosting raising pooled AUC by 4 points
moved the Hindi delay by zero. The residual errors are
concentrated in hesitant speakers who pause mid-phrase with level pitch and no
final lengthening, which is genuinely ambiguous from prosody alone — a human
listener resolves it lexically, from whether the clause is syntactically
complete.

**One more day —** I would attack that ambiguity where the headroom actually is:
a perfect ranker would score 100 ms against our 1228 ms, so nearly all the gap
is model quality rather than task structure. Concretely — train a small
1-D CNN or GRU directly on the causal log-mel prefix instead of 21 hand-designed
summary statistics, which is feasible on CPU at this data size and would let
the model learn turn-final contours I had to guess at; pool both languages with
turn-level augmentation to fight the 200-turn ceiling; and replace the fixed
1.5 s analysis window with a multi-scale one, since final lengthening lives at
200 ms and speaking-rate context lives at several seconds.
