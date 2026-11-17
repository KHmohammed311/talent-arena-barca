# EMI — Technical document

More than a Hack 2027, Phase 1: ball possession.

Repository: https://github.com/KHmohammed311/talent-arena-barca (code in `solution/`).

---

## 1. Summary

For each of the 199,319 frames of the tactical video we output `H`, `A`, `D` or `X`. The
pipeline answers two questions in order:

1. **Is the ball dead?** A two-state hidden Markov model (dead / alive), fitted on the match
   without any labels, followed by a refinement that moves each stoppage edge onto an observable
   event (ball leaving the pitch, ball vanishing, ball struck from rest).
2. **Which team has it?** Within each live stretch between two stoppages, a per-frame
   proximity score decoded with a "sticky" Viterbi decoder that charges a fixed cost for every
   change of team.

Inputs used: tactical tracking (ATD), Smart Tagging, broadcast tracking and broadcast homography
validity flags. No video is decoded, no model is trained on labels of the provided match, and
the public 5-minute ground truth is used only to check the pipeline.

Result on the public 5 minutes: **spellF1 0.640**, macroF1 0.908 (published baseline: 0.326 /
0.621; best of the four baselines on this window: 0.424 / 0.621).

## 2. How we framed the problem

spellF1 matches whole possessions by temporal overlap. That shaped every decision:

- a 1-second possession counts as much as a 40-second one;
- one wrong frame inside a possession splits it in two;
- a missed stoppage merges two possessions (`H H D D H H` is two, not one);
- timing matters more for short possessions than for long ones.

So we treated the task as **temporal segmentation**, not per-frame classification: first find
the stoppages, then find the hand-overs, and never let the output flicker.

## 3. What the data looks like

Measured on the unlabelled tracking of the match:

| fact | value | consequence |
|---|---|---|
| ball visible | 56.9% of frames, 59.3% of graded frames | the ball cannot be the only signal |
| ball gaps | ~150 in the graded frames, median ~11 s | the ball disappears for long stretches, mostly stoppages |
| player activity when the ball is unseen | median 1.0 m/s | players walk during stoppages |
| ball speed column | present on only 8.7% of visible frames | ball speed is computed from positions |
| track slots | 46 (24 home, 22 away) | slots are tracklets, not players: everything is per team |
| set-piece tags | 59 | fewer than the ~115 stoppages we find: tags are incomplete |
| set-piece tag length | 35 s (5 s for kick-offs) | a tag opens a few seconds before the restart and runs past it |

The key quantity is **activity**: the median speed of all tracked players in a frame, smoothed
over ±1 s. Open play is typically above 2 m/s; during a stoppage it sits around 1 m/s.

## 4. Assets: used and dropped

| asset | used for | status |
|---|---|---|
| tactical ATD (players + ball) | every feature | **used** |
| Smart Tagging | stoppage evidence, kick-offs, restart team | **used** |
| broadcast ATD + homography validity | synchronisation, replay detection | **used** |
| broadcast ball track | filling gaps of the tactical ball | **tried and dropped** |
| tactical homography | — | not needed (ATD is already in pitch coordinates) |
| both videos | — | not used |

Why the broadcast ball was dropped: it adds a ball on ~6,000 frames where the tactical one is
missing, but it is the less accurate of the two. Where the two tracks disagree by more than
2.5 m, the nearest-team rule is right 68% of the time with the tactical ball and 52% with the
broadcast ball. Using it lowered team accuracy.

## 5. Pipeline

```
tactical tracking ────┐
Smart Tagging ────────┼─► features ─► dead-ball HMM ─► boundary refinement ─┐
broadcast (synced) ───┘       │                                             ├─► labels ─► CSV
                              └────────► team evidence ─► sticky decoder ───┘
```

| file | role |
|---|---|
| `data.py` | load tracking, Smart Tagging and broadcast homography flags, indexed by video frame |
| `sync.py` | align the broadcast recording to the tactical timeline |
| `hmm.py` | two-state hidden Markov model, Baum–Welch fit and Viterbi decoding |
| `model.py` | features, dead-ball detection, boundary refinement, team decoding |
| `run.py` | entry point, writes the CSV |

### 5.1 Loading (`data.py`)

The ATD line format is `frame_id:player;player;...:ball`. `frame_id` counts from 1, the
submission from 0, so `row = frame_id − 1`. Coordinates are converted from 0–1 to metres
(105.3 × 68.0) so that distances are not distorted. Smart Tagging times convert as
`frame = round(seconds × 25)`. Parsed tracking is cached in `solution/.cache/` (git-ignored).

### 5.2 Broadcast synchronisation (`sync.py`)

The broadcast has its own frame counter, cuts and replays. Within a half,
`tactical_frame = broadcast_frame + offset`.

- **Offset search:** both recordings have a ball track in pitch coordinates. For windows of
  3,000 broadcast frames we score every candidate offset by the number of frames where the two
  balls are within 2.5 m. The right offset stands out sharply (~1,400 agreeing frames out of
  ~1,900). The median per half gives **4,014** frames (1st half) and **26,920** (2nd half).
- **Drift:** the offset is re-estimated in blocks of 1,500 frames within ±8 frames and
  median-smoothed.
- **Live or replay:** a broadcast frame is "live" if its homography is valid and the median
  distance from its players to the nearest tactical player is under 2 m (at least 6 players).
  61% of graded frames are live.

### 5.3 Features (`compute_features`)

| feature | definition |
|---|---|
| ball visible | tactical ball position present |
| ball speed | from positions 2 frames before and after, m/s |
| `dh`, `da` | distance from the ball to the nearest home / away player |
| outside | metres beyond the nearest pitch line (positive = out) |
| activity | median player speed, ±1 s (and a ±0.5 s version for timing restarts) |
| far-team distance | `max(dh, da)`: how far the second team is from the ball |

## 6. Dead-ball detection

### 6.1 The model

A stoppage is a state that lasts, so we use a two-state HMM (dead / alive) at 5 Hz. Each step
observes:

- three Gaussian variables: activity, log far-team distance, log ball speed;
- two binary variables: tactical ball unseen, ball outside the pitch (> 0.5 m).

The model is fitted with **Baum–Welch** on the unlabelled match. We only give a starting point
that names the states (dead ≈ players at 1.1 m/s, alive ≈ 2.3 m/s); after 40 iterations it
settles on its own values:

| state | activity | far-team distance | ball speed |
|---|---|---|---|
| dead | 1.01 m/s | ≈ 10 m | ≈ 2.2 m/s |
| alive | 2.39 m/s | ≈ 5.5 m | ≈ 6.2 m/s |

Each state persists ~99% of the time from one step to the next, so a stoppage must be supported
for several seconds to be accepted. Log-likelihoods are halved (`temper = 0.5`) because
consecutive 5 Hz steps are strongly correlated.

### 6.2 What failed first

1. **A single threshold on activity** (Otsu, cut at 2.1 m/s) labelled 60% of the match as dead:
   slow build-up play looks like a stoppage.
2. **A fully learned HMM** turned its two states into "ball seen / ball unseen" and missed every
   stoppage where the ball stays visible. Fix: the "ball unseen" probabilities are measured once
   on the match (55% when players walk, 10% when they run) and **frozen** during fitting.
3. **"Broadcast not live" as a per-frame feature** created false stoppages, because replays run
   on after play restarts (one 21-second replay covered 18 seconds of live play). Only the
   *start* of a replay is used (6.4).

### 6.3 Smart Tagging

- **Set pieces:** a tag opens a few seconds before the restart. From 2 s before to 2 s after
  each set-piece tag start, the dead state receives a large bonus. This recovers quick
  stoppages (fast throw-ins) where players never slow down. 57 of the 59 tag starts fall inside
  a predicted stoppage.
- **Kick-offs:** the kick is located inside the 5-second tag. Everything graded before the
  second-half kick-off is `D` (our detection lands within 8 frames of the published 113,742).
  After a goal, the 5 s before the kick are dead, and any stoppage running past the kick is
  ended at it.
- **Restart team:** corners, goal kicks and free kicks name the team taking them; that team gets
  a head start worth one switch over the 2 s after the restart (24 times in the match).
  Throw-ins are tagged `N/A` and get nothing.

### 6.4 Broadcast replays

A director cuts to a replay once play has stopped. The first frame of every stretch of at least
4 s where the broadcast is not live is treated like a set-piece tag start (bonus from 1 s before
to 2 s after). 112 replay starts in the match; most confirm stoppages already found.

### 6.5 Boundary refinement

Players react late: they keep running after the whistle and stand still until the restart.
So the edges of each HMM stoppage are moved onto observable events.

- **Start** (search ±12 s): the first frame the ball is seen more than 0.3 m outside the pitch;
  otherwise the first frame of a ball disappearance of at least 1 s; otherwise 1 s earlier.
- **End:** the last frame of the activity trough in the final 12 s, snapped to a **restart
  candidate** within 2 s: a ball struck from rest (< 1 m in the previous second, > 2.5 m in
  the next half second) or a ball reappearing after at least 1 s unseen. If a set-piece tag
  supports the stoppage, the first restart candidate 1–10 s after the tag start is used.
- Stoppages under 3 s without a tag or replay are dropped; live slivers under 1 s between two
  stoppages are absorbed.

## 7. Team in possession

### 7.1 Per-frame evidence

```
e = tanh((da − dh) / 1.5) × exp(−min(dh, da) / 4) × flight_gate(ball speed)
```

- **Who is nearer**, saturating: positive = home.
- **How close**: evidence fades when nobody is near the ball.
- **Flight gate**: 1 below 6 m/s, falling linearly to 0 at 12 m/s. A fast ball is in nobody's
  control, and for a ball in the air its ground position is wrong. On the public window the
  nearest-team rule is right 89% of the time below 3 m/s and 54% above 15 m/s.
- Ball unseen: evidence 0.

### 7.2 Sticky decoder

Each live stretch is decoded independently (a restart resets possession). We choose the
sequence of teams that maximises

```
sum of evidence agreeing with the chosen team − 20 × number of changes
```

with a two-state Viterbi. The other team needs roughly 1–2 s of clear evidence to take the
ball, which is how "a touch without control is not a change of possession" is implemented.
Frames without evidence inherit the team of their neighbours, so possession stays with the last
team while the ball is loose.

### 7.3 Post-processing rules

- **Free switches:** if a change of team falls inside a stretch with no evidence, it is placed
  in the middle when the ball was unseen, and at the end when the ball was in flight (the
  receiver gains possession on arrival, not when the ball was struck).
- **Final touch:** a spell under 2 s just before a stoppage is a player putting the ball out or
  fouling, not a controlled possession; it is given back to the previous team.

### 7.4 Touch detection: tried and dropped

We tested "possession goes to the last player who touched the ball", detecting touches as
sharp changes of ball velocity next to a player. It gave no gain on our data (see 8).

## 8. Training data and parameter choice

**No labels from the provided match were used for training or tuning.** The public 5 minutes
were used only to check frame alignment, file format and the meaning of the labels.

To choose the team-decoder settings we used other matches: the **Metrica Sports open sample
data**, https://github.com/metrica-sports/sample-data. The repository contains three anonymised
matches:

| match | format | use |
|---|---|---|
| Sample Game 1 | CSV tracking (home, away) + CSV events | **used** |
| Sample Game 2 | CSV tracking (home, away) + CSV events | **used** |
| Sample Game 3 | EPTS tracking + JSON events | not used (different format, not needed) |

**Labels** were derived by an algorithm from the published event data, using Metrica's event
definitions (`documentation/events-definitions.pdf` in that repository); nobody labelled
anything by hand:

- `D` where the tracking has no ball position (Metrica blanks the ball while it is out of play;
  every BALL OUT event lines up with the start of such a gap);
- otherwise the team of the most recent possession event (PASS, SHOT, BALL LOST, BALL OUT,
  SET PIECE, RECOVERY, FAULT RECEIVED). CHALLENGE and CARD are skipped.

This gives **760 possessions** over the two matches (median 4 s, a quarter to a third under
2 s).

**Making the test resemble our data.** The sample tracking is corrected; ours is automatic.
Measured on live play:

| | our match | clean sample |
|---|---|---|
| sharp ball velocity changes per minute | 3.7 | 29 |
| median distance ball to nearest player | 1.8 m | 0.3 m |

Averaging the sample ball over 1 s and adding a slowly varying 1 m position error reproduces
both figures. Settings were compared on that degraded version, with the true dead-ball mask
given so that only the team decision is tested.

**Results** (mean spellF1 over the two degraded matches):

| setting | cost 12 | cost 16 | cost 20 | cost 25 | cost 35 |
|---|---|---|---|---|---|
| no flight gate, no final-touch rule | 0.481 | 0.514 | 0.548 | 0.556 | 0.564 |
| flight gate 8–15 m/s | 0.550 | 0.566 | 0.580 | 0.584 | 0.577 |
| gate 8–15 + final-touch rule | 0.562 | 0.571 | 0.584 | 0.585 | 0.576 |
| **gate 6–12 + final-touch rule** | 0.586 | **0.598** | **0.596** | 0.590 | 0.576 |
| gate 10–20 + final-touch rule | 0.523 | 0.554 | 0.575 | 0.575 | 0.569 |

We kept **gate 6–12 m/s and switch cost 20**. Findings:

- the flight gate is worth about +0.03, the final-touch rule a little more;
- touch detection reaches 0.80 on the clean ball (against 0.69 for proximity) but is no better
  than proximity on the degraded ball: a one-second smoothing erases the velocity changes that
  mark a touch;
- even with a perfect dead-ball mask the degraded data tops out near 0.60, so short
  possessions are limited by the ball track, not by the decoder.

The comparison script is not part of the submission repository; it is available on request.

## 9. Possession rules implemented

| situation | our rule |
|---|---|
| ball goes out | dead from the first frame it is seen outside the pitch (or vanishes) |
| foul / offside / other stoppage | dead from when the players stop playing, refined to the ball event when visible |
| restart | live from the frame the ball is struck from rest or reappears |
| kick-off | located inside the Smart Tagging kick-off window; before it is dead |
| loose ball, ball in flight | possession stays with the last team |
| deflection or touch without control | no change unless the other team accumulates ~1–2 s of clear evidence |
| last touch before the ball goes out | under 2 s: not a possession, stays with the previous team |
| set piece with a named team | that team starts with an advantage at the restart |
| dead-ball ownership | not assigned (`D`), as the challenge asks |
| frames outside the two halves | `X` |

No `X` is written on graded frames: the decoder already carries the previous team through
frames with no evidence, which is what an `X` would mean for spellF1, and `X` costs macroF1.

## 10. Results on the public 5 minutes

| | ours | best baseline for that row |
|---|---|---|
| spellF1 | **0.640** | 0.424 |
| macroF1 | **0.908** | 0.621 |
| dead / alive F1 | 0.987 | ~0.90 |
| team accuracy while alive | 0.904 | 0.886 |
| possessions (truth 17) | 13 | 16 |
| possessions in the whole match | 242 | |

Bootstrap interval on spellF1: 0.45–0.81. With 17 possessions this is a check, not a forecast.

- All four stoppages are found, edges mostly within 1 s (one start 3 s late).
- Long possessions (7 s and more) are recovered well.
- Possessions of 0.8 s, 1.3 s and 2.8 s are missed; missing a 0.8 s possession also merges the
  two around it.

## 11. Limitations

1. **Short possessions.** Six of the 17 public possessions are under 4 s; we match two. The
   automatic ball track is too smooth to show the touches that start them (section 8).
   The fix would be a sharper ball track re-detected from the video.
2. **Proximity is not control.** A pass ending near an opponent, or a high ball whose ground
   position is wrong, gives misleading evidence.
3. **Slow restarts.** After a short goal kick nobody moves; the restart can be timed 1–2 s late.
4. **Stoppages with no visible cause and no tag** depend entirely on players slowing down.
5. **Live play with the ball unseen** (~9% of live frames) carries the neighbouring label.
6. **Parameters chosen on other matches** were chosen on corrected tracking degraded to look
   like ours, which matches on two statistics, not on everything.

With more time: ball re-detection from the tactical video, then touch-based possession, which
was clearly the better model on clean data.

## 12. Reproduction

1. Unpack the match bundle into `data/` (`tactical/` and `broadcast/`; the videos are not
   needed).
2. `pip install -r solution/requirements.txt` (numpy 2.3.5; pandas 2.3.3 for scoring only).
   Developed with Python 3.14.
3. `python solution/run.py` → `evaluation/results/submission.csv` (2–3 minutes; deterministic).
4. `python evaluation/validate_submission.py evaluation/results/submission.csv`
5. Optional: `python solution/run.py --no-broadcast` runs on the tactical recording alone.
