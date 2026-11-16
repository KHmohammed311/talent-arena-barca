"""Possession model: per-frame evidence -> dead-ball intervals -> sticky team decoding.

No ground truth is read anywhere in this module. Every constant is either a physical prior
(metres, seconds) or estimated from the unlabelled tracking itself (the activity threshold).
"""

import numpy as np

HOME, AWAY, DEAD = 0, 1, 2
FPS = 25


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------
def runs(mask):
    """[(start, end)) of every True run."""
    m = np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def box_mean(x, half):
    """Centred moving average ignoring NaN."""
    ok = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    cnt = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / cnt, np.nan)


# --------------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------------
def _nearest_by_team(P, ball_xy, team):
    """Distance from the ball to the nearest home player and to the nearest away player."""
    dist = np.sqrt(((P[..., :2] - ball_xy[:, None, :]) ** 2).sum(-1))
    dist = np.where(np.isnan(dist), np.inf, dist)
    return dist[:, team == HOME].min(1), dist[:, team == AWAY].min(1)


def compute_features(trk, bc=None, fill=True):
    """Per-frame features. `bc` is the broadcast recording aligned to tactical frames (sync.align),
    or None to run on the tactical tracking alone.

    With the broadcast there are two ball sensors. The tactical one is the more reliable and is
    always preferred; the broadcast one only fills frames where the tactical ball is unseen
    (`fill`). The broadcast also yields "replay starts": the first frame of every stretch of at
    least REPLAY_MIN frames in which it is not showing live play.
    """
    P, team = trk["players"], trk["team"]
    L, W = trk["pitch"]
    n = trk["n_frames"]
    B_tac = trk["ball"][:, :2]
    bv_tac = ~np.isnan(B_tac[:, 0])
    B_bc = np.full((n, 2), np.nan)
    replay_starts = np.zeros(0, dtype=int)
    if bc is not None:
        if fill:
            B_bc = bc["ball"]
        away = bc["covered"] & ~bc["live"]
        replay_starts = np.array([s for s, e in runs(away) if e - s >= REPLAY_MIN], dtype=int)
    bv_bc = ~np.isnan(B_bc[:, 0])

    # fused ball: tactical where seen, broadcast otherwise
    B = np.where(bv_tac[:, None], B_tac, B_bc)
    bv = bv_tac | bv_bc
    from_bc = bv & ~bv_tac

    # ball speed from positions, m/s, central difference over +-2 frames; undefined across a
    # change of sensor, where the jump between the two tracks is not motion
    vb = np.full(n, np.nan)
    vb[2:-2] = np.sqrt(((B[4:] - B[:-4]) ** 2).sum(1)) / 4.0 * FPS
    same = np.ones(n, bool)
    same[2:-2] = from_bc[4:] == from_bc[:-4]
    vb[~same] = np.nan

    dh, da = _nearest_by_team(P, B, team)

    # metres outside the pitch (positive = out)
    outside = np.maximum.reduce([-B[:, 0], B[:, 0] - L, -B[:, 1], B[:, 1] - W])
    outside = np.where(bv, outside, np.nan)

    # collective activity: median tracked-player speed, smoothed over +-1 s
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        act_raw = np.nanmedian(P[..., 2], axis=1)
    n_tracked = (~np.isnan(P[..., 0])).sum(1)
    act_raw = np.where(n_tracked >= 8, act_raw, np.nan)
    act = box_mean(act_raw, FPS)

    return {"bv": bv, "bv_tac": bv_tac, "vb": vb, "dh": dh, "da": da,
            "outside": outside, "act": act, "act_fast": box_mean(act_raw, FPS // 2),
            "replay_starts": replay_starts, "n_tracked": n_tracked, "ball_xy": B}


# --------------------------------------------------------------------------------------------
# dead ball
# --------------------------------------------------------------------------------------------
HMM_STEP = 5  # the dead/alive model runs at 5 Hz
MIN_CORE = 3 * FPS       # a stoppage shorter than 3 s is not a stoppage
TAG_BEFORE = 2 * FPS     # frames before a set-piece tag start that are treated as dead
TAG_AFTER = 2 * FPS      # ... and after it
TAG_BONUS = 6.0          # log-evidence per 5 Hz step inside that window (decisive)
FLIGHT_SLOW = 6.0        # m/s: below this the ball is being carried or passed short
FLIGHT_FAST = 12.0       # m/s: above this it has been struck and is in flight
SWITCH_COST = 20.0       # evidence the other team must accumulate to take possession
LAST_TOUCH_MAX = 2 * FPS  # a spell this short that ends in a stoppage is a touch, not a possession
REPLAY_MIN = 4 * FPS     # the broadcast leaving live play for this long counts as a replay
REPLAY_BEFORE = 1 * FPS  # frames before a replay start that are treated as dead
REPLAY_AFTER = 2 * FPS   # ... and after it


def _hmm_blocks(f, a, b):
    """5 Hz observations for one half: Gaussian [activity, log far-team distance, log ball speed],
    Bernoulli [tactical ball unseen, ball outside the pitch]."""
    T = (b + 1 - a) // HMM_STEP

    def sl(x):
        return x[a:a + T * HMM_STEP].reshape(T, HMM_STEP)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xg = np.stack([np.nanmean(sl(f["act"]), 1), np.nanmean(sl(f["lfar"]), 1),
                       np.nanmean(sl(f["lvb"]), 1)], 1)
        miss = (sl(~f["bv_tac"]).mean(1) > 0.5).astype(float)
        unseen = sl(~f["bv"]).mean(1) > 0.5
        out = np.nanmax(sl(np.where(f["bv"], f["outside"], -99.0)), 1)
    outb = np.where(unseen, np.nan, (out > 0.5).astype(float))
    return xg, np.stack([miss, outb], 1)


def dead_intervals(f, halves, tags):
    """Boolean array, True where the ball is judged dead.

    Core: a two-state HMM fitted without labels (see hmm.py) on collective player activity, how
    far the second team is from the ball, ball speed, ball unseen and ball outside the pitch.
    Start refinement: play stops before the players do, so the start is moved to the frame the
    ball left the pitch or vanished when such an event is found nearby.
    End refinement: play restarts before the players speed up, so the end is moved to the last
    time the ball was struck from rest, or reappeared, near the end of the core.
    """
    from . import hmm
    n = len(f["act"])
    act = f["act"]
    graded = np.zeros(n, bool)
    for _, a, b in halves:
        graded[a:b + 1] = True

    far = np.maximum(f["dh"], f["da"])
    far[~np.isfinite(far)] = np.nan
    f["lfar"] = np.log(np.clip(far, 0.5, 60.0))
    f["lvb"] = np.log(np.clip(box_mean(f["vb"], FPS // 2), 0.2, 30.0))

    # How often the ball is unseen when the players walk vs when they run, measured on the
    # unlabelled match. Frozen during EM (see hmm.fit).
    walk, run = 1.1, 2.3  # m/s: initial state means for collective activity
    a_ok = graded & np.isfinite(act)
    lo_m, hi_m = a_ok & (act < 0.5 * (walk + run)), a_ok & (act >= 0.5 * (walk + run))
    p_miss = [float((~f["bv_tac"])[lo_m].mean()), float((~f["bv_tac"])[hi_m].mean())]

    seqs = [_hmm_blocks(f, a, b) for _, a, b in halves]
    init = dict(mu=[[walk, 2.7, 0.7], [run, 1.6, 1.8]], sd=[[0.3, 0.6, 0.8], [0.8, 0.6, 0.8]],
                pb=[[p_miss[0], 0.1], [p_miss[1], 0.01]], stay=[0.99, 0.99])
    params = hmm.fit(seqs, init, n_iter=40, temper=0.5, freeze_bern=(0,))

    # Smart Tagging set pieces. A set-piece tag opens a few seconds BEFORE its restart is taken,
    # so the frames around the tag start are dead ball. Fed to the decoder as extra evidence:
    # this recovers short stoppages (quick throw-ins) the players never slow down for.
    set_pieces = [ev for ev in tags if ev["code"] == "SET PIECES"]
    tag_starts = np.array(sorted(ev["start_frame"] for ev in set_pieces), dtype=int)

    dead = np.zeros(n, bool)
    restarts = []  # (frame, came_from_tag)
    # Broadcast replays. A director cuts to a replay once play has stopped, so the START of a
    # replay is dead ball. Its end is not: replays routinely run on after the restart.
    replay_starts = f["replay_starts"]
    anchors = np.array(sorted(set(tag_starts.tolist()) | set(replay_starts.tolist())), dtype=int)

    info = {"p_miss": np.round(p_miss, 3).tolist(), "mu": params["mu"].round(2).tolist(),
            "cores": 0, "short_cores_dropped": 0, "replay_starts": int(len(replay_starts))}
    for (_, a, b), (xg, xb) in zip(halves, seqs):
        extra = np.zeros((len(xg), 2))
        for ts in tag_starts:
            lo, hi = (ts - TAG_BEFORE - a) // HMM_STEP, (ts + TAG_AFTER - a) // HMM_STEP
            if hi > 0 and lo < len(xg):
                extra[max(lo, 0):hi, hmm.DEAD] += TAG_BONUS
        for rs in replay_starts:
            lo, hi = (rs - REPLAY_BEFORE - a) // HMM_STEP, (rs + REPLAY_AFTER - a) // HMM_STEP
            if hi > 0 and lo < len(xg):
                extra[max(lo, 0):hi, hmm.DEAD] = np.maximum(extra[max(lo, 0):hi, hmm.DEAD], TAG_BONUS)
        path = hmm.viterbi(xg, xb, params, extra=extra)
        core = np.zeros(b + 1 - a, bool)
        core[:len(path) * HMM_STEP] = np.repeat(path == hmm.DEAD, HMM_STEP)
        for s, e in runs(core):
            s, e = s + a, e + a
            tagged = tag_starts[(tag_starts >= e - 4 * FPS) & (tag_starts <= e)]
            if e - s < MIN_CORE and not len(anchors[(anchors >= s - FPS) & (anchors <= e)]):
                info["short_cores_dropped"] += 1
                continue
            info["cores"] += 1
            start = refine_start(f, s, a) if s > a else a
            end = None
            if len(tagged) and e < b + 1:
                # the restart follows the tag start: first strike / reappearance after it
                ts = int(tagged[-1])
                cands = [t for t in restart_candidates(f, ts + FPS, ts + 10 * FPS) if t > start + FPS]
                if cands:
                    end = cands[0]
            if end is None:
                end = refine_end(f, e, start, b + 1) if e < b + 1 else b + 1
            in_core = tag_starts[(tag_starts >= s - FPS) & (tag_starts <= e)]
            if len(in_core):
                # whatever the refinements say, the frames just before a tag start are dead
                start = min(start, max(a, int(in_core[0]) - TAG_BEFORE))
                end = max(end, min(b + 1, int(in_core[-1]) + FPS))
            if end - start >= 2 * FPS:
                dead[start:end] = True
                restarts.append(end)

    # Kick-offs. The tag is a short window around the kick itself, so the kick is located inside
    # it (where the players start moving) and the frames before it are dead: back to the start of
    # the half for a "Kick Off (Start)", back to the stoppage already found for one after a goal.
    for ev in set_pieces:
        kind = ev["tags"].get("Type", "").lower()
        if not kind.startswith("kick off"):
            continue
        ts, te = ev["start_frame"], ev["end_frame"]
        for _, a, b in halves:
            if not (a < te and ts <= b):
                continue
            ko = refine_end(f, min(te, b + 1), max(ts - FPS, a - FPS), b + 1, back=te - ts)
            ko = int(min(max(ko, a), b + 1))
            if kind == "kick off (start)":
                dead[a:ko] = True
            else:
                dead[max(a, ts - 5 * FPS):ko] = True
            # the kick-off IS the restart: a stoppage the model lets run on past it (teams often
            # restart at walking pace) ends here instead
            stop = min(te, b + 1)
            while stop < b + 1 and dead[stop] and stop - ko < 30 * FPS:
                stop += 1
            dead[ko:stop] = False
            restarts = [r for r in restarts if not ts - 5 * FPS <= r <= te + 20 * FPS] + [ko]
    # a live sliver under a second between two stoppages is an artefact of the refinements
    for _, a, b in halves:
        for s, e in runs(~dead[a:b + 1]):
            if e - s < FPS and s > 0 and e < b + 1 - a:
                dead[a + s:a + e] = True
    info["restarts"] = sorted(r for r in restarts if r < n and not dead[r])
    return dead, info


def refine_start(f, s, half_start, window=12 * FPS, lag=FPS):
    """Move a stoppage start onto the event that caused it, if one is visible nearby."""
    lo, hi = max(half_start, s - window), min(len(f["bv"]) - 1, s + window)
    out, bv = f["outside"], f["bv"]
    idx = np.arange(lo, hi)
    # 1. the ball crossed a line: first frame of the out-of-pitch episode closest to s
    is_out = bv[lo:hi] & (np.nan_to_num(out[lo:hi], nan=-99.0) > 0.3)
    eps = runs(is_out)
    if eps:
        first = [idx[a] for a, _ in eps]
        return int(min(first, key=lambda t: abs(t - s)))
    # 2. the ball vanished (kicked into the stands, picked up): first unseen frame closest to s
    gaps = [idx[a] for a, b in runs(~bv[lo:hi]) if b - a >= FPS and a > 0]
    if gaps:
        return int(min(gaps, key=lambda t: abs(t - s)))
    return max(half_start, s - lag)


def restart_candidates(f, lo, hi, rest_m=1.0, go_m=2.5):
    """Frames where the ball is struck from rest (net displacement under `rest_m` metres over the
    previous second, over `go_m` in the next half second), or reappears after being unseen."""
    B, bv = f["ball_xy"], f["bv"]
    out = []
    for t in range(max(lo, FPS), min(hi, len(bv) - FPS)):
        if not bv[t]:
            continue
        if not bv[t - 1] and not bv[max(0, t - FPS):t].any():
            out.append(t)  # reappearance
            continue
        if bv[t - FPS] and bv[t + FPS // 2]:
            before = np.hypot(*(B[t] - B[t - FPS]))
            after = np.hypot(*(B[t + FPS // 2] - B[t]))
            if before < rest_m and after > go_m:
                out.append(t)
    return out


def refine_end(f, e, start, half_end, back=12 * FPS, snap=2 * FPS):
    """Move a stoppage end onto the restart.

    Players stand still waiting for a restart and start moving when it is taken, so the restart
    is where collective activity leaves its trough near the end of the stoppage. If the ball is
    seen being struck from rest, or reappearing, within `snap` frames of that, use that frame.
    """
    lo = max(start + FPS, e - back)
    hi = min(half_end, e)
    if hi - lo < 3:
        return hi
    act = f["act_fast"][lo:hi]
    if np.isfinite(act).any():
        floor = np.nanmin(act)
        onset = lo + int(np.where(np.nan_to_num(act, nan=np.inf) <= floor + 0.1)[0][-1])
    else:
        onset = hi - FPS
    cands = restart_candidates(f, onset - snap, onset + snap)
    cands = [t for i, t in enumerate(cands) if i == 0 or t - cands[i - 1] > 1]  # one per strike
    cands = [t for t in cands if start + FPS <= t <= half_end]
    if cands:
        return int(min(cands, key=lambda t: abs(t - onset)))
    return int(min(max(onset, start + 1), half_end))


# --------------------------------------------------------------------------------------------
# team in possession
# --------------------------------------------------------------------------------------------
def team_evidence(f, margin=1.5, reach=4.0, flight_gate=True):
    """Signed per-frame evidence, positive = home.

    Three factors: which team is nearer and by how much; how close the nearest player is; and,
    with `flight_gate`, how fast the ball is travelling. A ball moving faster than FLIGHT_FAST
    has just been struck and is in nobody's control, so who it happens to be passing says nothing
    (and for a ball in the air the tracked ground position is wrong anyway). The weight falls
    linearly from 1 at FLIGHT_SLOW to 0 at FLIGHT_FAST. Zero where the ball is unseen: those
    frames keep whatever the neighbours say."""
    dh, da = f["dh"], f["da"]
    near = np.minimum(dh, da)
    with np.errstate(invalid="ignore"):
        e = np.tanh((da - dh) / margin) * np.exp(-near / reach)
    if flight_gate:
        vb = np.nan_to_num(f["vb"], nan=0.0)
        e = e * np.clip((FLIGHT_FAST - vb) / (FLIGHT_FAST - FLIGHT_SLOW), 0.0, 1.0)
    e[~f["bv"]] = 0.0
    e[~np.isfinite(e)] = 0.0
    return e


def viterbi_two_state(e, switch_cost):
    """argmax over sequences s_t in {+1 (home), -1 (away)} of sum(s_t * e_t) - cost * #switches."""
    n = len(e)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    back = np.zeros((n, 2), dtype=np.int8)
    sh, sa = e[0], -e[0]
    for t in range(1, n):
        if sa - switch_cost > sh:
            back[t, 0] = 1
            nh = sa - switch_cost
        else:
            nh = sh
        if sh - switch_cost > sa:
            back[t, 1] = 0
            na = sh - switch_cost
        else:
            back[t, 1] = 1
            na = sa
        sh, sa = nh + e[t], na - e[t]
    out = np.zeros(n, dtype=np.int8)
    s = 0 if sh >= sa else 1
    for t in range(n - 1, -1, -1):
        out[t] = s
        s = back[t, s]
    return out


def decode(trk, tags, bc=None, switch_cost=SWITCH_COST, fill=False, flight_gate=True, absorb_last_touch=True):
    """Full pipeline -> array of 'H' / 'A' / 'D' / 'X' over every video frame.

    `bc` is the broadcast recording aligned to the tactical timeline (sync.align), or None."""
    f = compute_features(trk, bc, fill=fill)
    n = trk["n_frames"]
    dead, info = dead_intervals(f, trk["halves"], tags)
    e = team_evidence(f, flight_gate=flight_gate)

    # Smart Tagging names the team taking a corner, goal kick or free kick. That team has the
    # ball at the restart, so it gets a head start (worth one switch) over the next two seconds.
    home_name = trk["team_names"][0].strip().lower()
    restarts = np.array(info.pop("restarts"), dtype=int)
    info["tag_team_priors"] = 0
    for ev in tags:
        kind = ev["tags"].get("Type", "").lower()
        team = ev["tags"].get("Team", "").strip().lower()
        if ev["code"] != "SET PIECES" or kind not in ("corner kick", "goal kick", "free kick"):
            continue
        if team in ("", "n/a") or not len(restarts):
            continue
        after = restarts[(restarts >= ev["start_frame"] - FPS) & (restarts <= ev["start_frame"] + 12 * FPS)]
        if len(after):
            r = int(after[0])
            e[r:r + 2 * FPS] += (1.0 if team == home_name else -1.0) * switch_cost / (2 * FPS)
            info["tag_team_priors"] += 1

    labels = np.full(n, "X", dtype="<U1")
    for _, a, b in trk["halves"]:
        seg_dead = dead[a:b + 1]
        lab = np.full(b + 1 - a, "D", dtype="<U1")
        # each live stretch between stoppages is decoded on its own: a restart resets possession
        for s, t in runs(~seg_dead):
            ev_seg = e[a + s:a + t]
            st = viterbi_two_state(ev_seg, switch_cost)
            st = place_free_switches(st, ev_seg, f["bv"][a + s:a + t])
            if absorb_last_touch and t < len(seg_dead):
                st = absorb_final_touch(st)
            lab[s:t] = np.where(st == HOME, "H", "A")
        labels[a:b + 1] = lab
    return labels, f, info


def place_free_switches(states, e, seen, min_len=10):
    """A switch decided inside a stretch with no evidence could sit anywhere in it at equal cost.

    If the ball was unseen there, put the switch in the middle: the least-wrong guess for both
    possessions. If the ball was seen (it was in flight), put it at the end: the receiving team
    gains possession when the ball arrives, not when it was struck."""
    states = states.copy()
    for s, t in runs(e == 0):
        if t - s < min_len or s == 0 or t == len(states):
            continue
        before, after = states[s - 1], states[t]
        if before != after:
            cut = (s + t) // 2 if seen[s:t].mean() < 0.5 else t
            states[s:cut] = before
            states[cut:t] = after
        else:
            states[s:t] = before
    return states


def absorb_final_touch(states):
    """The last spell before a stoppage, if it lasts under LAST_TOUCH_MAX frames, is a player
    putting the ball out or committing the foul, not a possession: by the challenge's definition
    a touch without control does not change possession. It is given back to the previous team."""
    states = states.copy()
    change = np.where(states[1:] != states[:-1])[0] + 1
    if len(change) and len(states) - change[-1] < LAST_TOUCH_MAX:
        states[change[-1]:] = states[change[-1] - 1]
    return states
