"""Put the broadcast recording on the tactical timeline.

The broadcast feed is a second recording of the same match with its own frame counter. During
live play it shows the same instants as the tactical camera, so within a half the two differ by
a (nearly) constant number of frames:   tactical_frame = broadcast_frame + offset.

The offset is found from the data: both recordings carry a ball track in pitch coordinates, and
at the right offset the two tracks coincide. Nothing here is hand-set per match.

What comes out, indexed by TACTICAL frame:
  ball      broadcast ball position in metres (NaN where unusable)
  live      True where the broadcast is verifiably showing the live match at that instant
  covered   True where a broadcast frame exists for that instant at all
A broadcast frame that is not "live" is a replay, a close-up or a crowd shot. Directors cut to
those mostly when the ball is dead, which makes ~live a dead-ball cue.
"""

import numpy as np

FPS = 25
AGREE_M = 2.5          # two ball positions closer than this "agree"
COARSE_WINDOW = 3000   # broadcast frames per coarse search window
COARSE_STEP = 5
LOCAL_BLOCK = 1500     # tactical frames per local refinement block
LOCAL_RANGE = 8        # +- frames searched around the half offset
LIVE_MATCH_M = 2.0     # median broadcast-player -> nearest tactical-player distance for "live"
LIVE_MIN_PLAYERS = 6


def _agreement(tb, bb, i0, width, d):
    """How many frames of broadcast window [i0, i0+width) have a ball within AGREE_M of the
    tactical ball at offset d."""
    j0 = i0 + d
    if j0 < 0 or j0 + width > len(tb) or i0 < 0 or i0 + width > len(bb):
        return 0
    a, b = bb[i0:i0 + width], tb[j0:j0 + width]
    ok = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
    if ok.sum() < 50:
        return 0
    return int((np.hypot(a[ok, 0] - b[ok, 0], a[ok, 1] - b[ok, 1]) < AGREE_M).sum())


def half_offsets(tactical, broadcast):
    """One offset per half: the best-agreeing offset of every coarse window, then the median of
    the windows that land in that half."""
    tb, bb = tactical["ball"][:, :2], broadcast["ball"][:, :2]
    lo, hi = -len(bb), len(tb)
    found = []
    for i0 in range(0, len(bb) - COARSE_WINDOW, COARSE_WINDOW * 2):
        best_d, best_s = None, 0
        for d in range(max(lo, -i0), min(hi, len(tb) - i0 - COARSE_WINDOW), COARSE_STEP):
            s = _agreement(tb, bb, i0, COARSE_WINDOW, d)
            if s > best_s:
                best_d, best_s = d, s
        if best_d is None or best_s < 200:
            continue
        for d in range(best_d - COARSE_STEP, best_d + COARSE_STEP + 1):
            s = _agreement(tb, bb, i0, COARSE_WINDOW, d)
            if s > best_s:
                best_d, best_s = d, s
        found.append((i0 + COARSE_WINDOW // 2 + best_d, best_d))
    out = {}
    for half, a, b in tactical["halves"]:
        ds = [d for centre, d in found if a <= centre <= b]
        if ds:
            out[half] = int(np.median(ds))
    return out


def align(tactical, broadcast, valid):
    """Arrays over tactical frames (see module docstring), plus the offsets used."""
    n = tactical["n_frames"]
    tb, bb = tactical["ball"][:, :2], broadcast["ball"][:, :2]
    offsets = half_offsets(tactical, broadcast)

    # per-frame offset: the half's offset, refined block by block (the two clocks drift by a
    # few frames over a half)
    src = np.full(n, -1, dtype=np.int64)  # broadcast frame for each tactical frame
    for half, a, b in tactical["halves"]:
        if half not in offsets:
            continue
        base = offsets[half]
        starts = list(range(a, b + 1, LOCAL_BLOCK))
        local = []
        for s in starts:
            width = min(LOCAL_BLOCK, b + 1 - s)
            best_d, best_s = base, 0
            for d in range(base - LOCAL_RANGE, base + LOCAL_RANGE + 1):
                sc = _agreement(tb, bb, s - d, width, d)
                if sc > best_s:
                    best_d, best_s = d, sc
            local.append(best_d if best_s >= 100 else np.nan)
        local = np.array(local, dtype=float)
        # fill blocks with too little ball overlap from their neighbours, then median-smooth
        idx = np.arange(len(local))
        good = ~np.isnan(local)
        local = np.interp(idx, idx[good], local[good]) if good.any() else np.full(len(local), base)
        smooth = np.array([np.median(local[max(0, i - 1):i + 2]) for i in idx])
        for s, d in zip(starts, np.round(smooth).astype(int)):
            e = min(s + LOCAL_BLOCK, b + 1)
            src[s:e] = np.arange(s, e) - d
    src[(src < 0) | (src >= broadcast["n_frames"])] = -1
    covered = src >= 0

    ball = np.full((n, 2), np.nan)
    live = np.zeros(n, dtype=bool)
    tp, bp = tactical["players"][..., :2], broadcast["players"][..., :2]
    rows = np.where(covered)[0]
    for c0 in range(0, len(rows), 4000):
        r = rows[c0:c0 + 4000]
        q = src[r]
        v = valid[q]
        a = bp[q]                       # [m, slots_b, 2]
        b = tp[r]                       # [m, slots_t, 2]
        d = np.sqrt(((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1))
        d = np.where(np.isnan(d), np.inf, d).min(2)        # nearest tactical player
        seen = ~np.isnan(a[..., 0])
        d = np.where(seen & np.isfinite(d), d, np.nan)
        enough = (~np.isnan(d)).sum(1) >= LIVE_MIN_PLAYERS
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(d, axis=1)
        is_live = v & enough & (np.nan_to_num(med, nan=np.inf) < LIVE_MATCH_M)
        live[r] = is_live
        ball[r[is_live]] = bb[q[is_live]]

    info = {"offsets": offsets, "covered": int(covered.sum()), "live": int(live.sum())}
    return {"ball": ball, "live": live, "covered": covered, "info": info}
