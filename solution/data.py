"""Load the provided assets into numpy arrays, indexed by tactical video frame.

Everything returned here is indexed so that row i is video frame i, i.e. submission row i.
The parsed tracking is cached under solution/.cache (gitignored) because streaming the raw
ATD takes about a minute.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(HERE, ".cache")
HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    """0 for home slots, 1 for away slots. First team in the metadata is home (EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    names = [team.findtext("Name") for team in root.iter("Team")]
    slots = [player.get("teamId") for player in root.iter("Player")]
    team = np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)
    return team, names


def _group(text):
    parts = text.split(",")
    if len(parts) < 2 or parts[0][:1] in ("N", "n", ""):
        return (np.nan, np.nan, np.nan)
    x, y = float(parts[0]), float(parts[1])
    s = float(parts[2]) if len(parts) > 2 and parts[2].strip()[:1] not in ("N", "n", "") else np.nan
    return (x, y, s)


def parse_atd(raw_path, n_frames, n_slots):
    """players[n_frames, n_slots, 3] (x, y, speed), ball[n_frames, 3]. Normalised 0-1, NaN = missing."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1  # frame_id counts from 1
            if not 0 <= row < n_frames:
                continue
            groups = parts[1].split(";")
            for slot in range(min(n_slots, len(groups))):
                g = groups[slot]
                if g[:1] != "N":
                    players[row, slot] = _group(g)
            if parts[2][:1] != "N":
                ball[row] = _group(parts[2])
    return players, ball


def load_homography_valid(game=GAME, data_dir=None, source="broadcast"):
    """Boolean array over that source's frames: True where a pitch transform exists."""
    path = require(["homography"], game, data_dir, source)[0]
    n = source_frames(source, game)
    valid = np.zeros(n, dtype=bool)
    with open(path) as handle:
        header = handle.readline().strip().split(",")
        i_frame, i_valid = header.index("frame"), header.index("valid")
        for line in handle:
            parts = line.rstrip("\n").split(",")
            frame = int(parts[i_frame])
            if 0 <= frame < n:
                valid[frame] = parts[i_valid].strip() in ("1", "1.0", "True", "true")
    return valid


def load_tracking(game=GAME, data_dir=None, use_cache=True, source="tactical"):
    """dict(players, ball, team, pitch, n_frames). Positions are converted to METRES.

    source="broadcast" loads the TV feed's tracking, which is on its OWN timeline (see sync.py).
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    n_frames = source_frames(source, game)
    team, names = team_of_each_track(meta_path)
    cache = os.path.join(CACHE_DIR, "{}_{}_atd.npz".format(game, source))

    if use_cache and os.path.isfile(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        z = np.load(cache)
        players, ball = z["players"], z["ball"]
    else:
        players, ball = parse_atd(raw_path, n_frames, len(team))
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache, players=players, ball=ball)

    length, width = GAMES[game]["pitch"]
    players = players.astype(np.float64)
    ball = ball.astype(np.float64)
    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return {"players": players, "ball": ball, "team": team, "team_names": names,
            "pitch": (length, width), "n_frames": n_frames, "fps": GAMES[game]["fps"],
            "halves": half_video_blocks(game)}


def load_smart_tagging(game=GAME, data_dir=None):
    """[{code, start_frame, end_frame, tags}] with frames on the tactical video timeline."""
    path = require(["smart_tagging"], game, data_dir)[0]
    fps = GAMES[game]["fps"]
    out = []
    for inst in ET.parse(path).iter("instance"):
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start_frame": int(round(float(inst.findtext("start")) * fps)),
            "end_frame": int(round(float(inst.findtext("end")) * fps)),
            "tags": {(lab.findtext("group") or "").strip(): (lab.findtext("text") or "").strip()
                     for lab in inst.findall("label")},
        })
    return out
