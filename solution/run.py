"""Regenerate the submission from the provided data.

    python solution/run.py                       # writes evaluation/results/submission.csv
    python solution/run.py --name my_team        # writes evaluation/results/my_team.csv
    python solution/run.py --no-broadcast        # tactical recording only
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from solution import sync  # noqa: E402
from solution.data import load_homography_valid, load_smart_tagging, load_tracking  # noqa: E402
from solution.model import decode, runs  # noqa: E402


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def build(use_broadcast=True, **kwargs):
    """Labels for every video frame, plus diagnostics."""
    trk = load_tracking()
    tags = load_smart_tagging()
    bc = None
    if use_broadcast:
        broadcast = load_tracking(source="broadcast")
        bc = sync.align(trk, broadcast, load_homography_valid(source="broadcast"))
    labels, _, info = decode(trk, tags, bc, **kwargs)
    if bc is not None:
        info["sync"] = bc["info"]
    return labels, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="submission")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    parser.add_argument("--no-broadcast", action="store_true")
    args = parser.parse_args()

    labels, info = build(use_broadcast=not args.no_broadcast)

    path = os.path.join(args.out, args.name + ".csv")
    write_submission(path, labels)
    counts = {k: int((labels == k).sum()) for k in "HADX"}
    n_poss = len(runs(labels == "H")) + len(runs(labels == "A"))
    print("info:", info)
    print("labels:", counts, " possessions:", n_poss, " dead stretches:", len(runs(labels == "D")))
    print("wrote", path)


if __name__ == "__main__":
    main()
