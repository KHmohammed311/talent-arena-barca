# Ball possession pipeline

Produces the `frame,label` submission for More than a Hack 2027, Phase 1, from the provided
match data. No training data, no ground truth and no video are used.

## Reproduce the submission

1. Unpack the match bundle into `data/` as described in `data/README.md`. The pipeline reads:
   - `data/tactical/BAR-ATH_tactical_atd.txt` and `.xml`
   - `data/tactical/BAR-ATH_tactical_smart_tagging.xml`
   - `data/broadcast/BAR-ATH_broadcast_atd.txt` and `.xml`
   - `data/broadcast/BAR-ATH_broadcast_homography.csv`
2. Install the dependencies:

   ```bash
   pip install -r solution/requirements.txt
   ```
3. Run:

   ```bash
   python solution/run.py
   ```

   This writes `evaluation/results/submission.csv` (199,319 rows). It takes two to three
   minutes the first time; the parsed tracking is cached in `solution/.cache/`.
4. Check and score it:

   ```bash
   python evaluation/validate_submission.py evaluation/results/submission.csv
   python evaluation/run_evaluation.py
   ```

The pipeline is deterministic: the same data gives the same file.

## Files

| file | role |
|---|---|
| `data.py` | load tracking, Smart Tagging and the broadcast homography flags |
| `sync.py` | align the broadcast recording to the tactical timeline |
| `hmm.py` | two-state hidden Markov model, fitted without labels |
| `model.py` | features, dead-ball detection, team decoding |
| `run.py` | entry point |
| `WALKTHROUGH.md` | how and why everything works |

## Data

The match data is not in this repository and must not be committed. `data/` and
`solution/.cache/` are excluded by `.gitignore`.
