#!/usr/bin/env bash
# PR-7 step 3: pairA clean-control write-path twin run (PR7_DESIGN §5/§13.2).
#
# The smallest governed-vs-baseline twin: the `clean_control` cell (pairA, the
# `clean` arm) run twice over the SAME input stream — `--govern none` (the
# ungoverned baseline) and `--govern annotate` (the null-action floor) — read
# out with the FROZEN shadow-governance scorer. Only the `--govern` flag differs
# (§5.3). annotate must cost EXACTLY nothing: every scored artifact stays
# byte-identical to the baseline, with only the summary's `govern` provenance
# block added (§4). The expected verdict is a zero-delta pass.
#
# NOT a governance experiment beyond the null-action floor: no quarantine, no
# refuse, no trust action, no static geometry gate, no retrieval change, no new
# observable. Same fixed protocol as the committed pair-A clean arms
# (pr3c_run_matrix.sh): COMMON flags, 3 seeds, --arm clean, classes 0,8,19,33,
# attractor 71. Geometry is provenance only, never an admission gate (PR-5
# step 1 closed static geometry as a safety predictor). The two engine files
# stay byte-frozen (PR7_DESIGN §1).
#
# Run from the repo root on gentoo with the venv active. Outputs land in
# results/issue_failure_mode_blindness/pr7/twin/clean_control/{none,annotate}/.
# After the run, copy that dir back to the canonical Darwin host and re-verify
# there: python benchmarks/pr7_twin_delta.py --govern annotate ... + pytest.
set -euo pipefail

ROOT=results/issue_failure_mode_blindness/pr7/twin/clean_control
COMMON="--vision --epochs 12 --supersede-epoch 6"
PAIR_A="0,8,19,33"; ATTR_A=71

for gov in none annotate; do
  OUT="$ROOT/$gov"
  mkdir -p "$OUT"
  for s in 0 1 2; do
    stem="clean_s${s}_pairA"
    echo "=== govern=$gov $stem"
    python benchmarks/failure_mode_probe.py $COMMON \
        --arm clean --seed "$s" \
        --vision-classes "$PAIR_A" --vision-attractor-class "$ATTR_A" \
        --govern "$gov" \
        --out "$OUT/per_probe_${stem}.csv"
  done

  # --- shadow-governance scoring: emit the per-stem governance.json -----------
  # failure_mode_probe.py emits per_probe/summary/topk/per_slot/fork_events only.
  # The per-stem governance.json (the `none` / `mode-conditioned-trust` /
  # `_router` policy table the analyzer reads) is produced by the FROZEN
  # shadow-governance scorer, byte-identical to analyze_fork_governance.main()'s
  # per-stem write loop (indent=1, sort_keys=True) — same call path as
  # pr6_step3_run_matrix.sh. The #87 detector is reproduced from the committed
  # fit set, independent of the run under test.
  python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
from benchmarks.analyze_fork_governance import load_run, score_run
from benchmarks.score_frozen_detector import (
    FIT_CSV, FIT_SUMMARY, reproduce_frozen_detector)

OUT = Path(sys.argv[1])
det, thr, prov = reproduce_frozen_detector(Path(FIT_CSV), Path(FIT_SUMMARY))
print("frozen-detector fit:", json.dumps(prov, sort_keys=True)[:160])
stems = sorted(p for p in OUT.glob("per_probe_*.csv")
               if not p.name.endswith((".per_slot.csv", ".fork_events.csv",
                                       ".topk.csv")))
for stem in stems:
    table = score_run(load_run(stem), det, thr)
    with open(stem.with_suffix(".governance.json"), "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
    print("scored", stem.stem)
PY
done

echo "=== run matrix complete: $ROOT/{none,annotate}/"
