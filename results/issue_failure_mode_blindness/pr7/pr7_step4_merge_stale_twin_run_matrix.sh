#!/usr/bin/env bash
# PR-7 step 4: pairD merge_path_stale annotate-only STRESS twin (PR7_DESIGN
# §5/§13.2).
#
# The annotate null-action floor on the cell where it actually does something:
# the soft-payload ("merge-path") stale arm HAS supersession (merge-suspect
# absorb) events, so `--govern annotate` stamps them (annotated_events > 0) —
# unlike the step-3 clean arm where annotated_events was 0. This is the real
# floor stress: stamping merge-suspect events must STILL change nothing the
# writer commits, because GovernanceHook.decide() returns ALLOW regardless.
# Every scored artifact must stay byte-identical to the ungoverned baseline,
# with only the summary's `govern` provenance block added (§4).
#
# The merge_path_stale cell guard is `capture_stable`: the write-time
# merge-suspect capture must hold (192/seed) and read-time damage must not
# worsen — annotate, the floor, is expected to give a zero delta on both.
#
# NOT a governance experiment beyond the null-action floor: no quarantine, no
# refuse, no trust action, no static geometry gate, no retrieval change, no new
# observable. Same fixed protocol as the committed pairD merge-path stale arms
# (pr6_step3_run_matrix.sh / pr6/stale_de/per_probe_stale-soft_s{0,1,2}_pairD):
# --arm stale --payload-mode soft, classes 10,28,32,95, attractor 52,
# --epochs 12 --supersede-epoch 6, 3 seeds. Geometry is provenance only, never
# an admission gate (PR-5 step 1). The two engine files stay byte-frozen (§1).
#
# Run from the repo root on gentoo with the venv active. Outputs land in
# results/issue_failure_mode_blindness/pr7/twin/merge_path_stale/{none,annotate}/.
# After the run, copy that dir back to the canonical Darwin host and re-verify
# there: python benchmarks/pr7_twin_delta.py --govern annotate ... + pytest.
set -euo pipefail

ROOT=results/issue_failure_mode_blindness/pr7/twin/merge_path_stale
COMMON="--vision --epochs 12 --supersede-epoch 6"
PAIR_D="10,28,32,95"; ATTR_D=52

for gov in none annotate; do
  OUT="$ROOT/$gov"
  mkdir -p "$OUT"
  for s in 0 1 2; do
    stem="stale-soft_s${s}_pairD"
    echo "=== govern=$gov $stem"
    python benchmarks/failure_mode_probe.py $COMMON \
        --arm stale --payload-mode soft --seed "$s" \
        --vision-classes "$PAIR_D" --vision-attractor-class "$ATTR_D" \
        --govern "$gov" \
        --out "$OUT/per_probe_${stem}.csv"
  done

  # --- shadow-governance scoring: emit the per-stem governance.json -----------
  # Same frozen-scorer per-stem path as pr6_step3_run_matrix.sh /
  # pr7_step3_clean_twin_run_matrix.sh: reproduce_frozen_detector + load_run +
  # score_run, byte-identical to analyze_fork_governance.main()'s per-stem write
  # loop (indent=1, sort_keys=True). The #87 detector is reproduced from the
  # committed fit set, independent of the run under test.
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
