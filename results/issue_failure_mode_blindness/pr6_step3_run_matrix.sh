#!/usr/bin/env bash
# PR-6 step 3 run matrix: complete the merge_path_stale cell by measuring the
# soft-payload ("merge-path") stale arm on the geometries PR-3c never ran —
# pairD {10,28,32,95}, pairE {47,56,61,76}, and pairB {5,27,48,86}. This is the
# SMALLEST run that seeds the cell's required label (D/E) and, in the same pass,
# drains the residual pair-B note in missing_evidence.
#
# NOT a governance experiment:
#   * same protocol/config as pr3c_run_matrix.sh — COMMON flags, 3 seeds, the
#     soft-payload stale arm only;
#   * the frozen `mode-conditioned-trust` probe is the measurand, emitted by
#     failure_mode_probe.py's governance.json (no policy sweep, no thresholds,
#     no new observables, no retrieval change);
#   * only the class set / attractor differ from the committed pair-A arms —
#     which IS the missing measurement (PR6_DESIGN §8; PR3C_RESULT.md §2).
#
# Geometry here is provenance only ("what was measured on what"), never an
# admission gate (PR-5 step 1 closed geometry as a safety predictor). PR-3c is
# left byte-for-byte untouched; the new arms land in their own directory.
#
# Run from the repo root on gentoo with the venv active. Outputs land in
# results/issue_failure_mode_blindness/pr6/stale_de/. After the run, copy that
# directory back to the canonical Darwin host and re-verify there (schema check
# below + pytest tests/test_pr6_stale_cell.py).
set -euo pipefail

OUT=results/issue_failure_mode_blindness/pr6/stale_de
mkdir -p "$OUT"

# Same fixed protocol as the committed PR-3c stale-soft arms.
COMMON="--vision --epochs 12 --supersede-epoch 6"

# class set -> attractor class (PR-5 hazard-index values, pr5_hazard_index.py:76-78;
# pairB attractor 13 matches pr3c_run_matrix.sh ATTR_B). pairA is already
# committed in pr3c/ and is NOT re-run.
PAIR_D="10,28,32,95"; ATTR_D=52
PAIR_E="47,56,61,76"; ATTR_E=1
PAIR_B="5,27,48,86";  ATTR_B=13

run() {  # run <pair-name> <classes> <attractor> <seed>
  local pair=$1 classes=$2 attr=$3 s=$4
  local stem="stale-soft_s${s}_${pair}"
  echo "=== $stem"
  python benchmarks/failure_mode_probe.py $COMMON \
      --arm stale --payload-mode soft --seed "$s" \
      --vision-classes "$classes" --vision-attractor-class "$attr" \
      --out "$OUT/per_probe_${stem}.csv"
}

# --- the 9 missing merge-path stale arms: 3 geometries x 3 seeds -------------
for s in 0 1 2; do
  run pairD "$PAIR_D" "$ATTR_D" "$s"
  run pairE "$PAIR_E" "$ATTR_E" "$s"
  run pairB "$PAIR_B" "$ATTR_B" "$s"
done

# --- self-check: provenance + schema the PR-6 panel consumes -----------------
# Mirrors pr3c's byte-identity guard, but these are new geometries with no prior
# baseline to compare against, so instead we assert each arm emitted exactly the
# fields read_merge_path_stale_evidence / _stale_soft_seed_label require.
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1])
EXPECT = {
    "pairD": [10, 28, 32, 95],
    "pairE": [47, 56, 61, 76],
    "pairB": [5, 27, 48, 86],
}
GOV_KEYS = {"none", "mode-conditioned-trust", "_router"}
fail = 0
for pair, classes in EXPECT.items():
    for s in (0, 1, 2):
        stem = f"per_probe_stale-soft_s{s}_{pair}"
        gov = json.loads((OUT / f"{stem}.governance.json").read_text())
        summ = json.loads((OUT / f"{stem}.summary.json").read_text())
        missing = GOV_KEYS - set(gov)
        if missing:
            print(f"FAIL {stem}: governance missing {sorted(missing)}"); fail = 1
        if "n_merge_suspect_events" not in gov.get("_router", {}):
            print(f"FAIL {stem}: _router missing n_merge_suspect_events"); fail = 1
        if summ.get("arm") != "stale-soft" or summ.get("payload_mode") != "soft":
            print(f"FAIL {stem}: arm/payload "
                  f"{summ.get('arm')!r}/{summ.get('payload_mode')!r}"); fail = 1
        if summ.get("classes") != classes:
            print(f"FAIL {stem}: classes {summ.get('classes')} != {classes}"); fail = 1
print("SCHEMA/PROVENANCE: OK" if not fail else "SCHEMA/PROVENANCE: FAILED")
sys.exit(fail)
PY

echo "ALL RUNS DONE: $(ls "$OUT"/per_probe_stale-soft_s*_pair*.csv 2>/dev/null \
  | grep -Evc 'per_slot|fork_events|topk') per-probe CSVs"
