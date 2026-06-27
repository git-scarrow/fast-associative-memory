#!/usr/bin/env bash
# PR-7 step 5: pairD merge_path_stale REFUSE twin — the first acting arm
# (PR7_DESIGN.md §4/§13).
#
# `--govern refuse` skips the already-classified write-time merge_suspect
# (supersession) write BEFORE it commits; non-suspect writes (clean) are allowed
# unchanged. This is the first action that actually changes the write state, so —
# unlike annotate — the scored artifacts are EXPECTED to diverge from baseline.
# The experiment measures whether refusing the supersession write drains
# read-time stale/broken without harming clean traffic, AND what it does to the
# write-time merge-suspect capture the merge_path_stale cell guards on.
#
# Protocol = the committed pairD merge-path stale arm (pr6/stale_de /
# pr7_step4): --arm stale --payload-mode soft, classes 10,28,32,95, attractor
# 52, --epochs 12 --supersede-epoch 6, 3 seeds. Only the --govern flag differs.
# The deployed engine stays byte-frozen; the refuse decision lives only in the
# experimental GovernanceHook (opt-in --govern path), never in retrieval.
#
# The `none` baseline is ALREADY committed at
# results/.../pr7/twin/merge_path_stale/none/ (step 4) and the analyzer reuses
# it for every action. This script (a) produces the refuse arm and (b) RE-RUNS
# none into a throwaway verify dir and byte-compares it to the committed none —
# proving the step-5 code change left the baseline write path byte-identical.
#
# Run from the repo root on gentoo with the venv active. The refuse arm lands in
# results/.../pr7/twin/merge_path_stale/refuse/; copy ONLY that dir back to
# Darwin (the committed none is unchanged) and re-verify there.
set -euo pipefail

ROOT=results/issue_failure_mode_blindness/pr7/twin/merge_path_stale
COMMON="--vision --epochs 12 --supersede-epoch 6"
PAIR_D="10,28,32,95"; ATTR_D=52

score_dir() {  # frozen shadow-governance scoring -> per-stem governance.json
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
from benchmarks.analyze_fork_governance import load_run, score_run
from benchmarks.score_frozen_detector import (
    FIT_CSV, FIT_SUMMARY, reproduce_frozen_detector)

OUT = Path(sys.argv[1])
det, thr, prov = reproduce_frozen_detector(Path(FIT_CSV), Path(FIT_SUMMARY))
stems = sorted(p for p in OUT.glob("per_probe_*.csv")
               if not p.name.endswith((".per_slot.csv", ".fork_events.csv",
                                       ".topk.csv")))
for stem in stems:
    table = score_run(load_run(stem), det, thr)
    with open(stem.with_suffix(".governance.json"), "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
    print("scored", stem.stem)
PY
}

# --- the refuse arm (the new measurement) -----------------------------------
OUT="$ROOT/refuse"
mkdir -p "$OUT"
for s in 0 1 2; do
  stem="stale-soft_s${s}_pairD"
  echo "=== govern=refuse $stem"
  python benchmarks/failure_mode_probe.py $COMMON \
      --arm stale --payload-mode soft --seed "$s" \
      --vision-classes "$PAIR_D" --vision-attractor-class "$ATTR_D" \
      --govern refuse \
      --out "$OUT/per_probe_${stem}.csv"
done
score_dir "$OUT"

# --- baseline-invariance check: re-run none, byte-compare to committed none --
VERIFY="$ROOT/_none_verify"
mkdir -p "$VERIFY"
for s in 0 1 2; do
  stem="stale-soft_s${s}_pairD"
  echo "=== govern=none (verify) $stem"
  python benchmarks/failure_mode_probe.py $COMMON \
      --arm stale --payload-mode soft --seed "$s" \
      --vision-classes "$PAIR_D" --vision-attractor-class "$ATTR_D" \
      --govern none \
      --out "$VERIFY/per_probe_${stem}.csv"
done
score_dir "$VERIFY"

echo "=== baseline invariance: re-run none vs committed none ==="
fail=0
for s in 0 1 2; do
  stem="stale-soft_s${s}_pairD"
  for ext in csv governance.json per_slot.csv fork_events.csv; do
    if ! cmp -s "$ROOT/none/per_probe_${stem}.${ext}" \
                "$VERIFY/per_probe_${stem}.${ext}"; then
      echo "DIFFER $stem.$ext"; fail=1
    fi
  done
  # committed none topk is gzipped; verify is plain — compare decompressed.
  if ! cmp -s <(gunzip -c "$ROOT/none/per_probe_${stem}.topk.csv.gz") \
              "$VERIFY/per_probe_${stem}.topk.csv"; then
    echo "DIFFER $stem.topk"; fail=1
  fi
done
[ "$fail" = 0 ] && echo "BASELINE-INVARIANT: re-run none byte-identical to committed none"
rm -rf "$VERIFY"

echo "=== run matrix complete: $ROOT/refuse/ (none baseline reused, unchanged) ==="
