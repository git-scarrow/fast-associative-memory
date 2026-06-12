#!/usr/bin/env bash
# PR-4 step 4 gentoo run matrix (PR4_DESIGN.md §4, §8 step 4; pair set
# fixed at gate 1, Addendum A.3). Same cache, config, epochs and protocol
# as pr3c_run_matrix.sh — only the class sets are new.
#
# Pair A (all arms x seeds 0-2) and pair B s0 clean/contra/stale/mixed are
# NOT re-run: the PR-3c artifacts in pr3c/ are reused verbatim (anchors,
# leakage-tainted per §7 — they cannot count toward H1/H3 anyway). Two
# protocol byte-identity re-runs (mixed_s0 pair A, mixed_s0 pair B) prove
# the driver still reproduces the pinned PR-3c CSVs bit-exactly, then are
# deleted. New cells: pair B completion (s1-2 of the four PR-3c arms,
# soft/one-shot s0-2) and pairs C/D/E full grids — 68 analysis runs.
#
# Run from the repo root on gentoo with the venv active. Outputs land in
# results/issue_failure_mode_blindness/pr4/; topk side tables are gzipped
# at the end (load_run reads .gz transparently).
set -euo pipefail

OUT=results/issue_failure_mode_blindness/pr4
PR3C=results/issue_failure_mode_blindness/pr3c
COMMON="--vision --epochs 12 --supersede-epoch 6"

# Gate-1 pair selection (Addendum A.3); A/B verbatim from pr3c_run_matrix.sh.
PAIR_A="0,8,19,33";   ATTR_A=71
PAIR_B="5,27,48,86";  ATTR_B=13
PAIR_C="10,29,42,67"; ATTR_C=69
PAIR_D="10,28,32,95"; ATTR_D=52
PAIR_E="47,56,61,76"; ATTR_E=1

run() {  # run <outfile-stem> <args...>
  local stem=$1; shift
  echo "=== $stem"
  python benchmarks/failure_mode_probe.py $COMMON "$@" \
      --out "$OUT/per_probe_${stem}.csv"
}

# arms <pairTag> <classes> <attr> <seeds...> — the six §4 arms per seed
arms() {
  local tag=$1 cls=$2 attr=$3; shift 3
  for s in "$@"; do
    run "clean_${tag}_s${s}" --arm clean --seed $s \
        --vision-classes $cls --vision-attractor-class $attr
    run "contra_${tag}_s${s}" --arm contra --rate 0.15 --seed $s \
        --vision-classes $cls --vision-attractor-class $attr
    run "stale_${tag}_s${s}" --arm stale --seed $s \
        --vision-classes $cls --vision-attractor-class $attr
    run "mixed_${tag}_s${s}" --arm mixed --rate 0.15 --seed $s \
        --vision-classes $cls --vision-attractor-class $attr
    run "stale-oneshot_${tag}_s${s}" --arm stale --one-shot --seed $s \
        --vision-classes $cls --vision-attractor-class $attr
    run "stale-soft_${tag}_s${s}" --arm stale --payload-mode soft --seed $s \
        --vision-classes $cls --vision-attractor-class $attr
  done
}

mkdir -p "$OUT"

# --- protocol byte-identity vs the pinned PR-3c artifacts -------------------
run "bytecheck_mixed_pairA_s0" --arm mixed --rate 0.15 --seed 0 \
    --vision-classes $PAIR_A --vision-attractor-class $ATTR_A
run "bytecheck_mixed_pairB_s0" --arm mixed --rate 0.15 --seed 0 \
    --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
fail=0
cmp -s "$PR3C/per_probe_mixed_s0.csv" \
       "$OUT/per_probe_bytecheck_mixed_pairA_s0.csv" || \
  { echo "MISMATCH: mixed_s0 pair A vs PR-3c"; fail=1; }
cmp -s "$PR3C/per_probe_mixed_s0_pairB.csv" \
       "$OUT/per_probe_bytecheck_mixed_pairB_s0.csv" || \
  { echo "MISMATCH: mixed_s0 pair B vs PR-3c"; fail=1; }
if [ "$fail" -ne 0 ]; then
  echo "PROTOCOL BYTE-IDENTITY: FAILED — aborting before the matrix"
  exit 1
fi
echo "PROTOCOL BYTE-IDENTITY: OK"
rm -f "$OUT"/per_probe_bytecheck_*

# --- pair B completion (PR-3c covers clean/contra/stale/mixed s0) -----------
for s in 1 2; do
  run "clean_pairB_s${s}" --arm clean --seed $s \
      --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
  run "contra_pairB_s${s}" --arm contra --rate 0.15 --seed $s \
      --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
  run "stale_pairB_s${s}" --arm stale --seed $s \
      --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
  run "mixed_pairB_s${s}" --arm mixed --rate 0.15 --seed $s \
      --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
done
for s in 0 1 2; do
  run "stale-oneshot_pairB_s${s}" --arm stale --one-shot --seed $s \
      --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
  run "stale-soft_pairB_s${s}" --arm stale --payload-mode soft --seed $s \
      --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
done

# --- pairs C, D, E: full grids ----------------------------------------------
arms pairC $PAIR_C $ATTR_C 0 1 2
arms pairD $PAIR_D $ATTR_D 0 1 2
arms pairE $PAIR_E $ATTR_E 0 1 2

gzip -f "$OUT"/per_probe_*.topk.csv
echo "ALL RUNS DONE: $(ls $OUT/per_probe_*.csv | grep -Evc 'per_slot|fork_events|topk') per-probe CSVs"
