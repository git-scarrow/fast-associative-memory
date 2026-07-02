#!/usr/bin/env bash
# PR-10 step 2 gentoo run matrix (PR10_READTIME_ABSTENTION_GATE.md §2).
#
# GOVERNED (--read-govern merge-abstain) arms ONLY. The committed PR-4/PR-3c
# baselines are the comparison targets and are NEVER regenerated (gate memo
# §6). Same cache, config, epochs and protocol as pr4_run_matrix.sh — the
# ONLY new flag is --read-govern merge-abstain; the write seam --govern is
# absent (none) everywhere.
#
# Cells: full grids for pairs A-E (six §4 arms x seeds 0-2 = 90) plus the two
# pairA jitter runs — pairA (18) and pairB-s0 clean/contra/stale/mixed (4) and
# jitter (2) are report-only anchors per the gate memo §2; the other 68 are
# the fresh verdict cells matching pr9/abstention_envelope.json cells_fresh.
#
# Prologue gates (abort before the matrix on failure):
#   1. protocol byte-identity — the merged driver with the seam OFF (no
#      --read-govern flag) still reproduces the pinned PR-3c mixed_s0 pair A/B
#      CSVs bit-exactly on this host;
#   2. G5 same-seed stability — the governed stale-soft pairD s0 cell is run
#      twice; the duplicate is kept as g5twin_* for the reader's governed_twin
#      check.
#
# Run from the repo root on gentoo with the venv active. Outputs land in
# results/issue_failure_mode_blindness/pr10/governed/; topk side tables are
# gzipped at the end (the reader compares decompressed content).
set -euo pipefail

OUT=results/issue_failure_mode_blindness/pr10/governed
PR3C=results/issue_failure_mode_blindness/pr3c
COMMON="--vision --epochs 12 --supersede-epoch 6"
GOV="--read-govern merge-abstain"

# Pair selection verbatim from pr4_run_matrix.sh (gate 1, Addendum A.3).
PAIR_A="0,8,19,33";   ATTR_A=71
PAIR_B="5,27,48,86";  ATTR_B=13
PAIR_C="10,29,42,67"; ATTR_C=69
PAIR_D="10,28,32,95"; ATTR_D=52
PAIR_E="47,56,61,76"; ATTR_E=1

run() {  # run <outfile-stem> <args...>   (governed)
  local stem=$1; shift
  echo "=== $stem"
  python benchmarks/failure_mode_probe.py $COMMON $GOV "$@" \
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

# --- gate 1: protocol byte-identity with the seam OFF ------------------------
python benchmarks/failure_mode_probe.py $COMMON --arm mixed --rate 0.15 \
    --seed 0 --vision-classes $PAIR_A --vision-attractor-class $ATTR_A \
    --out "$OUT/bytecheck_mixed_pairA_s0.csv"
python benchmarks/failure_mode_probe.py $COMMON --arm mixed --rate 0.15 \
    --seed 0 --vision-classes $PAIR_B --vision-attractor-class $ATTR_B \
    --out "$OUT/bytecheck_mixed_pairB_s0.csv"
fail=0
cmp -s "$PR3C/per_probe_mixed_s0.csv" \
       "$OUT/bytecheck_mixed_pairA_s0.csv" || \
  { echo "MISMATCH: mixed_s0 pair A vs PR-3c"; fail=1; }
cmp -s "$PR3C/per_probe_mixed_s0_pairB.csv" \
       "$OUT/bytecheck_mixed_pairB_s0.csv" || \
  { echo "MISMATCH: mixed_s0 pair B vs PR-3c"; fail=1; }
if [ "$fail" -ne 0 ]; then
  echo "PROTOCOL BYTE-IDENTITY: FAILED — aborting before the matrix"
  exit 1
fi
echo "PROTOCOL BYTE-IDENTITY: OK"
rm -f "$OUT"/bytecheck_*

# --- gate 2: G5 same-seed stability (duplicate kept for the reader) ----------
run "g5twin_stale-soft_pairD_s0" --arm stale --payload-mode soft --seed 0 \
    --vision-classes $PAIR_D --vision-attractor-class $ATTR_D

# --- fresh verdict cells: pair B completion + pairs C/D/E full grids ---------
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
arms pairC $PAIR_C $ATTR_C 0 1 2
arms pairD $PAIR_D $ATTR_D 0 1 2
arms pairE $PAIR_E $ATTR_E 0 1 2

# --- report-only anchors: pairA full grid + pairB s0 basics + jitter ---------
arms pairA $PAIR_A $ATTR_A 0 1 2
for s in 0; do
  run "clean_pairB_s${s}" --arm clean --seed $s \
      --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
  run "contra_pairB_s${s}" --arm contra --rate 0.15 --seed $s \
      --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
  run "stale_pairB_s${s}" --arm stale --seed $s \
      --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
  run "mixed_pairB_s${s}" --arm mixed --rate 0.15 --seed $s \
      --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
done
run "stale-jitter0.05_pairA_s0" --arm stale --key-jitter 0.05 --seed 0 \
    --vision-classes $PAIR_A --vision-attractor-class $ATTR_A
run "stale-jitter0.15_pairA_s0" --arm stale --key-jitter 0.15 --seed 0 \
    --vision-classes $PAIR_A --vision-attractor-class $ATTR_A

gzip -f "$OUT"/per_probe_*.topk.csv
echo "ALL RUNS DONE: $(ls $OUT/per_probe_*.csv | grep -Evc 'per_slot|fork_events|topk') per-probe CSVs (expect 93 = 90 grid + 2 jitter + 1 g5twin)"
