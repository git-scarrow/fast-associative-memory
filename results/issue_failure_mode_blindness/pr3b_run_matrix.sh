#!/usr/bin/env bash
# PR-3b gentoo run matrix (PR3_DESIGN.md §5 PR-3b). Run from the repo root
# on gentoo with the venv active. Every run: vitl14_cifar100_train cache,
# #87 config, stationary (contraction 0.0), 12 epochs, supersede epoch 6,
# rate 0.15 where applicable. Outputs land in
# results/issue_failure_mode_blindness/pr3b/ with the per_slot / fork_events
# side tables next to each per-probe CSV.
set -euo pipefail

OUT=results/issue_failure_mode_blindness/pr3b
PAIR_A="0,8,19,33";  ATTR_A=71
PAIR_B="5,27,48,86"; ATTR_B=13
COMMON="--vision --epochs 12 --supersede-epoch 6"

run() {  # run <outfile-stem> <args...>
  local stem=$1; shift
  echo "=== $stem"
  python benchmarks/failure_mode_probe.py $COMMON "$@" \
      --out "$OUT/per_probe_${stem}.csv"
}

# --- pair A: variant arms -------------------------------------------------
for s in 0 1 2; do
  run "mixed_s${s}"         --arm mixed --rate 0.15 --seed $s \
      --vision-classes $PAIR_A --vision-attractor-class $ATTR_A
  run "stale-oneshot_s${s}" --arm stale --one-shot --seed $s \
      --vision-classes $PAIR_A --vision-attractor-class $ATTR_A
  run "stale-soft_s${s}"    --arm stale --payload-mode soft --seed $s \
      --vision-classes $PAIR_A --vision-attractor-class $ATTR_A
done
for eps in 0.05 0.15; do
  run "stale-jitter${eps}_s0" --arm stale --key-jitter $eps --seed 0 \
      --vision-classes $PAIR_A --vision-attractor-class $ATTR_A
done

# --- pair A: PR-2 replication seeds (PR-2b/2c were seed 0 only) -----------
for s in 1 2; do
  run "clean_s${s}"  --arm clean              --seed $s \
      --vision-classes $PAIR_A --vision-attractor-class $ATTR_A
  run "contra_s${s}" --arm contra --rate 0.15 --seed $s \
      --vision-classes $PAIR_A --vision-attractor-class $ATTR_A
  run "stale_s${s}"  --arm stale              --seed $s \
      --vision-classes $PAIR_A --vision-attractor-class $ATTR_A
done

# --- pair B: second class set, seed 0 --------------------------------------
for arm in clean contra stale mixed; do
  extra=""
  [ "$arm" = contra ] || [ "$arm" = mixed ] && extra="--rate 0.15"
  run "${arm}_s0_pairB" --arm $arm $extra --seed 0 \
      --vision-classes $PAIR_B --vision-attractor-class $ATTR_B
done

echo "ALL RUNS DONE: $(ls $OUT/per_probe_*.csv | grep -Evc 'per_slot|fork_events') per-probe CSVs"
