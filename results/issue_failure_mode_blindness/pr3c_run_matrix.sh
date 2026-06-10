#!/usr/bin/env bash
# PR-3c gentoo run matrix: the exact PR-3b 21-run protocol re-executed with
# topk.csv logging (the shadow-governance basis), plus the three pair-A
# seed-0 PR-2 baselines. Same cache, config, seeds, epochs as
# pr3b_run_matrix.sh — the per-probe CSVs must come out byte-identical to
# the pinned PR-3b/PR-2 baselines (verified after the run); only the new
# <out>.topk.csv side table is additional information. Run from the repo
# root on gentoo with the venv active. Outputs land in
# results/issue_failure_mode_blindness/pr3c/.
set -euo pipefail

OUT=results/issue_failure_mode_blindness/pr3c
PAIR_A="0,8,19,33";  ATTR_A=71
PAIR_B="5,27,48,86"; ATTR_B=13
COMMON="--vision --epochs 12 --supersede-epoch 6"

run() {  # run <outfile-stem> <args...>
  local stem=$1; shift
  echo "=== $stem"
  python benchmarks/failure_mode_probe.py $COMMON "$@" \
      --out "$OUT/per_probe_${stem}.csv"
}

# --- pair A: PR-2 seed-0 baselines (byte-compare vs per_probe_vision_*) ----
run "clean_s0"  --arm clean              --seed 0 \
    --vision-classes $PAIR_A --vision-attractor-class $ATTR_A
run "contra_s0" --arm contra --rate 0.15 --seed 0 \
    --vision-classes $PAIR_A --vision-attractor-class $ATTR_A
run "stale_s0"  --arm stale              --seed 0 \
    --vision-classes $PAIR_A --vision-attractor-class $ATTR_A

# --- pair A: variant arms (byte-compare vs pr3b/) ---------------------------
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

# --- pair A: PR-2 replication seeds ----------------------------------------
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

# --- byte-identity vs the pinned baselines ----------------------------------
fail=0
for f in results/issue_failure_mode_blindness/pr3b/per_probe_*.csv; do
  base=$(basename "$f")
  case "$base" in *.per_slot.csv|*.fork_events.csv) continue;; esac
  if ! cmp -s "$f" "$OUT/$base"; then echo "MISMATCH: $base"; fail=1; fi
done
for arm in clean contra stale; do
  if ! cmp -s "results/issue_failure_mode_blindness/per_probe_vision_${arm}.csv" \
              "$OUT/per_probe_${arm}_s0.csv"; then
    echo "MISMATCH: ${arm}_s0 vs PR-2 baseline"; fail=1
  fi
done
[ "$fail" -eq 0 ] && echo "BASELINE BYTE-IDENTITY: OK" || echo "BASELINE BYTE-IDENTITY: FAILED"
echo "ALL RUNS DONE: $(ls $OUT/per_probe_*.csv | grep -Evc 'per_slot|fork_events|topk') per-probe CSVs"
