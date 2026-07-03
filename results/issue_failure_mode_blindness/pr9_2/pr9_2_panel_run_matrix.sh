#!/usr/bin/env bash
# PR-9.2: §9A shadow-certification panel (PR9_2_IDENTITY_CERT.md §3).
#
# The frozen §8 panel (PR8_QUARANTINE_REPLACEMENT_GATE.md §4.5), seeds {0,1,2}:
#   direct_harm(pairD) / collateral_harm(pairB,pairE) / clean_control(pairA,
#   pairC) / merge_path_stale(pairA,pairB,pairD,pairE); one_shot_ambiguity is
#   observe-only (no arm). Arms:
#   * none      — the COMMITTED baselines, copied byte-verified into the panel
#                 tree (never re-generated);
#   * shadow    — new runs (expected flagged 192/seed stale-soft, 0 clean);
#   * quarantine— instrumented RE-RUNS; retrieval artifacts must be
#                 byte-identical to the committed §8 arms (fidelity check).
#
# Prologue (STOP on any diff — a gentoo stack change since §8 is a
# re-baselining event, PR-10 result memo): fresh `none` re-runs of pairD
# stale-soft s0 and pairA clean s0 byte-compared against the committed stems.
# Determinism: same-seed shadow twin (pairD stale-soft s0) byte-compared.
#
# Run from the repo root on gentoo with the venv active. Engine byte-frozen;
# no --read-govern anywhere; committed artifacts are read-only inputs.
set -euo pipefail

ROOT=results/issue_failure_mode_blindness/pr9_2/panel
RUNS=results/issue_failure_mode_blindness/pr9_2/runs
C7=results/issue_failure_mode_blindness/pr7/twin
C4=results/issue_failure_mode_blindness/pr4
COMMON="--vision --epochs 12 --supersede-epoch 6"

PAIR_A="0,8,19,33";   ATTR_A=71
PAIR_B="5,27,48,86";  ATTR_B=13
PAIR_C="10,29,42,67"; ATTR_C=69
PAIR_D="10,28,32,95"; ATTR_D=52
PAIR_E="47,56,61,76"; ATTR_E=1

classes_of() { case "$1" in A) echo "$PAIR_A";; B) echo "$PAIR_B";; C) echo "$PAIR_C";; D) echo "$PAIR_D";; E) echo "$PAIR_E";; esac; }
attr_of()    { case "$1" in A) echo "$ATTR_A";; B) echo "$ATTR_B";; C) echo "$ATTR_C";; D) echo "$ATTR_D";; E) echo "$ATTR_E";; esac; }

cmp_stem() {  # cmp_stem <stemA> <stemB>  — retrieval artifacts byte/content
  local a="$1" b="$2"
  cmp -s "$a.csv" "$b.csv" || return 1
  cmp -s "$a.per_slot.csv" "$b.per_slot.csv" || return 1
  cmp -s "$a.fork_events.csv" "$b.fork_events.csv" || return 1
  local ta tb rc=0
  ta=$(mktemp); tb=$(mktemp)
  if [ -f "$a.topk.csv.gz" ]; then gunzip -c "$a.topk.csv.gz" > "$ta"; else cat "$a.topk.csv" > "$ta"; fi
  if [ -f "$b.topk.csv.gz" ]; then gunzip -c "$b.topk.csv.gz" > "$tb"; else cat "$b.topk.csv" > "$tb"; fi
  cmp -s "$ta" "$tb" || rc=1
  rm -f "$ta" "$tb"
  return $rc
}

run_probe() {  # run_probe <arm-flags...> — wrapper kept trivial on purpose
  python benchmarks/failure_mode_probe.py "$@"
}

# ---------------------------------------------------------------------- 0
echo "=== prologue: protocol byte-identity vs committed §8 stems"
P=$RUNS/prologue; mkdir -p "$P"
run_probe $COMMON --arm stale --payload-mode soft --seed 0 \
    --vision-classes "$PAIR_D" --vision-attractor-class "$ATTR_D" \
    --govern none --out "$P/per_probe_stale-soft_s0_pairD.csv"
cmp_stem "$P/per_probe_stale-soft_s0_pairD" \
         "$C7/direct_harm/none/per_probe_stale-soft_s0_pairD" \
  || { echo "PROLOGUE FAIL: pairD stale-soft none != committed"; exit 1; }
run_probe $COMMON --arm clean --seed 0 \
    --vision-classes "$PAIR_A" --vision-attractor-class "$ATTR_A" \
    --govern none --out "$P/per_probe_clean_s0_pairA.csv"
cmp_stem "$P/per_probe_clean_s0_pairA" \
         "$C7/clean_control/none/per_probe_clean_s0_pairA" \
  || { echo "PROLOGUE FAIL: pairA clean none != committed"; exit 1; }
echo "prologue OK"

# ---------------------------------------------------------------------- 1
echo "=== distinct runs: shadow + instrumented quarantine"
for gov in shadow quarantine; do
  for p in A B D E; do
    for s in 0 1 2; do
      stem="stale-soft_s${s}_pair${p}"
      out="$RUNS/$gov/per_probe_${stem}.csv"
      mkdir -p "$RUNS/$gov"
      echo "--- $gov stale-soft pair$p s$s"
      run_probe $COMMON --arm stale --payload-mode soft --seed "$s" \
          --vision-classes "$(classes_of $p)" \
          --vision-attractor-class "$(attr_of $p)" \
          --govern "$gov" --out "$out"
    done
  done
  for p in A C; do
    for s in 0 1 2; do
      stem="clean_s${s}_pair${p}"
      out="$RUNS/$gov/per_probe_${stem}.csv"
      echo "--- $gov clean pair$p s$s"
      run_probe $COMMON --arm clean --seed "$s" \
          --vision-classes "$(classes_of $p)" \
          --vision-attractor-class "$(attr_of $p)" \
          --govern "$gov" --out "$out"
    done
  done
done

# ---------------------------------------------------------------------- 2
echo "=== same-seed shadow twin (determinism)"
T=$RUNS/twin; mkdir -p "$T"
run_probe $COMMON --arm stale --payload-mode soft --seed 0 \
    --vision-classes "$PAIR_D" --vision-attractor-class "$ATTR_D" \
    --govern shadow --out "$T/per_probe_stale-soft_s0_pairD.csv"
cmp_stem "$T/per_probe_stale-soft_s0_pairD" \
         "$RUNS/shadow/per_probe_stale-soft_s0_pairD" \
  || { echo "TWIN FAIL: same-seed shadow not byte-identical"; exit 1; }
cmp -s "$T/per_probe_stale-soft_s0_pairD.summary.json" \
       "$RUNS/shadow/per_probe_stale-soft_s0_pairD.summary.json" \
  || { echo "TWIN FAIL: shadow summary not byte-identical"; exit 1; }
echo "twin OK"

# ---------------------------------------------------------------------- 3
echo "=== populate the panel tree (committed none + new arms)"
copy_stem() {  # copy_stem <src-stem> <dst-dir>
  local src="$1" dst="$2"; mkdir -p "$dst"
  for suf in .csv .per_slot.csv .fork_events.csv .summary.json; do
    cp "$src$suf" "$dst/"
  done
  if [ -f "$src.topk.csv.gz" ]; then cp "$src.topk.csv.gz" "$dst/"; else cp "$src.topk.csv" "$dst/"; fi
}
cell_of() {  # cell_of <cell> <committed-none-stem> <run-stem>
  local cell="$1" cstem="$2" rstem="$3"
  copy_stem "$cstem" "$ROOT/$cell/none"
  copy_stem "$RUNS/shadow/$rstem" "$ROOT/$cell/shadow"
  copy_stem "$RUNS/quarantine/$rstem" "$ROOT/$cell/quarantine"
}
for s in 0 1 2; do
  cell_of direct_harm "$C7/direct_harm/none/per_probe_stale-soft_s${s}_pairD" \
          "per_probe_stale-soft_s${s}_pairD"
  for p in B E; do
    cell_of collateral_harm \
            "$C7/collateral_harm/none/per_probe_stale-soft_s${s}_pair${p}" \
            "per_probe_stale-soft_s${s}_pair${p}"
  done
  for p in A B D E; do
    cell_of merge_path_stale \
            "$C7/merge_path_stale/none/per_probe_stale-soft_s${s}_pair${p}" \
            "per_probe_stale-soft_s${s}_pair${p}"
  done
  cell_of clean_control "$C7/clean_control/none/per_probe_clean_s${s}_pairA" \
          "per_probe_clean_s${s}_pairA"
  # pairC clean: committed baseline lives in the pr4 grid (pr4 stem naming);
  # copy it under its own name and place the new arms beside it.
  copy_stem "$C4/per_probe_clean_pairC_s${s}" "$ROOT/clean_control/none"
  mkdir -p "$ROOT/clean_control/shadow" "$ROOT/clean_control/quarantine"
  for suf in .csv .per_slot.csv .fork_events.csv .summary.json .topk.csv .topk.csv.gz; do
    [ -f "$RUNS/shadow/per_probe_clean_s${s}_pairC$suf" ] && \
      cp "$RUNS/shadow/per_probe_clean_s${s}_pairC$suf" \
         "$ROOT/clean_control/shadow/per_probe_clean_pairC_s${s}$suf" || true
    [ -f "$RUNS/quarantine/per_probe_clean_s${s}_pairC$suf" ] && \
      cp "$RUNS/quarantine/per_probe_clean_s${s}_pairC$suf" \
         "$ROOT/clean_control/quarantine/per_probe_clean_pairC_s${s}$suf" || true
  done
done

# ---------------------------------------------------------------------- 4
echo "=== governance readout: current frozen scorer, ALL arms, panel tree"
python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
from benchmarks.analyze_fork_governance import load_run, score_run
from benchmarks.score_frozen_detector import (
    FIT_CSV, FIT_SUMMARY, reproduce_frozen_detector)

ROOT = Path(sys.argv[1])
det, thr, prov = reproduce_frozen_detector(Path(FIT_CSV), Path(FIT_SUMMARY))
print("frozen-detector fit:", json.dumps(prov, sort_keys=True)[:160])
stems = sorted(p for p in ROOT.rglob("per_probe_*.csv")
               if not p.name.endswith((".per_slot.csv", ".fork_events.csv",
                                       ".topk.csv")))
for stem in stems:
    table = score_run(load_run(stem), det, thr)
    with open(stem.with_suffix(".governance.json"), "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
    print("scored", stem.parent.parent.name + "/" + stem.parent.name + "/"
          + stem.stem)
PY

# ---------------------------------------------------------------------- 5
echo "=== gentoo sha256 manifest over the panel tree"
( cd "$ROOT" && find . -type f | sort | xargs sha256sum ) \
  > results/issue_failure_mode_blindness/pr9_2/panel.gentoo.sha256

echo "ALL RUNS DONE"
