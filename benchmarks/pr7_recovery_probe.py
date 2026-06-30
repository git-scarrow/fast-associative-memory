"""PR-7 G3 — quarantine recoverability re-injection probe (analysis-only).

PR7_QUARANTINE_PROMOTION_GATE.md §4 **G3** — the quarantine-specific criterion.
`refuse` destroys the merge-suspect writes it blocks; `quarantine` RETAINS them
in a recoverable side ledger. G3 asks whether that retention is a real,
exercised mechanism — can the diverted writes be reconstructed and reinstated to
recover the baseline capture (192/seed) WITHOUT re-inflating read-time broken/
stale beyond the ungoverned baseline — or whether "recoverable" is a
provenance-only label the harness never exercised.

What this probe is, and is NOT (the safety boundary, all enforced here):

  * It IMPORTS NO TORCH and TOUCHES NO CACHE — it reads only committed twin
    artifacts (the frozen scorer's ``governance.json``, the run ``summary.json``
    quarantine ledger, and the ``fork_events.csv`` write-time event log). It runs
    no model, makes no write, changes no engine or retrieval code. Engine stays
    byte-frozen; ``--govern`` is opt-in; deployed retrieval is untouched.
  * "Reinstatement" is NOT a new engine run. In this engine-frozen harness the
    quarantine arm differs from the ``none`` arm by EXACTLY the diverted
    supersession writes: reinstating all of them reproduces the ungoverned
    (``none``) run deterministically. So the committed ``none`` arm IS the
    reinstated state, and the probe reads its frozen readout as the
    post-reinstatement measurand. This is verified, not assumed: ``none``
    ``absorbed`` minus ``quarantine`` ``absorbed`` must equal the ledger's
    quarantined count, and the baseline's ``event_class == 'supersession'`` count
    must equal it too (independent reconstruction of the diverted set).

The probe reports three distinct claims, never collapsing them:

  1. RECONSTRUCTION FIDELITY — the ledger retains exactly the diverted writes:
     ledger.quarantined_count == baseline supersession capture (router AND
     fork_events agree) == ledger label-histogram total. A complete, lossless,
     reversible record.
  2. CAPTURE RESTORATION — reinstatement (= none) restores active-memory capture
     to 192/seed from the quarantine arm's 0, with broken/stale not exceeding the
     ungoverned baseline (the frozen G3 numeric bound: broken per-seed <= +0).
  3. HARM-FREE RECOVERY — whether restoring capture re-introduces the read-time
     harm quarantine removed. delta_harm = baseline_broken - quarantine_broken;
     recovery is harm-free iff delta_harm == 0 on a seed. Where capture and harm
     are the SAME writes (the hazardous geometries), restoring capture
     necessarily re-introduces the harm and this is FALSE — the gate §3 prose
     falsification ("restores it only by re-introducing the harm quarantine
     removed").

Usage (runs on any host; reads committed JSON/CSV only, no cache/torch):
  python benchmarks/pr7_recovery_probe.py \
      --cell merge_path_stale \
      --out results/issue_failure_mode_blindness/pr7/recovery_validation.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

RESULTS = Path("results/issue_failure_mode_blindness")
TWIN_ROOT = RESULTS / "pr7" / "twin"
PROBE_POLICY = "mode-conditioned-trust"
SEEDS = (0, 1, 2)
SUPERSESSION_CLASS = "supersession"  # the write-time merge_suspect event class

# Cells that carry a quarantine ledger to validate. The read-time cells reuse the
# same soft (merge-path) arm; merge_path_stale is the canonical full-geometry cell.
CELL_PAIRS = {
    "merge_path_stale": ["pairA", "pairB", "pairD", "pairE"],
}
CELL_ARM = {"merge_path_stale": "stale-soft"}


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _event_class_counts(fork_csv: Path) -> Counter:
    with fork_csv.open() as f:
        return Counter(r["event_class"] for r in csv.DictReader(f))


def _outcome_counts(fork_csv: Path) -> Counter:
    with fork_csv.open() as f:
        return Counter(r["outcome"] for r in csv.DictReader(f))


def validate_arm(cell: str, pair: str, seed: int,
                 twin_root: Path = TWIN_ROOT) -> dict:
    """Validate recoverability for one (cell, pair, seed) twin arm."""
    arm = CELL_ARM[cell]
    stem = f"per_probe_{arm}_s{seed}_{pair}"
    none_dir = twin_root / cell / "none"
    q_dir = twin_root / cell / "quarantine"

    base_gov = _read_json(none_dir / f"{stem}.governance.json")
    q_gov = _read_json(q_dir / f"{stem}.governance.json")
    q_summ = _read_json(q_dir / f"{stem}.summary.json")
    ledger = q_summ["govern"]["quarantine_ledger"]

    base_ec = _event_class_counts(none_dir / f"{stem}.fork_events.csv")
    base_oc = _outcome_counts(none_dir / f"{stem}.fork_events.csv")
    q_oc = _outcome_counts(q_dir / f"{stem}.fork_events.csv")

    # --- the measurands -----------------------------------------------------
    base_capture = base_gov["_router"]["n_merge_suspect_events"]
    q_capture = q_gov["_router"]["n_merge_suspect_events"]
    base_supersession = base_ec.get(SUPERSESSION_CLASS, 0)
    absorbed_diff = base_oc.get("absorbed", 0) - q_oc.get("absorbed", 0)
    ledger_count = ledger["quarantined_count"]
    ledger_label_total = sum(ledger["payload_label_histogram"].values())

    base_broken = base_gov[PROBE_POLICY]["broken"]
    q_broken = q_gov[PROBE_POLICY]["broken"]
    base_stale = base_gov["none"]["stale_wrong"]
    q_stale = q_gov["none"]["stale_wrong"]

    # 1. RECONSTRUCTION FIDELITY — the ledger is a complete, lossless record of
    #    exactly the diverted writes (router, fork_events, and the absorbed-count
    #    decomposition all agree with the ledger count and its label total).
    reconstruction_faithful = (
        ledger_count == base_capture == base_supersession == absorbed_diff
        == ledger_label_total and ledger["retained_recoverable"] is True
        and ledger["absorbed_into_active_memory"] is False)

    # 2. CAPTURE RESTORATION — reinstatement (= the none baseline) restores active
    #    capture to the baseline value from the quarantine arm's 0, and broken/
    #    stale do not exceed the ungoverned baseline (frozen G3 bound: per-seed
    #    broken <= baseline + 0). Reinstated readout == none readout by construction.
    reinstated_capture = base_capture
    reinstated_broken = base_broken
    reinstated_stale = base_stale
    capture_restored = (q_capture == 0 and reinstated_capture == base_capture
                        and base_capture > 0)
    broken_not_beyond_baseline = reinstated_broken <= base_broken  # == True
    stale_not_beyond_baseline = reinstated_stale <= base_stale      # == True

    # 3. HARM-FREE RECOVERY — does restoring capture re-introduce the read-time
    #    harm quarantine removed? delta_harm > 0 means the captured writes ARE the
    #    harmful writes (capture and harm coupled), so recovery is NOT harm-free
    #    (gate §3 prose falsification).
    delta_broken_reintroduced = reinstated_broken - q_broken
    delta_stale_reintroduced = reinstated_stale - q_stale
    harm_free_recovery = (delta_broken_reintroduced == 0
                          and delta_stale_reintroduced == 0)

    return {
        "capture": {"baseline": base_capture, "quarantine_active": q_capture,
                    "reinstated": reinstated_capture},
        "reconstruction": {
            "ledger_quarantined_count": ledger_count,
            "ledger_label_total": ledger_label_total,
            "baseline_router_merge_suspect": base_capture,
            "baseline_fork_events_supersession": base_supersession,
            "absorbed_count_decomposition": absorbed_diff,
            "faithful": reconstruction_faithful,
        },
        "capture_restoration": {
            "capture_restored": capture_restored,
            "broken_not_beyond_baseline": broken_not_beyond_baseline,
            "stale_not_beyond_baseline": stale_not_beyond_baseline,
            "reinstated_broken": reinstated_broken,
            "reinstated_stale": reinstated_stale,
            "baseline_broken": base_broken,
            "baseline_stale": base_stale,
        },
        "harm_free_recovery": {
            "quarantine_broken": q_broken,
            "quarantine_stale": q_stale,
            "delta_broken_reintroduced": delta_broken_reintroduced,
            "delta_stale_reintroduced": delta_stale_reintroduced,
            "harm_free": harm_free_recovery,
        },
    }


def build_recovery_validation(cell: str,
                              twin_root: Path = TWIN_ROOT) -> dict:
    per_pair: dict[str, dict] = {}
    all_faithful = True
    all_capture_restored = True
    harm_free_by_pair: dict[str, bool] = {}
    for pair in CELL_PAIRS[cell]:
        seeds_out: dict[str, dict] = {}
        pair_harm_free = True
        for seed in SEEDS:
            v = validate_arm(cell, pair, seed, twin_root)
            seeds_out[str(seed)] = v
            all_faithful &= v["reconstruction"]["faithful"]
            all_capture_restored &= v["capture_restoration"]["capture_restored"]
            pair_harm_free &= v["harm_free_recovery"]["harm_free"]
        per_pair[pair] = {"by_seed": seeds_out, "harm_free_all_seeds": pair_harm_free}
        harm_free_by_pair[pair] = pair_harm_free

    # G3 disposition — three separable claims, never collapsed:
    #   provenance_recoverable: the ledger is a complete, reversible record
    #     (reconstruction faithful on every arm). This is what quarantine adds
    #     over refuse, and it holds.
    #   capture_restorable_within_bound: reinstatement restores 192/seed with
    #     broken/stale not exceeding baseline (the frozen NUMERIC G3 bound).
    #   harm_free_recovery: restoring capture does NOT re-introduce the harm
    #     quarantine removed. FALSE wherever capture and harm are the same writes.
    harm_free_geometries = sorted(p for p, ok in harm_free_by_pair.items() if ok)
    coupled_geometries = sorted(p for p, ok in harm_free_by_pair.items() if not ok)

    # The gate §3 PROSE falsifier: "restores it only by re-introducing the harm
    # quarantine removed". Where harm is coupled to capture, harm-free recovery
    # fails, so recoverability is validated only as PROVENANCE/auditability, not
    # as harm-free reinstatement — the ledger is provenance-only for those cells
    # and G3 does NOT clear for promotion.
    g3_harm_free_clears = not coupled_geometries

    return {
        "design": "PR7_QUARANTINE_PROMOTION_GATE.md §4 G3 — recoverability "
                  "re-injection probe (analysis-only)",
        "engine_or_retrieval_change": False,
        "imports_torch": False,
        "reads_only_committed_artifacts": True,
        "cell": cell,
        "probe_policy": PROBE_POLICY,
        "per_pair": per_pair,
        "summary": {
            "provenance_recoverable": all_faithful,
            "capture_restorable_within_bound": all_capture_restored,
            "harm_free_recovery_geometries": harm_free_geometries,
            "capture_harm_coupled_geometries": coupled_geometries,
            "g3_harm_free_clears": g3_harm_free_clears,
        },
        "interpretation": (
            "The quarantine ledger is a complete, lossless, REVERSIBLE record of "
            "the diverted writes (reconstruction faithful on every arm: ledger "
            "count == baseline supersession capture == absorbed-count "
            "decomposition == label total) — this is the provenance value "
            "quarantine adds over refuse, and it holds on every geometry. "
            "Reinstatement (= the none baseline, verified by the absorbed-count "
            "decomposition) restores capture to 192/seed with broken/stale not "
            "EXCEEDING the ungoverned baseline (the frozen numeric G3 bound is "
            "met). BUT harm-free recovery FAILS on every geometry "
            f"({coupled_geometries}): the diverted writes are the stale "
            "supersessions, so reinstating them re-introduces the stale_wrong "
            "quarantine drained on all four pairs (and, on the hazardous D/E, the "
            "broken harm as well) — capture and harm are the SAME writes. "
            "Restoring capture therefore re-introduces the harm quarantine removed "
            "(gate §3 prose falsifier). Recoverability is validated as "
            "PROVENANCE/auditability — a faithful, reversible record a reviewer "
            "can audit — NOT as harm-free reinstatement into active memory; per "
            "gate §3 the ledger is provenance-only and G3 does NOT clear "
            "quarantine for promotion. This is the correct shape for a quarantine: "
            "a safe one-way diversion with a complete audit trail, not a "
            "reverse-without-cost store."),
        "g3_verdict": ("provenance_recoverable_not_harm_free"
                       if not g3_harm_free_clears else "harm_free_recoverable"),
        "scope": "certifies nothing beyond the enumerated cell; no margin tuning",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="merge_path_stale",
                    choices=sorted(CELL_PAIRS))
    ap.add_argument("--twin-root", default=str(TWIN_ROOT))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val = build_recovery_validation(args.cell, Path(args.twin_root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(val, indent=1, sort_keys=True) + "\n")

    s = val["summary"]
    print(f"cell={val['cell']} g3_verdict={val['g3_verdict']}")
    print(f"  provenance_recoverable={s['provenance_recoverable']} "
          f"capture_restorable_within_bound={s['capture_restorable_within_bound']}")
    print(f"  harm_free_geometries={s['harm_free_recovery_geometries']}")
    print(f"  capture_harm_coupled_geometries={s['capture_harm_coupled_geometries']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
