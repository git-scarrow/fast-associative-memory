"""tests/test_read_path_invariants.py — temporal/provenance invariant audit (PR-3a).

PR-3 plans to use ``last_seen`` (write recency), ``hit_counts`` (maturity),
``usage``, and ``slot_records`` (write lineage) as temporal/provenance
evidence for fork governance (PR3_DESIGN.md §5 PR-3a(i)). That evidence is
only valid if NO read/inference path mutates those fields: a ``last_seen``
that updates on retrieval is read recency, not write recency, and every
recency feature built on it would be invalid.

These tests pin the invariant the design memo's code reading suggested
(all mutation sites in ``learn_local``'s hit/miss blocks; ``forward()``
documented read-only): after exercising every read path the failure-mode
study uses, the audited fields — plus keys/values/occupied for good
measure — are bit-identical to their pre-read snapshot.

A positive control pins that the snapshot machinery DOES detect mutation
(``learn_local`` changes the fields), so a pass is not vacuous.

If any of these tests ever fails after an engine change, the affected
field must be cut from the PR-3 feature set (PR3_DESIGN.md §10), not
papered over.

Hermetic: CPU, seconds, no caches, no GPU, no network.
"""
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from associative_core import ContinuousCAM  # noqa: E402
from dynamic_vigilance import (  # noqa: E402
    DynamicVigilance, RetrievalFloorPolicy)
from benchmarks.failure_mode_probe import (  # noqa: E402
    FailureModeRegistry, build_synthetic_stream, label_probes)

# The fields PR-3 wants as temporal/provenance evidence (the audit's subject)
AUDITED_TENSORS = ("last_seen", "hit_counts", "usage")
# Storage fields included for completeness: a read path that rewrote a key
# or payload would invalidate far more than the temporal features.
STORAGE_TENSORS = ("keys", "values", "occupied")


def _populated_memory(C=4, dim=32, seed=0):
    """A #87-config memory (margin vigilance + live-Δ floor + provenance)
    populated through the normal learn path, including a contradiction fork
    and a supersession group, so slot_records carry non-trivial state."""
    batches, hq, hlab = build_synthetic_stream(
        num_classes=C, dim=dim, epochs=3, seed=seed)
    registry = FailureModeRegistry()
    mem = ContinuousCAM(key_dim=dim, value_dim=C, max_entries=256,
                        dynamic_vigilance=DynamicVigilance(),
                        retrieval_floor_policy=RetrievalFloorPolicy(),
                        track_provenance=True)
    rng = torch.Generator().manual_seed(seed + 1)
    group = registry.new_stale_group("audit", pre_label=0, post_label=C - 1)
    for epoch, (q, y, lab) in enumerate(batches):
        g0 = lab == 0
        if epoch < 2:
            registry.write_phase1(mem, group, q[g0], y[g0])
        else:
            if not group.superseded:
                registry.supersede(group, epoch)
            y2 = F.one_hot(torch.full((int(g0.sum()),), C - 1), C).float()
            registry.write_phase2(mem, group, q[g0], y2)
        ids = registry.next_ids("clean", int((~g0).sum()))
        mem.learn_local(q[~g0], y[~g0], record_ids=ids)
        registry.inject_contradictions(mem, q, lab, C, rate=0.2, rng=rng)
    truth = torch.where(hlab == 0, torch.full_like(hlab, C - 1), hlab)
    return mem, registry, hq, truth


def _snapshot(mem):
    snap = {name: getattr(mem, name).clone()
            for name in AUDITED_TENSORS + STORAGE_TENSORS}
    snap["slot_records"] = copy.deepcopy(mem.slot_records)
    return snap


def _assert_unchanged(mem, snap, context):
    for name in AUDITED_TENSORS + STORAGE_TENSORS:
        assert torch.equal(getattr(mem, name), snap[name]), (
            f"{name} mutated by read path: {context} — if intentional, this "
            f"field must be cut from the PR-3 temporal/provenance feature "
            f"set (PR3_DESIGN.md §10).")
    assert mem.slot_records == snap["slot_records"], (
        f"slot_records mutated by read path: {context}")


def test_read_paths_leave_temporal_provenance_fields_untouched():
    """Every read path the failure-mode study uses, exercised twice
    (a stateful mutation often shows up only on the second pass)."""
    mem, registry, hq, truth = _populated_memory()
    occ_idx = mem.occupied.nonzero(as_tuple=True)[0]
    snap = _snapshot(mem)

    for rep in range(2):
        mem.forward(hq)
        mem.forward(hq, trace=True)
        mem.probe_cross_class_similarity(hq, truth, return_per_probe=True)
        mem.get_stats()
        mem.effective_slot_labels(occ_idx)
        mem._get_nearest_batch(hq)
        mem._active_sim_floor()
        mem._coverage_eviction_score(occ_idx)
        for slot in occ_idx.tolist():
            mem.records_for(slot)
        mem.records_for_slots(occ_idx.tolist())
        _assert_unchanged(mem, snap, f"engine read paths (pass {rep + 1})")


def test_driver_read_paths_leave_fields_untouched():
    """The PR-2a driver's read-side machinery: label_probes (which replicates
    candidate selection) and the registry's slot-set recomputations."""
    mem, registry, hq, truth = _populated_memory()
    snap = _snapshot(mem)

    for rep in range(2):
        label_probes(mem, hq, truth, registry)
        registry.contra_fork_slots(mem)
        registry.stale_slots(mem)
        _assert_unchanged(mem, snap, f"driver read paths (pass {rep + 1})")


def test_snapshot_detects_write_mutation():
    """Positive control: learn_local DOES move the audited fields, so the
    equality assertions above cannot pass vacuously."""
    mem, registry, hq, truth = _populated_memory()
    snap = _snapshot(mem)

    q = F.normalize(torch.randn(4, mem.key_dim,
                                generator=torch.Generator().manual_seed(99)),
                    dim=-1)
    y = F.one_hot(torch.tensor([0, 1, 2, 3]), mem.value_dim).float()
    mem.learn_local(q, y, record_ids=registry.next_ids("control", 4))

    changed = [name for name in AUDITED_TENSORS + ("occupied",)
               if not torch.equal(getattr(mem, name), snap[name])]
    changed += (["slot_records"]
                if mem.slot_records != snap["slot_records"] else [])
    assert changed, ("learn_local changed nothing the snapshot tracks — "
                     "the audit would be vacuous.")
    # last_seen specifically must move on a write: it is the write-recency
    # field PR-3's recency features depend on.
    assert not torch.equal(mem.last_seen, snap["last_seen"]), (
        "learn_local did not update last_seen — write-recency semantics "
        "changed; PR-3 recency features are invalid.")
