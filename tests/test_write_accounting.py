"""
tests/test_write_accounting.py — ContinuousCAM.last_write_stats correctness.

Guards the literal per-call write accounting that probe_contraction.py reports
(issue #73 instrumentation). The counts are derived from the hits/misses masks
and the actual allocation, NOT from occupancy deltas — the property under test
is exactly the one the occupancy-delta proxy got wrong: once the table
saturates and reuses evicted slots, occupancy stops growing while writes keep
allocating, so an occupancy-delta would mislabel allocations as merges.
"""
import torch
import torch.nn.functional as F
import pytest

from associative_core import ContinuousCAM
from dynamic_vigilance import DynamicVigilance


def _batch(n, dim, classes, seed):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(n, dim, generator=g), dim=-1)
    y = torch.zeros(n, classes)
    y[torch.arange(n), torch.randint(0, classes, (n,), generator=g)] = 1.0
    return q, y


def _mem(max_entries, dim=16, classes=4):
    return ContinuousCAM(key_dim=dim, value_dim=classes, max_entries=max_entries,
                         dynamic_vigilance=DynamicVigilance())


def test_counts_sum_to_written():
    """written == merged + allocated + dropped on every call."""
    mem = _mem(max_entries=256)
    for seed in range(5):
        q, y = _batch(32, 16, 4, seed)
        mem.learn_local(q, y)
        ws = mem.last_write_stats
        assert ws["written"] == 32
        assert ws["merged"] + ws["allocated"] + ws["dropped"] == ws["written"]


def test_first_batch_all_allocations():
    """Into empty memory every row is a miss → all allocated, none merged."""
    mem = _mem(max_entries=256)
    q, y = _batch(32, 16, 4, seed=0)
    mem.learn_local(q, y)
    ws = mem.last_write_stats
    assert ws["merged"] == 0
    assert ws["allocated"] == 32
    assert ws["dropped"] == 0


def test_rewrite_produces_merges():
    """Re-writing identical (query, label) pairs must register as merges, not
    new allocations (best_sim≈1.0 ≥ vigilance, same class → hit)."""
    mem = _mem(max_entries=256)
    q, y = _batch(32, 16, 4, seed=1)
    mem.learn_local(q, y)
    occ_after_first = int(mem.get_stats()["n_occupied"])
    mem.learn_local(q, y)
    ws = mem.last_write_stats
    assert ws["merged"] > 0, "identical re-writes should merge into prototypes"
    # No net occupancy growth from a pure-merge batch.
    assert int(mem.get_stats()["n_occupied"]) == occ_after_first


def test_allocations_counted_under_saturation():
    """The regression the proxy failed: once occupancy is pinned at capacity,
    fresh misses are still allocated (via eviction/reuse). last_write_stats must
    report those as allocations even though occupancy does not change."""
    cap = 40
    mem = _mem(max_entries=cap)
    # Saturate with distinct random batches (random labels rarely re-hit).
    for seed in range(3):
        q, y = _batch(32, 16, 4, seed)
        mem.learn_local(q, y)
    occ = int(mem.get_stats()["n_occupied"])
    assert occ == cap, "precondition: table saturated"

    # A fresh batch of misses: occupancy can't grow, but allocations happen.
    q, y = _batch(32, 16, 4, seed=99)
    mem.learn_local(q, y)
    ws = mem.last_write_stats
    occ_delta = int(mem.get_stats()["n_occupied"]) - occ
    assert occ_delta == 0, "saturated table: occupancy frozen"
    # Literal allocations are non-zero despite zero occupancy delta — the exact
    # case where merged = written - occ_delta would have been wrong.
    assert ws["allocated"] + ws["dropped"] == ws["written"] - ws["merged"]
    assert ws["allocated"] > 0 or ws["dropped"] > 0
    proxy_merged = ws["written"] - occ_delta  # the old, broken proxy
    assert proxy_merged != ws["merged"], "this scenario must expose the proxy bug"


def test_write_accounting_reports_slot_reuse_as_eviction():
    cam = ContinuousCAM(
        key_dim=2, value_dim=2, max_entries=1,
        adaptive_eviction=False, use_lfu=False,
        track_provenance=True,
    )
    cam.learn_local(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), record_ids=["old"])
    cam.learn_local(
        torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 1.0]]),
        record_ids=["new"], write_mode="allocate-only",
    )
    assert cam.last_write_stats["evicted"] == 1
    assert cam.records_for(0) == {"new"}


def test_learn_local_rejects_unknown_write_mode():
    cam = ContinuousCAM(key_dim=2, value_dim=1)
    with pytest.raises(ValueError, match="write_mode"):
        cam.learn_local(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0]]), write_mode="merge-ish")
