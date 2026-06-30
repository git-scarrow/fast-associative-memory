"""Probe instrumentation: persist per-event diverted keys in the quarantine
ledger (PR: "persist diverted event keys for audit identity").

Proves the quarantine ledger became event-addressable WITHOUT changing
quarantine behavior, retrieval/readout artifacts, or the aggregate ledger
fields. The keys live ONLY in summary.json's ledger (never in the retrieval
CSVs), captured from pre-write observables already in hand at diversion time, so
the emitted retrieval artifacts are byte-identical to the pre-instrumentation
quarantine arm.
"""
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from benchmarks.failure_mode_probe import (  # noqa: E402
    EVENT_SUPERSESSION, QUARANTINE_DIVERTED_JOIN_KEY, GovernanceHook,
    run_synthetic, run_vision)

_SUFFIXES = (".csv", ".per_slot.csv", ".fork_events.csv", ".topk.csv")
_CLASSES = [0, 8]
_ATTRACTOR = 71


def _emitted(out: Path) -> dict:
    base = out.with_suffix("")
    return {s: Path(f"{base}{s}").read_bytes() for s in _SUFFIXES}


def _make_cache(path: Path, dim=24, n_per=24, noise=0.05, seed=0) -> Path:
    g = torch.Generator().manual_seed(seed)
    ids = _CLASSES + [_ATTRACTOR]
    centers = F.normalize(torch.randn(len(ids), dim, generator=g), dim=-1)
    embeds, labels = [], []
    for ci, cid in enumerate(ids):
        x = centers[ci] + noise * torch.randn(n_per, dim, generator=g)
        embeds.append(F.normalize(x, dim=-1))
        labels += [cid] * n_per
    p = path / "tiny_cache.pt"
    torch.save({"embeds": torch.cat(embeds),
                "labels": torch.tensor(labels)}, p)
    return p


def _vision_common(cache):
    return dict(epochs=6, cache_path=str(cache), classes=_CLASSES,
                attractor_class=_ATTRACTOR, samples_per_class=8,
                held_out_per_class=8, contraction=0.0, seed=0,
                supersede_epoch=3, payload_mode="soft")


# ---------------------------------------------------------------------------
# 1. Aggregate-consumer compatibility: the existing aggregate path is unchanged
#    and diverted_events is purely additive.
# ---------------------------------------------------------------------------
def test_legacy_aggregate_shape_unchanged_without_keys():
    """Calling record_quarantine WITHOUT diverted_events reproduces the exact
    pre-instrumentation aggregate ledger; the new field is present but empty."""
    h = GovernanceHook("quarantine")
    h.record_quarantine(EVENT_SUPERSESSION, [3, 3, 7])
    h.record_quarantine(EVENT_SUPERSESSION, [3])
    led = h.provenance()["quarantine_ledger"]
    # aggregate fields byte-for-byte as before this PR
    assert led["opportunity_count"] == 4
    assert led["quarantined_count"] == 4
    assert led["payload_label_histogram"] == {3: 3, 7: 1}
    assert led["retained_recoverable"] is True
    assert led["absorbed_into_active_memory"] is False
    assert "recoverable" in led["reason"]
    # additive instrumentation: present, declared, but empty (no keys supplied)
    assert led["diverted_events"] == []
    assert led["diverted_event_join_key"] == list(QUARANTINE_DIVERTED_JOIN_KEY)


def test_keys_appended_with_run_global_event_index():
    """diverted_events carries the supplied per-event keys plus a run-global
    event_index; the aggregate histogram is still derived from labels."""
    h = GovernanceHook("quarantine")

    def ev(bi, slot, seq, label):
        return {"epoch": 3, "event_class": EVENT_SUPERSESSION, "batch_index": bi,
                "payload_label": label, "incumbent_slot": slot,
                "incumbent_last_write_seq": seq}

    h.record_quarantine(EVENT_SUPERSESSION, [3, 7],
                        diverted_events=[ev(0, 5, 10, 3), ev(1, 6, 11, 7)])
    h.record_quarantine(EVENT_SUPERSESSION, [3],
                        diverted_events=[ev(0, 9, 20, 3)])
    led = h.provenance()["quarantine_ledger"]
    evs = led["diverted_events"]
    assert led["quarantined_count"] == 3
    assert [e["event_index"] for e in evs] == [0, 1, 2]  # run-global ordinal
    for e in evs:
        assert all(k in e for k in QUARANTINE_DIVERTED_JOIN_KEY)
    # aggregate histogram unchanged in semantics; matches per-event labels
    assert led["payload_label_histogram"] == {3: 2, 7: 1}
    assert Counter(e["payload_label"] for e in evs) == \
        led["payload_label_histogram"]


# ---------------------------------------------------------------------------
# 2. Retrieval/readout unchanged: the capture is summary-only.
# ---------------------------------------------------------------------------
def test_quarantine_retrieval_byte_identical_with_and_without_capture(
        tmp_path, monkeypatch):
    """Every emitted retrieval artifact (.csv/.per_slot/.fork_events/.topk) of a
    quarantine run is byte-identical whether or not the per-event keys are
    captured — proving the instrumentation touches only the summary ledger and
    does not perturb the write path / RNG / memory state."""
    import benchmarks.failure_mode_probe as fmp

    out_cap = tmp_path / "withcap.csv"
    n1 = run_synthetic("stale", rate=0.0, epochs=6, supersede_epoch=3,
                       out_path=out_cap, seed=0, govern="quarantine")
    with_cap = _emitted(out_cap)

    # simulate the pre-instrumentation path: ignore the diverted keys entirely
    orig = fmp.GovernanceHook.record_quarantine

    def no_capture(self, event_class, labels, diverted_events=None):
        return orig(self, event_class, labels, diverted_events=None)

    monkeypatch.setattr(fmp.GovernanceHook, "record_quarantine", no_capture)
    out_nocap = tmp_path / "nocap.csv"
    n2 = run_synthetic("stale", rate=0.0, epochs=6, supersede_epoch=3,
                       out_path=out_nocap, seed=0, govern="quarantine")
    no_cap = _emitted(out_nocap)

    assert n1 == n2 and n1 > 0
    assert with_cap == no_cap, (
        "capturing diverted keys changed an emitted retrieval artifact — the "
        "instrumentation must be summary-only")


def test_quarantine_divert_behavior_unchanged(tmp_path):
    """Behavior invariant: quarantine still removes EVERY supersession write from
    the emitted artifacts (diverted, not committed) and leaves non-suspect
    writes — the divert decision is untouched by the key capture."""
    import csv
    out = tmp_path / "q.csv"
    run_synthetic("stale", rate=0.0, epochs=6, supersede_epoch=3,
                  out_path=out, seed=0, govern="quarantine")
    rows = list(csv.DictReader(
        (out.with_suffix(".fork_events.csv")).read_text().splitlines()))
    classes = Counter(r["event_class"] for r in rows)
    assert classes.get(EVENT_SUPERSESSION, 0) == 0   # all diverted
    assert sum(classes.values()) > 0                 # non-suspect writes remain


# ---------------------------------------------------------------------------
# 3. Vision: the ledger is now event-addressable on real probe output.
# ---------------------------------------------------------------------------
def test_vision_quarantine_ledger_is_event_addressable(tmp_path):
    cache = _make_cache(tmp_path)
    _, s = run_vision("stale", rate=0.0, out_path=tmp_path / "q.csv",
                      govern="quarantine", **_vision_common(cache))
    led = s["govern"]["quarantine_ledger"]
    evs = led["diverted_events"]
    # one per-event record per diverted write; counts agree with the aggregate
    assert len(evs) == led["quarantined_count"] > 0
    assert s["govern"]["quarantined_events"] == len(evs)
    assert [e["event_index"] for e in evs] == list(range(len(evs)))
    for e in evs:
        assert e["event_class"] == EVENT_SUPERSESSION
        assert all(k in e for k in QUARANTINE_DIVERTED_JOIN_KEY)
    # aggregate histogram is reproducible from the per-event payload labels
    assert Counter(e["payload_label"] for e in evs) == \
        led["payload_label_histogram"]
    # aggregate fields preserve their committed semantics
    assert led["retained_recoverable"] is True
    assert led["absorbed_into_active_memory"] is False
    assert led["opportunity_count"] == led["quarantined_count"]


def test_vision_clean_quarantine_has_empty_diverted_events(tmp_path):
    cache = _make_cache(tmp_path)
    _, s = run_vision("clean", rate=0.0, out_path=tmp_path / "c.csv",
                      govern="quarantine", **_vision_common(cache))
    led = s["govern"]["quarantine_ledger"]
    assert led["quarantined_count"] == 0
    assert led["diverted_events"] == []
    assert led["payload_label_histogram"] == {}
