"""tests/test_pr7_refuse_behavior.py — PR-7 step 5: the first ACTING arm,
`refuse` (PR7_DESIGN.md §4/§13).

`refuse` skips the already-classified write-time merge_suspect (supersession)
write BEFORE it commits, and ONLY that write class — non-suspect traffic (clean
/ contradiction) is allowed unchanged, the deployed read-time path is untouched,
and the engine stays byte-frozen (pinned in test_pr7_govern_noop.py). These
tests run hermetically (synthetic CPU + a tiny in-memory vision cache); no
Gentoo cache and no torch-GPU is required. Pinned behaviors:

  * GovernanceHook.allow_write refuses ONLY (action=refuse, supersession); every
    other action and event class is allowed (so none/annotate/quarantine keep
    the baseline write path byte-for-byte);
  * on a supersession arm, refuse changes the emitted artifacts and drops every
    supersession write event, while keeping the non-suspect write events;
  * refuse provenance records the refused count, the refused event class, and a
    human reason; unimplemented `quarantine` stays a no-op.
"""
import csv
import io
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.failure_mode_probe import (
    EVENT_SUPERSESSION, GOVERN_ALLOW, GOVERN_REFUSE, GOVERN_REFUSE_EVENT_CLASS,
    GovernanceHook, run_synthetic, run_vision)

ROOT = Path(__file__).resolve().parent.parent

_SUFFIXES = (".csv", ".per_slot.csv", ".fork_events.csv", ".topk.csv")


def _emitted_bytes(out: Path) -> dict:
    base = out.with_suffix("")
    return {sfx: Path(f"{base}{sfx}").read_bytes() for sfx in _SUFFIXES}


def _event_class_counts(emitted: dict) -> dict:
    rows = list(csv.DictReader(io.StringIO(emitted[".fork_events.csv"].decode())))
    counts: dict = {}
    for r in rows:
        counts[r["event_class"]] = counts.get(r["event_class"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# 1. GovernanceHook: refuse is selective; other actions are unaffected
# ---------------------------------------------------------------------------
def test_allow_write_refuses_only_merge_suspect():
    assert GOVERN_REFUSE_EVENT_CLASS == EVENT_SUPERSESSION
    refuse = GovernanceHook("refuse")
    # the ONLY refused case: refuse action on the supersession (merge_suspect)
    # write class.
    assert refuse.allow_write(EVENT_SUPERSESSION) == GOVERN_REFUSE
    # every non-suspect class is allowed (incl. one-shot, which stays
    # observe-only and is never refused, PR7_DESIGN §12).
    for ev in ("initial", "duplicate-rewrite", "clean-rewrite", "contradiction",
               "one-shot-ambiguous"):
        assert refuse.allow_write(ev) == GOVERN_ALLOW
    # the null-action actions never divert, even on the supersession class.
    for action in ("none", "annotate"):
        assert GovernanceHook(action).allow_write(EVENT_SUPERSESSION) \
            == GOVERN_ALLOW


def test_refuse_provenance_records_count_and_reason():
    h = GovernanceHook("refuse")
    h.record_refusal(EVENT_SUPERSESSION, 5)
    h.record_refusal(EVENT_SUPERSESSION, 3)
    h.decide("clean-rewrite", "forked")  # one allowed write
    prov = h.provenance()
    assert prov["action"] == "refuse"
    assert prov["step"] == "pr7-step5-refuse"
    assert prov["implemented"] is True
    assert prov["refused_events"] == 8
    assert prov["refused_event_class"] == EVENT_SUPERSESSION
    assert "merge_suspect" in prov["reason"] and "supersession" in prov["reason"]
    # events_seen counts allowed (decide) + refused (record_refusal) writes.
    assert prov["events_seen"] == 9


def test_quarantine_diverts_only_merge_suspect():
    # quarantine (step 6) is the second acting arm: it diverts the supersession
    # (merge_suspect) write to the recoverable ledger, and ONLY that class.
    from benchmarks.failure_mode_probe import (
        GOVERN_QUARANTINE, GOVERN_QUARANTINE_EVENT_CLASS)
    assert GOVERN_QUARANTINE_EVENT_CLASS == EVENT_SUPERSESSION
    q = GovernanceHook("quarantine")
    assert q.allow_write(EVENT_SUPERSESSION) == GOVERN_QUARANTINE
    assert q.implemented is True
    assert q.provenance()["step"] == "pr7-step6-quarantine"
    # every non-suspect class is allowed (incl. one-shot, never quarantined).
    for ev in ("initial", "duplicate-rewrite", "clean-rewrite", "contradiction",
               "one-shot-ambiguous"):
        assert q.allow_write(ev) == GOVERN_ALLOW
    # refuse never diverts to quarantine, and quarantine never returns the refuse
    # decision — the two acting arms are distinct dispositions.
    assert GovernanceHook("refuse").allow_write(EVENT_SUPERSESSION) != \
        GOVERN_QUARANTINE


def test_quarantine_ledger_records_count_reason_and_payload():
    """The quarantine ledger RETAINS the diverted writes recoverable: it records
    opportunity/quarantined counts, a reason, and a per-label payload histogram
    (the accounting that distinguishes quarantine from refuse, which discards)."""
    h = GovernanceHook("quarantine")
    h.record_quarantine(EVENT_SUPERSESSION, [3, 3, 7])
    h.record_quarantine(EVENT_SUPERSESSION, [3])
    h.decide("clean-rewrite", "forked")  # one allowed (committed) write
    prov = h.provenance()
    assert prov["action"] == "quarantine"
    assert prov["step"] == "pr7-step6-quarantine"
    assert prov["implemented"] is True
    assert prov["quarantined_events"] == 4
    assert prov["quarantined_event_class"] == EVENT_SUPERSESSION
    assert "merge_suspect" in prov["reason"] and "recoverable" in prov["reason"]
    ledger = prov["quarantine_ledger"]
    assert ledger["opportunity_count"] == 4
    assert ledger["quarantined_count"] == 4
    assert ledger["retained_recoverable"] is True
    assert ledger["absorbed_into_active_memory"] is False
    # payload accounting: per-label histogram of the diverted (retained) rows.
    assert ledger["payload_label_histogram"] == {3: 3, 7: 1}
    assert "recoverable" in ledger["reason"]
    # events_seen counts allowed (decide) + quarantined (diverted) writes.
    assert prov["events_seen"] == 5
    # no refusal tally on a quarantine arm (it retains, it does not discard).
    assert "refused_events" not in prov


# ---------------------------------------------------------------------------
# 2. Synthetic run: refuse skips supersession writes, keeps the rest
# ---------------------------------------------------------------------------
def test_refuse_drops_supersession_events_only(tmp_path):
    """On the synthetic stale arm, refuse changes the emitted artifacts and
    removes every supersession write event, while the non-suspect write events
    (clean/initial/duplicate) remain — refuse touches only merge_suspect."""
    base_out = tmp_path / "none.csv"
    ref_out = tmp_path / "refuse.csv"
    n_base = run_synthetic("stale", rate=0.0, epochs=6, supersede_epoch=3,
                           out_path=base_out, seed=0, govern="none")
    n_ref = run_synthetic("stale", rate=0.0, epochs=6, supersede_epoch=3,
                          out_path=ref_out, seed=0, govern="refuse")
    assert n_base > 0 and n_ref > 0
    base, refused = _emitted_bytes(base_out), _emitted_bytes(ref_out)

    base_counts = _event_class_counts(base)
    ref_counts = _event_class_counts(refused)
    # baseline supersedes; refuse skips every one of those writes.
    assert base_counts.get(EVENT_SUPERSESSION, 0) > 0
    assert ref_counts.get(EVENT_SUPERSESSION, 0) == 0
    # non-suspect write events are still emitted under refuse.
    non_suspect = set(base_counts) - {EVENT_SUPERSESSION}
    assert non_suspect and all(ref_counts.get(c, 0) > 0 for c in non_suspect)
    # and the skipped writes changed the run's emitted artifacts.
    assert refused != base


# ---------------------------------------------------------------------------
# 3. Vision summary (hermetic tiny cache): refuse provenance on a real run
# ---------------------------------------------------------------------------
_CLASSES = [0, 8]
_ATTRACTOR = 71


def _make_cache(path: Path, dim=24, n_per=24, noise=0.05, seed=0) -> Path:
    g = torch.Generator().manual_seed(seed)
    ids = _CLASSES + [_ATTRACTOR]
    centers = F.normalize(torch.randn(len(ids), dim, generator=g), dim=-1)
    embeds, labels = [], []
    for ci, cid in enumerate(ids):
        x = centers[ci] + noise * torch.randn(n_per, dim, generator=g)
        embeds.append(F.normalize(x, dim=-1))
        labels += [cid] * n_per
    blob = {"embeds": torch.cat(embeds), "labels": torch.tensor(labels)}
    p = path / "tiny_cache.pt"
    torch.save(blob, p)
    return p


def test_vision_refuse_summary_and_artifacts(tmp_path):
    """A hermetic merge-path (stale-soft) vision run: refuse records a non-zero
    refused count in provenance and changes the emitted artifacts vs baseline,
    while the baseline summary carries no govern block."""
    cache = _make_cache(tmp_path)
    common = dict(epochs=6, cache_path=str(cache), classes=_CLASSES,
                  attractor_class=_ATTRACTOR, samples_per_class=8,
                  held_out_per_class=8, contraction=0.0, seed=0,
                  supersede_epoch=3, payload_mode="soft")
    _, s_base = run_vision("stale", rate=0.0, out_path=tmp_path / "b.csv",
                           govern="none", **common)
    _, s_ref = run_vision("stale", rate=0.0, out_path=tmp_path / "r.csv",
                          govern="refuse", **common)
    assert "govern" not in s_base
    gov = s_ref["govern"]
    assert gov["action"] == "refuse"
    assert gov["refused_events"] > 0
    assert gov["refused_event_class"] == EVENT_SUPERSESSION
    # refuse skipped real writes, so the emitted artifacts diverge from baseline.
    assert _emitted_bytes(tmp_path / "r.csv") != _emitted_bytes(tmp_path / "b.csv")


# ---------------------------------------------------------------------------
# 4. Quarantine (step 6): diverts supersession writes to the recoverable ledger
# ---------------------------------------------------------------------------
def test_quarantine_diverts_supersession_events_only(tmp_path):
    """On the synthetic stale arm, quarantine (like refuse) removes every
    supersession write event from the emitted artifacts — those writes are kept
    out of the active memory state — while non-suspect write events remain. The
    difference from refuse is the recoverable ledger, asserted on the summary."""
    base_out = tmp_path / "none.csv"
    q_out = tmp_path / "quarantine.csv"
    n_base = run_synthetic("stale", rate=0.0, epochs=6, supersede_epoch=3,
                           out_path=base_out, seed=0, govern="none")
    n_q = run_synthetic("stale", rate=0.0, epochs=6, supersede_epoch=3,
                        out_path=q_out, seed=0, govern="quarantine")
    assert n_base > 0 and n_q > 0
    base, quar = _emitted_bytes(base_out), _emitted_bytes(q_out)

    base_counts = _event_class_counts(base)
    q_counts = _event_class_counts(quar)
    # baseline supersedes; quarantine diverts every one of those writes out of
    # the active state, so no supersession write event is emitted.
    assert base_counts.get(EVENT_SUPERSESSION, 0) > 0
    assert q_counts.get(EVENT_SUPERSESSION, 0) == 0
    # non-suspect write events are still emitted under quarantine (only the
    # merge_suspect class is touched).
    non_suspect = set(base_counts) - {EVENT_SUPERSESSION}
    assert non_suspect and all(q_counts.get(c, 0) > 0 for c in non_suspect)
    # the diverted writes changed the run's emitted artifacts.
    assert quar != base


def test_vision_quarantine_summary_and_ledger(tmp_path):
    """A hermetic merge-path (stale-soft) vision run: quarantine records a
    non-zero recoverable ledger (opportunity/quarantined counts + payload
    histogram) in provenance and changes the emitted artifacts vs baseline, while
    the baseline summary carries no govern block."""
    cache = _make_cache(tmp_path)
    common = dict(epochs=6, cache_path=str(cache), classes=_CLASSES,
                  attractor_class=_ATTRACTOR, samples_per_class=8,
                  held_out_per_class=8, contraction=0.0, seed=0,
                  supersede_epoch=3, payload_mode="soft")
    _, s_base = run_vision("stale", rate=0.0, out_path=tmp_path / "b.csv",
                           govern="none", **common)
    _, s_q = run_vision("stale", rate=0.0, out_path=tmp_path / "q.csv",
                        govern="quarantine", **common)
    assert "govern" not in s_base
    gov = s_q["govern"]
    assert gov["action"] == "quarantine"
    assert gov["quarantined_events"] > 0
    assert gov["quarantined_event_class"] == EVENT_SUPERSESSION
    ledger = gov["quarantine_ledger"]
    assert ledger["quarantined_count"] == gov["quarantined_events"]
    assert ledger["opportunity_count"] == gov["quarantined_events"]
    assert ledger["retained_recoverable"] is True
    assert ledger["absorbed_into_active_memory"] is False
    # payload accounting retained: the histogram totals the quarantined rows.
    assert sum(ledger["payload_label_histogram"].values()) == \
        gov["quarantined_events"]
    # quarantine diverted real writes, so artifacts diverge from baseline.
    assert _emitted_bytes(tmp_path / "q.csv") != _emitted_bytes(tmp_path / "b.csv")
