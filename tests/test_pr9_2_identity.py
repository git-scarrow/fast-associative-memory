"""tests/test_pr9_2_identity.py — hermetic gates for PR-9.2.

The registered change (PR9_2_IDENTITY_CERT.md; gate memo §8's required
pre-registration): the §9A flag-set-identity join moves from the
state-contaminated incumbent key (QUARANTINE_DIVERTED_JOIN_KEY — refuted
8/24 in the committed identity smoke, divergence entirely in
``incumbent_last_write_seq``) to the write-event-INTRINSIC key
``(epoch, event_class, batch_index)`` with ``payload_label`` as a
consistency-check field, persisted explicitly on BOTH ledger sides:

  * driver: ``record_shadow_flag`` gains an optional ``flagged_events``
    per-event list mirroring ``record_quarantine``'s ``diverted_events``
    (same fields, same pre-write capture point); the shadow ledger gains
    ``flagged_event_join_key`` / ``flagged_event_check_fields`` /
    ``flagged_event_records`` — all additive, retrieval byte-inert;
  * harness: when both per-event lists exist, identity is decided on the
    intrinsic key; a duplicate key on either side is a STOP condition
    (identity-collision), never a joinable multiset; the incumbent key is
    demoted to a reported diagnostic; legacy artifacts (no shadow
    records) still take the incumbent path unchanged.

CPU-only, no GPU, no network.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmarks.pr8_shadow_audit_cert as cert  # noqa: E402
from benchmarks.failure_mode_probe import (  # noqa: E402
    EVENT_SUPERSESSION, PR92_INTRINSIC_CHECK_FIELDS, PR92_INTRINSIC_JOIN_KEY,
    GovernanceHook)
from tests.test_pr8_shadow_audit_cert import (  # noqa: E402
    _build_smoke_tree, _quarantine_govern, _shadow_govern, _write_run)


# ---------------------------------------------------------------------------
# 1. constants pinned engine <-> harness
# ---------------------------------------------------------------------------
def test_intrinsic_constants_match_engine():
    assert tuple(cert.INTRINSIC_JOIN_KEY) == tuple(PR92_INTRINSIC_JOIN_KEY)
    assert tuple(cert.INTRINSIC_CHECK_FIELDS) == \
        tuple(PR92_INTRINSIC_CHECK_FIELDS)
    led = GovernanceHook("shadow").provenance()["quarantine_ledger"]
    assert led["flagged_event_join_key"] == list(PR92_INTRINSIC_JOIN_KEY)
    assert led["flagged_event_check_fields"] == \
        list(PR92_INTRINSIC_CHECK_FIELDS)
    assert led["flagged_event_records"] == []


# ---------------------------------------------------------------------------
# 2. hook: shadow per-event records are additive, aggregate path unchanged
# ---------------------------------------------------------------------------
def _flag_ev(bi, label, slot=5, seq=10, epoch=3):
    return {"epoch": epoch, "event_class": EVENT_SUPERSESSION,
            "batch_index": bi, "payload_label": label,
            "incumbent_slot": slot, "incumbent_last_write_seq": seq}


def test_shadow_aggregate_shape_unchanged_without_records():
    h = GovernanceHook("shadow")
    h.record_shadow_flag(EVENT_SUPERSESSION, [3, 3, 7])
    led = h.provenance()["quarantine_ledger"]
    assert led["flagged_count"] == 3
    assert led["payload_label_histogram"] == {3: 2, 7: 1}
    assert led["disposition"] == "flagged_not_diverted"
    assert led["flagged_event_records"] == []  # additive, empty when omitted


def test_shadow_records_appended_with_run_global_event_index():
    h = GovernanceHook("shadow")
    h.record_shadow_flag(EVENT_SUPERSESSION, [3, 7],
                         flagged_events=[_flag_ev(0, 3), _flag_ev(1, 7)])
    h.record_shadow_flag(EVENT_SUPERSESSION, [3],
                         flagged_events=[_flag_ev(0, 3, epoch=4)])
    led = h.provenance()["quarantine_ledger"]
    recs = led["flagged_event_records"]
    assert [e["event_index"] for e in recs] == [0, 1, 2]
    assert led["flagged_count"] == 3
    for e in recs:
        assert all(k in e for k in PR92_INTRINSIC_JOIN_KEY)
        assert all(k in e for k in PR92_INTRINSIC_CHECK_FIELDS)


# ---------------------------------------------------------------------------
# 3. harness fixtures — the intrinsic join semantics
# ---------------------------------------------------------------------------
def _shadow_govern_with_records(flagged, hist, records):
    gov = _shadow_govern(flagged, hist)
    gov["quarantine_ledger"]["flagged_event_join_key"] = \
        list(cert.INTRINSIC_JOIN_KEY)
    gov["quarantine_ledger"]["flagged_event_check_fields"] = \
        list(cert.INTRINSIC_CHECK_FIELDS)
    gov["quarantine_ledger"]["flagged_event_records"] = records
    return gov


def _intrinsic_events(n, *, epoch=3, label=3, seqs=None, batch_indices=None,
                      labels=None):
    """Per-event dicts whose intrinsic key is (epoch, class, batch_index).
    ``seqs`` sets incumbent_last_write_seq per event (state divergence knob)."""
    out = []
    for i in range(n):
        out.append({
            "event_index": i, "epoch": epoch,
            "event_class": cert.SUPERSESSION_CLASS,
            "batch_index": (batch_indices[i] if batch_indices else i),
            "payload_label": (labels[i] if labels else label),
            "incumbent_slot": 100 + i,
            "incumbent_last_write_seq": (seqs[i] if seqs else 100 + i)})
    return out


def _trio(root, *, shadow_records, q_diverted, super_seqs=(100, 101, 102, 103)):
    """merge_path_stale/pairD/s0 none+shadow+quarantine with explicit
    per-event records on both sides."""
    base = {"arm": "stale", "n_probes": 1, "readout": "frozen-87"}
    d = root / "merge_path_stale"
    n = len(shadow_records)
    hist = {"3": n} if n else {}
    _write_run(d / "none" / "stale_s0_pairD", fork_super_seqs=list(super_seqs),
               govern=None, base_summary=base)
    _write_run(d / "shadow" / "stale_s0_pairD",
               fork_super_seqs=list(super_seqs),
               govern=_shadow_govern_with_records(n, hist, shadow_records),
               base_summary=base)
    _write_run(d / "quarantine" / "stale_s0_pairD", fork_super_seqs=[],
               govern=_quarantine_govern(len(q_diverted),
                                         {"3": len(q_diverted)} or {},
                                         diverted_events=q_diverted),
               base_summary=base)
    return {"panel": "pr8-9a", "readout": "frozen-87", "mode": "smoke",
            "artifact_root": str(root), "seeds": [0],
            "runs": [{"cell": "merge_path_stale", "pair": "pairD", "seed": 0,
                      "stems": {
                          "none": "merge_path_stale/none/stale_s0_pairD",
                          "shadow": "merge_path_stale/shadow/stale_s0_pairD",
                          "quarantine":
                              "merge_path_stale/quarantine/stale_s0_pairD"}}]}


def test_intrinsic_proven_despite_incumbent_divergence(tmp_path):
    """THE design claim: identical intrinsic keys + check fields decide
    identity-proven even when incumbent_last_write_seq diverges on 3 of 4
    events (the §8 refutation mechanism). The incumbent diagnostic records
    the divergence without gating."""
    shadow = _intrinsic_events(4, seqs=[100, 999, 998, 997])
    quarantine = _intrinsic_events(4, seqs=[100, 101, 102, 103])
    rep = cert.run_certification(_trio(tmp_path, shadow_records=shadow,
                                       q_diverted=quarantine))
    fid = rep["runs"][0]["flag_set_identity"]
    assert fid["identity_status"] == "identity-proven", fid
    assert fid["status"] == cert.PASS
    assert fid["intrinsic_join_key"] == list(cert.INTRINSIC_JOIN_KEY)
    diag = fid["diagnostic_incumbent_key_agreement"]
    assert diag == {**diag, "agree": 1, "joined": 4}  # divergence recorded
    assert fid["check_field_mismatches"] == []
    assert fid["shadow_ledger_fork_events_link_ok"] is True


def test_intrinsic_collision_is_stop_condition(tmp_path):
    quarantine = _intrinsic_events(4, batch_indices=[0, 1, 2, 2])  # duplicate
    shadow = _intrinsic_events(4)
    rep = cert.run_certification(_trio(tmp_path, shadow_records=shadow,
                                       q_diverted=quarantine))
    fid = rep["runs"][0]["flag_set_identity"]
    assert fid["identity_status"] == "identity-collision"
    assert fid["status"] == cert.FAIL
    assert fid["collisions"]["quarantine"], fid["collisions"]
    assert rep["verdict"] == cert.FAIL


def test_intrinsic_check_field_contradiction_violates(tmp_path):
    shadow = _intrinsic_events(4)
    quarantine = _intrinsic_events(4, labels=[3, 3, 3, 7])  # payload differs
    rep = cert.run_certification(_trio(tmp_path, shadow_records=shadow,
                                       q_diverted=quarantine))
    fid = rep["runs"][0]["flag_set_identity"]
    assert fid["identity_status"] == "identity-violated"
    assert fid["check_field_mismatches"] and \
        fid["check_field_mismatches"][0]["field"] == "payload_label"


def test_intrinsic_missing_event_violates(tmp_path):
    shadow = _intrinsic_events(4)
    quarantine = _intrinsic_events(4, batch_indices=[0, 1, 2, 9])
    rep = cert.run_certification(_trio(tmp_path, shadow_records=shadow,
                                       q_diverted=quarantine))
    fid = rep["runs"][0]["flag_set_identity"]
    assert fid["identity_status"] == "identity-violated"
    assert fid["diverted_minus_flagged"] and fid["shadow_flagged_minus_diverted"]


def test_legacy_shadow_without_records_takes_incumbent_path(tmp_path):
    """Pre-PR-9.2 shadow artifacts (no flagged_event_records) score exactly as
    before: the incumbent-key multiset join decides."""
    manifest = _build_smoke_tree(tmp_path, quarantine_diverted="match")
    rep = cert.run_certification(manifest)
    fid = next(r for r in rep["runs"]
               if r["cell"] == "merge_path_stale")["flag_set_identity"]
    assert fid["identity_status"] == "identity-proven"
    assert "intrinsic_join_key" not in fid  # legacy path, not intrinsic


# ---------------------------------------------------------------------------
# 4. quarantine re-run fidelity vs committed §8 arm
# ---------------------------------------------------------------------------
def _fidelity_pair(root):
    base = {"arm": "stale", "n_probes": 1, "readout": "frozen-87"}
    committed = root / "committed" / "stale_s0_pairD"
    rerun = root / "rerun" / "stale_s0_pairD"
    _write_run(committed, fork_super_seqs=[],
               govern=_quarantine_govern(4, {"3": 4}), base_summary=base)
    _write_run(rerun, fork_super_seqs=[],
               govern=_quarantine_govern(4, {"3": 4},
                                         diverted_events=_intrinsic_events(4)),
               base_summary=base)
    return committed, rerun


def test_fidelity_passes_on_additive_only_ledger_growth(tmp_path):
    committed, rerun = _fidelity_pair(tmp_path)
    out = cert.check_quarantine_rerun_fidelity(cert.ShaLog(), rerun, committed)
    assert out["status"] == cert.PASS, out


def test_fidelity_ignores_registry_version_but_not_run_fields(tmp_path):
    """`implemented_actions` records the driver's action REGISTRY (which grew
    when PR-8 registered `shadow`), not the run — the §8-era committed
    ledgers list 4 actions, an instrumented re-run lists 5. Fidelity must
    pass on that skew alone, and still fail on any run-describing field."""
    committed, rerun = _fidelity_pair(tmp_path)
    cs = committed.parent / (committed.name + ".summary.json")
    summ = json.loads(cs.read_text())
    summ["govern"]["implemented_actions"] = \
        ["none", "annotate", "quarantine", "refuse"]  # the §8-era registry
    cs.write_text(json.dumps(summ, indent=2))
    out = cert.check_quarantine_rerun_fidelity(cert.ShaLog(), rerun, committed)
    assert out["status"] == cert.PASS, out
    # a run-describing field still gates
    summ["govern"]["events_seen"] = 999
    cs.write_text(json.dumps(summ, indent=2))
    out = cert.check_quarantine_rerun_fidelity(cert.ShaLog(), rerun, committed)
    assert out["status"] == cert.FAIL


def test_fidelity_fails_on_retrieval_byte_drift(tmp_path):
    committed, rerun = _fidelity_pair(tmp_path)
    f = rerun.parent / (rerun.name + ".csv")
    f.write_text(f.read_text().replace("0,3,1", "0,3,0"))
    out = cert.check_quarantine_rerun_fidelity(cert.ShaLog(), rerun, committed)
    assert out["status"] == cert.FAIL


def test_fidelity_dimension_only_when_declared(tmp_path):
    """Manifests that never declare quarantine_committed keep the pre-PR-9.2
    dimension set; declaring it adds the dimension and gates on it."""
    manifest = _build_smoke_tree(tmp_path, quarantine_diverted="match")
    rep = cert.run_certification(manifest)
    assert "quarantine_rerun_fidelity" not in rep["dimensions"]
    # build a committed-era counterpart: identical artifacts, aggregate-only
    # ledger (no diverted_events / join-key fields) — the §8 shape
    base = {"arm": "stale", "n_probes": 1, "readout": "frozen-87"}
    _write_run(tmp_path / "committed" / "stale_s0_pairD", fork_super_seqs=[],
               govern=_quarantine_govern(4, {"3": 4}), base_summary=base)
    manifest["runs"][0]["stems"]["quarantine_committed"] = \
        "committed/stale_s0_pairD"
    rep2 = cert.run_certification(manifest)
    assert rep2["dimensions"]["quarantine_rerun_fidelity"] == cert.PASS, \
        rep2["runs"][0]["quarantine_rerun_fidelity"]


# ---------------------------------------------------------------------------
# 5. cross-host via gentoo sha256 manifest (the PR-10-consistent reading)
# ---------------------------------------------------------------------------
def test_cross_host_sha_manifest_pass_and_fail(tmp_path):
    manifest = _build_smoke_tree(tmp_path, quarantine_diverted="match")
    rep = cert.run_certification(manifest)  # collect the read-set first
    import hashlib
    lines = []
    for p in sorted(tmp_path.rglob("*")):
        if p.is_file():
            rel = p.relative_to(tmp_path)
            lines.append(
                f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}")
    shas = tmp_path / "gentoo.sha256"
    shas.write_text("\n".join(lines) + "\n")
    manifest["gentoo_sha_manifest"] = str(shas)
    rep = cert.run_certification(manifest)
    assert rep["cross_host_determinism"]["status"] == cert.PASS
    assert rep["cross_host_determinism"]["method"] == "gentoo-sha-manifest"
    # corrupt one digest -> FAIL
    text = shas.read_text().splitlines()
    first_digest = text[0].split("  ")[0]
    flipped = ("0" if first_digest[0] != "0" else "1") + first_digest[1:]
    text[0] = flipped + "  " + text[0].split("  ", 1)[1]
    shas.write_text("\n".join(text) + "\n")
    rep = cert.run_certification(manifest)
    assert rep["cross_host_determinism"]["status"] == cert.FAIL


# ---------------------------------------------------------------------------
# 6. integration: real probe trio — intrinsic identity PROVEN on divergent
#    runs where the incumbent key is refuted (the committed-smoke mirror)
# ---------------------------------------------------------------------------
def test_vision_trio_intrinsic_proven_where_incumbent_refuted(tmp_path):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from benchmarks.failure_mode_probe import run_vision

    classes, attractor, dim, n_per = [0, 8], 71, 24, 24
    g = torch.Generator().manual_seed(0)
    ids = classes + [attractor]
    centers = F.normalize(torch.randn(len(ids), dim, generator=g), dim=-1)
    embeds, labels = [], []
    for ci, cid in enumerate(ids):
        x = centers[ci] + 0.05 * torch.randn(n_per, dim, generator=g)
        embeds.append(F.normalize(x, dim=-1))
        labels += [cid] * n_per
    cache = tmp_path / "cache.pt"
    torch.save({"embeds": torch.cat(embeds),
                "labels": torch.tensor(labels)}, cache)
    common = dict(epochs=6, cache_path=str(cache), classes=classes,
                  attractor_class=attractor, samples_per_class=8,
                  held_out_per_class=8, contraction=0.0, seed=0,
                  supersede_epoch=3, payload_mode="soft")

    stems = {}
    for arm in ("none", "shadow", "quarantine"):
        stem = tmp_path / "merge_path_stale" / arm / "stale_s0_pairD"
        stem.parent.mkdir(parents=True, exist_ok=True)
        _, summary = run_vision("stale", rate=0.0,
                                out_path=stem.with_suffix(".csv"),
                                govern=arm, **common)
        stem.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2))
        stems[arm] = f"merge_path_stale/{arm}/stale_s0_pairD"

    manifest = {"panel": "pr8-9a", "readout": "frozen-87", "mode": "smoke",
                "artifact_root": str(tmp_path), "seeds": [0],
                "runs": [{"cell": "merge_path_stale", "pair": "pairD",
                          "seed": 0, "stems": stems}]}
    rep = cert.run_certification(manifest)
    run = rep["runs"][0]
    assert run["inertness"]["status"] == cert.PASS
    fid = run["flag_set_identity"]
    # the intrinsic path ran and PROVED identity on genuinely divergent runs
    assert fid["identity_status"] == "identity-proven", fid
    assert fid["intrinsic_join_key"] == list(cert.INTRINSIC_JOIN_KEY)
    assert fid["shadow_record_count"] == fid["quarantine_record_count"] == 24
    # the incumbent diagnostic reproduces the committed smoke's refutation
    # shape: agreement only on the FIRST supersession epoch (8 of 24)
    diag = fid["diagnostic_incumbent_key_agreement"]
    assert diag["joined"] == 24 and diag["agree"] == 8, diag
