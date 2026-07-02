"""PR-8 §9A shadow-quarantine certification harness — smoke + teeth.

These tests prove the HARNESS MECHANICS only; they do NOT certify §9A (no full
panel, single host, and — as the harness itself reports — event-level flag-set
identity is unprovable from current committed artifacts). The expected verdict on
every fixture here is therefore ``incomplete``, never ``pass``.

Two layers:
  A. torch-free hand-crafted fixtures — fast, fully deterministic, and adversarial
     (each broken variant MUST flip the relevant dimension to ``fail``), plus a
     hypothetical "§8 keys present" fixture proving the identity check can both
     certify and refute when per-event keys actually exist on both sides;
  B. a real-probe vision smoke (one stale + one clean arm) proving the harness
     reads genuine failure_mode_probe artifacts;
  C. a constants pin so the harness's hardcoded engine strings cannot drift.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import pr8_shadow_audit_cert as cert  # noqa: E402

# ---------------------------------------------------------------------------
# Hand-crafted fixture builders (no torch)
# ---------------------------------------------------------------------------
_FORK_HEADER = ("arm,epoch,event_class,record_tag,record_seq,outcome,pre_sim,"
                "payload_cos_incumbent,effective_vigilance,incumbent_slot,"
                "incumbent_hit_counts,incumbent_last_write_seq,"
                "incumbent_n_records,owner_slot,injected_label")


def _fork_row(event_class, seq, *, slot, label=3, arm="stale"):
    return (f"{arm},3,{event_class},k{seq},{seq},absorbed,0.9,0.9,0.5,{slot},"
            f"1,{seq},1,{slot},{label}")


def _write_fork(path: Path, super_seqs, *, arm="stale"):
    rows = [_FORK_HEADER]
    # shared non-supersession traffic (identical across arms)
    for s in (10, 11, 12):
        rows.append(_fork_row("initial", s, slot=s, label=0, arm=arm))
    # supersession rows — present on committed (none/shadow) runs only
    for s in super_seqs:
        rows.append(_fork_row(cert.SUPERSESSION_CLASS, s, slot=s, label=3, arm=arm))
    path.write_text("\n".join(rows) + "\n")


def _shadow_govern(flagged, hist):
    return {
        "action": cert.SHADOW_ACTION, "step": cert.SHADOW_STEP,
        "implemented": True, "events_seen": 100,
        "flagged_events": flagged,
        "flagged_event_class": cert.SUPERSESSION_CLASS,
        "quarantine_ledger": {
            "opportunity_count": flagged, "flagged_count": flagged,
            "quarantined_count": 0, "retained_recoverable": False,
            "absorbed_into_active_memory": True,
            "payload_label_histogram": hist,
            "disposition": cert.FLAGGED_DISPOSITION, "reason": "audit-only"},
        "reason": "audit-only shadow quarantine"}


def _diverted_events(seqs, *, epoch=3):
    """Per-event diverted keys keyed so JOIN_KEY (epoch, event_class,
    incumbent_slot, incumbent_last_write_seq) matches the shadow fork rows built
    by `_fork_row` (which set incumbent_slot == incumbent_last_write_seq == seq)."""
    return [{"event_index": i, "epoch": epoch,
             "event_class": cert.SUPERSESSION_CLASS, "batch_index": i,
             "payload_label": 3, "incumbent_slot": s,
             "incumbent_last_write_seq": s} for i, s in enumerate(seqs)]


def _quarantine_govern(quarantined, hist, diverted_events=None):
    ledger = {
        "opportunity_count": quarantined, "quarantined_count": quarantined,
        "retained_recoverable": True, "absorbed_into_active_memory": False,
        "payload_label_histogram": hist, "reason": "diverted"}
    if diverted_events is not None:
        # post-instrumentation ledger: additive per-event keys
        ledger["diverted_event_join_key"] = list(cert.JOIN_KEY)
        ledger["diverted_events"] = diverted_events
    return {
        "action": cert.QUARANTINE_ACTION, "step": "pr7-step6-quarantine",
        "implemented": True, "events_seen": 100,
        "quarantined_events": quarantined,
        "quarantined_event_class": cert.SUPERSESSION_CLASS,
        "quarantine_ledger": ledger, "reason": "quarantine"}


def _write_run(stem: Path, *, fork_super_seqs, govern, base_summary,
               arm="stale", topk_gz=False):
    """Write the five emitted artifacts for one run stem. none/shadow share
    byte-identical csv/per_slot/topk/fork; only summary differs (govern block)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    (stem.parent / (stem.name + ".csv")).write_text(
        "probe_index,vote_pred_label,vote_correct\n0,3,1\n")
    (stem.parent / (stem.name + ".per_slot.csv")).write_text(
        "arm,epoch,slot,decode\nstale,3,7,3\n")
    topk_text = "arm,epoch,probe_index,rank,slot,sim\nstale,3,0,0,7,0.9\n"
    if topk_gz:
        import gzip
        (stem.parent / (stem.name + ".topk.csv.gz")).write_bytes(
            gzip.compress(topk_text.encode()))
    else:
        (stem.parent / (stem.name + ".topk.csv")).write_text(topk_text)
    _write_fork(stem.parent / (stem.name + ".fork_events.csv"),
                fork_super_seqs, arm=arm)
    summary = dict(base_summary)
    if govern is not None:
        summary["govern"] = govern
    (stem.parent / (stem.name + ".summary.json")).write_text(
        json.dumps(summary, indent=2))


def _build_smoke_tree(root: Path, *, quarantine_diverted="none"):
    """One stale (merge_path_stale/pairD) + one clean (clean_control/pairA),
    seed 0, arms none/shadow/quarantine.

    ``quarantine_diverted`` controls the quarantine ledger's per-event keys:
      * "none"     — legacy aggregate-only ledger (no diverted_events) -> the
                     harness reports identity-unprovable;
      * "match"    — diverted_events whose JOIN_KEY matches the shadow flagged
                     rows -> identity-proven;
      * "mismatch" — one diverted key differs -> identity-violated.
    A diverted write emits NO fork_events row, so the quarantine fork is always
    empty; the per-event keys live only in the ledger (as in the real probe)."""
    base = {"arm": "stale", "n_probes": 1, "readout": "frozen-87"}
    base_clean = {"arm": "clean", "n_probes": 1, "readout": "frozen-87"}
    stale_seqs = [100, 101, 102, 103]
    if quarantine_diverted == "match":
        q_div, clean_div = _diverted_events(stale_seqs), []
    elif quarantine_diverted == "mismatch":
        q_div, clean_div = _diverted_events([100, 101, 102, 999]), []
    else:  # "none": pre-instrumentation ledger
        q_div, clean_div = None, None
    # ----- merge_path_stale / pairD / s0 -----
    d = root / "merge_path_stale"
    _write_run(d / "none" / "stale_s0_pairD", fork_super_seqs=stale_seqs,
               govern=None, base_summary=base)
    _write_run(d / "shadow" / "stale_s0_pairD", fork_super_seqs=stale_seqs,
               govern=_shadow_govern(4, {"3": 4}), base_summary=base)
    _write_run(d / "quarantine" / "stale_s0_pairD", fork_super_seqs=[],
               govern=_quarantine_govern(4, {"3": 4}, diverted_events=q_div),
               base_summary=base)
    # ----- clean_control / pairA / s0 (clean: zero supersession; gz topk) -----
    c = root / "clean_control"
    _write_run(c / "none" / "clean_s0_pairA", fork_super_seqs=[],
               govern=None, base_summary=base_clean, arm="clean", topk_gz=True)
    _write_run(c / "shadow" / "clean_s0_pairA", fork_super_seqs=[],
               govern=_shadow_govern(0, {}), base_summary=base_clean,
               arm="clean", topk_gz=True)
    _write_run(c / "quarantine" / "clean_s0_pairA", fork_super_seqs=[],
               govern=_quarantine_govern(0, {}, diverted_events=clean_div),
               base_summary=base_clean, arm="clean", topk_gz=True)
    return {
        "panel": "pr8-9a", "readout": "frozen-87", "mode": "smoke",
        "artifact_root": str(root), "seeds": [0],
        "runs": [
            {"cell": "merge_path_stale", "pair": "pairD", "seed": 0,
             "stems": {"none": "merge_path_stale/none/stale_s0_pairD",
                       "shadow": "merge_path_stale/shadow/stale_s0_pairD",
                       "quarantine":
                           "merge_path_stale/quarantine/stale_s0_pairD"}},
            {"cell": "clean_control", "pair": "pairA", "seed": 0,
             "stems": {"none": "clean_control/none/clean_s0_pairA",
                       "shadow": "clean_control/shadow/clean_s0_pairA",
                       "quarantine":
                           "clean_control/quarantine/clean_s0_pairA"}},
        ]}


# ---------------------------------------------------------------------------
# A. Hand-crafted smoke (the honest "incomplete" outcome)
# ---------------------------------------------------------------------------
def test_handcrafted_smoke_is_incomplete_not_pass(tmp_path):
    manifest = _build_smoke_tree(tmp_path)
    rep = cert.run_certification(manifest)

    # The dimensions the seam CAN prove pass; identity/panel/host cannot.
    assert rep["dimensions"]["inertness"] == cert.PASS
    assert rep["dimensions"]["ledger_coverage"] == cert.PASS
    assert rep["dimensions"]["clean_control"] == cert.PASS
    assert rep["dimensions"]["flag_set_identity"] == cert.INCOMPLETE
    assert rep["dimensions"]["panel_coverage"] == cert.INCOMPLETE
    assert rep["dimensions"]["cross_host_determinism"] == cert.INCOMPLETE

    # Overall: NEVER pass; identity explicitly unprovable.
    assert rep["verdict"] == cert.INCOMPLETE
    assert rep["certifies"] is None
    assert rep["flag_set_identity_status"] == \
        "identity-unprovable with current artifacts"

    # The provable adjacent claim DID pass (shadow flags exactly the baseline
    # supersession set, by per-event key), and counts/histograms agree as
    # explicitly non-certifying support.
    fid = rep["runs"][0]["flag_set_identity"]
    assert fid["provable_subclaim_shadow_equals_baseline_supersession"] is True
    assert fid["shadow_self_consistent"] is True
    assert fid["shadow_flagged_distinct_keys"] == 4
    assert fid["quarantine_diverted_keys_available"] is False
    assert fid["intersection_size"] is None
    assert fid["non_certifying_count_equal"] is True
    assert fid["non_certifying_histogram_equal"] is True


def test_report_files_written_and_renderable(tmp_path):
    manifest = _build_smoke_tree(tmp_path)
    out_json = tmp_path / "rep.json"
    out_md = tmp_path / "rep.md"
    cert.run_certification(manifest, manifest_path=str(tmp_path / "m.json"),
                           out_json=str(out_json), out_md=str(out_md))
    loaded = json.loads(out_json.read_text())
    assert loaded["verdict"] == cert.INCOMPLETE
    assert loaded["sha256_manifest"]            # non-empty hash manifest
    md = out_md.read_text()
    assert "INCOMPLETE" in md
    assert "identity-unprovable with current artifacts" in md
    assert "NOT a certification" in md


# ---------------------------------------------------------------------------
# A (teeth). Each break MUST flip the right dimension to fail.
# ---------------------------------------------------------------------------
def test_inertness_break_fails(tmp_path):
    manifest = _build_smoke_tree(tmp_path)
    # perturb the shadow per_probe csv -> retrieval no longer byte-inert
    bad = tmp_path / "merge_path_stale" / "shadow" / "stale_s0_pairD.csv"
    bad.write_text("probe_index,vote_pred_label,vote_correct\n0,9,0\n")
    rep = cert.run_certification(manifest)
    assert rep["dimensions"]["inertness"] == cert.FAIL
    assert rep["verdict"] == cert.FAIL


def test_ledger_break_fails(tmp_path):
    manifest = _build_smoke_tree(tmp_path)
    f = tmp_path / "merge_path_stale" / "shadow" / "stale_s0_pairD.summary.json"
    s = json.loads(f.read_text())
    s["govern"]["quarantine_ledger"]["quarantined_count"] = 5  # not audit-only!
    f.write_text(json.dumps(s, indent=2))
    rep = cert.run_certification(manifest)
    assert rep["dimensions"]["ledger_coverage"] == cert.FAIL
    assert rep["verdict"] == cert.FAIL


def test_clean_control_flag_leak_fails(tmp_path):
    manifest = _build_smoke_tree(tmp_path)
    f = tmp_path / "clean_control" / "shadow" / "clean_s0_pairA.summary.json"
    s = json.loads(f.read_text())
    s["govern"]["flagged_events"] = 2
    s["govern"]["quarantine_ledger"]["flagged_count"] = 2
    s["govern"]["quarantine_ledger"]["opportunity_count"] = 2
    s["govern"]["quarantine_ledger"]["payload_label_histogram"] = {"3": 2}
    f.write_text(json.dumps(s, indent=2))
    rep = cert.run_certification(manifest)
    assert rep["dimensions"]["ledger_coverage"] == cert.FAIL
    assert rep["dimensions"]["clean_control"] == cert.FAIL
    assert rep["verdict"] == cert.FAIL


def test_summary_non_govern_drift_fails_inertness(tmp_path):
    """If shadow changes a NON-govern summary field, inertness must catch it —
    the additive govern block is the ONLY allowed summary difference."""
    manifest = _build_smoke_tree(tmp_path)
    f = tmp_path / "merge_path_stale" / "shadow" / "stale_s0_pairD.summary.json"
    s = json.loads(f.read_text())
    s["n_probes"] = 999  # a readout field — must match baseline
    f.write_text(json.dumps(s, indent=2))
    rep = cert.run_certification(manifest)
    assert rep["dimensions"]["inertness"] == cert.FAIL
    assert rep["verdict"] == cert.FAIL


# ---------------------------------------------------------------------------
# A (identity has teeth WHEN §8 keys exist). Hypothetical future ledger that
# persists per-event diverted keys: the harness must then certify or refute.
# ---------------------------------------------------------------------------
def test_identity_proven_when_quarantine_keys_match(tmp_path):
    # ledger diverted_events JOIN_KEY matches the shadow flagged rows -> proven
    manifest = _build_smoke_tree(tmp_path, quarantine_diverted="match")
    rep = cert.run_certification(manifest)
    fid = rep["runs"][0]["flag_set_identity"]
    assert fid["quarantine_diverted_keys_available"] is True
    assert fid["quarantine_diverted_event_records"] == 4
    assert fid["identity_status"] == "identity-proven"
    assert fid["intersection_size"] == 4
    assert fid["diverted_minus_flagged"] == []
    assert fid["shadow_flagged_minus_diverted"] == []
    assert rep["dimensions"]["flag_set_identity"] == cert.PASS


def test_identity_violated_when_quarantine_keys_differ(tmp_path):
    # ledger diverted a DIFFERENT incumbent than shadow flagged -> violated/fail
    manifest = _build_smoke_tree(tmp_path, quarantine_diverted="mismatch")
    rep = cert.run_certification(manifest)
    fid = rep["runs"][0]["flag_set_identity"]
    assert fid["identity_status"] == "identity-violated"
    assert fid["intersection_size"] == 3            # 3 of 4 keys coincide
    assert len(fid["diverted_minus_flagged"]) == 1  # the 999 incumbent
    assert len(fid["shadow_flagged_minus_diverted"]) == 1  # the 103 incumbent
    assert rep["dimensions"]["flag_set_identity"] == cert.FAIL
    assert rep["verdict"] == cert.FAIL


# ---------------------------------------------------------------------------
# A (cross-host determinism). Mirror tree pass/fail.
# ---------------------------------------------------------------------------
def test_cross_host_pass_and_fail(tmp_path):
    import shutil
    darwin = tmp_path / "darwin"
    manifest = _build_smoke_tree(darwin)
    gentoo = tmp_path / "gentoo"
    shutil.copytree(darwin, gentoo)
    manifest["gentoo_root"] = str(gentoo)
    rep = cert.run_certification(manifest)
    assert rep["dimensions"]["cross_host_determinism"] == cert.PASS

    # flip one byte on gentoo -> sha256 mismatch -> fail
    (gentoo / "merge_path_stale" / "shadow" / "stale_s0_pairD.fork_events.csv"
     ).write_text("corrupted\n")
    rep2 = cert.run_certification(manifest)
    assert rep2["dimensions"]["cross_host_determinism"] == cert.FAIL
    assert rep2["verdict"] == cert.FAIL


# ---------------------------------------------------------------------------
# A (panel mode). Missing artifacts are a FAIL in panel mode (not smoke).
# ---------------------------------------------------------------------------
def test_panel_mode_missing_cells_fail(tmp_path):
    manifest = _build_smoke_tree(tmp_path)
    manifest["mode"] = "panel"          # claim a full panel from a 2-run subset
    manifest["seeds"] = [0, 1, 2]
    rep = cert.run_certification(manifest)
    assert rep["dimensions"]["panel_coverage"] == cert.FAIL
    assert rep["verdict"] == cert.FAIL
    assert rep["panel_coverage"]["missing_cells"]   # e.g. direct_harm/pairD/s1


# ---------------------------------------------------------------------------
# B. Real-probe vision smoke (torch) — proves harness reads genuine artifacts.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0])
def test_real_probe_vision_smoke(tmp_path, seed):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from benchmarks.failure_mode_probe import run_vision

    # tiny hermetic #87-shaped cache (mirrors test_pr7_refuse_behavior)
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
                  held_out_per_class=8, contraction=0.0, seed=seed,
                  supersede_epoch=3, payload_mode="soft")

    def _run(cell, arm, govern, run_arm):
        stem = tmp_path / cell / arm / f"{run_arm}_s{seed}_pairD"
        stem.parent.mkdir(parents=True, exist_ok=True)
        _, summary = run_vision(run_arm, rate=0.0, out_path=stem.with_suffix(".csv"),
                                govern=govern, **common)
        # mirror main(): write the summary.json next to the CSVs
        stem.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
        return f"{cell}/{arm}/{run_arm}_s{seed}_pairD"

    runs = []
    for cell, run_arm in (("merge_path_stale", "stale"),
                          ("clean_control", "clean")):
        stems = {arm: _run(cell, arm, arm if arm != "none" else "none", run_arm)
                 for arm in ("none", "shadow", "quarantine")}
        runs.append({"cell": cell, "pair": "pairD" if cell == "merge_path_stale"
                     else "pairA", "seed": seed, "stems": stems})

    manifest = {"panel": "pr8-9a", "readout": "frozen-87", "mode": "smoke",
                "artifact_root": str(tmp_path), "seeds": [seed], "runs": runs}
    rep = cert.run_certification(manifest)

    # Real artifacts: inertness + ledger + clean still certify.
    assert rep["dimensions"]["inertness"] == cert.PASS, rep["runs"][0]["inertness"]
    assert rep["dimensions"]["ledger_coverage"] == cert.PASS
    assert rep["dimensions"]["clean_control"] == cert.PASS
    stale = next(r for r in rep["runs"] if r["cell"] == "merge_path_stale")
    clean = next(r for r in rep["runs"] if r["cell"] == "clean_control")
    assert stale["ledger"]["flagged_events"] > 0
    assert clean["ledger"]["flagged_events"] == 0

    # The instrumentation made identity DECIDABLE on real artifacts: no longer
    # "unprovable" — the harness now returns a proven/violated decision from the
    # ledger's per-event keys. (On real divergent runs the incumbent-keyed join
    # typically REFUTES once shadow commits and quarantine does not — the "or
    # refute" half of the goal; the fixture tests own the proven case.)
    fid = stale["flag_set_identity"]
    assert fid["quarantine_diverted_keys_available"] is True
    assert fid["quarantine_diverted_event_records"] == stale["ledger"][
        "flagged_events"]
    assert fid["identity_status"] in ("identity-proven", "identity-violated")
    assert fid["intersection_size"] is not None  # a real join ran


# ---------------------------------------------------------------------------
# C. Constants pin — the harness's hardcoded engine strings cannot drift.
# ---------------------------------------------------------------------------
def test_harness_constants_match_engine():
    from benchmarks.failure_mode_probe import (
        EVENT_SUPERSESSION, QUARANTINE_DIVERTED_JOIN_KEY, GovernanceHook)
    assert cert.SUPERSESSION_CLASS == EVENT_SUPERSESSION
    assert tuple(cert.JOIN_KEY) == tuple(QUARANTINE_DIVERTED_JOIN_KEY)
    prov = GovernanceHook("shadow").provenance()
    assert cert.SHADOW_ACTION == prov["action"]
    assert cert.SHADOW_STEP == prov["step"]
    assert cert.FLAGGED_DISPOSITION == \
        prov["quarantine_ledger"]["disposition"]
