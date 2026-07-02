"""tests/test_pr10_read_seam.py — PR-10 step 1: the read-time abstention seam
(PR10_READTIME_ABSTENTION_GATE.md §1/§3), the delta reader, and the boundary.

The seam serves the frozen scorer's ``merge-abstain`` policy as a reader-facing
outcome — ``--read-govern {none, merge-abstain}``, consulted AFTER the deployed
vote is computed and scored, BEFORE the per-probe row is emitted — and must
perturb NOTHING else. These gates pin, hermetically (CPU, synthetic + the
tiny-vision cache, no GPU/network — the PR-8 test pattern):

  (a) ``none`` arm identity — the default invocation and ``--read-govern
      none`` emit the pre-seam schema and bytes; the seam is never consulted;
  (b) a governed soft run (synthetic AND tiny-vision) abstains EXACTLY on the
      merge-suspect-led rows the frozen scorer's own ``merge-abstain`` policy
      abstains on (scorer code, not a reimplementation, is the oracle here);
      the two columns are additive and every pre-existing byte is preserved;
  (c) governed clean / contra / one-shot runs: zero abstentions;
  (d) write-stream artifacts (fork_events / per_slot / topk) byte-identical
      between same-seed none and governed twins; vision summary identical
      after removing the ``read_govern`` block;
  (e) NO tie trigger — an exact 0.5/0.5 tie with no merge suspect answers;
      a merge suspect that does not lead the vote answers;
  (f) the engine files are byte-frozen (sha256 == the pr7_twin_delta
      baselines);
  (g) ``pr10_readout_delta`` certifies the hermetic twins
      (``readout-certified``) and fails on a mutated fixture; its mirrored
      constants are pinned equal to the live scorer/driver values.
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from benchmarks import analyze_fork_governance as afg  # noqa: E402
from benchmarks import failure_mode_probe as fmp  # noqa: E402
from benchmarks import pr10_readout_delta as prd  # noqa: E402
from benchmarks.analyze_fork_governance import (  # noqa: E402
    apply_policy, build_writetime_router, load_run, _vote)
from benchmarks.failure_mode_probe import (  # noqa: E402
    ABSTAIN_REASON_MERGE_LED, OUT_COLS, READ_GOVERN_ACTIONS, READ_GOVERN_COLS,
    SERVED_ABSTAIN, SERVED_ANSWER, ReadGovernanceHook, run_synthetic,
    run_vision)
from benchmarks.pr7_twin_delta import ENGINE_SHA256_BASELINE  # noqa: E402

_SUFFIXES = (".csv", ".per_slot.csv", ".fork_events.csv", ".topk.csv")
_WRITE_STREAM = _SUFFIXES[1:]
_CLASSES = [0, 8]
_ATTRACTOR = 71

# Hermetic soft configurations chosen so the merge-abstain trigger actually
# fires (nonzero abstentions) — the synthetic soft arm needs an early
# supersession epoch for the EMA-merge slot to lead held-out votes.
_SYN_SOFT = dict(arm="stale", rate=0.0, epochs=6, supersede_epoch=1, seed=1,
                 payload_mode="soft")
_VIS_COMMON = dict(rate=0.0, epochs=6, classes=_CLASSES,
                   attractor_class=_ATTRACTOR, samples_per_class=8,
                   held_out_per_class=8, contraction=0.0, seed=0,
                   supersede_epoch=3, payload_mode="soft")


def _emitted(out: Path) -> dict:
    base = out.with_suffix("")
    return {s: Path(f"{base}{s}").read_bytes() for s in _SUFFIXES}


def _rows(out: Path) -> list[dict]:
    with open(out, newline="") as f:
        return list(csv.DictReader(f))


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


def _scorer_merge_abstain_rows(stem: Path) -> set:
    """(epoch, probe_index) rows the FROZEN SCORER's merge-abstain policy
    abstains on — computed with the scorer's own code (load_run,
    build_writetime_router, apply_policy), the anti-drift oracle for the seam
    and the reader."""
    run = load_run(stem)
    router = build_writetime_router(run["events"], run["slot_obs"])
    abstained = set()
    for p in run["probes"]:
        epoch, pi = int(float(p["epoch"])), int(p["probe_index"])
        cands = run["topk"][(epoch, pi)]
        none_answer, _ = _vote(cands, run["value_dim"])
        ans, acted, detail = apply_policy(
            "merge-abstain", cands, [], none_answer, run["value_dim"],
            epoch, run["slot_obs"], True, router)
        if ans is None:
            assert detail == {"trigger": "merge"}
            abstained.add((epoch, pi))
    return abstained


def _served_rows(out: Path) -> tuple[set, set]:
    """(abstained keys, answered keys) from the governed per-probe CSV, with
    the trigger-purity checks applied to every row."""
    abstained, answered = set(), set()
    for r in _rows(out):
        key = (int(float(r["epoch"])), int(r["probe_index"]))
        if r["served_outcome"] == SERVED_ABSTAIN:
            assert r["abstain_reason"] == ABSTAIN_REASON_MERGE_LED
            abstained.add(key)
        else:
            assert r["served_outcome"] == SERVED_ANSWER
            assert r["abstain_reason"] == ""
            answered.add(key)
    return abstained, answered


def _dump_summary(out: Path, summary: dict) -> None:
    """Mirror the CLI's summary emission (main() writes indent=2 JSON)."""
    with open(out.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    """Same-seed none/governed twins: synthetic soft (plus a governed
    same-seed re-run for G5) and tiny-vision soft (with summaries on disk,
    as the CLI would leave them)."""
    tmp = tmp_path_factory.mktemp("pr10_twins")
    t = {"dir": tmp}

    t["syn_none"] = tmp / "syn_none.csv"
    t["syn_gov"] = tmp / "syn_gov.csv"
    t["syn_gov_twin"] = tmp / "syn_gov_twin.csv"
    run_synthetic(out_path=t["syn_none"], **_SYN_SOFT)
    run_synthetic(out_path=t["syn_gov"], read_govern="merge-abstain",
                  **_SYN_SOFT)
    run_synthetic(out_path=t["syn_gov_twin"], read_govern="merge-abstain",
                  **_SYN_SOFT)

    cache = _make_cache(tmp)
    t["vis_none"] = tmp / "vis_none.csv"
    t["vis_gov"] = tmp / "vis_gov.csv"
    _, s_none = run_vision("stale", out_path=t["vis_none"],
                           cache_path=str(cache), **_VIS_COMMON)
    _, s_gov = run_vision("stale", out_path=t["vis_gov"],
                          cache_path=str(cache),
                          read_govern="merge-abstain", **_VIS_COMMON)
    _dump_summary(t["vis_none"], s_none)
    _dump_summary(t["vis_gov"], s_gov)
    t["vis_summary_none"], t["vis_summary_gov"] = s_none, s_gov
    return t


# ---------------------------------------------------------------------------
# (a) none-arm identity: pre-seam schema and bytes, seam never consulted
# ---------------------------------------------------------------------------
def test_none_arm_bytes_and_schema_unchanged(tmp_path, monkeypatch):
    default = tmp_path / "default.csv"
    run_synthetic(out_path=default, **_SYN_SOFT)

    # the seam must never be consulted on a none/default run
    def _boom(self, *a, **k):  # pragma: no cover - failure path
        raise AssertionError("read seam consulted on a --read-govern none run")
    monkeypatch.setattr(fmp.ReadGovernanceHook, "serve_epoch", _boom)
    explicit = tmp_path / "none.csv"
    run_synthetic(out_path=explicit, read_govern="none", **_SYN_SOFT)
    monkeypatch.undo()

    assert _emitted(default) == _emitted(explicit)
    header = default.read_text().splitlines()[0]
    assert header == ",".join(OUT_COLS)  # pre-seam schema, no served columns
    assert not any(c in header.split(",") for c in READ_GOVERN_COLS)


def test_none_vision_summary_has_no_read_govern(twins):
    assert "read_govern" not in twins["vis_summary_none"]


# ---------------------------------------------------------------------------
# (b) governed soft runs abstain EXACTLY on the scorer's merge-suspect-led
#     rows; columns additive; every pre-existing byte preserved
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("none_key,gov_key", [("syn_none", "syn_gov"),
                                              ("vis_none", "vis_gov")])
def test_governed_soft_abstains_exactly_on_merge_led_rows(twins, none_key,
                                                          gov_key):
    abstained, answered = _served_rows(twins[gov_key])
    assert len(abstained) > 0  # the trigger actually fires on both fixtures
    assert abstained == _scorer_merge_abstain_rows(twins[gov_key])
    # the reader's no-torch recomputation is pinned to the same oracle
    assert abstained == prd.merge_led_rows(twins[gov_key].with_suffix(""))


@pytest.mark.parametrize("none_key,gov_key", [("syn_none", "syn_gov"),
                                              ("vis_none", "vis_gov")])
def test_governed_columns_additive_and_fields_preserved(twins, none_key,
                                                        gov_key):
    gov_lines = twins[gov_key].read_bytes().split(b"\r\n")
    none_lines = twins[none_key].read_bytes().split(b"\r\n")
    assert len(gov_lines) == len(none_lines)
    header = gov_lines[0].rsplit(b",", 2)
    assert header[0] == none_lines[0]
    assert [header[1], header[2]] == [c.encode() for c in READ_GOVERN_COLS]
    for gl, nl in zip(gov_lines[1:], none_lines[1:]):
        if gl == b"" and nl == b"":
            continue
        # every pre-existing byte of every row preserved (gate G2)
        assert gl.rsplit(b",", 2)[0] == nl


def test_governed_vision_summary_read_govern_block(twins):
    abstained, answered = _served_rows(twins["vis_gov"])
    rg = twins["vis_summary_gov"]["read_govern"]
    assert rg["action"] == "merge-abstain"
    assert rg["abstained"] == len(abstained) > 0
    assert rg["answered"] == len(answered)
    assert rg["probes_seen"] == len(abstained) + len(answered)
    assert rg["abstain_reason_histogram"] == \
        {ABSTAIN_REASON_MERGE_LED: len(abstained)}


# ---------------------------------------------------------------------------
# (c) governed clean / contra / one-shot runs: zero abstentions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm,kw", [
    ("clean", dict(rate=0.0, epochs=6, supersede_epoch=3)),
    ("contra", dict(rate=0.15, epochs=6, supersede_epoch=3)),
    ("stale", dict(rate=0.0, epochs=7, supersede_epoch=3, one_shot=True)),
])
def test_governed_nonsoft_zero_abstentions(tmp_path, arm, kw):
    out = tmp_path / f"{arm}.csv"
    n = run_synthetic(arm, out_path=out, seed=0,
                      read_govern="merge-abstain", **kw)
    assert n > 0
    abstained, answered = _served_rows(out)
    assert abstained == set()
    assert len(answered) == n
    # exactness, not just zero: the scorer's own policy also never fires here
    assert _scorer_merge_abstain_rows(out) == set()


# ---------------------------------------------------------------------------
# (d) write-stream byte-identity between same-seed none and governed twins
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("none_key,gov_key", [("syn_none", "syn_gov"),
                                              ("vis_none", "vis_gov")])
def test_write_stream_byte_identical_between_twins(twins, none_key, gov_key):
    e_none, e_gov = _emitted(twins[none_key]), _emitted(twins[gov_key])
    for suf in _WRITE_STREAM:
        assert e_none[suf] == e_gov[suf], (
            f"{suf} differs between none and governed twins — the read seam "
            "leaked into the write path")
    assert e_none[".csv"] != e_gov[".csv"]  # only the two appended columns


def test_vision_summary_identical_after_removing_read_govern(twins):
    gov = json.loads(json.dumps(twins["vis_summary_gov"]))
    gov.pop("read_govern")
    assert gov == twins["vis_summary_none"]


def test_governed_same_seed_twin_byte_identical(twins):
    assert _emitted(twins["syn_gov"]) == _emitted(twins["syn_gov_twin"])


# ---------------------------------------------------------------------------
# (e) no tie trigger; merge suspect that does not LEAD never fires
# ---------------------------------------------------------------------------
def test_no_tie_trigger_on_forced_tie_fixture():
    """Exact 0.5/0.5 tie with no merge suspect: the seam ANSWERS (the refuted
    abstain-tie component is demonstrably not part of the served policy).
    A merge suspect that does not lead the vote also answers; only the
    leading slot triggers (first-rank tie-break, exactly the scorer's max)."""
    tie = [{"rank": 0, "slot": 1, "sim": "0.9", "surviving": 1,
            "weight": "0.5", "decode": 0},
           {"rank": 1, "slot": 2, "sim": "0.9", "surviving": 1,
            "weight": "0.5", "decode": 1}]
    hook = ReadGovernanceHook("merge-abstain")
    assert hook.decide(tie, set()) == (SERVED_ANSWER, "")
    # suspect present but NOT leading (slot 1 wins the tie at first rank)
    assert hook.decide(tie, {2}) == (SERVED_ANSWER, "")
    # suspect LEADS -> abstain with the one recorded reason
    assert hook.decide(tie, {1}) == (SERVED_ABSTAIN,
                                     ABSTAIN_REASON_MERGE_LED)
    assert hook.probes_seen == 3
    assert hook.abstained == 1
    assert hook.reason_counts == {ABSTAIN_REASON_MERGE_LED: 1}


def test_one_shot_exact_ties_never_abstain(tmp_path):
    """Integration form of (e): the one-shot arm sits at exact 0.5/0.5 vote
    ties (PR-3b) and has zero merge-suspect events — the governed run must
    answer every row (also covered by (c); asserted here against the tie
    mechanism explicitly)."""
    out = tmp_path / "oneshot.csv"
    run_synthetic("stale", rate=0.0, epochs=7, supersede_epoch=3,
                  out_path=out, seed=0, one_shot=True,
                  read_govern="merge-abstain")
    run = load_run(out)
    router = build_writetime_router(run["events"], run["slot_obs"])
    assert router["merge"] == []          # no merge suspects on this arm
    abstained, _ = _served_rows(out)
    assert abstained == set()


# ---------------------------------------------------------------------------
# (f) engine byte-freeze
# ---------------------------------------------------------------------------
def test_engine_files_sha256_frozen():
    for fname, expected in ENGINE_SHA256_BASELINE.items():
        digest = hashlib.sha256((ROOT / fname).read_bytes()).hexdigest()
        assert digest == expected, (
            f"{fname} drifted from the pr7_twin_delta baseline — PR-10 "
            "requires the engine byte-frozen")


def test_engine_files_do_not_reference_the_read_seam():
    for fname in ENGINE_SHA256_BASELINE:
        text = (ROOT / fname).read_text()
        assert "read_govern" not in text
        assert "ReadGovernanceHook" not in text


# ---------------------------------------------------------------------------
# (g) the delta reader: mirrored-constant pins, certification, and failure
# ---------------------------------------------------------------------------
def test_reader_constants_pinned_to_live_modules():
    assert prd.MERGE_SUSPECT_COS == afg.MERGE_SUSPECT_COS
    assert prd.ABSORBED_OUTCOME == fmp.ABSORBED
    assert prd.SERVED_ANSWER == fmp.SERVED_ANSWER
    assert prd.SERVED_ABSTAIN == fmp.SERVED_ABSTAIN
    assert prd.ABSTAIN_REASON_MERGE_LED == fmp.ABSTAIN_REASON_MERGE_LED
    assert tuple(prd.READ_GOVERN_COLS) == tuple(fmp.READ_GOVERN_COLS)
    assert prd.READ_GOVERN_ACTION in READ_GOVERN_ACTIONS
    assert READ_GOVERN_ACTIONS == ("none", "merge-abstain")
    assert not set(READ_GOVERN_COLS) & set(OUT_COLS)


def _manifest(twins, tmp_path, extra_cells=(), envelope=None):
    cells = [
        {"cell": "hermetic/synthetic-soft/s1", "soft": True,
         "baseline": str(twins["syn_none"].with_suffix("")),
         "governed": str(twins["syn_gov"].with_suffix("")),
         "governed_twin": str(twins["syn_gov_twin"].with_suffix(""))},
        {"cell": "hermetic/vision-soft/s0", "soft": True,
         "baseline": str(twins["vis_none"].with_suffix("")),
         "governed": str(twins["vis_gov"].with_suffix(""))},
    ] + list(extra_cells)
    manifest = {"cells": cells}
    if envelope is not None:
        manifest["envelope"] = str(envelope)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return manifest, path


def test_readout_delta_certifies_hermetic_twins(twins, tmp_path):
    # a non-soft cell (zero-abstention rule) + the envelope count path,
    # exercised with an envelope entry holding the measured exact count
    clean_none = tmp_path / "clean_none.csv"
    clean_gov = tmp_path / "clean_gov.csv"
    kw = dict(rate=0.0, epochs=6, supersede_epoch=3, seed=0)
    run_synthetic("clean", out_path=clean_none, **kw)
    run_synthetic("clean", out_path=clean_gov, read_govern="merge-abstain",
                  **kw)
    abstained, _ = _served_rows(twins["vis_gov"])
    envelope = tmp_path / "envelope.json"
    envelope.write_text(json.dumps(
        {"cells_fresh": {"hermetic/vision-soft/s0":
                         {"abstained_merge": len(abstained)}}}))
    extra = [{"cell": "hermetic/synthetic-clean/s0", "soft": False,
              "baseline": str(clean_none.with_suffix("")),
              "governed": str(clean_gov.with_suffix(""))}]
    manifest, mpath = _manifest(twins, tmp_path, extra_cells=extra,
                                envelope=envelope)
    manifest["cells"][1]["envelope_cell"] = "hermetic/vision-soft/s0"
    mpath.write_text(json.dumps(manifest, indent=1))

    delta = prd.build_readout_delta(manifest, mpath.parent)
    assert delta["verdict"] == prd.VERDICT_CERTIFIED
    assert delta["n_cells"] == 3
    for name, cell in delta["cells"].items():
        assert cell["pass"], (name, cell["gates"])
    g5 = delta["cells"]["hermetic/synthetic-soft/s1"]["gates"]["G5_determinism"]
    assert g5["detail"]["twin"]["byte_identical"] is True
    env_check = delta["cells"]["hermetic/vision-soft/s0"]["gates"][
        "G3_abstention_set_exactness"]["detail"]["envelope_count"]
    assert env_check == {"expected_abstained_merge": len(abstained),
                         "equal": True}


def _copy_run(src_stem: Path, dst_stem: Path):
    for suf in _SUFFIXES:
        shutil.copyfile(f"{src_stem}{suf}", f"{dst_stem}{suf}")


def test_readout_delta_fails_on_mutated_fixture(twins, tmp_path):
    src = twins["syn_gov"].with_suffix("")
    base = twins["syn_none"].with_suffix("")

    # mutation 1: one served abstention flipped to answer -> G3 fails
    # (G2/G4 stay clean: pre-existing bytes and the reason vocabulary hold)
    flip = tmp_path / "flip"
    flip.mkdir()
    _copy_run(src, flip / "gov")
    csv_path = flip / "gov.csv"
    data = csv_path.read_bytes()
    needle = (",%s,%s\r\n" % (SERVED_ABSTAIN,
                              ABSTAIN_REASON_MERGE_LED)).encode()
    assert needle in data
    csv_path.write_bytes(data.replace(
        needle, (",%s,\r\n" % SERVED_ANSWER).encode(), 1))
    manifest = {"cells": [{"cell": "mutated/flip", "soft": True,
                           "baseline": str(base),
                           "governed": str(flip / "gov")}]}
    delta = prd.build_readout_delta(manifest, tmp_path)
    assert delta["verdict"] == prd.VERDICT_FAIL
    gates = delta["cells"]["mutated/flip"]["gates"]
    assert not gates["G3_abstention_set_exactness"]["pass"]
    assert gates["G2_answered_stream_byte_identity"]["pass"]
    assert gates["G4_trigger_purity"]["pass"]

    # mutation 2: one write-stream byte changed -> G1 fails
    leak = tmp_path / "leak"
    leak.mkdir()
    _copy_run(src, leak / "gov")
    ps = leak / "gov.per_slot.csv"
    ps.write_bytes(ps.read_bytes() + b"tampered\r\n")
    manifest = {"cells": [{"cell": "mutated/leak", "soft": True,
                           "baseline": str(base),
                           "governed": str(leak / "gov")}]}
    delta = prd.build_readout_delta(manifest, tmp_path)
    assert delta["verdict"] == prd.VERDICT_FAIL
    assert not delta["cells"]["mutated/leak"]["gates"][
        "G1_write_stream_byte_identity"]["pass"]

    # mutation 3: a pre-existing field altered -> G2 fails
    edit = tmp_path / "edit"
    edit.mkdir()
    _copy_run(src, edit / "gov")
    cp = edit / "gov.csv"
    lines = cp.read_bytes().split(b"\r\n")
    first = lines[1].split(b",")
    first[2] = b"99"  # epoch column of the first data row
    lines[1] = b",".join(first)
    cp.write_bytes(b"\r\n".join(lines))
    manifest = {"cells": [{"cell": "mutated/edit", "soft": True,
                           "baseline": str(base),
                           "governed": str(edit / "gov")}]}
    delta = prd.build_readout_delta(manifest, tmp_path)
    assert delta["verdict"] == prd.VERDICT_FAIL
    assert not delta["cells"]["mutated/edit"]["gates"][
        "G2_answered_stream_byte_identity"]["pass"]


def test_readout_delta_empty_manifest_never_certifies(tmp_path):
    delta = prd.build_readout_delta({"cells": []}, tmp_path)
    assert delta["verdict"] == prd.VERDICT_FAIL
    assert delta["n_cells"] == 0
