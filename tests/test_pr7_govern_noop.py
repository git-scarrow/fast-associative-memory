"""tests/test_pr7_govern_noop.py — PR-7 step 1: the no-op governance boundary
scaffold (PR7_DESIGN.md §13.1, §11 tests 1-3 & 6).

PR-7 needs ONE opt-in seam where an experimental writer can later act on an
already-classified write (annotate / quarantine / refuse). Step 1 adds only
the seam and proves it is inert and boundary-safe BEFORE any governed run:

  1. no-op identity — `--govern none` (the baseline) is byte-identical to the
     pre-seam driver, and the default invocation is unchanged (§11 test 1);
  2. byte-identical emitted artifacts — EVERY action {none, annotate,
     quarantine, refuse} produces byte-identical write decisions, so all four
     emitted CSVs match the baseline exactly. `annotate` is the step-2
     null-action floor (it records a provenance annotation but commits the
     write exactly as baseline); `quarantine`/`refuse` stay recorded no-ops;
  3. engine-frozen boundary — the seam lives ONLY in the experimental driver;
     no deployed engine/retrieval file references it, and the two files that
     determine run output are sha256-unchanged from the PR-6 baseline
     (§11 test 2);
  4. determinism / deployment parity — a governed run is reproducible and its
     deployed vote is bit-identical to the baseline (§11 test 3);
  5. provenance — the (no-op) action is recorded in the run summary only when
     a non-baseline action is requested, so `none` stays byte-identical; the
     CLI entry point exposes `--govern` and rejects unknown values (§11 test 6).

This file pins the experimental BOUNDARY (the seam is inert on every emitted
artifact and never leaks into deployed retrieval). No write refusal, quarantine,
read-time / slot-granularity trust, geometry gate, or one-shot classification is
implemented (PR7_DESIGN.md §12). The analyzer-level provenance guards (stem ↔
classes ↔ govern, non-soft payload refused on the merge-path cell) live in
tests/test_pr7_twin_delta.py with `pr7_twin_delta.py` (step 2). The mechanism
regression (§11 test 5) is covered by tests/test_failure_mode_probe.py and
tests/test_failure_mode_vision.py, run unchanged.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.failure_mode_probe import (  # noqa: E402
    GOVERN_ACTIONS, GOVERN_ALLOW, GOVERN_IMPLEMENTED_ACTIONS, GovernanceHook,
    run_synthetic, run_vision)

PROBE = ROOT / "benchmarks" / "failure_mode_probe.py"

# The deployed engine/retrieval surface PR-7 must leave byte-frozen
# (PR7_DESIGN.md §1, §12). The governance seam may not appear in any of these.
ENGINE_FILES = ["associative_core.py", "fast_associative_memory.py",
                "query.py", "dynamic_vigilance.py", "adapter.py"]

# The two files whose bytes determine run output; PR-6 sha256-verified them
# identical across the canonical (Darwin) and compute (gentoo) hosts. PR-7
# step 1 must not change them — this pin fails loudly if it ever does.
ENGINE_SHA256_BASELINE = {
    "associative_core.py":
        "ce5a2785069905c0cebba43cf7642374a5c2834a60254bee007d5f0ae141c341",
    "fast_associative_memory.py":
        "38a601fc0819af0d19b877a07bc16c839839fb1dacf7b174ec0f2a573660d875",
}

# The four artifacts a synthetic run emits next to the main CSV.
_SUFFIXES = (".csv", ".per_slot.csv", ".fork_events.csv", ".topk.csv")


def _emitted_bytes(out: Path) -> dict:
    """All four emitted artifacts for one run, keyed by suffix -> raw bytes."""
    base = out.with_suffix("")
    return {sfx: Path(f"{base}{sfx}").read_bytes() for sfx in _SUFFIXES}


def _run_synth(tmp_path, arm, govern, *, name, payload_mode="onehot",
               supersede_epoch=3, epochs=6, seed=0, pass_govern=True):
    """Run the synthetic driver for one arm and return the emitted bytes.

    ``pass_govern=False`` omits the govern argument entirely, exercising the
    default (proving the *default invocation* path is unchanged)."""
    out = tmp_path / name
    kw = dict(out_path=out, seed=seed, payload_mode=payload_mode)
    if pass_govern:
        kw["govern"] = govern
    n = run_synthetic(arm, rate=0.30 if arm in ("contra", "mixed") else 0.0,
                      epochs=epochs, supersede_epoch=supersede_epoch, **kw)
    assert n > 0
    return _emitted_bytes(out)


# Arms chosen to route every write-classification path through the seam:
# clean (clean writes only), contra (contradiction injections), stale (one-hot
# supersession fork), stale-soft (merge-path EMA-freeze supersession).
_ARMS = [
    ("clean", "onehot", -1),
    ("contra", "onehot", -1),
    ("stale", "onehot", 3),
    ("stale", "soft", 3),       # merge-path stale (the soft EMA-merge arm)
    ("mixed", "onehot", 3),
]


# ---------------------------------------------------------------------------
# 1 + 2. No-op identity: baseline, default, and every action agree byte-for-byte
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm,payload_mode,supersede_epoch", _ARMS)
def test_default_invocation_unchanged(tmp_path, arm, payload_mode,
                                      supersede_epoch):
    """Adding the opt-in --govern arg (default 'none') must not change the
    default invocation: omitting govern == passing govern='none', byte-for-byte
    across all four emitted artifacts (§11 test 1)."""
    default = _run_synth(tmp_path, arm, None, name="default.csv",
                         payload_mode=payload_mode,
                         supersede_epoch=supersede_epoch, pass_govern=False)
    explicit_none = _run_synth(tmp_path, arm, "none", name="none.csv",
                               payload_mode=payload_mode,
                               supersede_epoch=supersede_epoch)
    assert default == explicit_none


@pytest.mark.parametrize("arm,payload_mode,supersede_epoch", _ARMS)
@pytest.mark.parametrize("action", ["annotate", "quarantine"])
def test_noop_governance_matches_baseline(tmp_path, arm, payload_mode,
                                          supersede_epoch, action):
    """`annotate` (the null-action floor) and `quarantine` (still a recorded
    no-op) make byte-identical write decisions to the ungoverned baseline on
    every arm, so all four emitted artifacts match exactly under a fixed seed
    (§11 test 1). `refuse` is the step-5 acting arm and is covered separately:
    a pure no-op only on arms with no merge_suspect (supersession) write."""
    baseline = _run_synth(tmp_path, arm, "none", name="base.csv",
                          payload_mode=payload_mode,
                          supersede_epoch=supersede_epoch)
    governed = _run_synth(tmp_path, arm, action, name=f"{action}.csv",
                          payload_mode=payload_mode,
                          supersede_epoch=supersede_epoch)
    assert governed == baseline, (
        f"{action} on arm {arm}/{payload_mode} changed an emitted artifact — "
        f"it must be a pure no-op")


# Arms with NO supersession write: refuse has nothing to skip, so it stays a
# pure no-op (proving non-suspect traffic is never harmed).
_NON_SUSPECT_ARMS = [("clean", "onehot", -1), ("contra", "onehot", -1)]


@pytest.mark.parametrize("arm,payload_mode,supersede_epoch", _NON_SUSPECT_ARMS)
def test_refuse_is_noop_on_non_suspect_arms(tmp_path, arm, payload_mode,
                                            supersede_epoch):
    """`refuse` acts ONLY on the write-time merge_suspect (supersession) class;
    on arms that never supersede it skips nothing, so every emitted artifact is
    byte-identical to the baseline — non-suspect writes are allowed unchanged
    (PR7_DESIGN.md §4/§13)."""
    baseline = _run_synth(tmp_path, arm, "none", name="base.csv",
                          payload_mode=payload_mode,
                          supersede_epoch=supersede_epoch)
    refused = _run_synth(tmp_path, arm, "refuse", name="refuse.csv",
                         payload_mode=payload_mode,
                         supersede_epoch=supersede_epoch)
    assert refused == baseline, (
        f"refuse on non-suspect arm {arm}/{payload_mode} changed an emitted "
        f"artifact — refuse must touch only merge_suspect writes")


def test_deployed_vote_column_identical_under_governance(tmp_path):
    """Deployment parity (§11 test 3): the deployed per-probe vote is unchanged
    by governance. Byte-identity of the main CSV implies the vote_pred_label
    column is identical to the baseline, and the baseline vote is pinned to
    forward().argmax by test_failure_mode_probe.test_vote_pred_pinned_to_forward
    — so the governed vote is bit-identical to deployment too."""
    base = _run_synth(tmp_path, "contra", "none", name="b.csv")
    gov = _run_synth(tmp_path, "contra", "refuse", name="g.csv")
    # The main CSV carries vote_pred_label / vote_correct; identical bytes ⇒
    # identical deployed vote on every probe.
    assert gov[".csv"] == base[".csv"]
    assert b"vote_pred_label" in base[".csv"]


# ---------------------------------------------------------------------------
# 3. Engine-frozen boundary — the seam lives only in the experimental driver
# ---------------------------------------------------------------------------
def test_governance_seam_absent_from_deployed_engine():
    """No deployed engine/retrieval file may reference governance: the seam is
    experimental-only, so the deployed forward()/learn_local path never reaches
    it (PR7_DESIGN.md §4, §12 — §11 test 2)."""
    for fname in ENGINE_FILES:
        text = (ROOT / fname).read_text().lower()
        assert "govern" not in text, (
            f"{fname} references governance — the seam leaked into the "
            f"deployed engine/retrieval surface")
    # ...and the seam IS importable from the experimental driver.
    assert GovernanceHook("none").action == "none"


def test_deployed_engine_sha256_unchanged():
    """The two files that determine run output are byte-frozen at the PR-6
    baseline (PR7_DESIGN.md §1 safety invariant — §11 test 2)."""
    for fname, expected in ENGINE_SHA256_BASELINE.items():
        got = hashlib.sha256((ROOT / fname).read_bytes()).hexdigest()
        assert got == expected, (
            f"{fname} sha256 changed from the PR-6 baseline ({expected}) to "
            f"{got} — PR-7 must keep deployed engine code byte-frozen")


def test_seam_does_not_import_engine_retrieval():
    """The GovernanceHook touches no engine state: constructing it and driving
    it requires nothing from the retrieval path, and it carries no reference to
    a ContinuousCAM / forward()."""
    h = GovernanceHook("quarantine")
    # decide() takes only the already-classified (event_class, outcome) — no
    # memory, no slots, no geometry.
    assert h.decide("contradiction", "forked") == GOVERN_ALLOW
    assert not any(isinstance(v, torch.Tensor) for v in vars(h).values())


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action", ["none", "annotate", "quarantine", "refuse"])
def test_governed_run_is_deterministic(tmp_path, action):
    """A governed run is reproducible: same seed → identical artifacts on
    re-run (determinism is a PR-7 precondition — §11 test 3)."""
    a = _run_synth(tmp_path, "mixed", action, name="a.csv")
    b = _run_synth(tmp_path, "mixed", action, name="b.csv")
    assert a == b


# ---------------------------------------------------------------------------
# 5. decide() ALLOWs every action (annotate is the null-action floor); only
#    'none' and 'annotate' are implemented (quarantine/refuse stay no-ops).
# ---------------------------------------------------------------------------
def test_every_action_decides_allow_only():
    """decide() returns ALLOW for every action and every outcome — `annotate`
    is the null-action floor (it commits the write exactly as baseline), and
    `quarantine`/`refuse` are still recorded no-ops. Step 2 implements `none`
    and `annotate` (PR7_DESIGN.md §4/§13.2)."""
    assert GOVERN_IMPLEMENTED_ACTIONS == ("none", "annotate", "refuse")
    expected_step = {"none": "pr7-step1-noop", "annotate": "pr7-step2-annotate",
                     "quarantine": "pr7-step1-noop", "refuse": "pr7-step5-refuse"}
    for action in GOVERN_ACTIONS:
        h = GovernanceHook(action)
        # decide() is the POST-write call: a write reaching it is always
        # committed (refuse's skip happens earlier, in allow_write). Driving it
        # with a non-suspect class (contradiction) is allowed for every action.
        for outcome in ("forked", "plain-miss", "absorbed", "dropped"):
            assert h.decide("contradiction", outcome) == GOVERN_ALLOW
        prov = h.provenance()
        assert prov["action"] == action
        assert prov["step"] == expected_step[action]
        assert prov["implemented"] is (action in ("none", "annotate", "refuse"))
        assert prov["implemented_actions"] == ["none", "annotate", "refuse"]
        assert prov["events_seen"] == 4
        assert sum(prov["outcome_counts"].values()) == 4
    # annotate counts the supersession events it stamps; non-supersession
    # outcomes above are not annotated, so the floor stays a pure record.
    h = GovernanceHook("annotate")
    h.decide("supersession", "absorbed")
    h.decide("contradiction", "forked")
    assert h.provenance()["annotated_events"] == 1


def test_invalid_govern_action_rejected():
    with pytest.raises(ValueError, match="--govern"):
        GovernanceHook("deprecate")
    assert "deprecate" not in GOVERN_ACTIONS


# ---------------------------------------------------------------------------
# 6. Provenance — recorded only when active, so 'none' stays byte-identical
# ---------------------------------------------------------------------------
CLASSES = [0, 8]
ATTRACTOR = 71


def _make_cache(path: Path, dim=24, n_per=20, noise=0.05, seed=0) -> Path:
    """Tiny clustered cache in the vitl14 blob format (mirrors the vision
    test's fixture) so run_vision's summary path is exercised hermetically."""
    g = torch.Generator().manual_seed(seed)
    ids = CLASSES + [ATTRACTOR]
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


def _run_vision(tmp_path, govern, name):
    cache = _make_cache(tmp_path)
    out = tmp_path / name
    n, summary = run_vision(
        "contra", rate=0.5, epochs=5, out_path=out, cache_path=str(cache),
        classes=CLASSES, attractor_class=ATTRACTOR, samples_per_class=8,
        held_out_per_class=8, contraction=0.0, seed=0, govern=govern)
    return n, summary, out


def test_vision_none_summary_byte_identical_to_default(tmp_path):
    """A baseline (govern='none') vision summary is byte-identical to omitting
    the flag, and carries NO 'govern' key (§11 test 1)."""
    _, s_none, _ = _run_vision(tmp_path, "none", "none.csv")
    cache = _make_cache(tmp_path)
    out = tmp_path / "default.csv"
    _, s_default = run_vision(
        "contra", rate=0.5, epochs=5, out_path=out, cache_path=str(cache),
        classes=CLASSES, attractor_class=ATTRACTOR, samples_per_class=8,
        held_out_per_class=8, contraction=0.0, seed=0)  # no govern arg
    assert "govern" not in s_none
    assert "govern" not in s_default
    assert s_none == s_default


def test_vision_summary_records_active_govern_but_not_csvs(tmp_path):
    """An active (non-baseline) action is recorded in the summary as a no-op
    provenance block, while every emitted CSV stays byte-identical to the
    baseline (§11 test 6 — provenance without behavior change)."""
    n0, s_base, out_base = _run_vision(tmp_path, "none", "base.csv")
    n1, s_gov, out_gov = _run_vision(tmp_path, "annotate", "gov.csv")
    assert n0 == n1

    gov = s_gov["govern"]
    assert gov["action"] == "annotate"
    assert gov["step"] == "pr7-step2-annotate"
    assert gov["implemented"] is True
    assert gov["implemented_actions"] == ["none", "annotate", "refuse"]
    assert gov["events_seen"] > 0

    # The provenance block is the ONLY summary difference.
    assert {k: v for k, v in s_gov.items() if k != "govern"} == s_base

    # Every emitted artifact is byte-identical: governance changed no write.
    assert _emitted_bytes(out_gov) == _emitted_bytes(out_base)


# ---------------------------------------------------------------------------
# 7. CLI entry point — the opt-in flag is wired and validated
# ---------------------------------------------------------------------------
def _cli(tmp_path, name, extra):
    out = tmp_path / name
    cmd = [sys.executable, str(PROBE), "--synthetic", "--arm", "clean",
           "--epochs", "3", "--out", str(out)] + extra
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return proc, out


def test_cli_govern_flag_default_and_noop(tmp_path):
    """The entry point exposes --govern; default == 'none' == 'annotate' for
    all emitted artifacts (the seam is opt-in and inert at the CLI level)."""
    p_default, out_default = _cli(tmp_path, "d.csv", [])
    p_none, out_none = _cli(tmp_path, "n.csv", ["--govern", "none"])
    p_ann, out_ann = _cli(tmp_path, "a.csv", ["--govern", "annotate"])
    assert p_default.returncode == 0, p_default.stderr
    assert p_none.returncode == 0, p_none.stderr
    assert p_ann.returncode == 0, p_ann.stderr
    assert _emitted_bytes(out_default) == _emitted_bytes(out_none)
    assert _emitted_bytes(out_ann) == _emitted_bytes(out_none)


def test_cli_rejects_unknown_govern(tmp_path):
    proc, _ = _cli(tmp_path, "x.csv", ["--govern", "deprecate"])
    assert proc.returncode != 0
    assert "govern" in proc.stderr.lower()
