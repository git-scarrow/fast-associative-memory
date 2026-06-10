"""tests/test_failure_mode_vision.py — hermetic tests for the PR-2b
cache-backed vision arm of benchmarks/failure_mode_probe.py.

The gentoo contradiction-arm result is only trustworthy if the vision wiring
does what it claims BEFORE any cache-backed run: loads a feature cache through
the exact #87 ``VisionDriftStream``, holds contraction fixed, stamps every
write, injects through the PR-2a machinery, and emits the PR-2a schema. These
tests pin that on a tiny synthetic cache FILE (torch.save'd embeds/labels —
the same blob format as vitl14_cifar100_train), CPU-only, no GPU, no network:

  1. the clean vision arm is a true negative control (no injection labels);
  2. the contra vision arm end-to-end: schema, label domain, strict=>lenient,
     stale never fires, injections accounted one outcome each, forks occur
     and surface as contradictory-flagged wrong probes;
  3. stale is refused (PR-2c, not wired);
  4. byte-identical determinism for a fixed seed (reproducible gentoo runs);
  5. the summary's per-epoch fork accounting matches the CSV's labels.
"""
import csv
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.failure_mode_probe import (  # noqa: E402
    MODE_PRECEDENCE, OUT_COLS, run_vision)

# Non-contiguous cached class ids on purpose — the real protocol uses
# {0, 8, 19, 33} with attractor 71, and VisionDriftStream must remap them.
CLASSES = [0, 8]
ATTRACTOR = 71


def _make_cache(path: Path, dim=24, n_per=20, noise=0.05, seed=0) -> Path:
    """Tiny clustered cache in the vitl14 blob format: {"embeds", "labels"}."""
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


def _run(arm, tmp_path, rate=0.5, epochs=5, seed=0, name="out.csv"):
    cache = _make_cache(tmp_path)
    out = tmp_path / name
    n, summary = run_vision(
        arm, rate=rate, epochs=epochs, out_path=out, cache_path=str(cache),
        classes=CLASSES, attractor_class=ATTRACTOR, samples_per_class=8,
        held_out_per_class=8, contraction=0.0, seed=seed)
    return n, summary, out


def test_vision_clean_arm_is_negative_control(tmp_path):
    n, summary, out = _run("clean", tmp_path)
    assert n > 0
    assert summary["n_injections"] == 0
    assert summary["injection_outcomes"] == {}
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == n
    for r in rows:
        assert r["arm"] == "clean"
        assert r["injection_rate"] == "0.0"
        assert r["failure_mode"] in ("CORRECT", "BLENDED", "OTHER_WRONG")
        assert r["contradictory_strict"] == "0"
        assert r["contradictory_lenient"] == "0"
        assert r["stale_strict"] == "0"
        assert r["stale_lenient"] == "0"


def test_vision_contra_arm_end_to_end(tmp_path):
    n, summary, out = _run("contra", tmp_path)
    assert n > 0
    rows = list(csv.DictReader(open(out)))
    assert set(rows[0].keys()) == set(OUT_COLS)

    # Every injection got exactly one outcome; the protocol size is exact.
    n_train = len(CLASSES) * 8
    expected = int(round(0.5 * n_train)) * 5
    assert summary["n_injections"] == expected
    assert sum(summary["injection_outcomes"].values()) == expected
    # Stationary tight clusters + disagreeing payloads must fork.
    assert summary["injection_outcomes"].get("forked", 0) > 0

    allowed = set(MODE_PRECEDENCE) | {"CORRECT"}
    n_contra = 0
    for r in rows:
        assert r["arm"] == "contra"
        assert r["supersede_epoch"] == "-1"
        assert float(r["contraction"]) == 0.0
        assert r["failure_mode"] in allowed
        assert int(r["contradictory_strict"]) <= int(r["contradictory_lenient"])
        if r["vote_correct"] == "1":
            assert r["failure_mode"] == "CORRECT"
        assert r["stale_lenient"] == "0"  # no stale protocol in this arm
        n_contra += int(r["contradictory_lenient"])
    assert n_contra > 0, "50% injection over 5 stationary epochs must " \
                         "surface contradictory-flagged wrong probes"


def test_vision_rejects_stale_arm(tmp_path):
    cache = _make_cache(tmp_path)
    with pytest.raises(ValueError, match="PR-2c"):
        run_vision("stale", rate=0.0, epochs=1, out_path=tmp_path / "x.csv",
                   cache_path=str(cache), classes=CLASSES,
                   attractor_class=ATTRACTOR, samples_per_class=8,
                   held_out_per_class=8)


def test_vision_run_is_deterministic(tmp_path):
    _, s1, out1 = _run("contra", tmp_path, name="a.csv")
    _, s2, out2 = _run("contra", tmp_path, name="b.csv")
    assert out1.read_bytes() == out2.read_bytes()
    assert s1 == s2


def test_vision_summary_matches_csv_labels(tmp_path):
    n, summary, out = _run("contra", tmp_path)
    rows = list(csv.DictReader(open(out)))
    by_epoch = {e["epoch"]: e for e in summary["per_epoch"]}
    assert sum(e["probes"] for e in summary["per_epoch"]) == n
    for epoch, log in by_epoch.items():
        er = [r for r in rows if int(r["epoch"]) == epoch]
        assert len(er) == log["probes"]
        assert sum(int(r["vote_correct"]) == 0 for r in er) == log["wrong"]
        assert (sum(int(r["contradictory_strict"]) for r in er)
                == log["contradictory_strict"])
        assert (sum(int(r["contradictory_lenient"]) for r in er)
                == log["contradictory_lenient"])
