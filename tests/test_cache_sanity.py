"""tests/test_cache_sanity.py — hermetic tests for tools/cache_sanity.py.

The real feature caches are gitignored and host-specific, so these tests build
tiny synthetic caches in tmp dirs and validate the inspector's verdicts:
ok / missing / broken-symlink / format-error / integrity-error, the
embeds-vs-features key detection, and the A1-style class-count check. No
network, no GPU, no real caches required.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import cache_sanity  # noqa: E402


def _spec(path, **kw):
    return {"id": "t", "tier": "test", "path": str(path), "consumers": [], **kw}


def _save(path: Path, feature_key="embeds", n=12, d=8, n_classes=3,
          inject_nan=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = torch.randn(n, d)
    if inject_nan:
        feats[0, 0] = float("nan")
    labels = torch.arange(n) % n_classes
    torch.save({feature_key: feats, "labels": labels.long()}, path)
    return feats, labels


def test_ok_cache_with_embeds_key(tmp_path):
    p = tmp_path / "cache.pt"
    feats, labels = _save(p)
    rec = cache_sanity.inspect_cache(_spec(p))
    assert rec["status"] == "ok"
    assert rec["feature_key"] == "embeds"
    assert rec["features"]["shape"] == [12, 8]
    assert rec["labels"]["n_classes_present"] == 3
    assert rec["checks"]["rows_match"] and rec["checks"]["finite"]


def test_features_key_detected(tmp_path):
    p = tmp_path / "dogs.pt"
    _save(p, feature_key="features")
    rec = cache_sanity.inspect_cache(_spec(p))
    assert rec["status"] == "ok"
    assert rec["feature_key"] == "features"


def test_missing_cache(tmp_path):
    rec = cache_sanity.inspect_cache(_spec(tmp_path / "nope.pt"))
    assert rec["status"] == "missing"
    assert "resolved_path" not in rec


def test_broken_symlink_reported(tmp_path):
    link_dir = tmp_path / "feature_cache_x"
    link_dir.symlink_to(tmp_path / "does-not-exist")
    # Symlink detection walks path components relative to REPO_ROOT; use an
    # absolute path so the component walk sees the link itself.
    rec = cache_sanity.inspect_cache(_spec(link_dir / "cache.pt"))
    assert rec["status"] in ("missing", "broken-symlink")


def test_nan_flags_integrity_error(tmp_path):
    p = tmp_path / "bad.pt"
    _save(p, inject_nan=True)
    rec = cache_sanity.inspect_cache(_spec(p))
    assert rec["status"] == "integrity-error"
    assert rec["features"]["nan_count"] == 1


def test_format_error_on_wrong_keys(tmp_path):
    p = tmp_path / "weird.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": torch.zeros(3)}, p)
    rec = cache_sanity.inspect_cache(_spec(p))
    assert rec["status"] == "format-error"


def test_required_class_check(tmp_path):
    p = tmp_path / "cifar.pt"
    # 100 samples of class 0, 100 of class 1, 1 of class 7 (attractor).
    feats = torch.randn(201, 8)
    labels = torch.cat([torch.zeros(100), torch.ones(100),
                        torch.full((1,), 7)]).long()
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeds": feats, "labels": labels}, p)

    ok_req = {"study": "A1-like", "classes": [0, 1],
              "min_per_class": 96, "attractor": 7}
    rec = cache_sanity.inspect_cache(_spec(p, required_classes=ok_req))
    assert rec["checks"]["study_requirements_met"] is True

    bad_req = {"study": "A1-like", "classes": [0, 1, 2],
               "min_per_class": 96, "attractor": 7}
    rec = cache_sanity.inspect_cache(_spec(p, required_classes=bad_req))
    assert rec["checks"]["study_requirements_met"] is False


def test_fallback_resolution(tmp_path):
    real = tmp_path / "real.pt"
    _save(real)
    rec = cache_sanity.inspect_cache(
        _spec(tmp_path / "primary-missing.pt", fallbacks=[str(real)]))
    assert rec["status"] == "ok"
    assert rec["resolved_path"] == str(real)
    assert len(rec["tried"]) == 2


def test_manifest_is_json_serializable(tmp_path):
    p = tmp_path / "c.pt"
    _save(p)
    manifest = cache_sanity.build_manifest(specs=[_spec(p)])
    text = json.dumps(manifest)
    assert "generated_at" in manifest and manifest["summary"] == {"ok": 1}
    assert isinstance(text, str)


def test_hash_option(tmp_path):
    p = tmp_path / "c.pt"
    _save(p)
    rec = cache_sanity.inspect_cache(_spec(p), do_hash=True)
    assert len(rec["sha256"]) == 64
