"""
tests/test_g26_cross_domain_matrix.py — Unit tests for Gauntlet 26.

Validates the G26 cross-domain regression matrix benchmark:
  - _format_matrix produces a well-formed table with expected markers
  - main() correctly loads adapters, handles missing files (warning + skip)
  - main() exits with code 0 when all adapters pass, 1 when any adapter kills
  - main() correctly evaluates adapters against synthetic datasets
"""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

# Allow importing from the repo root regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.g26_cross_domain_matrix import _format_matrix, main
from benchmarks.gauntlet_19_cross_domain import (
    accuracy_relative_drop,
    check_kill_condition,
)
from adapter import MetricAdapter


# ─────────────────────────────────────────────── fixtures & helpers ───

_EMBED_DIM = 32
_NUM_CLASSES = 5
_SAMPLES_PER_CLASS = 40
_TOTAL_TRAIN = _NUM_CLASSES * _SAMPLES_PER_CLASS
_TOTAL_TEST = _NUM_CLASSES * 10


def _make_generic_data(seed: int = 0):
    """Return (train_embeds, train_labels, test_embeds, test_labels) for a
    well-separated synthetic dataset."""
    torch.manual_seed(seed)
    centers = F.normalize(torch.eye(_NUM_CLASSES, _EMBED_DIM), dim=-1) * 5.0

    def _cluster(center, n, noise=0.05):
        return center + noise * torch.randn(n, _EMBED_DIM)

    train_e = torch.cat([_cluster(centers[c], _SAMPLES_PER_CLASS) for c in range(_NUM_CLASSES)])
    train_l = torch.arange(_NUM_CLASSES).repeat_interleave(_SAMPLES_PER_CLASS)
    test_e = torch.cat([_cluster(centers[c], 10) for c in range(_NUM_CLASSES)])
    test_l = torch.arange(_NUM_CLASSES).repeat_interleave(10)

    perm = torch.randperm(len(train_e))
    return train_e[perm], train_l[perm], test_e, test_l


def _save_feature_cache(tmpdir: str, name: str, embed_dim: int = _EMBED_DIM) -> str:
    """Save synthetic train/test .pt files into tmpdir and return tmpdir."""
    torch.manual_seed(0)
    train = {
        "embeds": torch.randn(_TOTAL_TRAIN, embed_dim),
        "labels": torch.arange(_NUM_CLASSES).repeat_interleave(_SAMPLES_PER_CLASS),
    }
    test = {
        "embeds": torch.randn(_TOTAL_TEST, embed_dim),
        "labels": torch.arange(_NUM_CLASSES).repeat_interleave(10),
    }
    torch.save(train, os.path.join(tmpdir, f"{name}_train.pt"))
    torch.save(test, os.path.join(tmpdir, f"{name}_test.pt"))
    return tmpdir


def _save_identity_adapter(path: str, embed_dim: int = _EMBED_DIM) -> None:
    """Save an identity-initialised adapter state-dict to *path*."""
    adapter = MetricAdapter(input_dim=embed_dim, output_dim=embed_dim)
    with torch.no_grad():
        adapter.net.weight.copy_(torch.eye(embed_dim))
        adapter.net.bias.zero_()
    torch.save(adapter.state_dict(), path)


# ──────────────────────────────────────── _format_matrix tests ───

class TestFormatMatrix:
    def _make_results(self, baseline=80.0, birds=78.0, cars=79.0):
        return {
            "CIFAR-100": {"baseline": baseline, "Birds": birds, "Cars": cars},
        }

    def test_returns_string(self):
        table, _ = _format_matrix(self._make_results(), ["Birds", "Cars"], threshold=5.0)
        assert isinstance(table, str)

    def test_contains_dataset_name(self):
        table, _ = _format_matrix(self._make_results(), ["Birds", "Cars"], threshold=5.0)
        assert "CIFAR-100" in table

    def test_contains_adapter_names(self):
        table, _ = _format_matrix(self._make_results(), ["Birds", "Cars"], threshold=5.0)
        assert "Birds" in table
        assert "Cars" in table

    def test_pass_all(self):
        table, failures = _format_matrix(self._make_results(), ["Birds", "Cars"], threshold=5.0)
        assert not failures
        assert "PASS" in table

    def test_kill_when_drop_exceeds_threshold(self):
        # Birds drops 10 pp → kill
        results = {"CIFAR-100": {"baseline": 80.0, "Birds": 70.0, "Cars": 79.0}}
        table, failures = _format_matrix(results, ["Birds", "Cars"], threshold=5.0)
        assert failures
        assert "KILL" in table
        assert any("Birds" in f for f in failures)

    def test_exact_threshold_is_pass(self):
        # Exactly 5.0 pp drop → pass (strict >)
        results = {"CIFAR-100": {"baseline": 80.0, "Birds": 75.0}}
        table, failures = _format_matrix(results, ["Birds"], threshold=5.0)
        assert not failures
        assert "PASS" in table

    def test_multiple_datasets(self):
        results = {
            "CIFAR-100": {"baseline": 80.0, "Birds": 79.0},
            "ImageNet-R": {"baseline": 70.0, "Birds": 69.0},
        }
        table, failures = _format_matrix(results, ["Birds"], threshold=5.0)
        assert "CIFAR-100" in table
        assert "ImageNet-R" in table
        assert not failures

    def test_missing_adapter_shown_as_na(self):
        # Only 'Birds' evaluated; 'Cars' absent → should show N/A
        results = {"CIFAR-100": {"baseline": 80.0, "Birds": 79.0}}
        table, failures = _format_matrix(results, ["Birds", "Cars"], threshold=5.0)
        assert "N/A" in table

    def test_drop_shown_in_table(self):
        results = {"CIFAR-100": {"baseline": 80.0, "Birds": 77.0}}
        table, _ = _format_matrix(results, ["Birds"], threshold=5.0)
        assert "+3.00" in table or "3.00" in table

    def test_negative_drop_shown(self):
        # Adapter improves accuracy → drop is negative
        results = {"CIFAR-100": {"baseline": 78.0, "Birds": 80.0}}
        table, failures = _format_matrix(results, ["Birds"], threshold=5.0)
        assert not failures
        assert "-2.00" in table or "−2.00" in table or "-2" in table

    def test_header_present(self):
        table, _ = _format_matrix(self._make_results(), ["Birds"], threshold=5.0)
        assert "G26" in table


# ──────────────────────────────────────── main() integration tests ───

class TestMainMissingAdapter:
    """main() should warn and skip missing adapter files without crashing."""

    def test_missing_adapter_skipped(self, tmp_path, capsys):
        cifar_dir = tmp_path / "cifar"
        cifar_dir.mkdir()
        _save_feature_cache(str(cifar_dir), "cifar100", embed_dim=_EMBED_DIM)

        imgr_dir = tmp_path / "imagenet_r"
        imgr_dir.mkdir()
        _save_feature_cache(str(imgr_dir), "imagenet_r", embed_dim=_EMBED_DIM)

        # All adapter paths point to non-existent files
        with pytest.raises(SystemExit) as exc:
            main([
                "--cifar-cache", str(cifar_dir),
                "--imagenet-r-cache", str(imgr_dir),
                "--adapter-birds", str(tmp_path / "nonexistent_birds.pt"),
                "--adapter-cars", str(tmp_path / "nonexistent_cars.pt"),
                "--adapter-aircraft", str(tmp_path / "nonexistent_aircraft.pt"),
                "--adapter-flowers", str(tmp_path / "nonexistent_flowers.pt"),
                "--input-dim", str(_EMBED_DIM),
                "--output-dim", str(_EMBED_DIM),
                "--hidden-dim", "0",
            ])
        # No adapters loaded → only baseline run → exit 0
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err


class TestMainAllPass:
    """main() should exit 0 when all adapters pass the kill condition."""

    def test_identity_adapters_pass(self, tmp_path):
        cifar_dir = tmp_path / "cifar"
        cifar_dir.mkdir()
        _save_feature_cache(str(cifar_dir), "cifar100", embed_dim=_EMBED_DIM)

        imgr_dir = tmp_path / "imagenet_r"
        imgr_dir.mkdir()
        _save_feature_cache(str(imgr_dir), "imagenet_r", embed_dim=_EMBED_DIM)

        # Save identity adapters for all 4 domains
        adapter_paths = {}
        for name in ["birds", "cars", "aircraft", "flowers"]:
            p = str(tmp_path / f"adapter_{name}.pt")
            _save_identity_adapter(p, embed_dim=_EMBED_DIM)
            adapter_paths[name] = p

        with pytest.raises(SystemExit) as exc:
            main([
                "--cifar-cache", str(cifar_dir),
                "--imagenet-r-cache", str(imgr_dir),
                "--adapter-birds", adapter_paths["birds"],
                "--adapter-cars", adapter_paths["cars"],
                "--adapter-aircraft", adapter_paths["aircraft"],
                "--adapter-flowers", adapter_paths["flowers"],
                "--input-dim", str(_EMBED_DIM),
                "--output-dim", str(_EMBED_DIM),
                "--hidden-dim", "0",
                "--threshold", "5.0",
            ])
        assert exc.value.code == 0


class TestMainKillCondition:
    """main() should exit 1 when any adapter exceeds the drop threshold."""

    def test_distorting_adapter_kills(self, tmp_path):
        """Mock eval_fam_accuracy to return values that trigger the kill condition."""
        cifar_dir = tmp_path / "cifar"
        cifar_dir.mkdir()
        _save_feature_cache(str(cifar_dir), "cifar100", embed_dim=_EMBED_DIM)

        imgr_dir = tmp_path / "imagenet_r"
        imgr_dir.mkdir()
        _save_feature_cache(str(imgr_dir), "imagenet_r", embed_dim=_EMBED_DIM)

        # Save identity adapters for all 4 domains
        adapter_paths = {}
        for name in ["birds", "cars", "aircraft", "flowers"]:
            p = str(tmp_path / f"adapter_{name}.pt")
            _save_identity_adapter(p, embed_dim=_EMBED_DIM)
            adapter_paths[name] = p

        # Mock eval_fam_accuracy to return: baseline=90% for adapter=None,
        # then 83% for Birds (drop=7pp > 5pp → KILL), 89% for others (pass).
        call_counter = {"n": 0}

        def _mock_eval(train_e, train_l, test_e, test_l, device, num_classes,
                       adapter=None, **kwargs):
            if adapter is None:
                return 90.0, 0.1
            call_counter["n"] += 1
            # Birds is loaded first; return low accuracy to trigger kill
            if call_counter["n"] % 4 == 1:
                return 83.0, 0.1  # 7pp drop — kill
            return 89.0, 0.1  # 1pp drop — pass

        with patch(
            "benchmarks.g26_cross_domain_matrix.eval_fam_accuracy",
            side_effect=_mock_eval,
        ):
            with pytest.raises(SystemExit) as exc:
                main([
                    "--cifar-cache", str(cifar_dir),
                    "--imagenet-r-cache", str(imgr_dir),
                    "--adapter-birds", adapter_paths["birds"],
                    "--adapter-cars", adapter_paths["cars"],
                    "--adapter-aircraft", adapter_paths["aircraft"],
                    "--adapter-flowers", adapter_paths["flowers"],
                    "--input-dim", str(_EMBED_DIM),
                    "--output-dim", str(_EMBED_DIM),
                    "--hidden-dim", "0",
                    "--threshold", "5.0",
                ])
        assert exc.value.code == 1


class TestMainMissingCacheDirectory:
    """main() should handle missing feature cache gracefully (skip dataset)."""

    def test_missing_cache_skipped(self, tmp_path, capsys):
        # No cache directories exist at all
        with pytest.raises(SystemExit) as exc:
            main([
                "--cifar-cache", str(tmp_path / "nonexistent_cifar"),
                "--imagenet-r-cache", str(tmp_path / "nonexistent_imagenet_r"),
                "--adapter-birds", str(tmp_path / "nonexistent.pt"),
                "--adapter-cars", str(tmp_path / "nonexistent.pt"),
                "--adapter-aircraft", str(tmp_path / "nonexistent.pt"),
                "--adapter-flowers", str(tmp_path / "nonexistent.pt"),
            ])
        # No datasets evaluated, no failures → exit 0
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
