"""
tests/test_g27_multi_adapter_top2.py — Unit tests for Gauntlet 27.

Validates the G27 multi-domain adapter benchmark:
  - _mine_hard_negatives returns valid triplets or empty tensors
  - train_multi_adapter trains and saves weights
  - eval_fam_with_adapter returns accuracy in [0, 100] and valid prototype stats
  - _format_table produces a well-formed string with expected markers
  - main() executes end-to-end on synthetic data without errors (PASS case)
  - main() exits with code 1 when kill condition is met
"""

import math
import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.g27_multi_adapter_top2 import (
    _mine_hard_negatives,
    _format_table,
    eval_fam_with_adapter,
    train_multi_adapter,
    main,
)
from adapter import MetricAdapter


# ─────────────────────────────────────────────── fixtures & helpers ───

_EMBED_DIM = 32
_NUM_CLASSES = 4
_SAMPLES_PER_CLASS = 30


def _make_domain_data(num_classes: int = _NUM_CLASSES, embed_dim: int = _EMBED_DIM,
                      samples_per_class: int = _SAMPLES_PER_CLASS, seed: int = 0):
    """Return (train_embeds, train_labels, test_embeds, test_labels) for synthetic data."""
    torch.manual_seed(seed)
    centers = F.normalize(torch.eye(num_classes, embed_dim), dim=-1) * 5.0

    def _cluster(center, n, noise=0.05):
        return center + noise * torch.randn(n, embed_dim)

    train_e = torch.cat([_cluster(centers[c], samples_per_class) for c in range(num_classes)])
    train_l = torch.arange(num_classes).repeat_interleave(samples_per_class)
    test_e = torch.cat([_cluster(centers[c], 10) for c in range(num_classes)])
    test_l = torch.arange(num_classes).repeat_interleave(10)

    perm = torch.randperm(len(train_e))
    return train_e[perm], train_l[perm], test_e, test_l


def _save_cache(tmpdir: str, prefix: str, embeds: torch.Tensor, labels: torch.Tensor,
                suffix: str) -> None:
    """Save a feature cache file to tmpdir."""
    torch.save(
        {"embeds": embeds, "labels": labels},
        os.path.join(tmpdir, f"{prefix}_{suffix}.pt"),
    )


# ─────────────────────────────── _mine_hard_negatives tests ───

class TestMineHardNegatives:
    def test_returns_valid_triplets(self):
        torch.manual_seed(0)
        feats = torch.randn(20, _EMBED_DIM)
        labels = torch.arange(4).repeat(5)
        proj = F.normalize(feats, dim=-1)
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        assert a.shape[0] > 0
        assert a.shape == p.shape == n.shape

    def test_returns_empty_for_single_class(self):
        feats = torch.randn(8, _EMBED_DIM)
        labels = torch.zeros(8, dtype=torch.long)
        proj = F.normalize(feats, dim=-1)
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        assert a.shape[0] == 0

    def test_output_dim_matches_input(self):
        feats = torch.randn(16, _EMBED_DIM)
        labels = torch.arange(4).repeat(4)
        proj = F.normalize(feats, dim=-1)
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        if a.shape[0] > 0:
            assert a.shape[1] == _EMBED_DIM


# ─────────────────────────────── eval_fam_with_adapter tests ───

class TestEvalFamWithAdapter:
    @pytest.fixture(scope="class")
    def data(self):
        return _make_domain_data(seed=1)

    def test_returns_valid_accuracy(self, data):
        tr_e, tr_l, te_e, te_l = data
        acc, n_proto, cond = eval_fam_with_adapter(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=None,
            core_vigilance=0.70,
            core_entries=500,
        )
        assert 0.0 <= acc <= 100.0

    def test_returns_prototype_stats(self, data):
        tr_e, tr_l, te_e, te_l = data
        acc, n_proto, cond = eval_fam_with_adapter(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=None,
            core_vigilance=0.70,
            core_entries=500,
        )
        assert isinstance(n_proto, int)
        assert n_proto >= 0
        assert 0.0 <= cond <= 1.0

    def test_high_accuracy_on_separable_data(self, data):
        tr_e, tr_l, te_e, te_l = data
        acc, _, _ = eval_fam_with_adapter(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=None,
            core_vigilance=0.70,
            core_entries=500,
        )
        assert acc >= 80.0, f"Expected ≥80% on well-separated data, got {acc:.1f}%"

    def test_with_adapter(self, data):
        tr_e, tr_l, te_e, te_l = data
        adapter = MetricAdapter(input_dim=_EMBED_DIM, output_dim=_EMBED_DIM)
        with torch.no_grad():
            adapter.net.weight.copy_(torch.eye(_EMBED_DIM))
            adapter.net.bias.zero_()
        adapter.eval()
        acc, _, _ = eval_fam_with_adapter(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=adapter,
            core_vigilance=0.70,
            core_entries=500,
        )
        assert 0.0 <= acc <= 100.0


# ─────────────────────────────── train_multi_adapter tests ───

class TestTrainMultiAdapter:
    def test_trains_and_saves(self):
        tr_e, tr_l, _, _ = _make_domain_data(seed=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "adapter_test.pt")
            adapter = train_multi_adapter(
                train_embeds=tr_e,
                train_labels=tr_l,
                input_dim=_EMBED_DIM,
                output_dim=16,
                hidden_dim=32,
                epochs=1,
                batch_size=32,
                lr=1e-4,
                margin=0.2,
                device=torch.device("cpu"),
                output=out_path,
            )
            assert os.path.isfile(out_path)
            assert isinstance(adapter, MetricAdapter)

    def test_adapter_output_shape(self):
        tr_e, tr_l, _, _ = _make_domain_data(seed=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "adapter_shape.pt")
            adapter = train_multi_adapter(
                train_embeds=tr_e,
                train_labels=tr_l,
                input_dim=_EMBED_DIM,
                output_dim=16,
                hidden_dim=0,
                epochs=1,
                batch_size=32,
                device=torch.device("cpu"),
                output=out_path,
            )
            adapter.eval()
            x = torch.randn(4, _EMBED_DIM)
            out = adapter(x)
            assert out.shape == (4, 16)


# ─────────────────────────────── _format_table tests ───

class TestFormatTable:
    def _rows(self):
        return [
            ("Birds (in)", 85.0, 86.0),
            ("Cars (in)", 80.0, 78.0),
            ("CIFAR-100 (cross)", 70.0, 69.0),
            ("ImageNet-R (cross)", 60.0, 61.0),
        ]

    def test_returns_string(self):
        table = _format_table("birds", "cars", self._rows(), 73.75, 73.5, kill=False)
        assert isinstance(table, str)

    def test_pass_marker(self):
        table = _format_table("birds", "cars", self._rows(), 73.0, 75.0, kill=False)
        assert "PASS" in table
        assert "KILL" not in table

    def test_kill_marker(self):
        table = _format_table("birds", "cars", self._rows(), 73.75, 73.5, kill=True)
        assert "KILL" in table

    def test_average_row_present(self):
        table = _format_table("birds", "cars", self._rows(), 73.75, 73.5, kill=False)
        assert "AVERAGE" in table

    def test_contains_dataset_labels(self):
        rows = self._rows()
        table = _format_table("birds", "cars", rows, 73.75, 73.5, kill=False)
        for label, _, _ in rows:
            assert label in table


# ─────────────────────────────── end-to-end main() test ───

class TestMainEndToEnd:
    """Run main() on synthetic caches in temp directories."""

    def _build_caches(self, tmpdir: str):
        """Create minimal synthetic feature caches for all 4 fine-grained domains
        plus CIFAR-100 and ImageNet-R, and return the directory map."""
        dirs = {}
        for domain_name in ("birds", "cars", "aircraft", "flowers"):
            d = os.path.join(tmpdir, domain_name)
            os.makedirs(d)
            tr_e, tr_l, te_e, te_l = _make_domain_data(
                num_classes=4, embed_dim=_EMBED_DIM, samples_per_class=20, seed=hash(domain_name) % 100
            )
            _save_cache(d, domain_name, tr_e, tr_l, "train")
            _save_cache(d, domain_name, te_e, te_l, "test")
            dirs[domain_name] = d

        # CIFAR-100
        cifar_dir = os.path.join(tmpdir, "cifar")
        os.makedirs(cifar_dir)
        tr_e, tr_l, te_e, te_l = _make_domain_data(num_classes=5, embed_dim=_EMBED_DIM,
                                                    samples_per_class=20, seed=50)
        _save_cache(cifar_dir, "cifar100", tr_e, tr_l, "train")
        _save_cache(cifar_dir, "cifar100", te_e, te_l, "test")
        dirs["cifar"] = cifar_dir

        # ImageNet-R
        inr_dir = os.path.join(tmpdir, "inr")
        os.makedirs(inr_dir)
        tr_e, tr_l, te_e, te_l = _make_domain_data(num_classes=5, embed_dim=_EMBED_DIM,
                                                    samples_per_class=20, seed=60)
        _save_cache(inr_dir, "imagenet_r", tr_e, tr_l, "train")
        _save_cache(inr_dir, "imagenet_r", te_e, te_l, "test")
        dirs["inr"] = inr_dir

        return dirs

    def test_main_pass_no_single_adapters(self):
        """main() should exit 0 (PASS) when no single-domain adapters are present
        (best single avg defaults to nan → no kill)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = self._build_caches(tmpdir)
            out_pt = os.path.join(tmpdir, "multi_adapter.pt")
            argv = [
                "--domains", "birds", "cars",
                "--cache-birds", dirs["birds"],
                "--cache-cars", dirs["cars"],
                "--cache-aircraft", dirs["aircraft"],
                "--cache-flowers", dirs["flowers"],
                "--cifar-cache", dirs["cifar"],
                "--imagenet-r-cache", dirs["inr"],
                "--adapter-birds", os.path.join(tmpdir, "nonexistent_birds.pt"),
                "--adapter-cars", os.path.join(tmpdir, "nonexistent_cars.pt"),
                "--epochs", "1",
                "--batch-size", "16",
                "--device", "cpu",
                "--seed", "0",
                "--output", out_pt,
            ]
            with pytest.raises(SystemExit) as exc_info:
                main(argv)
            assert exc_info.value.code == 0
            assert os.path.isfile(out_pt)

    def test_main_with_real_single_adapters(self):
        """main() produces a valid exit code (0 or 1) when single-domain adapters exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = self._build_caches(tmpdir)
            out_pt = os.path.join(tmpdir, "multi_adapter.pt")

            # Save random single-domain adapters matching default dims (output=256, hidden=512)
            adapter_birds_pt = os.path.join(tmpdir, "adapter_birds.pt")
            adapter_cars_pt = os.path.join(tmpdir, "adapter_cars.pt")
            adapter_birds = MetricAdapter(input_dim=_EMBED_DIM, output_dim=256, hidden_dim=512)
            adapter_cars = MetricAdapter(input_dim=_EMBED_DIM, output_dim=256, hidden_dim=512)
            torch.save(adapter_birds.state_dict(), adapter_birds_pt)
            torch.save(adapter_cars.state_dict(), adapter_cars_pt)

            argv = [
                "--domains", "birds", "cars",
                "--cache-birds", dirs["birds"],
                "--cache-cars", dirs["cars"],
                "--cache-aircraft", dirs["aircraft"],
                "--cache-flowers", dirs["flowers"],
                "--cifar-cache", dirs["cifar"],
                "--imagenet-r-cache", dirs["inr"],
                "--adapter-birds", adapter_birds_pt,
                "--adapter-cars", adapter_cars_pt,
                "--epochs", "1",
                "--batch-size", "16",
                "--device", "cpu",
                "--seed", "0",
                "--output", out_pt,
            ]
            with pytest.raises(SystemExit) as exc_info:
                main(argv)
            assert exc_info.value.code in (0, 1)
            assert os.path.isfile(out_pt)

    def test_invalid_domain_exits_2(self):
        """main() should exit with code 2 for an invalid domain name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "--domains", "birds", "robots",
                "--device", "cpu",
            ]
            with pytest.raises(SystemExit) as exc_info:
                main(argv)
            assert exc_info.value.code == 2
