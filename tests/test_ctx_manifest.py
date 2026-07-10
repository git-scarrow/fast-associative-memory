"""Tests for the committed, sealed PR-13 query manifest (memo §12 R-1).

These read the committed JSON artifact and check it against the
registered rule. They do NOT rebuild it from §7 material and they render
nothing: the manifest is the immutable input to the sealed replay, and
what matters is that the artifact on disk still satisfies its seal and
its registration.
"""

import hashlib
import json
import os

import pytest

from harness.ctx import cells, replay, sample

MANIFEST_PATH = os.path.join("harness", "ctx", "manifests",
                             "query_manifest_v1.json")


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def sample_policy():
    return sample.load_sample_policy()


def test_manifest_is_sealed_and_verifies(manifest):
    assert replay.verify_manifest(manifest) == manifest["manifest_sha256"]


def test_manifest_seal_detects_a_single_flipped_identity(manifest):
    tampered = json.loads(json.dumps(manifest))
    tampered["queries"][0] = tampered["queries"][0] + "x"
    with pytest.raises(replay.SealError):
        replay.verify_manifest(tampered)


def test_manifest_pins_the_registered_sample_policy(manifest, sample_policy):
    with open(os.path.join("harness", "ctx", "policy", "sample_v1.json"),
              "rb") as fh:
        want = hashlib.sha256(fh.read()).hexdigest()
    assert manifest["sample_policy"]["sha256"] == want
    assert manifest["sample_policy"]["salt"] == sample_policy["salt"] \
        == "pr13-sample-v1"
    assert manifest["sample_policy"]["cap_per_cell"] == 256


def test_manifest_pins_the_frozen_disposition_policy(manifest):
    with open(os.path.join("harness", "ctx", "policy",
                           "disposition_policy_v1.json"), "rb") as fh:
        want = hashlib.sha256(fh.read()).hexdigest()
    assert manifest["disposition_policy"]["sha256"] == want


def test_every_registered_fam_cell_is_present_and_capped(manifest):
    got = [c["cell_id"] for c in manifest["cells"]]
    assert got == [c[0] for c in cells.FAM_CELLS]
    assert len(got) == 13
    for cell in manifest["cells"]:
        assert cell["n_selected"] <= 256
        assert cell["n_selected"] == len(cell["queries"])
        assert sum(s["n_selected"] for s in cell["strata"]) == cell["n_selected"]
        assert sum(s["n_total"] for s in cell["strata"]) == cell["n_total"]


def test_every_nonempty_stratum_contributes_at_least_one_query(manifest):
    for cell in manifest["cells"]:
        for stratum in cell["strata"]:
            if stratum["n_total"] > 0:
                assert stratum["n_selected"] >= 1


def test_strata_never_carry_a_compiler_mechanism(manifest):
    seen = {s["disposition_class"] for c in manifest["cells"]
            for s in c["strata"]}
    assert seen <= {"withhold", "defer", "dual_present", "caveat", "assert"}


def test_harm_category_matches_the_registered_cell_role(manifest):
    roles = {c[0]: c[3] for c in cells.FAM_CELLS}
    for cell in manifest["cells"]:
        assert cell["role"] == roles[cell["cell_id"]]
        assert {s["harm_category"] for s in cell["strata"]} == {cell["role"]}


def test_selected_queries_are_in_salted_hash_order(manifest, sample_policy):
    salt = sample_policy["salt"]
    for cell in manifest["cells"]:
        hashes = [sample.order_hash(q, salt) for q in cell["queries"]]
        assert hashes == sorted(hashes)


def test_query_identities_are_unique_and_well_formed(manifest):
    queries = manifest["queries"]
    assert len(set(queries)) == len(queries)
    assert all(q.startswith(("fam-v1:", "organic:", "mt:")) for q in queries)


def test_organic_and_multiturn_are_retained_in_full(manifest):
    with open(cells.SYNTHETIC_QUERIES, encoding="utf-8") as fh:
        organic = json.load(fh)
    with open(cells.SYNTHETIC_SESSIONS, encoding="utf-8") as fh:
        sessions = json.load(fh)
    assert manifest["organic"]["n_queries"] == len(organic) == 20
    assert sorted(manifest["organic"]["queries"]) == \
        sorted(q["query_id"] for q in organic)
    assert manifest["multiturn"]["n_sessions"] == len(sessions) == 12
    assert all(len(s["turns"]) == 3 for s in manifest["multiturn"]["sessions"])


def test_organic_pins_the_committed_synthetic_ledger(manifest):
    with open(cells.SYNTHETIC_LEDGER, "rb") as fh:
        want = hashlib.sha256(fh.read()).hexdigest()
    assert manifest["organic"]["ledger_sha256"] == want
    assert manifest["organic"]["corpus"] == "synthetic"


def test_totals_reconcile_with_the_listed_identities(manifest):
    t = manifest["totals"]
    assert t["fam_queries_selected"] == sum(c["n_selected"]
                                            for c in manifest["cells"])
    assert t["queries"] == (t["fam_queries_selected"] + t["organic_queries"]
                            + t["multiturn_turns"]) == len(manifest["queries"])
    assert t["rows"] == t["queries"] * len(replay.ARM_PLAN) == 20304


def test_manifest_expands_to_the_registered_row_count(manifest):
    rows = replay.expand_rows(manifest)
    assert len(rows) == manifest["totals"]["rows"]
    assert len(set(r["row_id"] for r in rows)) == len(rows)
    arms = {r["arm"] for r in rows}
    assert arms == {"governed", "raw_matched", "raw_native", "none"}


def test_every_arm_sees_exactly_the_same_query_identities(manifest):
    rows = replay.expand_rows(manifest)
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], set()).add(r["query_id"])
    reference = set(manifest["queries"])
    assert all(ids == reference for ids in by_arm.values())
