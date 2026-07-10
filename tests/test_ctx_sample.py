"""Hermetic tests for the PR-13 registered sample (memo §12 R-1).

SELECTION-TIMING KILL COMPLIANCE (memo §8.1): every fixture is synthetic.
No §7 cell material is read, and no context block is rendered anywhere in
this file.
"""

import hashlib

import pytest

from harness.ctx.compile import load_policy
from harness.ctx.sample import (
    allocate,
    disposition_class,
    load_sample_policy,
    order_hash,
    sample_cell,
)


@pytest.fixture(scope="module")
def policy():
    return load_policy()


@pytest.fixture(scope="module")
def sample_policy():
    return load_sample_policy()


def make_item(item_id, state="agent-readable", signals=None, candidate_set=None):
    return {
        "item_id": item_id,
        "source_id": "test-v1",
        "content": f"content of {item_id}",
        "content_kind": "source-native",
        "content_sha256": hashlib.sha256(item_id.encode()).hexdigest(),
        "event_time": None,
        "ingest_time": None,
        "evidence": [{"adapter_id": "test-v1", "signal": s, "value": True,
                      "evidence_ptr": f"ptr:{item_id}:{s}",
                      "tier": "core-certified" if s == "merge_suspect"
                              else "harness-heuristic"}
                     for s in (signals or ["source_identity"])],
        "relations": {"supersedes": [], "contradicts": [],
                      "candidate_set_id": candidate_set},
        "state": state,
        "policy_version": "1.0",
    }


# --- registered artifact ---------------------------------------------------

def test_sample_policy_pins_the_registered_constants(sample_policy):
    assert sample_policy["salt"] == "pr13-sample-v1"
    assert sample_policy["cap_per_cell"] == 256
    assert sample_policy["hash_order"]["algorithm"] == "sha256"
    assert sample_policy["strata"]["keys"] == ["harm_category",
                                               "disposition_class"]


# --- hash order ------------------------------------------------------------

def test_order_hash_is_sha256_of_identity_then_salt():
    want = hashlib.sha256(b"fam-v1:c:e0:p1" + b"pr13-sample-v1").hexdigest()
    assert order_hash("fam-v1:c:e0:p1", "pr13-sample-v1") == want


def test_order_hash_depends_on_the_salt():
    assert order_hash("q", "a") != order_hash("q", "b")


# --- disposition class -----------------------------------------------------

def test_disposition_class_takes_the_highest_precedence_item(policy):
    items = [
        make_item("a"),                                    # assert
        make_item("b", signals=["superseded_by"]),         # caveat
        make_item("c", signals=["merge_suspect"]),         # defer
    ]
    assert disposition_class(items, policy) == "defer"


def test_disposition_class_is_budget_and_turn_free(policy):
    """It reads the rule table, not the compiler's budget loop: a query
    of clean items is `assert` no matter how large its content is."""
    items = [make_item("a"), make_item("b")]
    assert disposition_class(items, policy) == "assert"


def test_disposition_class_never_returns_a_compiler_mechanism(policy):
    for signals, state in ((["source_identity"], "agent-readable"),
                           (["superseded_by"], "stale"),
                           (["oneshot_tie"], "agent-readable"),
                           (["source_identity"], "quarantined")):
        cls = disposition_class([make_item("x", state=state, signals=signals,
                                           candidate_set="cs")], policy)
        assert cls not in ("withdraw", "summarize")


# --- allocation ------------------------------------------------------------

def test_allocate_returns_everything_when_under_cap():
    sizes = {("harm", "assert"): 10, ("harm", "caveat"): 5}
    assert allocate(256, sizes) == sizes


def test_allocate_is_proportional_and_sums_to_cap():
    sizes = {("harm", "assert"): 900, ("harm", "caveat"): 90,
             ("harm", "defer"): 10}
    got = allocate(100, sizes)
    assert sum(got.values()) == 100
    assert got[("harm", "assert")] == 90
    assert got[("harm", "caveat")] == 9
    assert got[("harm", "defer")] == 1


def test_allocate_gives_every_nonempty_stratum_at_least_one():
    """A stratum that rounds to zero is repaired from the largest."""
    sizes = {("harm", "assert"): 10_000, ("harm", "defer"): 3}
    got = allocate(256, sizes)
    assert got[("harm", "defer")] == 1
    assert got[("harm", "assert")] == 255
    assert sum(got.values()) == 256


def test_allocate_never_exceeds_stratum_size():
    sizes = {("harm", "assert"): 3, ("harm", "caveat"): 10_000}
    got = allocate(256, sizes)
    assert got[("harm", "assert")] <= 3
    assert sum(got.values()) == 256


def test_allocate_is_deterministic_under_dict_reordering():
    a = {("harm", "assert"): 500, ("harm", "caveat"): 500,
         ("harm", "defer"): 500}
    b = dict(reversed(list(a.items())))
    assert allocate(256, a) == allocate(256, b)


def test_allocate_ignores_empty_strata():
    sizes = {("harm", "assert"): 5, ("harm", "defer"): 0}
    assert allocate(256, sizes) == {("harm", "assert"): 5}


def test_allocate_rejects_a_cap_below_the_stratum_count():
    sizes = {("harm", d): 10 for d in ("assert", "caveat", "defer")}
    with pytest.raises(ValueError):
        allocate(2, sizes)


# --- cell sampling ---------------------------------------------------------

def _bundles(n, signals_for):
    return {f"fam-v1:cell:e0:p{i}": {"items": [make_item(f"i{i}",
                                                         signals=signals_for(i))]}
            for i in range(n)}


def test_sample_cell_retains_a_small_cell_whole(policy, sample_policy):
    bundles = _bundles(40, lambda i: ["source_identity"])
    out = sample_cell("cell", "harm", bundles, policy, sample_policy)
    assert out["n_total"] == 40 and out["n_selected"] == 40
    assert sorted(out["queries"]) == sorted(bundles)


def test_sample_cell_caps_and_covers_every_stratum(policy, sample_policy):
    # 1000 assert rows, 7 defer rows: the rare stratum must survive.
    bundles = _bundles(1007, lambda i: ["merge_suspect"] if i < 7
                       else ["source_identity"])
    out = sample_cell("cell", "harm", bundles, policy, sample_policy)
    assert out["n_selected"] == 256
    classes = {s["disposition_class"]: s for s in out["strata"]}
    assert classes["defer"]["n_total"] == 7
    assert classes["defer"]["n_selected"] >= 1
    assert sum(s["n_selected"] for s in out["strata"]) == 256
    assert sum(s["n_total"] for s in out["strata"]) == 1007


def test_sample_cell_is_byte_deterministic(policy, sample_policy):
    bundles = _bundles(600, lambda i: ["superseded_by"] if i % 3 else
                       ["source_identity"])
    a = sample_cell("cell", "harm", bundles, policy, sample_policy)
    b = sample_cell("cell", "harm", dict(reversed(list(bundles.items()))),
                    policy, sample_policy)
    assert a == b


def test_sample_cell_orders_queries_by_salted_hash(policy, sample_policy):
    bundles = _bundles(50, lambda i: ["source_identity"])
    out = sample_cell("cell", "harm", bundles, policy, sample_policy)
    salt = sample_policy["salt"]
    hashes = [order_hash(q, salt) for q in out["queries"]]
    assert hashes == sorted(hashes)


def test_sample_cell_selection_moves_with_the_salt(policy, sample_policy):
    """The salt is load-bearing: change it and a different subset is
    drawn. (Guards against a hash that silently ignores the salt.)"""
    bundles = _bundles(600, lambda i: ["source_identity"])
    a = sample_cell("cell", "harm", bundles, policy, sample_policy)
    other = dict(sample_policy, salt="a-different-salt")
    b = sample_cell("cell", "harm", bundles, policy, other)
    assert set(a["queries"]) != set(b["queries"])
    assert a["n_selected"] == b["n_selected"] == 256


def test_sample_cell_carries_the_registered_harm_category(policy, sample_policy):
    bundles = _bundles(10, lambda i: ["source_identity"])
    out = sample_cell("cell", "clean-control", bundles, policy, sample_policy)
    assert out["role"] == "clean-control"
    assert all(s["harm_category"] == "clean-control" for s in out["strata"])
