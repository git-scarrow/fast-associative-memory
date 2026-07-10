"""Hermetic tests for the PR-13 governed context compiler (memo §9,
integrity gates G-C1/G-C3/G-C4 in their build-checkpoint form).

SELECTION-TIMING KILL COMPLIANCE (memo §8.1): every fixture here is
synthetic. No §7 cell material — no FAM run artifact, no Shutter-Deck
row — is compiled by this suite; hermetic-test renders over synthetic
fixtures are explicitly outside the "first evaluation render".

Token counting uses a whitespace counter; the compiler takes the
tokenizer as an injected callable precisely so these tests are
independent of the consumer pin.
"""

import hashlib
import json

import pytest

from harness.ctx.compile import (
    PRECEDENCE,
    compile as ctx_compile,
    load_policy,
    render_raw_matched,
    summarize,
)


def wtok(text):
    """Deterministic whitespace token counter for hermetic tests."""
    return len(text.split())


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make_item(native_id, content, state="agent-readable", signals=None,
              candidate_set=None, event_time=None, tier="harness-heuristic"):
    evidence = [
        {"adapter_id": "test-v1", "signal": s, "value": True,
         "evidence_ptr": f"ptr:{native_id}:{s}", "tier": tier}
        for s in (signals or ["source_identity"])
    ]
    return {
        "item_id": f"test:{native_id}",
        "source_id": "test-v1",
        "content": content,
        "content_kind": "source-native",
        "content_sha256": sha(content),
        "event_time": event_time,
        "ingest_time": None,
        "evidence": evidence,
        "relations": {"supersedes": [], "contradicts": [],
                      "candidate_set_id": candidate_set},
        "state": state,
        "policy_version": "1.0",
    }


@pytest.fixture(scope="module")
def policy():
    return load_policy()


@pytest.fixture()
def basic_items():
    return [
        make_item("a1", "The deploy runs at nine. It uses the blue cluster."),
        make_item("a2", "The database was migrated last spring to the new host.",
                  signals=["stale_support"]),
        make_item("a3", "Old config value is twelve.", state="quarantined"),
        make_item("a4", "Escalation-only note about the ledger.",
                  state="human-review"),
    ]


def compile_once(items, policy, budget=200, turn_state=None, **kw):
    return ctx_compile(items, policy, budget, turn_state, wtok, **kw)


# --- G-C1: byte-determinism ---------------------------------------------

def test_double_run_byte_identity(policy, basic_items):
    b1, p1 = compile_once(basic_items, policy)
    b2, p2 = compile_once(list(reversed(basic_items)), policy)
    assert b1 == b2
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


# --- G-C3: budget integrity, no silent drops -----------------------------

def test_block_within_budget_and_all_candidates_audited(policy, basic_items):
    budget = 25
    block, packet = compile_once(basic_items, policy, budget=budget)
    assert wtok(block) <= budget
    audited = {row["item_id"] for row in packet["rows"]}
    assert audited == {i["item_id"] for i in basic_items}


def test_budget_pressure_summarizes_then_withholds(policy):
    items = [make_item(f"b{i}", "Sentence one is long enough here. "
                                "Sentence two adds much more detail text.")
             for i in range(8)]
    block, packet = compile_once(items, policy, budget=40)
    decisions = {row["budget_decision"] for row in packet["rows"]}
    assert "budget_withheld" in decisions or "summarized" in decisions
    assert wtok(block) <= 40
    # header must state the budget-withheld count (I6 / §6 no-silent-truncation)
    n_bw = sum(1 for row in packet["rows"]
               if row["budget_decision"] == "budget_withheld")
    assert f"budget-withheld:{n_bw}" in block.splitlines()[0]


def test_budget_smaller_than_header_fails_loudly(policy, basic_items):
    with pytest.raises(ValueError):
        compile_once(basic_items, policy, budget=2)


# --- G-C4: invariant suite (structural forms) -----------------------------

def test_i4_audit_count_includes_not_retrieved(policy, basic_items):
    nr = [{"native_id": "test:gone1", "reason": "index_miss"}]
    _, packet = compile_once(basic_items, policy, not_retrieved=nr)
    assert len(packet["rows"]) == len(basic_items) + 1
    nr_rows = [r for r in packet["rows"]
               if r["budget_decision"] == "not_retrieved"]
    assert len(nr_rows) == 1
    assert nr_rows[0]["reason_code"] == "not_retrieved:index_miss"


def test_i6_header_always_first_line(policy, basic_items):
    block, _ = compile_once(basic_items, policy)
    header = block.splitlines()[0]
    for field in ("shown:", "caveated:", "unresolved:", "withheld:",
                  "budget-withheld:"):
        assert field in header


# --- §3 semantics ---------------------------------------------------------

def test_assert_only_without_adverse_evidence(policy, basic_items):
    _, packet = compile_once(basic_items, policy)
    rows = {r["item_id"]: r for r in packet["rows"]}
    assert rows["test:a1"]["disposition"] == "assert"
    assert rows["test:a2"]["disposition"] == "caveat"
    assert rows["test:a3"]["disposition"] == "withhold"
    assert rows["test:a4"]["disposition"] == "defer"


def test_caveat_renders_registered_template(policy, basic_items):
    block, _ = compile_once(basic_items, policy)
    assert "[caveat: possibly stale: merged support]" in block


def test_withheld_content_absent_from_block(policy, basic_items):
    block, _ = compile_once(basic_items, policy)
    assert "Old config value" not in block
    assert "Escalation-only note" not in block  # defer renders a notice, not content


def test_precedence_withhold_beats_caveat(policy):
    item = make_item("p1", "Contested content.", state="quarantined",
                     signals=["stale_support"])
    _, packet = compile_once([item], policy)
    assert packet["rows"][0]["disposition"] == "withhold"
    assert packet["rows"][0]["reason_code"] == "quarantined"


def test_unmatched_evidence_fails_closed_to_defer(policy):
    item = make_item("u1", "Mystery signal content.", signals=["never_registered"])
    _, packet = compile_once([item], policy)
    assert packet["rows"][0]["disposition"] == "defer"
    assert packet["rows"][0]["reason_code"] == "unmatched_evidence_fail_closed"
    assert packet["anomalies"][0]["kind"] == "unmatched_evidence_fail_closed"


def test_dual_present_without_candidate_set_downgrades(policy):
    item = make_item("d0", "Tie with nobody.", signals=["oneshot_tie"])
    _, packet = compile_once([item], policy)
    assert packet["rows"][0]["disposition"] == "defer"
    assert any(a["kind"] == "dual_present_no_candidate_set"
               for a in packet["anomalies"])


def test_dual_present_renders_unasserted_candidates(policy):
    items = [
        make_item("c1", "Value is red.", signals=["witness_alt_candidate_set"],
                  candidate_set="cs1"),
        make_item("c2", "Value is blue.", signals=["witness_alt_candidate_set"],
                  candidate_set="cs1"),
    ]
    block, packet = compile_once(items, policy)
    assert "unresolved (2 candidates — neither asserted)" in block
    assert "(a) Value is blue." in block or "(a) Value is red." in block
    assert all(r["disposition"] == "dual_present" for r in packet["rows"])


def test_withdrawal_notice_on_multiturn(policy):
    item = make_item("w1", "Turn-one fact.", state="superseded",
                     signals=["superseded_by"])
    # By turn 3, adverse evidence has escalated the item to quarantined.
    item_now = dict(item, state="quarantined")
    turn_state = {"turn_index": 3, "prior_rendered": {"test:w1": 1}}
    block, packet = compile_once([item_now], policy, turn_state=turn_state)
    assert "WITHDRAWN: the item served at turn 1 (test:w1) is withdrawn" in block
    assert packet["rows"][0]["disposition"] == "withdraw"
    assert packet["rows"][0]["reason_code"].startswith("withdrawn:")


def test_abstention_passthrough_renders_before_items(policy):
    items = [
        make_item("m1", "Merge-suspect row.", signals=["merge_suspect"],
                  tier="core-certified"),
        make_item("m2", "Clean fact."),
    ]
    block, _ = compile_once(items, policy)
    lines = block.splitlines()
    abstain_line = next(i for i, l in enumerate(lines) if l.startswith("ABSTAIN"))
    first_item_line = next(i for i, l in enumerate(lines) if l.startswith("- "))
    assert abstain_line < first_item_line


def test_summarizer_is_deterministic_prefix(policy):
    text = "First sentence here. Second sentence follows. Third one."
    s1 = summarize(text, policy, wtok)
    s2 = summarize(text, policy, wtok)
    assert s1 == s2 == "First sentence here."


# --- §8.2 raw-matched arm --------------------------------------------------

def test_raw_matched_same_multiset_no_structure(policy, basic_items):
    budget = 200
    block, packet = compile_once(basic_items, policy, budget=budget)
    raw = render_raw_matched(basic_items, packet, budget, policy, wtok)
    # identical rendered multiset: a1 and a2 present, withheld/deferred absent
    assert "The deploy runs at nine" in raw
    assert "The database was migrated" in raw
    assert "Old config value" not in raw
    assert "Escalation-only note" not in raw
    # structure stripped
    assert "[governed context" not in raw
    assert "caveat" not in raw
    assert "unresolved" not in raw
    assert "WITHDRAWN" not in raw
    assert wtok(raw) <= budget


def test_raw_matched_dual_members_plain(policy):
    items = [
        make_item("c1", "Value is red.", signals=["witness_alt_candidate_set"],
                  candidate_set="cs1"),
        make_item("c2", "Value is blue.", signals=["witness_alt_candidate_set"],
                  candidate_set="cs1"),
    ]
    _, packet = compile_once(items, policy)
    raw = render_raw_matched(items, packet, 200, policy, wtok)
    assert "Value is red." in raw and "Value is blue." in raw
    assert "candidates" not in raw and "neither asserted" not in raw


# --- policy artifact sanity -------------------------------------------------

def test_policy_precedence_matches_memo():
    policy = load_policy()
    assert policy["precedence"] == PRECEDENCE


def test_policy_caveat_reasons_all_templated():
    policy = load_policy()
    for rule in policy["rules"]:
        if rule["disposition"] == "caveat":
            assert rule["reason_code"] in policy["caveat_templates"]


def test_schemas_parse_and_core_certified_enum_guard():
    import os
    schema_dir = os.path.join(os.path.dirname(__file__), "..", "harness",
                              "ctx", "schema")
    names = ["context_item", "adapter_output", "disposition_policy",
             "consumer_output"]
    for name in names:
        with open(os.path.join(schema_dir, f"{name}.schema.json")) as fh:
            json.load(fh)
    with open(os.path.join(schema_dir, "context_item.schema.json")) as fh:
        ci = json.load(fh)
    tiers = ci["$defs"]["evidence_record"]["properties"]["tier"]["enum"]
    assert tiers == ["core-certified", "harness-heuristic", "source-asserted"]
