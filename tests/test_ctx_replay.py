"""Hermetic tests for the PR-13 sealed replay runner (memo §8).

SELECTION-TIMING KILL COMPLIANCE (memo §8.1): every fixture here is
synthetic and throwaway. No §7 cell material is read, and no real
consumer is ever constructed — the one `is_real` object in this file is a
tripwire whose ``generate`` raises if the runner ever reaches it.
"""

import hashlib
import json
import os

import pytest

from harness.ctx import replay
from harness.ctx.compile import load_policy
from harness.ctx.fake_consumer import FakeConsumer


@pytest.fixture(scope="module")
def policy():
    return load_policy()


# --- fixtures ---------------------------------------------------------------

def make_item(native_id, content="a fact.", state="agent-readable",
              signals=None, candidate_set=None):
    return {
        "item_id": f"test:{native_id}",
        "source_id": "test-v1",
        "content": content,
        "content_kind": "source-native",
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "event_time": None,
        "ingest_time": None,
        "evidence": [{"adapter_id": "test-v1", "signal": s, "value": True,
                      "evidence_ptr": f"ptr:{native_id}:{s}",
                      "tier": "core-certified" if s == "merge_suspect"
                              else "harness-heuristic"}
                     for s in (signals or ["source_identity"])],
        "relations": {"supersedes": [], "contradicts": [],
                      "candidate_set_id": candidate_set},
        "state": state,
        "policy_version": "1.0",
    }


def bundle(items, query_text="What is the value?", turn=1):
    return {"items": items, "not_retrieved": [], "query_text": query_text,
            "turn_state": {"turn_index": turn, "prior_rendered": {}}}


def make_manifest(queries):
    return {"manifest_id": "pr13-query-manifest-test", "version": "1.0",
            "queries": list(queries),
            "arm_plan": [list(a) for a in replay.ARM_PLAN],
            "budgets": list(replay.BUDGETS), "sealed": False}


def sealed(queries):
    return replay.seal_manifest(make_manifest(queries))


def simple_sources(n=2):
    return {f"q{i}": bundle([make_item(f"a{i}"), make_item(f"b{i}", "another fact.")])
            for i in range(n)}


class TripwireConsumer:
    """`is_real` with no weights: if the runner ever calls it, the seal
    gate leaked."""
    is_real = True
    pin_id = "tripwire"

    @staticmethod
    def count_tokens(text):
        return len(text.split())

    def generate(self, prompt, max_new_tokens=256):
        raise AssertionError("a real render was attempted")


class FlakyConsumer(FakeConsumer):
    def __init__(self, fail_after):
        super().__init__("wellformed")
        self.fail_after = fail_after

    def generate(self, prompt, max_new_tokens=256):
        if self.calls >= self.fail_after:
            raise RuntimeError("simulated consumer crash")
        return super().generate(prompt, max_new_tokens)


def good_scoring_manifest(qm):
    return replay.seal_manifest({
        "manifest_id": "pr13-scoring-manifest-test",
        "query_manifest_sha256": qm["manifest_sha256"],
        "consumer": {"replay_ready": True},
        "decoding": {"max_new_tokens": 256},
    })


# --- (2) hard prevention of a real render before sealing --------------------

def test_unsealed_manifest_blocks_a_real_render(tmp_path, policy):
    out = str(tmp_path / "rows.jsonl")
    tripwire = TripwireConsumer()
    with pytest.raises(replay.SealError):
        # sources={} on purpose: the seal gate must fire BEFORE source
        # validation, before any compile, and before any generation.
        replay.run(make_manifest(["q0"]), tripwire, {}, out, policy)
    assert not os.path.exists(out)


def test_unsealed_manifest_blocks_even_the_fake(tmp_path, policy):
    out = str(tmp_path / "rows.jsonl")
    with pytest.raises(replay.SealError):
        replay.run(make_manifest(["q0"]), FakeConsumer(), simple_sources(1),
                   out, policy)
    assert not os.path.exists(out)


def test_mutation_after_sealing_is_detected(tmp_path, policy):
    m = sealed(["q0", "q1"])
    m["queries"].append("q_smuggled")
    with pytest.raises(replay.SealError, match="mutated after sealing"):
        replay.verify_manifest(m)


def test_forged_seal_flag_does_not_verify():
    m = make_manifest(["q0"])
    m["sealed"] = True                       # claimed, but never digested
    m["manifest_sha256"] = "0" * 64
    with pytest.raises(replay.SealError):
        replay.verify_manifest(m)


def test_real_consumer_needs_a_scoring_manifest(tmp_path, policy):
    m = sealed(["q0"])
    with pytest.raises(replay.SealError, match="real consumer"):
        replay.run(m, TripwireConsumer(), {}, str(tmp_path / "r.jsonl"), policy)


def test_scoring_manifest_must_pin_this_query_manifest(tmp_path, policy):
    m = sealed(["q0"])
    other = good_scoring_manifest(sealed(["q_other"]))
    with pytest.raises(replay.SealError, match="different query manifest"):
        replay.run(m, TripwireConsumer(), {}, str(tmp_path / "r.jsonl"),
                   policy, scoring_manifest=other)


def test_scoring_manifest_must_report_replay_ready(tmp_path, policy):
    m = sealed(["q0"])
    sm = replay.seal_manifest({
        "query_manifest_sha256": m["manifest_sha256"],
        "consumer": {"replay_ready": False},
        "decoding": {"max_new_tokens": 256}})
    with pytest.raises(replay.SealError, match="replay_ready"):
        replay.run(m, TripwireConsumer(), {}, str(tmp_path / "r.jsonl"),
                   policy, scoring_manifest=sm)


def test_scoring_manifest_output_limit_must_match_the_contract(tmp_path, policy):
    m = sealed(["q0"])
    sm = replay.seal_manifest({
        "query_manifest_sha256": m["manifest_sha256"],
        "consumer": {"replay_ready": True},
        "decoding": {"max_new_tokens": 512}})
    with pytest.raises(replay.SealError, match="output limit"):
        replay.run(m, TripwireConsumer(), {}, str(tmp_path / "r.jsonl"),
                   policy, scoring_manifest=sm)


def test_the_fake_consumer_can_never_claim_to_be_real():
    fake = FakeConsumer()
    assert fake.is_real is False
    with pytest.raises(AttributeError):
        fake.is_real = True


# --- row expansion: identical identities in every arm -----------------------

def test_expansion_is_six_rows_per_query_over_identical_identities():
    m = sealed(["q0", "q1", "q2"])
    rows = replay.expand_rows(m)
    assert len(rows) == 18
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r["query_id"])
    assert set(by_arm) == {"governed", "raw_matched", "raw_native", "none"}
    assert sorted(set(by_arm["governed"])) == sorted(set(by_arm["raw_matched"]))
    assert sorted(set(by_arm["governed"])) == sorted(set(by_arm["raw_native"]))
    assert sorted(set(by_arm["governed"])) == sorted(set(by_arm["none"]))
    assert len(set(r["row_id"] for r in rows)) == 18


def test_expansion_rejects_arm_motion_against_a_sealed_manifest():
    m = sealed(["q0"])
    m["arm_plan"] = [["governed", 800], ["none", None]]
    m = replay.seal_manifest(m)          # re-sealed: the seal is not the guard
    with pytest.raises(replay.IntegrityError, match="arm motion"):
        replay.expand_rows(m)


def test_expansion_rejects_budget_grid_motion():
    m = sealed(["q0"])
    m["budgets"] = [800, 1500, 3000]
    m = replay.seal_manifest(m)
    with pytest.raises(replay.IntegrityError, match="budget-grid motion"):
        replay.expand_rows(m)


def test_prompt_shas_cover_both_sealed_templates():
    shas = replay.prompt_shas()
    assert set(shas) == {"single_turn_v1.txt", "none_arm_v1.txt"}
    assert all(len(v) == 64 for v in shas.values())


def test_expansion_rejects_out_of_order_session_turns():
    m = sealed(["mt:withdrawal:p#t1", "mt:withdrawal:p#t3"])
    with pytest.raises(replay.IntegrityError, match="out of turn order"):
        replay.expand_rows(m)


def test_run_rejects_a_manifest_query_with_no_source(tmp_path, policy):
    m = sealed(["q0", "q_absent"])
    with pytest.raises(replay.IntegrityError, match="no source"):
        replay.run(m, FakeConsumer(), simple_sources(1),
                   str(tmp_path / "r.jsonl"), policy)


# --- parsing, extra prose, malformed output ---------------------------------

def _records(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _run_fake(tmp_path, policy, mode, sources=None, name="rows.jsonl"):
    sources = sources or simple_sources(1)
    m = sealed(sorted(sources))
    out = str(tmp_path / name)
    summary = replay.run(m, FakeConsumer(mode), sources, out, policy)
    return summary, _records(out)


def test_wellformed_output_parses(tmp_path, policy):
    summary, recs = _run_fake(tmp_path, policy, "wellformed")
    assert summary["rows"] == 6
    assert all(r["status"] == "ok" for r in recs)
    assert all(r["extra_prose"] is False for r in recs)
    assert all(isinstance(r["hedged"], bool) for r in recs)


def test_extra_prose_is_flagged_not_failed(tmp_path, policy):
    _summary, recs = _run_fake(tmp_path, policy, "extra_prose")
    assert all(r["status"] == "ok" for r in recs)
    assert all(r["extra_prose"] is True for r in recs)


@pytest.mark.parametrize("mode,reason", [
    ("no_object", "no_balanced_json_object"),
    ("extra_field", "wrong_field_set"),
    ("empty_answer", "bad_answer"),
    ("bad_hedged", "bad_hedged"),
    ("cap_truncated", "no_balanced_json_object"),
])
def test_malformed_output_is_unparseable_with_the_registered_reason(
        tmp_path, policy, mode, reason):
    _summary, recs = _run_fake(tmp_path, policy, mode, name=f"{mode}.jsonl")
    assert all(r["status"] == "unparseable" for r in recs)
    assert all(r["reason"] == reason for r in recs)
    assert all(r["answer"] is None for r in recs)


def test_status_counts_are_reported(tmp_path, policy):
    summary, recs = _run_fake(tmp_path, policy, "mixed", simple_sources(8))
    assert sum(summary["status_counts"].values()) == len(recs) == 48
    assert set(summary["status_counts"]) <= {"ok", "unparseable"}


# --- token boundaries -------------------------------------------------------

def _long_items(n=3, words=400):
    return [make_item(f"L{i}", " ".join([f"w{i}"] * words) + ".")
            for i in range(n)]


def test_every_context_arm_stays_within_its_registered_budget(tmp_path, policy):
    sources = {"q0": bundle(_long_items())}
    _summary, recs = _run_fake(tmp_path, policy, "wellformed", sources)
    for r in recs:
        if r["arm"] == "none":
            assert r["block_sha256"] is None and r["block_tokens"] == 0
        else:
            assert r["block_tokens"] <= r["budget"]
    # B=800 cannot hold 1,200 words: the compiler must have acted.
    at_800 = [r for r in recs if r["arm"] == "governed" and r["budget"] == 800]
    at_1500 = [r for r in recs if r["arm"] == "governed" and r["budget"] == 1500]
    assert at_800[0]["rendered_item_ids"] != at_1500[0]["rendered_item_ids"] \
        or at_800[0]["block_tokens"] < at_1500[0]["block_tokens"]


def test_an_over_budget_block_raises_rather_than_truncating(
        tmp_path, policy, monkeypatch):
    monkeypatch.setattr(replay, "render_raw_native",
                        lambda items, budget, count_tokens: "x " * (budget + 5))
    m = sealed(["q0"])
    with pytest.raises(replay.IntegrityError, match="G-C3"):
        replay.run(m, FakeConsumer(), simple_sources(1),
                   str(tmp_path / "r.jsonl"), policy)


def test_raw_native_carries_items_governance_withholds(policy):
    items = [make_item("q1", "quarantined content.", state="quarantined"),
             make_item("ok", "clean content.")]
    block = replay.render_raw_native(items, 1500, lambda t: len(t.split()))
    assert "quarantined content." in block
    assert "clean content." in block
    assert "[governed context" not in block


def test_raw_native_truncates_by_token_count_in_native_order(policy):
    items = [make_item("a", "one two three four five."),
             make_item("b", "six seven eight nine ten.")]
    block = replay.render_raw_native(items, 5, lambda t: len(t.split()))
    assert block == "one two three four five."


# --- evidence-set equality (§8.2 arm-equivalence) ---------------------------

def test_governed_and_raw_matched_render_the_same_item_multiset(
        tmp_path, policy):
    sources = {"q0": bundle(_long_items())}
    summary, recs = _run_fake(tmp_path, policy, "wellformed", sources)
    assert summary["evidence_set_pairs_checked"] == 2      # one per budget
    for budget in replay.BUDGETS:
        g = next(r for r in recs if r["arm"] == "governed"
                 and r["budget"] == budget)
        rm = next(r for r in recs if r["arm"] == "raw_matched"
                  and r["budget"] == budget)
        assert g["rendered_item_ids"] == rm["rendered_item_ids"]
        assert g["block_sha256"] != rm["block_sha256"]      # structure differs


def test_evidence_set_mismatch_is_an_integrity_error(tmp_path, policy):
    out = str(tmp_path / "rows.jsonl")
    m = sealed(["q0"])
    replay.run(m, FakeConsumer(), simple_sources(1), out, policy)
    expected = [r["row_id"] for r in replay.expand_rows(m)]
    rendered = {("q0", "governed", 800): ["test:a0", "test:b0"],
                ("q0", "raw_matched", 800): ["test:a0"]}
    with pytest.raises(replay.IntegrityError, match="item multisets differ"):
        replay._reconcile(out, expected, m["manifest_sha256"], rendered, 6, 0)


# --- resume integrity, duplicate and omission detection ---------------------

def test_resume_completes_the_run_without_repeating_rows(tmp_path, policy):
    sources = simple_sources(2)
    m = sealed(sorted(sources))
    out = str(tmp_path / "rows.jsonl")

    with pytest.raises(RuntimeError, match="simulated consumer crash"):
        replay.run(m, FlakyConsumer(fail_after=5), sources, out, policy)
    partial = _records(out)
    assert len(partial) == 5

    summary = replay.run(m, FakeConsumer(), sources, out, policy)
    assert summary["resumed"] == 5
    assert summary["executed"] == 7
    assert summary["rows"] == 12
    ids = [r["row_id"] for r in _records(out)]
    assert len(ids) == len(set(ids)) == 12


def test_resume_refuses_a_log_written_under_another_manifest(tmp_path, policy):
    sources = simple_sources(1)
    m = sealed(sorted(sources))
    out = str(tmp_path / "rows.jsonl")
    replay.run(m, FakeConsumer(), sources, out, policy)

    other = sealed(["q0", "q1"])
    with pytest.raises(replay.IntegrityError, match="written under manifest"):
        replay.run(other, FakeConsumer(), simple_sources(2), out, policy)


def test_resume_refuses_a_duplicated_row(tmp_path, policy):
    sources = simple_sources(1)
    m = sealed(sorted(sources))
    out = str(tmp_path / "rows.jsonl")
    replay.run(m, FakeConsumer(), sources, out, policy)
    with open(out, encoding="utf-8") as fh:
        first = fh.readline()
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(first)
    with pytest.raises(replay.IntegrityError, match="duplicate row"):
        replay.run(m, FakeConsumer(), sources, out, policy)


def test_resume_refuses_a_foreign_row(tmp_path, policy):
    sources = simple_sources(1)
    m = sealed(sorted(sources))
    out = str(tmp_path / "rows.jsonl")
    replay.run(m, FakeConsumer(), sources, out, policy)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"row_id": "q_ghost|governed|B800",
                             "manifest_sha256": m["manifest_sha256"]}) + "\n")
    with pytest.raises(replay.IntegrityError, match="foreign row"):
        replay.run(m, FakeConsumer(), sources, out, policy)


def test_omission_is_detected_by_reconciliation(tmp_path, policy):
    sources = simple_sources(1)
    m = sealed(sorted(sources))
    out = str(tmp_path / "rows.jsonl")
    replay.run(m, FakeConsumer(), sources, out, policy)
    expected = [r["row_id"] for r in replay.expand_rows(m)] + ["q0|extra|B800"]
    with pytest.raises(replay.IntegrityError, match="never executed"):
        replay._reconcile(out, expected, m["manifest_sha256"], {}, 6, 0)


# --- multi-turn threading ---------------------------------------------------

def test_multiturn_turn_three_carries_the_withdrawal_notice(tmp_path, policy):
    served = make_item("X", "the original fact.")
    superseded = make_item("X", "the original fact.", state="superseded",
                           signals=["superseded_by"])
    fresh = make_item("Y", "the replacement fact.")
    sources = {
        "mt:withdrawal:p#t1": bundle([served], turn=1),
        "mt:withdrawal:p#t2": bundle([served], turn=2),
        "mt:withdrawal:p#t3": bundle([superseded, fresh], turn=3),
    }
    m = sealed(["mt:withdrawal:p#t1", "mt:withdrawal:p#t2",
                "mt:withdrawal:p#t3"])
    out = str(tmp_path / "mt.jsonl")
    summary = replay.run(m, FakeConsumer(), sources, out, policy)
    assert summary["rows"] == 18

    recs = {r["row_id"]: r for r in _records(out)}
    t3 = recs["mt:withdrawal:p#t3|governed|B800"]
    assert t3["session_id"] == "mt:withdrawal:p" and t3["turn"] == 3
    # The withdrawn item is a notice, not a rendered item.
    assert "test:X" not in t3["rendered_item_ids"]
    assert t3["hedged"] is True          # the fake reacts to WITHDRAWN:

    t1 = recs["mt:withdrawal:p#t1|governed|B800"]
    assert t1["rendered_item_ids"] == ["test:X"]


def test_none_arm_gets_no_context_block(tmp_path, policy):
    _summary, recs = _run_fake(tmp_path, policy, "wellformed")
    none_rows = [r for r in recs if r["arm"] == "none"]
    assert len(none_rows) == 1
    assert none_rows[0]["block_sha256"] is None
    assert none_rows[0]["rendered_item_ids"] is None
    assert none_rows[0]["budget"] is None


# --- determinism ------------------------------------------------------------

def test_two_runs_of_the_same_manifest_agree_byte_for_byte(tmp_path, policy):
    sources = simple_sources(3)
    m = sealed(sorted(sources))
    a, b = str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")
    replay.run(m, FakeConsumer("mixed"), sources, a, policy)
    replay.run(m, FakeConsumer("mixed"), sources, b, policy)
    assert open(a, encoding="utf-8").read() == open(b, encoding="utf-8").read()
