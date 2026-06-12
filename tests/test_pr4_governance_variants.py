"""tests/test_pr4_governance_variants.py — hermetic gates for PR-4 step 3.

PR4_DESIGN.md §8 step 3: the two pre-registered guarded variants
(`trust-downweight`, `trust-guarded`) and the §3 collateral decomposition,
pinned BEFORE any PR-4 cache run on the tiny synthetic cache from
test_failure_mode_vision (CPU-only, no GPU, no network):

  1. the memo-frozen parameters are the ones in code (λ=0.25, θ=0.8) and
     the policy set contains exactly the gate-1 survivors (trust-record
     dropped at Addendum A.4 must NOT appear);
  2. observe-only stays metric-identical to none and the policy-`none`
     fidelity gate still runs (score_run raises on violation — these runs
     pin that it does not);
  3. the decomposition is exhaustive: direct_br + collateral_br == broken
     for every policy on every arm scored here;
  4. variant behavior on the routed supersession path: downweighting at
     λ=0.25 still flips the deprecated side (fixes all stale-wrong rows,
     breaks none), and the guard resolves to exclusion when the fork
     slots' traffic is wholly fork-party (synthetic geometry);
  5. inherited routes are intact: merge-suspect -> abstain on the soft
     arm, one-shot ambiguity -> observe-only (no intervention, no fiat
     election) on the one-shot arm;
  6. the guard proxy is label-free: GuardIndex is built from topk + pair
     membership only, and its fractions are well-formed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.analyze_fork_governance import (  # noqa: E402
    POLICIES, TRUST_DOWNWEIGHT_LAMBDA, TRUST_GUARD_THETA, TRUST_ROUTED,
    GuardIndex, _vote, apply_policy, build_writetime_router, load_run,
    pair_counterparts, pair_party_labels)
from tests.test_pr3c_shadow import _run, _score  # noqa: E402

VARIANTS = ("trust-downweight", "trust-guarded")


def test_memo_frozen_parameters_and_policy_set():
    assert TRUST_DOWNWEIGHT_LAMBDA == 0.25
    assert TRUST_GUARD_THETA == 0.8
    for v in VARIANTS:
        assert v in POLICIES and v in TRUST_ROUTED
    assert "mode-conditioned-trust" in TRUST_ROUTED
    assert "trust-record" not in POLICIES  # dropped at gate 1 (A.4)


def test_observe_only_still_identical_to_none_and_decomposition_sums(
        tmp_path):
    for arm, name, kw in (("mixed", "m.csv", {}),
                          ("stale", "s.csv", {})):
        _, table, _ = _score(arm, tmp_path, name, **kw)
        n, obs = table["none"], table["observe-only"]
        assert obs == {**n}, "observe-only must be metric-identical to none"
        for policy in POLICIES:
            m = table[policy]
            assert m["direct_br"] + m["collateral_br"] == m["broken"], \
                (arm, policy, m["direct_br"], m["collateral_br"],
                 m["broken"])
        assert n["broken"] == n["direct_br"] == n["collateral_br"] == 0


def _mk_cands(*rows):
    """rows: (slot, decode, weight). Builds a surviving candidate list in
    rank order with descending placeholder sims."""
    return [{"rank": r, "slot": s, "sim": 0.9 - 0.01 * r, "surviving": True,
             "weight": str(w), "decode": d}
            for r, (s, d, w) in enumerate(rows)]


def _mk_router(I=1, O=2, old_side=0, new_side=1, verdict="supersession"):
    counts = {"supersession": (0, 1), "contradiction": (1, 0),
              "ambiguous": (0, 0)}[verdict]
    return {"pairs": [{"I": I, "O": O, "epoch": 0,
                       "old_side": old_side, "new_side": new_side,
                       "verdict_by_epoch": {0: {
                           "verdict": verdict,
                           "old_n": counts[0], "new_n": counts[1]}}}],
            "merge": []}


def _apply(policy, cands, router, guard=None):
    cp = pair_counterparts(router)
    none_ans, _ = _vote(cands, 2)
    return apply_policy(policy, cands, [], none_ans, 2, 0, {}, True,
                        router, guard=guard, counterpart=cp, probe_index=0)


def test_downweight_attenuates_instead_of_excluding():
    """Unit pins of the λ semantics on a routed supersession (deprecate the
    old side, slot 1). Flip case (0.5/0.5): 0.5·0.25 = 0.125 < 0.5, so
    attenuation flips like exclusion. Separation case (0.8/0.2): exclusion
    flips the vote, attenuation leaves a 0.2/0.2 tie that argmax-first
    resolves to the old side — the two actions are measurably distinct."""
    router = _mk_router()
    flip = _mk_cands((1, 0, 0.5), (2, 1, 0.5))
    for policy in ("mode-conditioned-trust",) + VARIANTS:
        guard = GuardIndex({(0, 0): flip}, pair_counterparts(router))
        ans, acted, detail = _apply(policy, flip, router, guard)
        assert (ans, acted) == (1, True), policy
        assert detail["dropped"] == frozenset({1}), policy

    skew = _mk_cands((1, 0, 0.8), (2, 1, 0.2))
    guard = GuardIndex({(0, 0): skew}, pair_counterparts(router))
    assert _apply("mode-conditioned-trust", skew, router, guard)[0] == 1
    assert _apply("trust-downweight", skew, router, guard)[0] == 0


def test_guard_routes_by_fork_party_fraction():
    """θ semantics: a deprecated slot whose appearances are all fork-party
    (counterpart co-surviving) is EXCLUDED; one whose fork-party fraction
    sits below θ=0.8 is attenuated instead. Built on the 0.8/0.2 skew
    where exclusion and attenuation elect different answers, so the
    routing choice is visible in the vote."""
    router = _mk_router()
    skew = _mk_cands((1, 0, 0.8), (2, 1, 0.2))
    # all-fork-party history: every appearance of slot 1 has slot 2 too
    g_hi = GuardIndex({(0, 0): skew}, pair_counterparts(router))
    assert g_hi.fork_party_fraction(1, 0, 0) == 1.0
    assert _apply("trust-guarded", skew, router, g_hi)[0] == 1  # excluded
    # slot 1 mostly appears WITHOUT its counterpart (shared support):
    # 1 fork-party appearance out of 3 -> fraction 1/3 < 0.8 -> attenuate
    solo = _mk_cands((1, 0, 1.0))
    g_lo = GuardIndex({(0, -2): solo, (0, -1): solo, (0, 0): skew},
                      pair_counterparts(router))
    assert abs(g_lo.fork_party_fraction(1, 0, 0) - 1 / 3) < 1e-9
    ans, acted, detail = _apply("trust-guarded", skew, router, g_lo)
    assert (ans, acted) == (0, True)  # attenuated: 0.2/0.2 tie keeps old
    assert detail["dropped"] == frozenset({1})


def test_guard_detects_shared_support_on_synthetic_run(tmp_path):
    """Integration: on the repeated-supersession run the deprecated fork
    slots appear in top-k for OTHER keys' probes too (shared support — the
    §3 collateral mechanism), so the guard proxy sits strictly below 1 and
    the exposure block reports it; outcome metrics still match trust here
    because attenuation and exclusion elect the same answers. Engine-fact
    pin: the routed trust family breaks nothing on this arm."""
    _, table, _ = _score("stale", tmp_path, "g.csv", epochs=7,
                         supersede_epoch=3)
    assert table["trust-guarded"] == table["mode-conditioned-trust"]
    for policy in ("mode-conditioned-trust",) + VARIANTS:
        assert table[policy]["broken"] == 0, policy
    exp = table["trust-guarded"]["collateral_exposure"]
    assert exp["n_slots"] > 0
    assert any(r["proxy_fork_party_fraction"] < 1.0
               for r in exp["per_slot"])
    for row in exp["per_slot"]:
        assert 0.0 <= row["proxy_fork_party_fraction"] <= 1.0
        assert 0.0 <= row["registry_exposure"] <= 1.0


def test_variants_inherit_merge_suspect_abstention(tmp_path):
    """Required benchmark (§1): merge-path stale must stay captured by
    write-time abstention — a variant that loses it fails outright."""
    _, table, _ = _score("stale", tmp_path, "soft.csv", epochs=5,
                         supersede_epoch=3, payload_mode="soft")
    wrong = table["none"]["wrong_none"]
    assert wrong > 0
    for policy in ("mode-conditioned-trust",) + VARIANTS:
        m = table[policy]
        assert m["stale_wrong_abstained"] == wrong, policy
        assert m["broken"] == 0, policy


def test_variants_inherit_oneshot_observe_only(tmp_path):
    """One-shot ambiguity is an observe-only outcome (§1): no variant may
    intervene on (or fiat-elect a side of) a protocol-certified-ambiguous
    pair. The router yields only `ambiguous` here, so acted == 0."""
    _, table, _ = _score("stale", tmp_path, "one.csv", epochs=7,
                         supersede_epoch=3, one_shot=True)
    for policy in ("mode-conditioned-trust",) + VARIANTS:
        m = table[policy]
        assert m["acted"] == 0, policy
        assert m["abstained"] == 0, policy
        assert m["answered"] == m["n"], policy


def test_guard_index_is_label_free_and_well_formed(tmp_path):
    """GuardIndex consumes only topk + router pair membership — no probe
    labels, no registry roles — and its fractions are causal prefixes."""
    _, _, out = _run("stale", tmp_path, name="gi.csv", epochs=7,
                     supersede_epoch=3)
    run = load_run(out)
    router = build_writetime_router(run["events"], run["slot_obs"])
    cp = pair_counterparts(router)
    guard = GuardIndex(run["topk"], cp)  # builds with no label inputs
    party = pair_party_labels(router)
    assert set(cp) == set(party)
    saw = 0
    for s in cp:
        apps = guard.appearances(s)
        for key in apps:
            frac = guard.fork_party_fraction(s, *key)
            assert frac is not None and 0.0 <= frac <= 1.0
            saw += 1
        # before the first appearance the proxy is undefined, not 0
        if apps:
            e0, p0 = apps[0]
            assert guard.fork_party_fraction(s, e0 - 1, p0) is None
    assert saw > 0


def test_scoring_remains_deterministic_with_variants(tmp_path):
    _, t1, _ = _score("mixed", tmp_path, "d1.csv")
    _, t2, _ = _score("mixed", tmp_path, "d2.csv")
    assert t1 == t2


def test_pr4_geometry_criteria_unit_pins():
    """§6 evaluator pins (analyze_pr4_geometry): the collateral bar is 1%
    of none's CORRECT rows, and domination charges abstention-on-correct
    as harm (else abstain-only baselines are unbeatable by construction)."""
    from benchmarks.analyze_pr4_geometry import bar_ok, dominates
    none = {"n": 2500, "wrong_none": 500}
    assert bar_ok({"broken": 20}, none)        # 20 <= 1% of 2000
    assert not bar_ok({"broken": 21}, none)
    cand = {"fixed": 100, "contra_wrong_fixed": 90, "broken": 5,
            "abstain_on_correct": 0}
    abstainer = {"fixed": 0, "contra_wrong_fixed": 0, "broken": 0,
                 "abstain_on_correct": 500}
    assert dominates(cand, abstainer, "mixed")
    assert dominates(cand, abstainer, "contra")
    equal = dict(cand)
    assert not dominates(cand, equal, "mixed")  # ties are not strict
