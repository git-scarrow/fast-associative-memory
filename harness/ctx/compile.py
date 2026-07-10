#!/usr/bin/env python3
"""PR-13 governed context compiler — build checkpoint (memo §5).

Deterministic pure function over schema-valid context items: no wall
clock, no RNG, no model calls, no filesystem access inside ``compile``
(policy artifacts are loaded by the caller via ``load_policy``). Same
inputs → byte-identical ``context_block`` and audit packet (gate G-C1).

Normative artifacts (on divergence they win over this docstring):
    harness/ctx/schema/context_item.schema.json       (memo §2)
    harness/ctx/schema/disposition_policy.schema.json (memo §3)
    harness/ctx/policy/disposition_policy_v1.json     (memo §3, frozen v1)

Label-blindness (G-C2): no function in this module accepts, reads, or
references truth columns. The poisoned-label canary is the normative
enforcement; any import audit is advisory and function-granular.

Token counting is injected as ``count_tokens`` so hermetic tests run
without the pinned consumer tokenizer; at scoring time the caller must
inject the tokenizer pinned in harness/ctx/policy/consumer_pin.json
(memo §6/§8.1). ``budget_cost`` lives in the audit row, never on the
item (memo §2, r2 blocking edit 2).
"""

import json
import os
from datetime import datetime

# Fixed by memo §3; the loaded policy's precedence list must equal this.
PRECEDENCE = ["withdraw", "withhold", "defer", "dual_present", "caveat", "summarize", "assert"]
_PREC_RANK = {d: i for i, d in enumerate(PRECEDENCE)}

TIER_RANK = {"core-certified": 0, "harness-heuristic": 1, "source-asserted": 2}

# Dispositions that count as adverse for the §6 priority key and for the
# assert-only-when-clean rule (enforced structurally by precedence).
ADVERSE = {"withdraw", "withhold", "defer", "dual_present", "caveat"}

_POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy")

# Recency sentinel: items with no usable timestamp sort after all dated
# items (memo §2: null is legal and rendered as unknown, never defaulted).
_NO_TIME = float("inf")


def load_policy(path=None):
    """Load and structurally check a disposition policy instance."""
    if path is None:
        path = os.path.join(_POLICY_DIR, "disposition_policy_v1.json")
    with open(path, "r", encoding="utf-8") as fh:
        policy = json.load(fh)
    if policy["precedence"] != PRECEDENCE:
        raise ValueError("policy precedence diverges from memo §3 order")
    for rule in policy["rules"]:
        if rule["disposition"] not in ("assert", "caveat", "dual_present", "withhold", "defer"):
            raise ValueError(f"rule {rule['rule_id']}: table rules may not output "
                             f"{rule['disposition']} (summarize/withdraw are compiler mechanisms)")
        if rule["disposition"] == "caveat" and rule["reason_code"] not in policy["caveat_templates"]:
            raise ValueError(f"rule {rule['rule_id']}: caveat reason_code "
                             f"{rule['reason_code']!r} has no registered template")
    return policy


def _rule_matches(rule, state, ev):
    m = rule["match"]
    if "state" in m and m["state"] != state:
        return False
    if "signal" in m and (ev is None or m["signal"] != ev["signal"]):
        return False
    if "tier" in m and (ev is None or m["tier"] != ev["tier"]):
        return False
    return True


def _resolve_item(item, policy):
    """Map one item's (state, evidence records) through the rule table.

    Split evaluation semantics (fail-closed on both dimensions):
      * the STATE is mapped once, through state-only rules (match has
        only a ``state`` key); an unmapped state fail-closes to defer;
      * each EVIDENCE RECORD is mapped by its first matching
        signal-bearing rule (match has a ``signal`` key, optionally
        state/tier constraints); an unmatched record fail-closes to
        defer with an anomaly.
    State-only rules never shadow evidence records — otherwise a benign
    state would silently assert items carrying unknown (possibly
    adverse) signals, defeating the memo §3 fail-closed rule.

    Cross-contribution conflicts resolve by precedence; because assert
    has lowest precedence, "assert only when no adverse evidence record
    exists" holds structurally.
    """
    state_rules = [r for r in policy["rules"] if "signal" not in r["match"]]
    signal_rules = [r for r in policy["rules"] if "signal" in r["match"]]
    contributions = []
    anomalies = []

    first_ptr = item["evidence"][0]["evidence_ptr"]
    first_tier = item["evidence"][0]["tier"]
    state_rule = next((r for r in state_rules
                       if r["match"]["state"] == item["state"]), None)
    if state_rule is None:
        anomalies.append({
            "kind": "unmatched_state_fail_closed",
            "item_id": item["item_id"],
            "state": item["state"],
        })
        contributions.append(("defer", "unmatched_state_fail_closed",
                              first_ptr, first_tier))
    else:
        contributions.append((state_rule["disposition"], state_rule["reason_code"],
                              first_ptr, first_tier))

    for ev in item["evidence"]:
        matched = None
        for rule in signal_rules:
            if _rule_matches(rule, item["state"], ev):
                matched = rule
                break
        if matched is None:
            anomalies.append({
                "kind": "unmatched_evidence_fail_closed",
                "item_id": item["item_id"],
                "signal": ev["signal"],
                "state": item["state"],
            })
            contributions.append(("defer", "unmatched_evidence_fail_closed",
                                  ev["evidence_ptr"], ev["tier"]))
        else:
            contributions.append((matched["disposition"], matched["reason_code"],
                                  ev["evidence_ptr"], ev["tier"]))
    # Highest precedence wins; first contribution in evidence order breaks
    # ties deterministically.
    winner = min(contributions, key=lambda c: _PREC_RANK[c[0]])
    disposition, reason_code, evidence_ptr, tier = winner

    # dual_present requires a candidate set; without one it is
    # unrenderable and fail-closes to defer with an anomaly (memo §3).
    if disposition == "dual_present" and not item["relations"].get("candidate_set_id"):
        anomalies.append({
            "kind": "dual_present_no_candidate_set",
            "item_id": item["item_id"],
        })
        disposition, reason_code = "defer", "dual_present_no_candidate_set"
    return disposition, reason_code, evidence_ptr, tier, anomalies


def _recency_key(item):
    """Newer first; null timestamps last (never defaulted)."""
    best = _NO_TIME
    for field in ("event_time", "ingest_time"):
        value = item.get(field)
        if value:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            best = min(best, -ts)
    return best


def _priority_key(item, adverse):
    """Memo §6 registered priority key: (tier rank, adverse-evidence-free
    first, recency, item_id). Lower sorts first = higher priority."""
    tier = min(TIER_RANK[ev["tier"]] for ev in item["evidence"])
    return (tier, 1 if adverse else 0, _recency_key(item), item["item_id"])


def _first_k_sentences(text, k):
    out, count, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in ".!?":
            out.append(text[start:i + 1].strip())
            count += 1
            start = i + 1
            if count >= k:
                break
    if count < k and start < len(text):
        out.append(text[start:].strip())
    return " ".join(s for s in out if s)


def _truncate_to_tokens(text, cap, count_tokens):
    """Deterministic word-boundary truncation to ≤ cap tokens."""
    words = text.split(" ")
    while words and count_tokens(" ".join(words)) > cap:
        words.pop()
    return " ".join(words)


def summarize(content, policy, count_tokens):
    """Memo §3: deterministic extractive rule — first k-sentence prefix
    under the token cap; never a model."""
    cfg = policy["summarizer"]
    prefix = _first_k_sentences(content, cfg["k_sentences"])
    return _truncate_to_tokens(prefix, cfg["token_cap"], count_tokens)


def _caveat_line(content, reason_code, policy):
    return f"{content} [caveat: {policy['caveat_templates'][reason_code]}]"


def _dual_line(members):
    letters = "abcdefghijklmnopqrstuvwxyz"
    parts = [f"({letters[i]}) {m['content']}" for i, m in enumerate(members)]
    return (f"unresolved ({len(members)} candidates — neither asserted): "
            + " ".join(parts))


def compile(items, policy, budget, turn_state, count_tokens,
            query_id="q0", not_retrieved=(), authorization="pr13-replay"):
    """Memo §5: compile(items, policy, budget, turn_state) →
    (context_block, audit_packet).

    ``count_tokens``, ``query_id``, ``not_retrieved`` and
    ``authorization`` elaborate the memo signature: the tokenizer is
    injected (pinned at scoring time per consumer_pin.json), and the
    audit fields the memo requires per row are threaded through.
    Deterministic: no clock, no RNG, no I/O, no model.
    """
    turn_state = turn_state or {"turn_index": 1, "prior_rendered": {}}
    prior_rendered = turn_state.get("prior_rendered", {})

    # --- 1. dispositions -------------------------------------------------
    resolved = {}          # item_id -> dict
    anomalies = []
    for item in sorted(items, key=lambda x: x["item_id"]):
        if item["item_id"] in resolved:
            raise ValueError(f"duplicate item_id {item['item_id']}")
        disposition, reason, ptr, tier, item_anoms = _resolve_item(item, policy)
        anomalies.extend(item_anoms)
        resolved[item["item_id"]] = {
            "item": item, "disposition": disposition, "reason_code": reason,
            "evidence_ptr": ptr, "tier": tier,
        }

    # --- 2. withdrawal notices (memo §3: record only grows) --------------
    # Trigger: an item rendered at an earlier turn now resolves to
    # withhold/defer, OR its state has moved to superseded/quarantined —
    # both are "adverse evidence arriving between turns" (§3). The state
    # clause is what lets ledger supersession (which resolves at caveat
    # level in single-turn cells) revoke a previously served item.
    withdrawals = []
    for iid, r in sorted(resolved.items()):
        if iid in prior_rendered and (
                r["disposition"] in ("withhold", "defer")
                or r["item"]["state"] in ("superseded", "quarantined")):
            withdrawals.append({
                "item_id": iid,
                "prior_turn": prior_rendered[iid],
                "underlying_reason": r["reason_code"],
            })
            r["reason_code"] = "withdrawn:" + r["reason_code"]
            r["disposition"] = "withdraw"

    # --- 3. abstention passthrough (renders before all dispositions) -----
    abstain_notices = []
    for iid, r in sorted(resolved.items()):
        for ev in r["item"]["evidence"]:
            if ev["signal"] == "merge_suspect" and ev["tier"] == "core-certified":
                abstain_notices.append(
                    f"ABSTAIN (merge-suspect, PR-10 passthrough): {ev['evidence_ptr']}")

    # --- 4. renderable units in §6 priority order -------------------------
    # A unit is a single assert/caveat item or a whole dual_present
    # candidate set (one rendered line covering its members).
    units = []
    seen_sets = {}
    for iid, r in sorted(resolved.items()):
        adverse = r["disposition"] in ADVERSE
        if r["disposition"] in ("assert", "caveat"):
            units.append({"kind": "single", "members": [r],
                          "key": _priority_key(r["item"], adverse)})
        elif r["disposition"] == "dual_present":
            set_id = r["item"]["relations"]["candidate_set_id"]
            if set_id in seen_sets:
                group = seen_sets[set_id]
                group["members"].append(r)
                group["key"] = min(group["key"], _priority_key(r["item"], adverse))
            else:
                group = {"kind": "dual", "set_id": set_id, "members": [r],
                         "key": _priority_key(r["item"], adverse)}
                seen_sets[set_id] = group
                units.append(group)
    for group in seen_sets.values():
        group["members"].sort(key=lambda m: m["item"]["item_id"])
    units.sort(key=lambda u: u["key"])

    # --- 5. budget loop: summarize lowest-priority summarizable, then ----
    # --- budget-withhold, until the whole block fits (memo §6) -----------
    def unit_line(unit):
        if unit["kind"] == "dual":
            return _dual_line([m["item"] for m in unit["members"]])
        member = unit["members"][0]
        if member.get("summarized"):
            body = summarize(member["item"]["content"], policy, count_tokens) + " (condensed)"
        else:
            body = member["item"]["content"]
        if member["disposition"] == "caveat" and not member.get("summarized"):
            return _caveat_line(body, member["reason_code"], policy)
        return body

    def render_block(active_units):
        defer_count = sum(1 for r in resolved.values() if r["disposition"] == "defer")
        caveated = sum(1 for u in active_units for m in u["members"]
                       if m["disposition"] == "caveat")
        shown = sum(len(u["members"]) for u in active_units)
        budget_withheld = sum(1 for r in resolved.values()
                              if r.get("budget_decision") == "budget_withheld")
        withheld = sum(1 for r in resolved.values()
                       if r["disposition"] == "withhold"
                       and r.get("budget_decision") != "budget_withheld")
        dual_sets_active = sum(1 for u in active_units if u["kind"] == "dual")
        lines = [(f"[governed context | shown:{shown} caveated:{caveated} "
                  f"unresolved:{defer_count + dual_sets_active} withheld:{withheld} "
                  f"budget-withheld:{budget_withheld} | policy:{policy['version']}]")]
        lines.extend(abstain_notices)
        for w in withdrawals:
            lines.append(f"WITHDRAWN: the item served at turn {w['prior_turn']} "
                         f"({w['item_id']}) is withdrawn; do not rely on it.")
        for unit in active_units:
            lines.append("- " + unit_line(unit))
        if defer_count:
            lines.append(f"{defer_count} item(s) unresolved; escalation available.")
        return "\n".join(lines)

    active = list(units)
    block = render_block(active)
    # summarize pass, lowest priority (end of list) first
    for unit in reversed(active):
        if count_tokens(block) <= budget:
            break
        if unit["kind"] == "single" and not unit["members"][0].get("summarized"):
            unit["members"][0]["summarized"] = True
            block = render_block(active)
    # budget-withhold pass
    while count_tokens(block) > budget and active:
        dropped = active.pop()
        for member in dropped["members"]:
            member["disposition"] = "withhold"
            member["reason_code"] = "budget_exceeded"
            member["budget_decision"] = "budget_withheld"
            member.pop("summarized", None)
        block = render_block(active)
    if count_tokens(block) > budget:
        # Even the bare governance header exceeds B: fail loudly (G-C3
        # forbids silent truncation of any kind).
        raise ValueError("budget smaller than the governance header; "
                         "no compliant block exists")

    # --- 6. audit packet (I4: one row per candidate + per not-retrieved) --
    render_index = {}
    for idx, unit in enumerate(active):
        for member in unit["members"]:
            render_index[member["item"]["item_id"]] = idx

    audit = []
    for iid, r in sorted(resolved.items()):
        idx = render_index.get(iid)
        if r.get("budget_decision") == "budget_withheld":
            decision, cost = "budget_withheld", 0
        elif r["disposition"] in ("assert", "caveat", "dual_present"):
            unit = active[idx]
            decision = "summarized" if r.get("summarized") else "rendered"
            cost = count_tokens(unit_line(unit))
        elif r["disposition"] in ("defer", "withdraw"):
            decision, cost = "notice", 0
        else:
            decision, cost = "not_rendered", 0
        audit.append({
            "query_id": query_id,
            "item_id": iid,
            "state": r["item"]["state"],
            "disposition": r["disposition"],
            "reason_code": r["reason_code"],
            "evidence_ptr": r["evidence_ptr"],
            "tier": r["tier"],
            "policy_version": r["item"]["policy_version"],
            "authorization": authorization,
            "budget_cost": cost,
            "budget_decision": decision,
            "render_index": idx,
        })
    for rec in not_retrieved:
        audit.append({
            "query_id": query_id,
            "item_id": rec["native_id"],
            "state": None,
            "disposition": None,
            "reason_code": "not_retrieved:" + rec["reason"],
            "evidence_ptr": None,
            "tier": None,
            "policy_version": policy["version"],
            "authorization": authorization,
            "budget_cost": 0,
            "budget_decision": "not_retrieved",
            "render_index": None,
        })
    return block, {"rows": audit, "anomalies": anomalies}


def render_raw_matched(items, governed_audit, budget, policy, count_tokens):
    """Memo §8.2 raw-matched arm: identical item multiset to the governed
    render, all governance structure stripped.

    Withheld items (evidence- or budget-withheld) are absent from both
    arms; summarized items are replaced by a deterministic raw prefix at
    the same token cost; dual-present candidate sets appear as plain
    unmarked items ordered by item_id; no header, no qualifiers, no
    notices of any kind.
    """
    by_id = {item["item_id"]: item for item in items}
    rendered = [row for row in governed_audit["rows"]
                if row["budget_decision"] in ("rendered", "summarized")]
    rendered.sort(key=lambda row: (row["render_index"], row["item_id"]))
    lines = []
    for row in rendered:
        content = by_id[row["item_id"]]["content"]
        if row["budget_decision"] == "summarized":
            content = _truncate_to_tokens(content, row["budget_cost"], count_tokens)
        lines.append(content)
    block = "\n".join(lines)
    if count_tokens(block) > budget:
        raise ValueError("raw-matched block exceeds B; governed render "
                         "invariant violated")
    return block
