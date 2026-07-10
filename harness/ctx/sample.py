#!/usr/bin/env python3
"""PR-13 deterministic stratified sample (memo §12 R-1).

Normative artifact: ``harness/ctx/policy/sample_v1.json``. This module
implements it and nothing else; on divergence the JSON wins.

The sample exists because §7's 13 FAM cells hold 35,007 probe rows and
§8.3 spends six model calls per query. It bounds the run's size. It
changes the **power** of every gate and the **constant** of none.

What the sampler is allowed to see: adapter output and the frozen
disposition rule table. What it never sees: a truth column (G-C2), a
budget (the stratum is computed before the §6 budget loop), a turn
state, or any consumer output. Applying the rule table to an item is not
a render — no context block is produced, and no consumer is touched.

Determinism: hash order over a committed salt; largest-remainder
allocation with fully specified tie-breaks and repair. Same inputs →
byte-identical manifest.
"""

import hashlib
import json
import os

# Single source of truth for the disposition table — imported, never
# re-implemented (memo's no-mirrored-logic rule). ``_resolve_item`` is
# the exact function ``compile()`` uses before the budget loop.
from harness.ctx.compile import PRECEDENCE, _resolve_item

_POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy")
_PREC_RANK = {d: i for i, d in enumerate(PRECEDENCE)}


def load_sample_policy(path=None):
    if path is None:
        path = os.path.join(_POLICY_DIR, "sample_v1.json")
    with open(path, "r", encoding="utf-8") as fh:
        sp = json.load(fh)
    if sp["strata"]["keys"] != ["harm_category", "disposition_class"]:
        raise ValueError("sample policy strata diverge from memo §12 R-1")
    return sp


def disposition_class(items, policy):
    """The query's stratum disposition: the highest-precedence (lowest
    rank) disposition the frozen rule table gives any of its items."""
    best = None
    for item in items:
        disposition, _reason, _ptr, _tier, _anoms = _resolve_item(item, policy)
        rank = _PREC_RANK[disposition]
        if best is None or rank < best:
            best = rank
    if best is None:
        raise ValueError("query has no items; not a query")
    return PRECEDENCE[best]


def order_hash(query_id, salt):
    """SHA-256(utf8(query_id) ‖ utf8(salt)), lowercase hex."""
    return hashlib.sha256(query_id.encode("utf-8")
                          + salt.encode("utf-8")).hexdigest()


def _stratum_sort_key(key):
    """(harm_category asc, disposition precedence rank asc)."""
    harm, disposition = key
    return (harm, _PREC_RANK[disposition])


def allocate(cap, sizes):
    """Largest-remainder allocation of ``cap`` across ``{stratum: n}``.

    Registered exactly in sample_v1.json: floor quotas, leftovers to the
    largest remainders (skipping strata already at full size), ties by
    stratum key ascending, then a repair pass giving every nonempty
    stratum at least one query.
    """
    sizes = {k: n for k, n in sizes.items() if n > 0}
    if not sizes:
        return {}
    total = sum(sizes.values())
    if total <= cap:
        return dict(sizes)
    if len(sizes) > cap:
        raise ValueError(f"cap {cap} below stratum count {len(sizes)}")

    keys = sorted(sizes, key=_stratum_sort_key)
    quota = {k: (cap * sizes[k]) // total for k in keys}
    remainder = {k: (cap * sizes[k]) % total for k in keys}

    leftover = cap - sum(quota.values())
    # Largest remainder first; ties by stratum key ascending. Strata
    # already holding their full size cannot absorb more.
    for k in sorted(keys, key=lambda k: (-remainder[k], _stratum_sort_key(k))):
        if leftover == 0:
            break
        if quota[k] < sizes[k]:
            quota[k] += 1
            leftover -= 1

    # Repair: every nonempty stratum gets at least one.
    while True:
        zeros = [k for k in keys if quota[k] == 0]
        if not zeros:
            break
        recipient = min(zeros, key=lambda k: (-remainder[k], _stratum_sort_key(k)))
        donors = [k for k in keys if quota[k] > 1]
        if not donors:
            raise ValueError("cannot give every nonempty stratum a query")
        donor = min(donors, key=lambda k: (-quota[k], _stratum_sort_key(k)))
        quota[donor] -= 1
        quota[recipient] += 1

    assert sum(quota.values()) == cap
    assert all(0 < quota[k] <= sizes[k] for k in keys)
    return quota


def sample_cell(cell_id, role, bundles, policy, sample_policy):
    """Select the registered sample from one FAM cell.

    Returns ``{"cell_id", "role", "n_total", "n_selected", "strata",
    "queries"}`` with ``queries`` in ascending hash order (so the
    manifest's row order is itself a registered, salt-derived order and
    cannot encode anything about outcomes).
    """
    salt = sample_policy["salt"]
    cap = sample_policy["cap_per_cell"]

    members = {}
    hashes = {}
    for qid, bundle in bundles.items():
        key = (role, disposition_class(bundle["items"], policy))
        members.setdefault(key, []).append(qid)
        hashes[qid] = order_hash(qid, salt)

    for qid_list in members.values():
        qid_list.sort(key=lambda q: hashes[q])

    quota = allocate(cap, {k: len(v) for k, v in members.items()})

    selected = []
    strata = []
    for key in sorted(members, key=_stratum_sort_key):
        take = quota.get(key, 0)
        selected.extend(members[key][:take])
        strata.append({"harm_category": key[0], "disposition_class": key[1],
                       "n_total": len(members[key]), "n_selected": take})

    selected.sort(key=lambda q: hashes[q])
    return {
        "cell_id": cell_id,
        "role": role,
        "n_total": len(bundles),
        "n_selected": len(selected),
        "strata": strata,
        "queries": selected,
    }
