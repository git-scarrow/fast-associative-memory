"""PR-3c — classifiers + SHADOW governance over the PR-3c artifacts.

Executes PR3_DESIGN.md §5 PR-3c: the H1 read-time mode classifier, the H2
write-time fork-event classifier, and the H3 shadow-governance comparison
table. Everything here is offline analysis of the logged tables; the
deployed ``forward()`` is never altered and no engine state is touched.

PR-4 step 3 (PR4_DESIGN.md §8) extends the policy set with the two
pre-registered guarded variants (`trust-downweight`, `trust-guarded`;
parameters frozen in the memo) and adds the §3 collateral decomposition:
per-policy `direct_br` / `collateral_br` columns and a per-slot
`collateral_exposure` block (registry-scored exposure next to the
label-free guard proxy). `trust-record` was dropped at gate 1 (memo
Addendum A.4) — per-record slot composition is not shadow-reconstructable.

Shadow-readout fidelity (§10): the policy-`none` vote is recomputed from
``<stem>.topk.csv[.gz]`` with the exact aggregation the driver verified at
write time (``vote_pred_from_candidates``) and compared to the deployed
``vote_pred_label`` on EVERY row — any mismatch raises.

Feature/label separation (§10 label circularity): policies and classifier
features read ONLY engine-observable state — the topk candidate
composition (slot, sim, weight, decode), per-slot observables
(hit_counts, last_write_seq, usage, n_records), write-event observables
(outcome, pre_sim, payload_cos_incumbent, effective_vigilance, incumbent
maturity/recency/lineage), and the frozen #87 two-axis score. Registry
ground truth (failure labels, slot roles, event_class) appears only as
scoring targets. POLICY_VISIBLE_* below are the enforced allowlists
(pinned by test).

A-priori analysis constants (fixed before looking at PR-3c outputs, from
PR-2/PR-3b mechanics):

* WITNESS_SIM_WINDOW = 0.05 — surviving candidates within 0.05 raw cosine
  of the surviving top-1 form the co-residence window; a fork witness is
  >= 2 distinct decode classes inside it. Fork-path conflicts re-write the
  same key (sim gap 0) and the strongest jitter arm (eps 0.15) moves the
  gap to ~0.011, so 0.05 covers every instantiated fork while staying far
  from the ~0.3 cross-class gap.
* TIE_MARGIN = 0.10 — abstain-tie triggers when the top-2 decode-class
  vote masses inside the witness window differ by less than this.
* CONFLICT_COS = 0.5 — the engine's own bipartite payload boundary
  (associative_core); a forked write at or below it is a conflict fork.
* MERGE_SUSPECT_COS = 0.9 — an ABSORBED write whose payload cosine to the
  incumbent is below this is a merge-suspect (clean duplicates sit at
  ~1.0; the soft arm's conflicting absorbs at ~0.6).

Write-time router (the rule the `mode-conditioned` policy executes): a
conflict-fork pair (incumbent I, fork O) is classified per probe epoch E
from traffic at epochs <= E AFTER the fork: if the OLD payload side is
written again, the old value is still being asserted -> contradiction ->
quarantine both sides; if only the NEW side is reinforced -> supersession
-> deprecate I; if neither -> one-shot ambiguous -> observe-only
(primary) / abstain (secondary). An absorbed conflict (merge-suspect)
leaves no fork to elect, so the only honest action is abstaining when the
suspect slot leads the vote. Event-locally a supersession fork and a
contradiction fork are IDENTICAL (forked, payload cos ~0, mature
incumbent) — the H2 "event-local" classifier variant quantifies exactly
that, and the timeline variant shows the separation the router uses.

Usage:
  python benchmarks/analyze_fork_governance.py \
      --dir results/issue_failure_mode_blindness/pr3c
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.analyze_calibration import auc_mann_whitney  # noqa: E402
from benchmarks.failure_mode_probe import (  # noqa: E402
    vote_pred_from_candidates)
from benchmarks.heldout_abstention import (  # noqa: E402
    MANIFOLD_SUPPORT, RANK_GAP, AbstentionDetector, abstain_mask)
from benchmarks.score_frozen_detector import (  # noqa: E402
    FIT_CSV, FIT_SUMMARY, reproduce_frozen_detector)

WITNESS_SIM_WINDOW = 0.05
TIE_MARGIN = 0.10
CONFLICT_COS = 0.5
MERGE_SUSPECT_COS = 0.9

# PR-4 (PR4_DESIGN.md §5) — parameters frozen in the design memo BEFORE any
# PR-4 cache run; changing either after the first run invalidates the
# affected variant's pre-registered status (§7 variant garden).
TRUST_DOWNWEIGHT_LAMBDA = 0.25
TRUST_GUARD_THETA = 0.8

# Enforced policy-visible allowlists (no registry-derived value may appear).
POLICY_VISIBLE_TOPK = ("rank", "slot", "sim", "surviving", "weight",
                       "decode")
POLICY_VISIBLE_SLOT = ("hit_counts", "last_write_seq", "usage", "n_records")
POLICY_VISIBLE_EVENT = ("epoch", "record_seq", "outcome", "pre_sim",
                        "payload_cos_incumbent", "effective_vigilance",
                        "incumbent_slot", "incumbent_hit_counts",
                        "incumbent_last_write_seq", "incumbent_n_records",
                        "owner_slot")

H1_FEATURES = ["top1_top2_margin", "top1_sim", "top2_sim", "vote_entropy",
               "effective_support", "max_vote_weight"]
H2_LOCAL_FEATURES = ["pre_sim", "payload_cos_incumbent",
                     "effective_vigilance", "incumbent_hit_counts",
                     "incumbent_n_records", "seq_gap", "outcome_forked"]
H2_TIMELINE_FEATURES = H2_LOCAL_FEATURES + ["subs_old", "subs_new"]

POLICIES = ["none", "observe-only", "entropy-abstain", "abstain-tie",
            "recency-naive", "quarantine-naive",
            "mode-conditioned-observe", "mode-conditioned-abstain",
            "mode-conditioned-trust", "trust-downweight", "trust-guarded"]
# mode-conditioned-trust was an EXPLORATORY refinement in PR-3c (post-hoc
# there; promoted to pre-registered for PR-4 H1, frozen verbatim):
# instead of the design memo's contradiction -> quarantine-both, it
# deprecates only the fork side that subsequent traffic did NOT reinforce
# (quarantining both only when both sides keep being written). The
# pre-registered quarantine-both variants are reported unchanged alongside.
#
# PR-4 guarded variants (PR4_DESIGN.md §5; routing identical to
# mode-conditioned-trust, only the DEPRECATION action differs — the
# quarantine-both branch, merge-suspect -> abstain and ambiguous ->
# observe-only routes are inherited unchanged):
#   trust-downweight — the deprecated side's vote weight is multiplied by
#       TRUST_DOWNWEIGHT_LAMBDA instead of excluded (H2-policy branch:
#       is exclusion-vs-attenuation the collateral driver?).
#   trust-guarded    — a slot is excluded only if its label-free fork-party
#       fraction (GuardIndex: share of its surviving top-k appearances, up
#       to and including the current probe, on probes where a counterpart
#       slot of one of its routed pairs is also a surviving candidate) is
#       >= TRUST_GUARD_THETA; otherwise it is downweighted at
#       TRUST_DOWNWEIGHT_LAMBDA. The proxy reads topk + router output
#       only — never the registry (§7 guard circularity).
TRUST_ROUTED = ("mode-conditioned-trust", "trust-downweight",
                "trust-guarded")
# Policies whose interventions trigger ONLY on read-time fork topology
# (the witness window) — the family that is structurally blind to
# merge-path stale (single merged slot, no co-resident competitor).
READ_TIME_FORK_ONLY = ("abstain-tie", "recency-naive", "quarantine-naive")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _open_maybe_gz(path: Path):
    gz = path.with_name(path.name + ".gz")
    if path.exists():
        return open(path, newline="")
    if gz.exists():
        return gzip.open(gz, "rt", newline="")
    raise FileNotFoundError(f"{path} (or .gz)")


def load_run(stem: Path) -> dict:
    """All four tables for one run stem (the per-probe CSV path)."""
    with open(stem, newline="") as f:
        probes = list(csv.DictReader(f))
    with _open_maybe_gz(stem.with_suffix(".topk.csv")) as f:
        topk = defaultdict(list)
        for r in csv.DictReader(f):
            topk[(int(r["epoch"]), int(r["probe_index"]))].append({
                "rank": int(r["rank"]), "slot": int(r["slot"]),
                "sim": float(r["sim"]), "surviving": r["surviving"] == "1",
                "weight": r["weight"], "decode": int(r["decode"])})
        for cands in topk.values():
            cands.sort(key=lambda c: c["rank"])
    slot_obs, slot_roles = {}, {}
    with open(stem.with_suffix(".per_slot.csv"), newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["epoch"]), int(r["slot"]))
            slot_obs[key] = {k: int(float(r[k])) if k != "usage"
                             else float(r[k]) for k in POLICY_VISIBLE_SLOT}
            slot_obs[key]["decode"] = int(r["decode"])  # engine state
            slot_roles[key] = r["role"]                  # scoring only
    with open(stem.with_suffix(".fork_events.csv"), newline="") as f:
        events = []
        for r in csv.DictReader(f):
            ev = {k: r[k] for k in ("outcome",)}
            for k in ("epoch", "record_seq", "incumbent_slot",
                      "incumbent_hit_counts", "incumbent_last_write_seq",
                      "incumbent_n_records", "owner_slot"):
                ev[k] = int(float(r[k]))
            for k in ("pre_sim", "payload_cos_incumbent",
                      "effective_vigilance"):
                ev[k] = float(r[k])
            ev["event_class"] = r["event_class"]  # scoring only
            events.append(ev)
    events.sort(key=lambda e: e["record_seq"])
    value_dim = 1 + max(
        max((c["decode"] for cs in topk.values() for c in cs), default=0),
        max((int(float(p["true_label"])) for p in probes), default=0))
    return {"probes": probes, "topk": topk, "slot_obs": slot_obs,
            "slot_roles": slot_roles, "events": events,
            "value_dim": value_dim}


# ---------------------------------------------------------------------------
# Read-time, label-free structure
# ---------------------------------------------------------------------------
def fork_witness(cands: list[dict]) -> list[dict]:
    """Surviving candidates inside the co-residence window of the surviving
    top-1, iff they span >= 2 decode classes; else []."""
    surv = [c for c in cands if c["surviving"]]
    if not surv:
        return []
    top = max(c["sim"] for c in surv)
    win = [c for c in surv if c["sim"] >= top - WITNESS_SIM_WINDOW]
    return win if len({c["decode"] for c in win}) >= 2 else []


def _vote(cands, value_dim, excluded=frozenset(), scaled=None):
    """(answer, forced_abstain) over the candidate set minus exclusions.

    ``scaled`` maps slot -> attenuation factor (PR-4 trust-downweight /
    trust-guarded): the candidate's weight is multiplied in float32 (the
    vote's accumulation dtype) instead of zeroed. With no exclusions and
    no scaling the emitted weight TEXT passes through untouched, so the
    policy-`none` bit-fidelity path is unchanged.
    """
    scaled = scaled or {}
    weights = []
    for c in cands:
        if c["slot"] in excluded:
            weights.append("0")
        elif c["slot"] in scaled:
            weights.append(float(np.float32(float(c["weight"]))
                                 * np.float32(scaled[c["slot"]])))
        else:
            weights.append(c["weight"])
    if not any(c["surviving"] and c["slot"] not in excluded for c in cands):
        return None, True
    return vote_pred_from_candidates(
        weights, [c["decode"] for c in cands], value_dim), False


def _class_mass(cands):
    mass = defaultdict(float)
    for c in cands:
        mass[c["decode"]] += float(c["weight"])
    return mass


# ---------------------------------------------------------------------------
# Write-time router (mode-conditioned policy; label-free)
# ---------------------------------------------------------------------------
def build_writetime_router(events: list[dict], slot_obs: dict) -> dict:
    """Classify every conflict from write-time observables, per epoch.

    Returns {"pairs": [...], "merge": [(epoch, slot), ...]} where each pair
    carries per-epoch verdicts under ``verdict_by_epoch``.
    """
    decode_at = _decode_at_fn(slot_obs)

    conflicts = [e for e in events
                 if e["outcome"] == "forked" and e["incumbent_slot"] >= 0
                 and np.isfinite(e["payload_cos_incumbent"])
                 and e["payload_cos_incumbent"] <= CONFLICT_COS]
    merges = [(e["epoch"], e["owner_slot"]) for e in events
              if e["outcome"] == "absorbed"
              and np.isfinite(e["payload_cos_incumbent"])
              and e["payload_cos_incumbent"] < MERGE_SUSPECT_COS]
    max_epoch = max((e["epoch"] for e in events), default=-1)

    pairs = []
    for c in conflicts:
        I, O = c["incumbent_slot"], c["owner_slot"]
        old_side = decode_at(c["epoch"], I)
        new_side = decode_at(c["epoch"], O)
        if old_side is None or new_side is None:
            continue  # evicted before any snapshot; nothing to route
        verdicts = {}
        later = [e for e in events
                 if e["record_seq"] > c["record_seq"]
                 and e["incumbent_slot"] in (I, O)
                 and e["outcome"] in ("absorbed", "forked")]
        for E in range(c["epoch"], max_epoch + 1):
            old_n = new_n = 0
            for e in later:
                if e["epoch"] > E:
                    continue
                side = decode_at(
                    e["epoch"],
                    e["owner_slot"] if e["outcome"] == "forked"
                    else e["incumbent_slot"])
                if side == old_side:
                    old_n += 1
                elif side == new_side:
                    new_n += 1
            if old_n > 0:
                v = "contradiction"
            elif new_n > 0:
                v = "supersession"
            else:
                v = "ambiguous"
            verdicts[E] = {"verdict": v, "old_n": old_n, "new_n": new_n}
        pairs.append({"I": I, "O": O, "epoch": c["epoch"],
                      "old_side": old_side, "new_side": new_side,
                      "verdict_by_epoch": verdicts})
    return {"pairs": pairs, "merge": merges}


def router_state(router: dict, epoch: int, trust: bool = False) -> dict:
    quarantine, deprecate, ambiguous = set(), set(), set()
    for p in router["pairs"]:
        v = p["verdict_by_epoch"].get(epoch)
        if v is None:
            continue
        if v["verdict"] == "contradiction":
            if trust and v["new_n"] == 0:
                deprecate.add(p["O"])  # old side corroborated; drop the fork
            else:
                quarantine |= {p["I"], p["O"]}
        elif v["verdict"] == "supersession":
            deprecate.add(p["I"])
        else:
            ambiguous |= {p["I"], p["O"]}
    merge = {s for (e, s) in router["merge"] if e <= epoch}
    # conservative precedence: a quarantined slot stays quarantined
    deprecate -= quarantine
    ambiguous -= (quarantine | deprecate)
    return {"quarantine": quarantine, "deprecate": deprecate,
            "ambiguous": ambiguous, "merge": merge}


def pair_counterparts(router: dict) -> dict[int, set[int]]:
    """Static slot -> {counterpart slots} map over all routed conflict
    pairs (pair MEMBERSHIP never changes across epochs; only verdicts do)."""
    cp = defaultdict(set)
    for p in router["pairs"]:
        cp[p["I"]].add(p["O"])
        cp[p["O"]].add(p["I"])
    return dict(cp)


def pair_party_labels(router: dict) -> dict[int, set[int]]:
    """Slot -> decode classes that are PARTIES to the slot's fork(s)
    (old_side/new_side of every pair the slot is a member of). Used only
    for the §3 registry-scored collateral exposure, never by a policy."""
    party = defaultdict(set)
    for p in router["pairs"]:
        for s in (p["I"], p["O"]):
            party[s] |= {p["old_side"], p["new_side"]}
    return dict(party)


class GuardIndex:
    """Label-free fork-party appearance index (PR4_DESIGN.md §5).

    For every slot that is a member of a routed conflict pair: the ordered
    list of its surviving top-k appearances over the shadow log, each
    flagged fork-party iff a counterpart slot of one of its pairs is also
    a surviving candidate of the SAME probe (the probe queries the
    contested key region). ``fork_party_fraction(slot, epoch, probe)`` is
    the fraction over appearances at probes <= (epoch, probe) — inclusive
    of the current probe, so the proxy is defined on first appearance.

    Built from topk + the write-time router's pair membership only; no
    registry column is read (§7 guard circularity — the proxy is validated
    AGAINST the registry exposure measure, never built from it).
    """

    def __init__(self, topk: dict, counterpart: dict[int, set[int]]):
        self._keys = defaultdict(list)    # slot -> [(epoch, probe), ...]
        self._party = defaultdict(list)   # slot -> prefix fork-party counts
        for key in sorted(topk):
            surv = {c["slot"] for c in topk[key] if c["surviving"]}
            for s in surv:
                cps = counterpart.get(s)
                if not cps:
                    continue
                ps = self._party[s]
                self._keys[s].append(key)
                ps.append((ps[-1] if ps else 0) + int(bool(cps & surv)))

    def fork_party_fraction(self, slot: int, epoch: int,
                            probe_index: int) -> float | None:
        ks = self._keys.get(slot)
        if not ks:
            return None
        i = bisect.bisect_right(ks, (epoch, probe_index))
        return None if i == 0 else self._party[slot][i - 1] / i

    def appearances(self, slot: int) -> list[tuple[int, int]]:
        return self._keys.get(slot, [])


# ---------------------------------------------------------------------------
# Policies — each returns (answer | None, acted: bool, detail | None).
# ``detail`` describes a vote-altering intervention for the §3 collateral
# decomposition: {"dropped": slots excluded or attenuated, "counterpart":
# the static pair-counterpart map (None for witness-window policies, whose
# drops are by construction local to the probe's own key region)}.
# ---------------------------------------------------------------------------
def apply_policy(policy: str, cands: list[dict], witness: list[dict],
                 none_answer: int, value_dim: int, epoch: int,
                 slot_obs: dict, frozen_retain: bool,
                 router: dict | None, guard: "GuardIndex | None" = None,
                 counterpart: dict | None = None, probe_index: int = -1):
    if policy in ("none", "observe-only"):
        return none_answer, False, None
    if policy == "entropy-abstain":
        return (none_answer, False, None) if frozen_retain \
            else (None, True, None)
    if policy == "abstain-tie":
        if witness:
            mass = sorted(_class_mass(witness).values(), reverse=True)
            if len(mass) >= 2 and mass[0] - mass[1] < TIE_MARGIN:
                return None, True, None
        return none_answer, False, None
    if policy == "recency-naive":
        if not witness:
            return none_answer, False, None
        rec = defaultdict(lambda: -1)
        for c in witness:
            obs = slot_obs.get((epoch, c["slot"]))
            if obs is not None:
                rec[c["decode"]] = max(rec[c["decode"]],
                                       obs["last_write_seq"])
        best = max(rec.values())
        newest = {d for d, r in rec.items() if r == best}
        if len(newest) != 1:
            return none_answer, False, None  # recency tie: no basis to act
        drop = {c["slot"] for c in witness if c["decode"] not in newest}
        if not drop:
            return none_answer, False, None
        ans, forced = _vote(cands, value_dim, excluded=frozenset(drop))
        return ans, True, {"dropped": frozenset(drop), "counterpart": None}
    if policy == "quarantine-naive":
        if not witness:
            return none_answer, False, None
        drop = frozenset(c["slot"] for c in witness)
        ans, forced = _vote(cands, value_dim, excluded=drop)
        return ans, True, {"dropped": drop, "counterpart": None}
    if policy.startswith("mode-conditioned") or policy in TRUST_ROUTED:
        st = router_state(router, epoch, trust=policy in TRUST_ROUTED)
        surv = [c for c in cands if c["surviving"]]
        top1 = max(surv, key=lambda c: float(c["weight"]))
        if top1["slot"] in st["merge"]:
            # no fork to elect; abstain is the honest act
            return None, True, None
        surv_slots = {c["slot"] for c in surv}
        exclude = set(st["quarantine"] & surv_slots)
        dep = st["deprecate"] & surv_slots
        scaled = {}
        if policy == "trust-downweight":
            scaled = {s: TRUST_DOWNWEIGHT_LAMBDA for s in dep}
        elif policy == "trust-guarded":
            for s in sorted(dep):
                frac = guard.fork_party_fraction(s, epoch, probe_index)
                if frac is not None and frac >= TRUST_GUARD_THETA:
                    exclude.add(s)
                else:
                    scaled[s] = TRUST_DOWNWEIGHT_LAMBDA
        else:
            exclude |= dep
        if policy == "mode-conditioned-abstain" and any(
                c["slot"] in st["ambiguous"] for c in surv
                if c["slot"] not in exclude):
            return None, True, None
        if not exclude and not scaled:
            return none_answer, False, None
        ans, forced = _vote(cands, value_dim,
                            excluded=frozenset(exclude), scaled=scaled)
        return ans, True, {"dropped": frozenset(exclude) | set(scaled),
                           "counterpart": counterpart}
    raise ValueError(policy)


# ---------------------------------------------------------------------------
# H3 — shadow-governance scoring for one run
# ---------------------------------------------------------------------------
def frozen_scores(probes: list[dict], det, thr) -> np.ndarray:
    cols = {RANK_GAP: np.array([float(p[RANK_GAP]) for p in probes]),
            MANIFOLD_SUPPORT: np.array([float(p[MANIFOLD_SUPPORT])
                                        for p in probes])}
    mask = np.ones(len(probes), dtype=bool)
    return abstain_mask(det.score(cols, mask), thr)  # True = retain


def score_run(run: dict, det, thr) -> dict:
    probes, topk = run["probes"], run["topk"]
    value_dim, slot_obs = run["value_dim"], run["slot_obs"]
    router = build_writetime_router(run["events"], slot_obs)
    counterpart = pair_counterparts(router)
    guard = GuardIndex(topk, counterpart)
    retain = frozen_scores(probes, det, thr)

    metrics = {policy: {k: 0 for k in (
        "n", "wrong_none", "acted", "abstained", "abstain_on_correct",
        "abstain_on_wrong", "answered", "answered_correct", "fixed",
        "broken", "direct_br", "collateral_br", "tie_flips",
        "witness_probes",
        "stale_wrong", "stale_wrong_fixed", "stale_wrong_abstained",
        "contra_wrong", "contra_wrong_fixed", "contra_wrong_abstained")}
        for policy in POLICIES}
    dropped_ever = {policy: set() for policy in POLICIES}
    for i, p in enumerate(probes):
        epoch = int(float(p["epoch"]))
        probe_index = int(p["probe_index"])
        cands = topk[(epoch, probe_index)]
        truth = int(float(p["true_label"]))
        deployed = int(float(p["vote_pred_label"]))
        none_answer, _ = _vote(cands, value_dim)
        if none_answer != deployed:
            raise RuntimeError(
                f"shadow-readout fidelity violation at epoch {epoch} "
                f"probe {p['probe_index']}: none={none_answer} "
                f"deployed={deployed}")
        witness = fork_witness(cands)
        surv_slots = {c["slot"] for c in cands if c["surviving"]}
        top1_slot = max((c for c in cands if c["surviving"]),
                        key=lambda c: float(c["weight"]))["slot"]
        # exact vote tie (the one-shot signature): the §1 side-effect
        # channel — a policy "fixing" such a row flips a coin-flip
        # election by fiat, not inference; counted per run as tie_flips.
        mass = sorted(_class_mass(
            [c for c in cands if c["surviving"]]).values(), reverse=True)
        exact_tie = len(mass) >= 2 and abs(mass[0] - mass[1]) < 1e-9
        none_correct = deployed == truth
        is_stale_wrong = p["stale_lenient"] == "1" and not none_correct
        is_contra_wrong = (p["contradictory_lenient"] == "1"
                           and not none_correct)
        for policy in POLICIES:
            m = metrics[policy]
            ans, acted, detail = apply_policy(
                policy, cands, witness, none_answer, value_dim, epoch,
                slot_obs, bool(retain[i]), router, guard=guard,
                counterpart=counterpart, probe_index=probe_index)
            if detail:
                dropped_ever[policy] |= detail["dropped"]
            m["n"] += 1
            m["wrong_none"] += int(not none_correct)
            m["acted"] += int(acted)
            m["witness_probes"] += int(bool(witness))
            m["stale_wrong"] += int(is_stale_wrong)
            m["contra_wrong"] += int(is_contra_wrong)
            if ans is None:
                m["abstained"] += 1
                m["abstain_on_correct"] += int(none_correct)
                m["abstain_on_wrong"] += int(not none_correct)
                m["stale_wrong_abstained"] += int(is_stale_wrong)
                m["contra_wrong_abstained"] += int(is_contra_wrong)
                continue
            m["answered"] += 1
            m["tie_flips"] += int(exact_tie and ans != none_answer)
            ok = ans == truth
            m["answered_correct"] += int(ok)
            m["fixed"] += int(ok and not none_correct)
            if none_correct and not ok:
                # §3 decomposition: DIRECT iff the deployed top-1 itself was
                # dropped AND its fork concerns this probe's key region (a
                # pair counterpart is a surviving candidate here; witness-
                # window drops are local by construction). Else COLLATERAL —
                # the vote change came from deprecating slots whose fork
                # verdicts concern other keys (the pair-B mechanism).
                m["broken"] += 1
                direct = False
                if detail and top1_slot in detail["dropped"]:
                    cp = detail["counterpart"]
                    direct = (True if cp is None else
                              bool(cp.get(top1_slot, set()) & surv_slots))
                m["direct_br" if direct else "collateral_br"] += 1
            m["stale_wrong_fixed"] += int(is_stale_wrong and ok)
            m["contra_wrong_fixed"] += int(is_contra_wrong and ok)

    # §3 per-slot collateral exposure, per policy, for every slot the policy
    # ever excluded/attenuated: the fraction of the slot's surviving top-k
    # appearances (over the run) on probes whose registry key (true label)
    # is NOT a party to the slot's fork. Registry is used only here, for
    # scoring; alongside it the label-free guard proxy's final fork-party
    # fraction, so the §7 guard-circularity check (proxy validated AGAINST
    # the registry measure) is a column comparison in the result memo.
    party_labels = pair_party_labels(router)
    truth_at = {(int(float(p["epoch"])), int(p["probe_index"])):
                int(float(p["true_label"])) for p in probes}
    out = {}
    for policy, m in metrics.items():
        m["accuracy_effective"] = round(m["answered_correct"] / m["n"], 6)
        m["read_time_fork_only"] = policy in READ_TIME_FORK_ONLY
        per_slot, vals = [], []
        routed = sorted(s for s in dropped_ever[policy] if s in party_labels)
        for s in routed:
            apps = guard.appearances(s)
            if not apps:
                continue
            nonparty = sum(1 for key in apps
                           if truth_at.get(key) not in party_labels[s])
            x = nonparty / len(apps)
            proxy = guard.fork_party_fraction(s, *apps[-1])
            per_slot.append({"slot": s, "n_appearances": len(apps),
                             "registry_exposure": round(x, 6),
                             "proxy_fork_party_fraction": round(proxy, 6)})
            vals.append(x)
        m["collateral_exposure"] = {
            "n_slots": len(per_slot),
            "n_dropped_unrouted": len(dropped_ever[policy]) - len(routed),
            "mean": round(float(np.mean(vals)), 6) if vals else None,
            "median": round(float(np.median(vals)), 6) if vals else None,
            "max": round(float(np.max(vals)), 6) if vals else None,
            "per_slot": per_slot}
        out[policy] = m

    out["_router"] = {
        "n_conflict_pairs": len(router["pairs"]),
        "n_merge_suspect_events": len(router["merge"]),
        "final_epoch_verdicts": dict(
            sorted(_count_verdicts(router).items())),
    }
    return out


def _count_verdicts(router):
    counts = defaultdict(int)
    for p in router["pairs"]:
        if p["verdict_by_epoch"]:
            last = p["verdict_by_epoch"][max(p["verdict_by_epoch"])]
            counts[last["verdict"]] += 1
    return counts


# ---------------------------------------------------------------------------
# H1 — read-time mode classifier
# ---------------------------------------------------------------------------
def _h1_dataset(probes):
    cols = {f: np.array([float(p[f]) for p in probes]) for f in H1_FEATURES}
    wrong = np.array([p["vote_correct"] == "0" for p in probes])
    contra = np.array([p["contradictory_lenient"] == "1" for p in probes])
    stale = np.array([p["stale_lenient"] == "1" for p in probes])
    cols["is_contra"] = (contra & wrong & ~stale).astype(float)
    cols["epoch"] = np.array([float(p["epoch"]) for p in probes])
    mode_rows = wrong & (contra ^ stale)  # exactly one mode label
    cols["_mode_rows"] = mode_rows
    cols["_correct"] = ~wrong
    return cols


def h1_study(runs: dict[str, dict]) -> dict:
    """Binary contra-wrong vs stale-wrong, fit on mixed_s0 even epochs."""
    fit = _h1_dataset(runs["mixed_s0"]["probes"])
    even = (fit["epoch"] % 2 == 0)
    det = AbstentionDetector(H1_FEATURES)
    det.fit(fit, fit["_mode_rows"] & even, "is_contra")

    def auc_on(cols, mask):
        sel = cols["_mode_rows"] & mask
        n_contra = int(cols["is_contra"][sel].sum())
        n_stale = int(sel.sum()) - n_contra
        if n_contra == 0 or n_stale == 0:
            return {"auc": None, "n_contra": n_contra, "n_stale": n_stale}
        p = det.score(cols, sel)
        return {"auc": round(auc_mann_whitney(
                    p, cols["is_contra"][sel].astype(int)), 4),
                "n_contra": n_contra, "n_stale": n_stale}

    splits = {"mixed_s0_heldout_epochs": auc_on(fit, fit["epoch"] % 2 == 1)}
    for name in ("mixed_s1", "mixed_s2", "mixed_s0_pairB"):
        cols = _h1_dataset(runs[name]["probes"])
        splits[name] = auc_on(cols, np.ones(len(cols["epoch"]), bool))
    # pure-arm pools: contra rows from the contra run + stale rows from a
    # stale-protocol run of the same seed/pair (deployment-shaped traffic
    # never mixes them for us, but the classifier question is the same).
    pools = {
        "pure_pairA_s0": ("contra_s0", "stale_s0"),
        "pure_pairA_s1": ("contra_s1", "stale_s1"),
        "pure_pairA_s2": ("contra_s2", "stale_s2"),
        "pure_pairB_s0": ("contra_s0_pairB", "stale_s0_pairB"),
        "jitter0.05_vs_contra_s0": ("contra_s0", "stale-jitter0.05_s0"),
        "jitter0.15_vs_contra_s0": ("contra_s0", "stale-jitter0.15_s0"),
        "oneshot_s0_vs_contra_s0": ("contra_s0", "stale-oneshot_s0"),
        "soft_s0_vs_contra_s0": ("contra_s0", "stale-soft_s0"),
    }
    for name, (c_run, s_run) in pools.items():
        cols = _h1_dataset(runs[c_run]["probes"] + runs[s_run]["probes"])
        splits[name] = auc_on(cols, np.ones(len(cols["epoch"]), bool))

    # 3-way version including `correct` — deployment never knows a probe is
    # wrong, so the binary task overstates what a deployed classifier sees.
    def _xy(cols, mask):
        sel = mask & (cols["_mode_rows"] | cols["_correct"])
        X = np.column_stack([cols[f][sel] for f in H1_FEATURES])
        y = np.where(cols["_correct"][sel], 0,
                     np.where(cols["is_contra"][sel] > 0, 1, 2))
        return X, y

    X_tr, y_tr = _xy(fit, even)
    model, mean, std = _fit_multinomial(X_tr, y_tr)
    threeway = {}
    for name, cols, mask in (
            [("mixed_s0_heldout_epochs", fit, fit["epoch"] % 2 == 1)]
            + [(n, _h1_dataset(runs[n]["probes"]),
                np.ones(len(runs[n]["probes"]), bool))
               for n in ("mixed_s1", "mixed_s2", "mixed_s0_pairB")]):
        X, y = _xy(cols, mask)
        Xs = np.clip((X - mean) / std, -10, 10)
        pred = model.predict_proba(Xs).argmax(axis=1)
        names = ["correct", "contra-wrong", "stale-wrong"]
        threeway[name] = {
            "n": int(len(y)),
            "per_class": {names[c]: {
                "n": int((y == c).sum()),
                "as": {names[k]: int((pred[y == c] == k).sum())
                       for k in range(3)}} for c in range(3)
                if (y == c).any()}}
    return {"features": H1_FEATURES,
            "fit": "mixed_s0 even epochs, contra-wrong(1) vs stale-wrong(0),"
                   " lenient flags, overlap rows excluded",
            "splits": splits, "threeway": threeway}


# ---------------------------------------------------------------------------
# H2 — write-time fork-event classifier
# ---------------------------------------------------------------------------
def _decode_at_fn(slot_obs: dict):
    """Forward-filled (epoch, slot) -> decode lookup (nearest snapshot at
    or before the epoch)."""
    by_slot = defaultdict(dict)
    for (e, s), obs in slot_obs.items():
        by_slot[s][e] = obs["decode"]

    def decode_at(epoch, slot):
        snaps = by_slot.get(slot)
        if not snaps:
            return None
        best = max((e for e in snaps if e <= epoch), default=None)
        return None if best is None else snaps[best]
    return decode_at


def _h2_dataset(run: dict):
    if "_h2_rows" in run:
        return run["_h2_rows"]
    decode_at = _decode_at_fn(run["slot_obs"])
    conflict_rel = [e for e in run["events"]
                    if e["incumbent_slot"] >= 0
                    and e["outcome"] in ("absorbed", "forked")]
    by_incumbent = defaultdict(list)
    for e in conflict_rel:
        by_incumbent[e["incumbent_slot"]].append(e)
    events = [e for e in conflict_rel if np.isfinite(e["pre_sim"])]
    rows = []
    for ev in events:
        I, O = ev["incumbent_slot"], ev["owner_slot"]
        old_side = decode_at(ev["epoch"], I)
        new_side = decode_at(ev["epoch"], O)
        subs_old = subs_new = 0
        later = by_incumbent[I] + ([] if O == I else by_incumbent[O])
        for e2 in later:
            if e2["record_seq"] <= ev["record_seq"]:
                continue
            side = decode_at(
                e2["epoch"], e2["owner_slot"] if e2["outcome"] == "forked"
                else e2["incumbent_slot"])
            subs_old += int(side == old_side)
            subs_new += int(side == new_side and new_side != old_side)
        rows.append({
            "pre_sim": ev["pre_sim"],
            "payload_cos_incumbent": ev["payload_cos_incumbent"],
            "effective_vigilance": ev["effective_vigilance"],
            "incumbent_hit_counts": ev["incumbent_hit_counts"],
            "incumbent_n_records": ev["incumbent_n_records"],
            "seq_gap": ev["record_seq"] - ev["incumbent_last_write_seq"],
            "outcome_forked": float(ev["outcome"] == "forked"),
            "subs_old": float(subs_old), "subs_new": float(subs_new),
            "event_class": ev["event_class"]})
    run["_h2_rows"] = rows
    return rows


def _fit_multinomial(X, y):
    from sklearn.linear_model import LogisticRegression
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = np.clip((X - mean) / std, -10, 10)
    model = LogisticRegression(max_iter=2000).fit(Xs, y)
    return model, mean, std


def h2_study(runs: dict[str, dict]) -> dict:
    feature_sets = {"event_local": H2_LOCAL_FEATURES,
                    "with_timeline": H2_TIMELINE_FEATURES}
    train_rows = _h2_dataset(runs["mixed_s0"])
    classes = ["duplicate-rewrite", "contradiction", "supersession"]
    train = [r for r in train_rows if r["event_class"] in classes]
    y_tr = np.array([classes.index(r["event_class"]) for r in train])

    eval_runs = {
        "mixed_s1": ["mixed_s1"], "mixed_s2": ["mixed_s2"],
        "mixed_s0_pairB": ["mixed_s0_pairB"],
        "pure_pairA_s0": ["contra_s0", "stale_s0"],
        "pure_pairB_s0": ["contra_s0_pairB", "stale_s0_pairB"],
        "soft_s0-2": ["stale-soft_s0", "stale-soft_s1", "stale-soft_s2"],
        "jitter_s0": ["stale-jitter0.05_s0", "stale-jitter0.15_s0"],
    }
    oneshot_runs = ["stale-oneshot_s0", "stale-oneshot_s1",
                    "stale-oneshot_s2"]

    report = {"classes": classes, "fit": "mixed_s0 conflict-relevant events "
              "(absorbed/forked with a live incumbent)", "variants": {}}
    for vname, feats in feature_sets.items():
        X_tr = np.array([[r[f] for f in feats] for r in train])
        model, mean, std = _fit_multinomial(X_tr, y_tr)

        def predict(rows):
            X = np.array([[r[f] for f in feats] for r in rows])
            Xs = np.clip((X - mean) / std, -10, 10)
            return model.predict_proba(Xs)

        splits = {}
        for name, run_names in eval_runs.items():
            rows = [r for rn in run_names for r in _h2_dataset(runs[rn])
                    if r["event_class"] in classes]
            if not rows:
                splits[name] = {"n": 0}
                continue
            proba = predict(rows)
            pred = proba.argmax(axis=1)
            y = np.array([classes.index(r["event_class"]) for r in rows])
            conf = {}
            for ci, cname in enumerate(classes):
                sel = y == ci
                if not sel.any():
                    continue
                conf[cname] = {
                    "n": int(sel.sum()),
                    "correct": int((pred[sel] == ci).sum()),
                    "as": {classes[cj]: int((pred[sel] == cj).sum())
                           for cj in range(len(classes))}}
            splits[name] = {"n": len(rows),
                            "accuracy": round(float((pred == y).mean()), 4),
                            "per_class": conf}
        # one-shot routing: where do protocol-certified-ambiguous events go?
        os_rows = [r for rn in oneshot_runs for r in _h2_dataset(runs[rn])
                   if r["event_class"] == "one-shot-ambiguous"]
        routing = {"n": len(os_rows)}
        if os_rows:
            proba = predict(os_rows)
            pred = proba.argmax(axis=1)
            routing.update({
                "argmax_counts": {classes[c]: int((pred == c).sum())
                                  for c in range(len(classes))},
                "max_prob_median": round(float(
                    np.median(proba.max(axis=1))), 4),
                "max_prob_p90": round(float(
                    np.percentile(proba.max(axis=1), 90)), 4)})
        splits["one-shot_routing"] = routing
        report["variants"][vname] = {"features": feats, "splits": splits}
    return report


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True,
                    help="directory of PR-3c run artifacts")
    ap.add_argument("--fit-csv", default=FIT_CSV)
    ap.add_argument("--fit-summary", default=FIT_SUMMARY)
    args = ap.parse_args()

    d = Path(args.dir)
    stems = sorted(p for p in d.glob("per_probe_*.csv")
                   if not p.name.endswith((".per_slot.csv",
                                           ".fork_events.csv", ".topk.csv")))
    det, thr, provenance = reproduce_frozen_detector(
        Path(args.fit_csv), Path(args.fit_summary))

    runs, table = {}, {}
    for stem in stems:
        name = stem.stem.replace("per_probe_", "")
        print(f"[governance] {name}", flush=True)
        runs[name] = load_run(stem)
        table[name] = score_run(runs[name], det, thr)
        with open(stem.with_suffix(".governance.json"), "w") as f:
            json.dump(table[name], f, indent=1, sort_keys=True)

    print("[classifiers] H1 mode classifier", flush=True)
    h1 = h1_study(runs)
    print("[classifiers] H2 fork-event classifier", flush=True)
    h2 = h2_study(runs)

    out = {"frozen_detector": provenance,
           "constants": {"witness_sim_window": WITNESS_SIM_WINDOW,
                         "tie_margin": TIE_MARGIN,
                         "conflict_cos": CONFLICT_COS,
                         "merge_suspect_cos": MERGE_SUSPECT_COS,
                         "trust_downweight_lambda": TRUST_DOWNWEIGHT_LAMBDA,
                         "trust_guard_theta": TRUST_GUARD_THETA},
           "policies": POLICIES,
           "read_time_fork_only": list(READ_TIME_FORK_ONLY),
           "governance": table, "h1": h1, "h2": h2}
    out_path = d / "pr3c_governance_table.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
