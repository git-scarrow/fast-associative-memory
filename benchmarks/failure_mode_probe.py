#!/usr/bin/env python3
"""Failure-mode-labeled injection driver (PR-2a, issue: failure-mode blindness).

ANALYSIS ONLY. This driver introduces **no retrieval intervention** and **no
detector refitting**: it runs the same write/probe loop as
benchmarks/calibration_probe.py and additionally (a) injects controlled
CONTRADICTORY and STALE writes through the *normal* ``learn_local`` path, and
(b) stamps every per-probe row with a mechanically derived failure-mode label,
so the frozen #87 two-axis detector can later be scored per failure mode.

Mechanisms exploited (all pre-existing, none modified):

  * Contradiction fork — ``learn_local``'s bipartite class check demotes a hit
    whose payload cosine to the stored value is <= 0.5 into a miss, silently
    allocating the conflicting payload as a co-resident prototype
    (associative_core.py, ``same_class = payload_sims > 0.5``). An injected
    write lands as a fork exactly when its pre-write nearest sim clears the
    vigilance threshold AND its payload disagrees.
  * Provenance — ``track_provenance=True`` + per-row ``record_ids`` give every
    write a durable tag; ``slot_records`` is the ground-truth map from slots
    back to the writes that formed them. The driver stamps EVERY write (clean,
    contradictory, supersession) so a fresh allocation is exactly the slot
    whose record set is ``{id}`` after the call.
  * EMA freeze — a mature prototype's adaptive alpha is
    ``hebb_lr / (1 + ema_beta * hit_counts)``, so a same-slot supersession
    (payload cosine > 0.5, argmax differs) barely moves the stored value: the
    slot keeps decoding to the pre-update label. That is the stale-merge path.
    A supersession whose payload cosine is <= 0.5 instead takes the fork path,
    leaving the mature pre-update slot co-resident with the update.

Label definitions (see results/issue_failure_mode_blindness/SCHEMA.md):

  CONTRADICTORY_STRICT   vote wrong AND the raw cosine top-1 slot is a forked
                         contradictory slot.
  CONTRADICTORY_LENIENT  vote wrong AND any surviving top-k candidate is a
                         forked contradictory slot.
  STALE_STRICT           vote wrong AND the raw cosine top-1 slot is a stale
                         (superseded, still decoding to the pre-update label)
                         slot AND the vote elects that pre-update label.
  STALE_LENIENT          vote wrong AND any surviving top-k candidate is a
                         stale slot.

"top-1" follows the established per-probe convention (``top1_label`` in the
issue-#82 telemetry): the best raw cosine candidate before any floor masking.
On every emitted row this is also the maximum-weight voter (a row whose max
sim is floor-masked has no vote and is excluded; softmax weight is monotone
in sim), so strict = leading-support EXPOSURE — the strongest contributor to
the wrong vote is in the flagged set. STALE_STRICT additionally ties the
answer's identity to that slot (the vote elects its pre-update label);
CONTRADICTORY_STRICT does not require the elected class to equal the fork's
label. "Surviving top-k" = candidates with finite post-floor similarity (the
``n_surviving_votes`` set); lenient = top-k implication only, NOT causal
attribution — ``contra_vote_weight`` / ``stale_vote_weight`` quantify the
actual mass for dose-response analysis. Strict implies lenient by
construction.

No GPU results are produced by this file in PR-2a; the synthetic mode exists
so tests/test_failure_mode_probe.py can validate every mechanism hermetically
(CPU, seconds, no caches). The ``--vision`` wiring mirrors calibration_probe
and is first exercised in PR-2b on gentoo, after the mechanism tests pass.

Example (synthetic smoke, CPU):
    python benchmarks/failure_mode_probe.py --synthetic --arm contra \
        --rate 0.15 --epochs 6 --out /tmp/per_probe_injected.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from associative_core import ContinuousCAM  # noqa: E402
from dynamic_vigilance import (  # noqa: E402
    DynamicVigilance, RelativeVigilance, RetrievalFloorPolicy)
from benchmarks.calibration_probe import (  # noqa: E402
    ALL_COLS, CONTEXT_COLS, PER_PROBE_KEYS)
from benchmarks.probe_contraction import VisionDriftStream  # noqa: E402


# Injection outcomes (mechanical classification of each injected write).
FORKED = "forked"          # demotion fork: pre-write sim >= vigilance, payload disagreed
PLAIN_MISS = "plain-miss"  # below vigilance; new slot but NOT co-resident by the gate
ABSORBED = "absorbed"      # merged into an existing prototype (payload agreed)
DROPPED = "dropped"        # no slot obtained (capacity)

# Failure-mode precedence for the single `failure_mode` column. The boolean
# flag columns are NOT collapsed — analysis can ignore precedence entirely.
MODE_PRECEDENCE = ["CONTRADICTORY_STRICT", "STALE_STRICT",
                   "CONTRADICTORY_LENIENT", "STALE_LENIENT",
                   "BLENDED", "OTHER_WRONG"]

# Ground-truth write-event classes (PR-3b fork_events.csv; PR3_DESIGN.md §7).
# `initial` / `duplicate-rewrite` / `clean-rewrite` are protocol-certified
# non-conflict classes; `one-shot-ambiguous` is supersession whose single
# write the protocol certifies as observationally undecidable. payload-drift
# and key-drift are reserved in the design memo and NOT instantiable here.
EVENT_INITIAL = "initial"
EVENT_DUPLICATE = "duplicate-rewrite"   # vision: identical batch re-written
EVENT_CLEAN_REWRITE = "clean-rewrite"   # synthetic: fresh same-class samples
EVENT_CONTRADICTION = "contradiction"
EVENT_SUPERSESSION = "supersession"
EVENT_ONE_SHOT = "one-shot-ambiguous"

# fork_events.csv schema: write-time observables ONLY (label-free at the
# feature level) + the protocol's ground-truth event_class. No registry-
# derived failure labels may appear here (pinned by test). Recency is
# expressed as the incumbent's latest provenance sequence number
# (protocol-time write recency) because the engine's `last_seen` is
# wall-clock and would break byte-determinism.
EVENT_COLS = ["arm", "epoch", "event_class", "record_tag", "record_seq",
              "outcome", "pre_sim", "payload_cos_incumbent",
              "effective_vigilance", "incumbent_slot",
              "incumbent_hit_counts", "incumbent_last_write_seq",
              "incumbent_n_records", "owner_slot", "injected_label"]

# per_slot.csv schema: one row per occupied slot per probe epoch. Role flags
# are NOT mutually exclusive; `role` collapses them by the documented
# precedence (contra_fork > stale_superseded > merge_candidate >
# current_fork > clean). Ground-truth roles come from the registry;
# merge_candidate is observational (PR3_DESIGN.md §7).
SLOT_COLS = ["arm", "epoch", "slot", "decode", "hit_counts",
             "last_write_seq", "usage", "n_records", "is_contra_fork",
             "is_stale_superseded", "is_current_fork", "is_merge_candidate",
             "role"]

# topk.csv schema (PR-3c): the candidate composition of every deployed
# vote, one row per probe x top-k candidate. This is the logged basis for
# SHADOW governance (PR3_DESIGN.md §5 PR-3c): counterfactual readouts are
# recomputed offline from these rows; the deployed forward() is never
# altered. sim/weight are emitted at float32 round-trip precision ("%.9g")
# so the offline policy-`none` vote is bit-identical to deployment — the
# driver re-derives the vote from the emitted text at write time and
# raises on any mismatch (one-shot rows sit at exact 0.5/0.5 ties, so
# anything short of bit fidelity would silently flip elections).
TOPK_COLS = ["arm", "epoch", "probe_index", "rank", "slot", "sim",
             "surviving", "weight", "decode"]

NEW_COLS = ["arm", "injection_rate", "supersede_epoch", "probe_index",
            "top1_slot", "failure_mode",
            "contradictory_strict", "contradictory_lenient",
            "stale_strict", "stale_lenient",
            "n_contra_topk", "n_stale_topk",
            "contra_vote_weight", "stale_vote_weight"]
OUT_COLS = ALL_COLS + NEW_COLS


# ---------------------------------------------------------------------------
# PR-7 write-path governance seam.
#   step 1 (PR7_DESIGN.md §13.1) — add the seam, prove the boundary, change no
#     behavior: every action is a recorded no-op.
#   step 2 (PR7_DESIGN.md §13.2, §4) — implement the cheapest action,
#     ``annotate``: the NULL-ACTION FLOOR. It stamps a write-time merge_suspect
#     annotation on the supersession event and changes NOTHING at read time, so
#     every scored artifact stays byte-identical to the ungoverned baseline —
#     proving the harness itself adds no harm.
#   step 5 (PR7_DESIGN.md §13, §4) — implement the first ACTING arm, ``refuse``:
#     it skips the already-classified write-time merge_suspect (supersession)
#     write BEFORE it commits. Non-suspect writes (clean / contradiction) are
#     allowed unchanged, so none/annotate stay byte-identical; only the
#     supersession write path diverges. ``quarantine`` remains a recorded no-op
#     (allows every write). No read-time / slot-granularity trust, geometry
#     gate, or one-shot classification is implemented at any step (§12).
# ---------------------------------------------------------------------------
GOVERN_ALLOW = "allow"
GOVERN_REFUSE = "refuse-write"  # pre-write decision: skip this write (≠ action name)
GOVERN_ACTIONS = ("none", "annotate", "quarantine", "refuse")
# Actions whose behavior is implemented. Step 2 implements ``annotate`` (the
# null-action floor — commits the write exactly as baseline); step 5 implements
# ``refuse`` (skips the write-time merge_suspect/supersession write before it
# commits). ``quarantine`` remains a recorded no-op (it allows every write).
GOVERN_IMPLEMENTED_ACTIONS = ("none", "annotate", "refuse")
# The already-classified write-event class ``refuse`` acts on: the write-time
# merge_suspect candidate = the supersession (EMA-merge absorb) write. One-shot
# ambiguity stays observe-only and is NEVER refused (PR7_DESIGN.md §12).
GOVERN_REFUSE_EVENT_CLASS = EVENT_SUPERSESSION


class GovernanceHook:
    """Opt-in write-path governance seam — PR-7 (PR7_DESIGN.md §4).

    The PR-7 twin-run harness (PR7_DESIGN.md §5) needs ONE place where the
    *experimental* writer can act on an already-classified write event before
    it commits — annotate / quarantine / refuse it. This object is that place,
    and lives only in this experimental driver: it is **never** constructed or
    reached by the deployed ``forward()`` / ``learn_local`` retrieval path
    (PR7_DESIGN.md §4 — "where the action lives is the whole safety
    argument").

    :meth:`decide` returns a write decision for an already-classified event.
    ``none`` (baseline) and ``annotate`` (step 2) both return
    :data:`GOVERN_ALLOW`: the write is committed exactly as the baseline would.
    ``annotate`` is the NULL-ACTION FLOOR — it records a write-time
    merge_suspect annotation on the supersession (merge-suspect absorb) event
    but changes nothing the writer does, so every emitted/scored artifact is
    byte-identical to the ungoverned baseline (PR7_DESIGN.md §4: annotate "must
    cost nothing"). The annotation is write-path provenance only — NOT the
    record-granularity ledger PR-6 §3 deferred (path 2 stays closed), and the
    hook never reads, mutates, or imports engine retrieval state. ``quarantine``
    / ``refuse`` (the acting decisions) stay recorded no-ops here and are
    implemented in step 3; no read-time / slot-granularity trust, geometry gate,
    or one-shot classification is implemented at any step (PR7_DESIGN.md §12).
    """

    def __init__(self, action: str = "none"):
        if action not in GOVERN_ACTIONS:
            raise ValueError(
                f"--govern must be one of {GOVERN_ACTIONS}; got {action!r}")
        self.action = action
        self.events_seen = 0
        self.annotated_events = 0
        self.refused_events = 0
        self.outcome_counts: dict = {}

    @property
    def active(self) -> bool:
        """True iff a non-baseline action was requested. This only gates
        provenance emission so a baseline (``none``) run stays byte-identical to
        the pre-seam driver; ``annotate`` is active but, being the null-action
        floor, still emits byte-identical scored artifacts (only the summary's
        ``govern`` provenance block differs)."""
        return self.action != "none"

    @property
    def implemented(self) -> bool:
        """True iff this action's behavior is implemented (vs a recorded
        no-op held for a later step)."""
        return self.action in GOVERN_IMPLEMENTED_ACTIONS

    def allow_write(self, event_class: str) -> str:
        """Pre-write decision (PR-7 step 5) — consulted BEFORE the write commits.

        Returns :data:`GOVERN_REFUSE` ONLY for the ``refuse`` action on the
        write-time merge_suspect class (the supersession write); every other
        action (``none`` / ``annotate`` / ``quarantine``) and every other event
        class returns :data:`GOVERN_ALLOW`, so the baseline write path stays
        byte-for-byte identical. A refused write is recorded via
        :meth:`record_refusal`; ``decide`` is then never reached for it."""
        if self.action == "refuse" and event_class == GOVERN_REFUSE_EVENT_CLASS:
            return GOVERN_REFUSE
        return GOVERN_ALLOW

    def record_refusal(self, event_class: str, n: int) -> None:
        """Record ``n`` rows refused (skipped before commit) for one write event
        (PR-7 step 5). The refused write never reaches :meth:`decide`, so it is
        counted here: it still contributes to ``events_seen`` (writes the hook
        observed) and is tallied separately in ``refused_events``."""
        self.events_seen += n
        self.refused_events += n

    def decide(self, event_class: str, outcome: str) -> str:
        """Observe one already-classified, ALLOWED write event; return its write
        decision. Returns :data:`GOVERN_ALLOW` for every action (the pre-write
        refusal, if any, happened in :meth:`allow_write`, so a write reaching
        ``decide`` is always committed).

        ``annotate`` additionally records a write-time merge_suspect annotation
        on the supersession event — write-path provenance that changes nothing
        the writer does, so scored output stays byte-identical (the null-action
        floor, PR7_DESIGN.md §4)."""
        self.events_seen += 1
        self.outcome_counts[outcome] = self.outcome_counts.get(outcome, 0) + 1
        if self.action == "annotate" and event_class == EVENT_SUPERSESSION:
            self.annotated_events += 1
        return GOVERN_ALLOW

    def provenance(self) -> dict:
        """Write-path provenance for the run summary (emitted only when
        :attr:`active`, so ``none`` stays byte-identical to the old driver)."""
        step = {"annotate": "pr7-step2-annotate",
                "refuse": "pr7-step5-refuse"}.get(self.action, "pr7-step1-noop")
        prov = {
            "action": self.action,
            "step": step,
            "implemented": self.implemented,
            "implemented_actions": list(GOVERN_IMPLEMENTED_ACTIONS),
            "events_seen": self.events_seen,
            "outcome_counts": dict(sorted(self.outcome_counts.items())),
        }
        if self.action == "annotate":
            prov["annotated_events"] = self.annotated_events
            prov["annotation"] = (
                "write-time merge_suspect stamp on supersession; read-time "
                "no-op (null-action floor, PR7_DESIGN.md §4)")
        if self.action == "refuse":
            prov["refused_events"] = self.refused_events
            prov["refused_event_class"] = GOVERN_REFUSE_EVENT_CLASS
            prov["reason"] = (
                "write-time merge_suspect (supersession) writes skipped before "
                "commit; non-suspect writes (clean / contradiction) allowed "
                "unchanged; deployed read-time retrieval path untouched "
                "(PR7_DESIGN.md §4/§13).")
        return prov


@dataclass
class StaleGroup:
    """One supersession protocol instance: keys K written K→A, then K→B.

    ``phase1_ids`` are the record ids of every K→A write. After ``superseded``
    is set (phase 2 began), any occupied slot whose provenance intersects
    ``phase1_ids`` and whose payload still argmax-decodes to ``pre_label`` is a
    stale slot: mature pre-update support for a key region whose current
    ground truth is ``post_label``.
    """
    name: str
    pre_label: int
    post_label: int
    phase1_ids: set = field(default_factory=set)
    phase2_ids: set = field(default_factory=set)
    superseded: bool = False
    supersede_epoch: int = -1


class FailureModeRegistry:
    """Bookkeeping for injected writes and mechanically derived slot sets.

    The registry only ever READS the memory (buffers + provenance accessors);
    it never mutates engine state. All writes go through ``learn_local``.

    Invariant required for outcome classification: the driver stamps EVERY
    write with a record id, so after any call a freshly allocated slot is
    exactly a slot whose record set equals ``{id}``. ``stamp_clean_ids`` /
    ``inject_contradictions`` / the stale-phase helpers all uphold this.
    """

    def __init__(self):
        # record_id -> {"label": injected wrong label, "outcome": str}
        self.contra: dict = {}
        self.stale_groups: list[StaleGroup] = []
        self._seq = 0

    # -- record-id helpers -------------------------------------------------
    def next_ids(self, tag: str, n: int) -> list:
        ids = [(tag, self._seq + j) for j in range(n)]
        self._seq += n
        return ids

    # -- contradiction arm ---------------------------------------------------
    def inject_contradictions(self, mem: ContinuousCAM, queries: torch.Tensor,
                              true_labels: torch.Tensor, num_classes: int,
                              rate: float, rng: torch.Generator,
                              eligible: torch.Tensor | None = None,
                              events: list | None = None, epoch: int = -1,
                              hook: "GovernanceHook | None" = None):
        """Re-write ``rate`` of the batch rows with permuted-label payloads.

        Each injected row keeps its real query (key) but carries a one-hot
        target for a uniformly chosen WRONG class — the hallucinating-writer
        model. Writes go through the normal ``learn_local``; the outcome of
        each (forked / plain-miss / absorbed / dropped) is classified from the
        pre-write nearest sim, the effective vigilance, and provenance.

        ``eligible`` (optional bool mask over batch rows) restricts which rows
        may be injected — the mixed arm excludes the superseded group's
        pre-label rows so supersession ground truth on those keys stays pure.
        With ``eligible=None`` the selection is identical to the PR-2a/2b
        behavior (a permutation prefix). ``events`` (optional list) collects
        one write-event row per injection (ground-truth class
        ``contradiction``) with the same pre-write observables as
        ``logged_learn`` — PR-3b's fork_events.csv channel.
        """
        if mem.slot_records is None:
            raise RuntimeError(
                "FailureModeRegistry requires ContinuousCAM(track_provenance="
                "True): injection outcomes are classified from slot_records.")
        B = queries.size(0)
        n_inj = int(round(rate * B))
        if n_inj == 0:
            return []
        perm = torch.randperm(B, generator=rng)
        if eligible is not None:
            perm = perm[eligible[perm]]
        rows = perm[:min(n_inj, perm.numel())]
        inj_q = queries[rows]
        true = true_labels[rows]
        n_sel = rows.numel()
        # Uniform wrong label: shift by 1..C-1 (mod C) — never the true class.
        shift = torch.randint(1, num_classes, (n_sel,), generator=rng)
        wrong = (true + shift) % num_classes
        inj_y = F.one_hot(wrong, num_classes).float()

        ids = self.next_ids("contra", n_sel)
        pre = _pre_write_observables(mem, inj_q, inj_y, self)

        mem.learn_local(inj_q, inj_y, record_ids=ids)

        owners = _owner_slots(mem, ids)
        for j, rid in enumerate(ids):
            slot = owners.get(rid)
            if slot is None:
                outcome = DROPPED
            elif mem.records_for(slot) == {rid}:
                outcome = (FORKED
                           if float(pre["pre_sim"][j]) >= float(
                               pre["effective_vigilance"][j])
                           else PLAIN_MISS)
            else:
                outcome = ABSORBED
            self.contra[rid] = {"label": int(wrong[j]), "outcome": outcome}
            # PR-7 step-1 seam: route the already-classified write through
            # governance. The decision is a recorded no-op (always ALLOW), so
            # the write stands exactly as the baseline made it.
            if hook is not None:
                hook.decide(EVENT_CONTRADICTION, outcome)
            if events is not None:
                events.append(_event_row(
                    epoch, EVENT_CONTRADICTION, rid, outcome, pre, j,
                    owners.get(rid), injected_label=int(wrong[j])))
        return ids

    @staticmethod
    def _effective_vigilance(mem: ContinuousCAM, queries: torch.Tensor,
                             n: int) -> torch.Tensor:
        """Per-row vigilance threshold the NEXT learn_local call will apply.

        Mirrors learn_local's branch: the dynamic policy only runs when it is
        attached AND memory is occupied; otherwise the static ``vigilance``
        scalar applies. Recomputed read-only from the same buffers (no opt-in
        log attribute is attached, so engine telemetry is untouched).
        """
        if mem.dynamic_vigilance is not None and mem.occupied.any():
            valid_idx = mem.occupied.nonzero(as_tuple=True)[0]
            q_norm = F.normalize(queries.float(), dim=-1)
            sims_full = q_norm @ mem._keys_norm[valid_idx].float().T
            proto_labels = mem.effective_slot_labels(valid_idx)
            v_eff, _ = mem.dynamic_vigilance.compute(sims_full, proto_labels)
            return v_eff
        return torch.full((n,), float(mem.vigilance))

    def contra_fork_slots(self, mem: ContinuousCAM) -> set:
        """Occupied slots formed by a FORKED contradictory write that still
        decode to the injected (wrong) label. Recomputed from live provenance
        each call, so eviction/reuse (which replaces a slot's record set) and
        any later payload drift invalidate entries automatically."""
        out = set()
        fork_ids = {rid for rid, meta in self.contra.items()
                    if meta["outcome"] == FORKED}
        if not fork_ids:
            return out
        occ = mem.occupied.nonzero(as_tuple=True)[0].tolist()
        for slot in occ:
            recs = mem.records_for(slot)
            hit = recs & fork_ids
            if not hit:
                continue
            decoded = int(mem.values[slot].float().argmax().item())
            if any(self.contra[rid]["label"] == decoded for rid in hit):
                out.add(slot)
        return out

    # -- stale arm (supersession protocol) ---------------------------------
    def new_stale_group(self, name: str, pre_label: int,
                        post_label: int) -> StaleGroup:
        g = StaleGroup(name=name, pre_label=pre_label, post_label=post_label)
        self.stale_groups.append(g)
        return g

    def write_phase1(self, mem: ContinuousCAM, group: StaleGroup,
                     queries: torch.Tensor, targets: torch.Tensor):
        """Phase 1: write K→A through the normal learn path, tagging ids."""
        ids = self.next_ids(f"stale1:{group.name}", queries.size(0))
        group.phase1_ids.update(ids)
        mem.learn_local(queries, targets, record_ids=ids)
        return ids

    def supersede(self, group: StaleGroup, epoch: int):
        """Mark the group's ground truth as flipped (phase 2 begins)."""
        group.superseded = True
        group.supersede_epoch = epoch

    def write_phase2(self, mem: ContinuousCAM, group: StaleGroup,
                     queries: torch.Tensor, targets: torch.Tensor):
        """Phase 2: write K→B through the normal learn path (post-supersession).

        Whether this merges (payload cosine > 0.5: EMA-freeze stale) or forks
        (<= 0.5: co-resident stale) is decided by the engine, not the driver.
        """
        if not group.superseded:
            raise RuntimeError(f"group {group.name}: call supersede() before "
                               "write_phase2 — stale labels are undefined "
                               "while A is still current ground truth.")
        ids = self.next_ids(f"stale2:{group.name}", queries.size(0))
        group.phase2_ids.update(ids)
        mem.learn_local(queries, targets, record_ids=ids)
        return ids

    def stale_slots(self, mem: ContinuousCAM) -> set:
        """Occupied slots carrying mature pre-update support: provenance
        intersects a superseded group's phase-1 ids AND the payload still
        argmax-decodes to that group's pre-update label. Recomputed live, so
        a slot that absorbed enough updates to flip its decode — or was
        evicted and reused — drops out automatically."""
        out = set()
        groups = [g for g in self.stale_groups if g.superseded]
        if not groups:
            return out
        occ = mem.occupied.nonzero(as_tuple=True)[0].tolist()
        for slot in occ:
            recs = mem.records_for(slot)
            if not recs:
                continue
            decoded = int(mem.values[slot].float().argmax().item())
            for g in groups:
                if decoded == g.pre_label and (recs & g.phase1_ids):
                    out.add(slot)
                    break
        return out


def _owner_slots(mem: ContinuousCAM, ids: list) -> dict:
    """Map each record id to the slot whose provenance contains it (or None).

    Within a single learn_local call every id lands in at most one slot
    (misses allocate distinct slots; hits union into pre-batch slots), so the
    first match is the only match when called immediately after the write.
    """
    wanted = set(ids)
    found: dict = {}
    occ = mem.occupied.nonzero(as_tuple=True)[0].tolist()
    for slot in occ:
        if not wanted:
            break
        hit = mem.records_for(slot) & wanted
        for rid in hit:
            found[rid] = slot
        wanted -= hit
    return {rid: found.get(rid) for rid in ids}


# ---------------------------------------------------------------------------
# PR-3b: write-event log (fork_events.csv) and per-slot table (per_slot.csv).
# Both are READ-ONLY observers — every write still goes through learn_local.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _pre_write_observables(mem: ContinuousCAM, queries: torch.Tensor,
                           targets: torch.Tensor,
                           registry: "FailureModeRegistry") -> dict:
    """Per-row pre-write state vs the PRE-BATCH memory (the same convention
    as the PR-2a outcome classification): nearest slot/sim, the vigilance
    threshold the call will apply, the payload cosine between the incoming
    target and the incumbent's stored value, and incumbent maturity/recency/
    lineage stats. Within-batch interactions are NOT captured."""
    n = queries.size(0)
    if not mem.occupied.any():
        return {"pre_sim": [float("nan")] * n,
                "payload_cos": [float("nan")] * n,
                "effective_vigilance": [float(mem.vigilance)] * n,
                "incumbent_slot": [-1] * n,
                "incumbent_hits": [0] * n,
                "incumbent_last_write_seq": [-1] * n,
                "incumbent_n_records": [0] * n}
    pre_slots, pre_sims = mem._get_nearest_batch(queries)
    thresholds = registry._effective_vigilance(mem, queries, n)
    inc_vals = F.normalize(mem.values[pre_slots].float(), dim=-1)
    pay = F.normalize(targets.float(), dim=-1)
    pcos = (inc_vals * pay).sum(dim=-1)
    return {
        "pre_sim": [float(s) for s in pre_sims],
        "payload_cos": [float(c) for c in pcos],
        "effective_vigilance": [float(t) for t in thresholds],
        "incumbent_slot": [int(s) for s in pre_slots],
        "incumbent_hits": [int(mem.hit_counts[int(s)]) for s in pre_slots],
        "incumbent_last_write_seq": [_last_write_seq(mem, int(s))
                                     for s in pre_slots],
        "incumbent_n_records": [len(mem.records_for(int(s)))
                                for s in pre_slots],
    }


def _last_write_seq(mem: ContinuousCAM, slot: int) -> int:
    """Protocol-time write recency: the latest provenance sequence number in
    the slot's record set (every write is stamped, so this is the slot's
    last write). Used instead of the engine's wall-clock ``last_seen`` so
    the side tables stay byte-deterministic; the audit pins that both are
    write-path-only."""
    recs = mem.records_for(slot)
    return max((seq for _tag, seq in recs), default=-1)


def _event_row(epoch: int, event_class: str, rid, outcome: str, pre: dict,
               j: int, owner, injected_label: int = -1) -> dict:
    tag, seq = rid
    return {"epoch": epoch, "event_class": event_class,
            "record_tag": tag, "record_seq": seq, "outcome": outcome,
            "pre_sim": round(pre["pre_sim"][j], 6),
            "payload_cos_incumbent": round(pre["payload_cos"][j], 6),
            "effective_vigilance": round(pre["effective_vigilance"][j], 6),
            "incumbent_slot": pre["incumbent_slot"][j],
            "incumbent_hit_counts": pre["incumbent_hits"][j],
            "incumbent_last_write_seq": pre["incumbent_last_write_seq"][j],
            "incumbent_n_records": pre["incumbent_n_records"][j],
            "owner_slot": -1 if owner is None else int(owner),
            "injected_label": injected_label}


def logged_learn(mem: ContinuousCAM, registry: FailureModeRegistry,
                 queries: torch.Tensor, targets: torch.Tensor,
                 epoch: int, event_class: str, events: list | None,
                 write_fn, hook: "GovernanceHook | None" = None) -> list:
    """Wrap one write call with pre-write observables and outcome rows.

    ``write_fn(ids_or_none)`` performs the actual write and returns the
    record ids — either by consuming pre-generated ids (clean writes) or by
    generating its own (the registry's phase helpers). Outcome
    classification mirrors inject_contradictions: a slot whose record set is
    exactly {rid} is fresh (forked above the threshold, plain-miss below);
    anything else absorbed; no owner = dropped.

    ``hook`` (PR-7 step 1) routes each already-classified write through the
    governance seam; the step-1 decision is a recorded no-op (always ALLOW),
    so the write stands exactly as made."""
    pre = _pre_write_observables(mem, queries, targets, registry)
    # PR-7 step 5: pre-write refuse decision (consulted BEFORE the write
    # commits). For none/annotate/quarantine this is always ALLOW, so the path
    # below is byte-identical to the pre-refuse driver; only `refuse` on the
    # merge_suspect (supersession) class skips the write entirely. A skipped
    # write commits nothing and emits no fork-event row (no write occurred); it
    # is recorded only in governance provenance.
    if hook is not None and hook.allow_write(event_class) == GOVERN_REFUSE:
        hook.record_refusal(event_class, queries.size(0))
        return []
    ids = write_fn()
    if events is None and hook is None:
        return ids
    owners = _owner_slots(mem, ids)
    for j, rid in enumerate(ids):
        slot = owners.get(rid)
        if slot is None:
            outcome = DROPPED
        elif mem.records_for(slot) == {rid}:
            outcome = (FORKED
                       if pre["pre_sim"][j] == pre["pre_sim"][j]  # not NaN
                       and pre["pre_sim"][j] >= pre["effective_vigilance"][j]
                       else PLAIN_MISS)
        else:
            outcome = ABSORBED
        if hook is not None:
            hook.decide(event_class, outcome)
        if events is not None:
            events.append(_event_row(epoch, event_class, rid, outcome, pre, j,
                                     slot))
    return ids


@torch.no_grad()
def slot_table_rows(mem: ContinuousCAM, registry: FailureModeRegistry,
                    epoch: int) -> list[dict]:
    """One row per occupied slot at this probe epoch: engine state (decode,
    maturity, recency, usage, lineage size) + ground-truth role flags from
    the registry, plus the observational merge_candidate flag (provenance
    spans both phases of one group — only instantiable on the merge path)."""
    contra_slots = registry.contra_fork_slots(mem)
    stale_slots = registry.stale_slots(mem)
    groups = [g for g in registry.stale_groups if g.superseded]
    rows = []
    for slot in mem.occupied.nonzero(as_tuple=True)[0].tolist():
        recs = mem.records_for(slot)
        decode = int(mem.values[slot].float().argmax())
        is_contra = slot in contra_slots
        is_stale = slot in stale_slots
        is_current = any((recs & g.phase2_ids) and decode == g.post_label
                         for g in groups)
        is_merge = any((recs & g.phase1_ids) and (recs & g.phase2_ids)
                       for g in groups)
        role = ("contra_fork" if is_contra
                else "stale_superseded" if is_stale
                else "merge_candidate" if is_merge
                else "current_fork" if is_current
                else "clean")
        rows.append({"epoch": epoch, "slot": slot, "decode": decode,
                     "hit_counts": int(mem.hit_counts[slot]),
                     "last_write_seq": _last_write_seq(mem, slot),
                     "usage": float(mem.usage[slot]),
                     "n_records": len(recs),
                     "is_contra_fork": int(is_contra),
                     "is_stale_superseded": int(is_stale),
                     "is_current_fork": int(is_current),
                     "is_merge_candidate": int(is_merge),
                     "role": role})
    return rows


def _write_side_table(path: Path, cols: list[str], rows: list[dict],
                      arm_label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({"arm": arm_label, **r})


def _f32_text(x: float) -> str:
    """float32 round-trip text (9 significant decimal digits)."""
    return f"{float(x):.9g}"


def vote_pred_from_candidates(weights, decodes, value_dim: int) -> int:
    """Recompute the deployed vote from one probe's logged top-k rows.

    Accumulates float32 mass in rank order (the replicated vote's
    scatter_add order) and argmaxes with first-index tie-break — the exact
    aggregation ``label_probes`` performs, so a policy-`none` shadow
    readout built on topk.csv is bit-identical to deployment. Shared by
    the driver's write-time fidelity check and the PR-3c governance
    analyzer; any divergence raises there instead of mis-electing.
    """
    mass = torch.zeros(int(value_dim), dtype=torch.float32)
    for w, d in zip(weights, decodes):
        mass[int(d)] += torch.tensor(float(w), dtype=torch.float32)
    return int(torch.argmax(mass))


def topk_table_rows(labeled: dict, epoch: int) -> list[dict]:
    """Per-candidate rows for ``<out>.topk.csv`` (PR-3c shadow basis).

    Emits every replicated top-k candidate (including non-surviving ones,
    whose weight is exactly 0) and verifies, per probe, that the vote
    re-derived from the EMITTED TEXT equals the deployed vote — the
    shadow-readout fidelity requirement (PR3_DESIGN.md §10), enforced at
    generation time rather than discovered at analysis time.
    """
    V = labeled["n_rows"]
    if V == 0:
        return []
    pp = labeled["per_probe"]
    k = labeled["topk_slots"].shape[1]
    rows = []
    for i in range(V):
        ws, ds = [], []
        for r in range(k):
            row = {
                "epoch": epoch,
                "probe_index": int(labeled["probe_index"][i]),
                "rank": r,
                "slot": int(labeled["topk_slots"][i][r]),
                "sim": _f32_text(labeled["topk_sims"][i][r]),
                "surviving": int(labeled["topk_surviving"][i][r]),
                "weight": _f32_text(labeled["topk_weights"][i][r]),
                "decode": int(labeled["topk_decode"][i][r]),
            }
            rows.append(row)
            ws.append(row["weight"])
            ds.append(row["decode"])
        shadow = vote_pred_from_candidates(ws, ds, labeled["value_dim"])
        if shadow != int(pp["vote_pred_label"][i]):
            raise RuntimeError(
                f"topk shadow-vote mismatch at epoch {epoch} probe "
                f"{int(labeled['probe_index'][i])}: offline {shadow} vs "
                f"deployed {int(pp['vote_pred_label'][i])} — logged "
                f"precision is insufficient for shadow governance.")
    return rows


def resolve_fork_outcome(epoch_log: list[dict]) -> str:
    """Observational fork-resolution label for a run's supersession group,
    from the per-epoch superseded-key election counts (PR3_DESIGN.md §7:
    natural-history outcome, never a classifier target). Tie detection uses
    the median stale vote mass on superseded keys at the final epoch."""
    post = [e for e in epoch_log if e.get("superseded")]
    if not post or post[-1].get("superseded_key_probes", 0) == 0:
        return "unresolved"
    last = post[-1]
    n = last["superseded_key_probes"]
    frac_new = last["updated_selected"] / n
    frac_old = last["stale_selected"] / n
    w = last.get("stale_weight_median_superseded")
    if w is not None and 0.45 <= w <= 0.55:
        return "persistent-tie"
    if frac_new >= 0.9:
        return "later-dominates"
    if frac_old >= 0.9:
        return "old-persists"
    if frac_new > 0 and frac_old > 0:
        return "persistent-split"
    return "mixed"


@torch.no_grad()
def label_probes(mem: ContinuousCAM, probe_queries: torch.Tensor,
                 probe_labels: torch.Tensor, registry: FailureModeRegistry,
                 blend_eps: float = 0.10) -> dict:
    """Per-probe telemetry + mechanical failure-mode labels.

    Predictors and evaluation labels come UNCHANGED from
    ``probe_cross_class_similarity(return_per_probe=True)`` (the issue-#82
    telemetry that all existing analyzers consume). This function adds the
    slot-composition columns the probe does not expose (top-1 slot, surviving
    top-k slots) by replicating the probe's candidate selection read-only, and
    derives the four failure-mode flags from the registry's slot sets.

    Alignment between the replicated rows and the probe's rows is VERIFIED at
    runtime (same row count, top1_sim/margin allclose, vote_pred equality), so
    silent drift between the two computations raises instead of mislabeling.

    Scope guard: NSTP and support-truncation change candidate survival inside
    the probe; replicating them here is out of scope for PR-2a, so both must
    be detached (they are in every calibration-sequence configuration).
    """
    if mem.nstp is not None or mem.retrieval_truncation_policy is not None:
        raise RuntimeError(
            "label_probes replicates the probe's candidate selection only for "
            "the calibration-sequence configuration (no NSTP, no truncation "
            "policy). Detach them or extend the replication first.")
    if mem.slot_records is None:
        raise RuntimeError("label_probes requires track_provenance=True.")

    out = mem.probe_cross_class_similarity(
        probe_queries, probe_labels, blend_eps=blend_eps,
        return_per_probe=True)
    pp = out.get("per_probe")
    if pp is None or pp["true_label"].numel() == 0:
        return {"per_probe": pp, "aggregates": out, "n_rows": 0}

    # --- Replicate the probe's candidate selection (read-only) ------------
    device = mem.keys.device
    q = F.normalize(probe_queries.to(device).float(), dim=-1)
    labels = probe_labels.to(device).long()
    valid_idx = mem.occupied.nonzero(as_tuple=True)[0]
    keys_occ = mem._keys_norm[valid_idx].float()
    proto_labels = mem.effective_slot_labels(valid_idx)

    sims = q @ keys_occ.T
    same_class = proto_labels.unsqueeze(0) == labels.unsqueeze(1)
    very_low = -1e9
    sim_other, _ = sims.masked_fill(same_class, very_low).max(dim=1)
    sim_same, _ = sims.masked_fill(~same_class, very_low).max(dim=1)
    valid = (sim_other > very_low / 2) & (sim_same > very_low / 2)

    k = min(mem.inference_k, keys_occ.size(0))
    topk_sims, topk_locs = sims[valid].topk(k, dim=1)
    top1_raw = topk_sims[:, 0].clone()
    masked = topk_sims.clone()
    sim_floor = mem._active_sim_floor()
    if sim_floor > 0.0:
        masked = masked.masked_fill(masked < sim_floor, -float("inf"))
    row_has_vote = torch.isfinite(masked).any(dim=1)
    safe = masked.masked_fill(~row_has_vote.unsqueeze(1), 0.0)
    weights = F.softmax(safe / mem.inference_temp, dim=-1)
    surviving = torch.isfinite(masked)

    vote_rows = row_has_vote.nonzero(as_tuple=True)[0]
    probe_index = valid.nonzero(as_tuple=True)[0][vote_rows]
    topk_slots = valid_idx[topk_locs[vote_rows]]               # (V, k)
    w_v = weights[vote_rows]                                    # (V, k)
    surv_v = surviving[vote_rows]                               # (V, k)
    top1_slot = topk_slots[:, 0]                                # (V,)

    # --- Verify alignment against the probe's authoritative rows ----------
    V = pp["true_label"].numel()
    if vote_rows.numel() != V:
        raise RuntimeError(
            f"row reconstruction mismatch: replicated {vote_rows.numel()} "
            f"voting rows, probe emitted {V} — candidate selection drifted.")
    if not torch.allclose(top1_raw[vote_rows].cpu(), pp["top1_sim"],
                          atol=1e-5):
        raise RuntimeError("top1_sim mismatch vs per_probe telemetry — "
                           "candidate selection drifted.")
    class_mass = torch.zeros(V, mem.value_dim, device=w_v.device,
                             dtype=w_v.dtype)
    class_mass.scatter_add_(1, proto_labels[topk_locs[vote_rows]], w_v)
    vote_pred = class_mass.argmax(dim=1)
    if not torch.equal(vote_pred.cpu(), pp["vote_pred_label"]):
        raise RuntimeError("vote_pred mismatch vs per_probe telemetry — "
                           "vote replication drifted.")

    # --- Failure-mode flags from the registry's mechanical slot sets ------
    contra_slots = registry.contra_fork_slots(mem)
    stale_slots = registry.stale_slots(mem)
    slot_decode = mem.values.float().argmax(dim=-1)

    def _in_set(t: torch.Tensor, s: set) -> torch.Tensor:
        if not s:
            return torch.zeros(t.shape, dtype=torch.bool)
        ref = torch.tensor(sorted(s), dtype=t.dtype)
        return torch.isin(t.cpu(), ref)

    contra_topk = _in_set(topk_slots, contra_slots) & surv_v.cpu()  # (V, k)
    stale_topk = _in_set(topk_slots, stale_slots) & surv_v.cpu()    # (V, k)
    wrong = pp["vote_correct"] == 0                                  # (V,)

    top1_contra = _in_set(top1_slot, contra_slots)
    top1_stale = _in_set(top1_slot, stale_slots)
    top1_decode = slot_decode[top1_slot].cpu()

    c_strict = wrong.bool() & top1_contra
    c_lenient = wrong.bool() & contra_topk.any(dim=1)
    s_strict = (wrong.bool() & top1_stale
                & (pp["vote_pred_label"] == top1_decode))
    s_lenient = wrong.bool() & stale_topk.any(dim=1)
    blended = wrong.bool() & (pp["is_blended"] == 1)

    w_cpu = w_v.cpu()
    flags = {"contradictory_strict": c_strict, "contradictory_lenient": c_lenient,
             "stale_strict": s_strict, "stale_lenient": s_lenient}
    failure_mode = []
    for i in range(V):
        if not bool(wrong[i]):
            failure_mode.append("CORRECT")
            continue
        row = {"CONTRADICTORY_STRICT": bool(c_strict[i]),
               "STALE_STRICT": bool(s_strict[i]),
               "CONTRADICTORY_LENIENT": bool(c_lenient[i]),
               "STALE_LENIENT": bool(s_lenient[i]),
               "BLENDED": bool(blended[i]),
               "OTHER_WRONG": True}
        failure_mode.append(next(m for m in MODE_PRECEDENCE if row[m]))

    return {
        "per_probe": pp,
        "aggregates": out,
        "n_rows": V,
        "probe_index": probe_index.cpu(),
        "top1_slot": top1_slot.cpu(),
        "failure_mode": failure_mode,
        **{name: t.long() for name, t in flags.items()},
        "n_contra_topk": contra_topk.sum(dim=1).long(),
        "n_stale_topk": stale_topk.sum(dim=1).long(),
        "contra_vote_weight": (w_cpu * contra_topk.float()).sum(dim=1),
        "stale_vote_weight": (w_cpu * stale_topk.float()).sum(dim=1),
        # PR-3c shadow basis: the full replicated candidate composition,
        # rank-ordered, with the decode labels the vote actually used.
        "topk_slots": topk_slots.cpu(),
        "topk_sims": topk_sims[vote_rows].cpu(),
        "topk_surviving": surv_v.cpu(),
        "topk_weights": w_cpu,
        "topk_decode": proto_labels[topk_locs[vote_rows]].cpu(),
        "value_dim": mem.value_dim,
    }


def append_rows(writer: csv.DictWriter, ctx: dict, labeled: dict) -> int:
    """Write one CSV row per labeled voting probe (schema: OUT_COLS)."""
    V = labeled["n_rows"]
    if V == 0:
        return 0
    pp = labeled["per_probe"]
    for i in range(V):
        row = dict(ctx)
        for key in PER_PROBE_KEYS:
            row[key] = pp[key][i].item()
        row["probe_index"] = int(labeled["probe_index"][i])
        row["top1_slot"] = int(labeled["top1_slot"][i])
        row["failure_mode"] = labeled["failure_mode"][i]
        for f in ("contradictory_strict", "contradictory_lenient",
                  "stale_strict", "stale_lenient"):
            row[f] = int(labeled[f][i])
        row["n_contra_topk"] = int(labeled["n_contra_topk"][i])
        row["n_stale_topk"] = int(labeled["n_stale_topk"][i])
        row["contra_vote_weight"] = round(float(labeled["contra_vote_weight"][i]), 6)
        row["stale_vote_weight"] = round(float(labeled["stale_vote_weight"][i]), 6)
        writer.writerow(row)
    return V


def soft_supersession_targets(n: int, pre_label: int, post_label: int,
                              num_classes: int) -> torch.Tensor:
    """Non-orthogonal phase-2 targets for the merge-path stale arm:
    0.6*e_A + 0.8*e_B (unit norm). cosine to the stored A payload is 0.6 >
    0.5, so the engine's bipartite check takes the HIT path and EMA-merges
    into the mature slot; argmax is B, so ground truth still flips."""
    y = (0.6 * F.one_hot(torch.full((n,), pre_label), num_classes)
         + 0.8 * F.one_hot(torch.full((n,), post_label), num_classes))
    return y.float()


def run_epoch_writes(mem: ContinuousCAM, registry: FailureModeRegistry,
                     group: StaleGroup | None, arm: str, epoch: int,
                     q: torch.Tensor, y: torch.Tensor, lab: torch.Tensor, *,
                     rate: float, num_classes: int, rng: torch.Generator,
                     supersede_epoch: int, one_shot: bool = False,
                     key_jitter: float = 0.0,
                     jitter_gen: torch.Generator | None = None,
                     payload_mode: str = "onehot",
                     events: list | None = None,
                     clean_rewrite_class: str = EVENT_DUPLICATE,
                     hook: "GovernanceHook | None" = None):
    """All of one epoch's writes for any arm (shared by the synthetic and
    vision runners; behavior for the PR-2 arms is unchanged):

      clean  — every row through learn_local.
      contra — clean writes, then permuted-label injections at ``rate``.
      stale  — group rows through the supersession protocol (phase 1 before
               ``supersede_epoch``; phase 2 at/after — exactly once with
               ``one_shot``, with keys jittered by ``key_jitter``, with
               merge-path soft targets under ``payload_mode='soft'``);
               remaining rows clean.
      mixed  — stale protocol AND contra injections in the same memory;
               the group's pre-label rows are EXCLUDED from injection so
               supersession ground truth on those keys stays pure.
    """
    clean_cls = EVENT_INITIAL if epoch == 0 else clean_rewrite_class
    if group is not None:
        g0 = lab == group.pre_label
        if epoch < supersede_epoch:
            logged_learn(
                mem, registry, q[g0], y[g0], epoch, clean_cls, events,
                lambda: registry.write_phase1(mem, group, q[g0], y[g0]),
                hook=hook)
        else:
            first = not group.superseded
            if first:
                registry.supersede(group, epoch)
            if first or not one_shot:
                q2 = q[g0]
                if key_jitter > 0.0:
                    # direction-normalized noise: key_jitter IS the L2
                    # perturbation magnitude, independent of dim (raw randn
                    # scales with sqrt(dim) and would obliterate the key).
                    noise = F.normalize(
                        torch.randn(q2.shape, generator=jitter_gen), dim=-1)
                    q2 = F.normalize(q2 + key_jitter * noise, dim=-1)
                n2 = int(g0.sum())
                if payload_mode == "soft":
                    y2 = soft_supersession_targets(
                        n2, group.pre_label, group.post_label, num_classes)
                else:
                    y2 = F.one_hot(torch.full((n2,), group.post_label),
                                   num_classes).float()
                ev = EVENT_ONE_SHOT if one_shot else EVENT_SUPERSESSION
                logged_learn(
                    mem, registry, q2, y2, epoch, ev, events,
                    lambda: registry.write_phase2(mem, group, q2, y2),
                    hook=hook)
        rest = ~g0
        q_c, y_c = q[rest], y[rest]
    else:
        q_c, y_c = q, y

    def _clean_write():
        ids = registry.next_ids("clean", q_c.size(0))
        mem.learn_local(q_c, y_c, record_ids=ids)
        return ids

    logged_learn(mem, registry, q_c, y_c, epoch, clean_cls, events,
                 _clean_write, hook=hook)

    if arm in ("contra", "mixed"):
        eligible = (lab != group.pre_label) if group is not None else None
        registry.inject_contradictions(mem, q, lab, num_classes, rate, rng,
                                       eligible=eligible, events=events,
                                       epoch=epoch, hook=hook)


def arm_label_for(arm: str, one_shot: bool, key_jitter: float,
                  payload_mode: str) -> str:
    """CSV `arm` value encoding the protocol variant, so multi-file analyses
    group correctly (e.g. stale-oneshot, stale-jitter0.05, stale-soft)."""
    label = arm
    if one_shot:
        label += "-oneshot"
    if key_jitter > 0.0:
        label += f"-jitter{key_jitter:g}"
    if payload_mode == "soft":
        label += "-soft"
    return label


# ---------------------------------------------------------------------------
# Synthetic harness — used by the hermetic tests and the --synthetic CLI mode.
# ---------------------------------------------------------------------------
def build_synthetic_stream(num_classes=4, dim=32, n_per=24, held_per=16,
                           noise=0.20, epochs=6, seed=0):
    """Clustered unit vectors: (epoch batches, held-out queries/labels)."""
    g = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(num_classes, dim, generator=g), dim=-1)
    batches = []
    for _ in range(epochs):
        lab = torch.arange(num_classes).repeat_interleave(n_per)
        q = F.normalize(centers[lab] + noise * torch.randn(
            lab.numel(), dim, generator=g), dim=-1)
        y = F.one_hot(lab, num_classes).float()
        batches.append((q, y, lab))
    hlab = torch.arange(num_classes).repeat_interleave(held_per)
    hq = F.normalize(centers[hlab] + noise * torch.randn(
        hlab.numel(), dim, generator=g), dim=-1)
    return batches, hq, hlab


def run_synthetic(arm: str, rate: float, epochs: int, supersede_epoch: int,
                  out_path: Path, seed: int = 0, num_classes: int = 4,
                  dim: int = 32, one_shot: bool = False,
                  key_jitter: float = 0.0,
                  payload_mode: str = "onehot", govern: str = "none") -> int:
    """End-to-end synthetic run for one arm; returns rows written.

    clean  — no injections (negative control: no contra/stale labels may fire)
    contra — permuted-label injections at ``rate`` per epoch
    stale  — class 0 superseded by ``num_classes - 1`` at ``supersede_epoch``;
             held-out class-0 ground truth flips at that epoch
    mixed  — stale protocol + contra injections (pre-label rows excluded)

    PR-3b variants (stale/mixed): ``one_shot`` writes phase 2 exactly once;
    ``key_jitter`` perturbs phase-2 keys; ``payload_mode='soft'`` takes the
    EMA-merge path. Side tables ``<out>.per_slot.csv`` and
    ``<out>.fork_events.csv`` are always written next to the main CSV.
    """
    batches, hq, hlab = build_synthetic_stream(
        num_classes=num_classes, dim=dim, epochs=epochs, seed=seed)
    rng = torch.Generator().manual_seed(seed + 1)
    jitter_gen = torch.Generator().manual_seed(seed + 2)
    registry = FailureModeRegistry()
    hook = GovernanceHook(govern)  # PR-7 write-path seam (annotate=floor; q/r no-op)
    mem = ContinuousCAM(key_dim=dim, value_dim=num_classes, max_entries=1024,
                        dynamic_vigilance=DynamicVigilance(),
                        retrieval_floor_policy=RetrievalFloorPolicy(),
                        track_provenance=True)
    group = None
    if arm in ("stale", "mixed"):
        group = registry.new_stale_group("synthetic-class0",
                                         pre_label=0,
                                         post_label=num_classes - 1)
    label = arm_label_for(arm, one_shot, key_jitter, payload_mode)
    events: list = []
    slot_rows: list = []
    topk_rows: list = []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        writer.writeheader()
        for epoch, (q, y, lab) in enumerate(batches):
            mem.reset_dynamic_vigilance_stats()
            truth = hlab.clone()
            run_epoch_writes(
                mem, registry, group, arm, epoch, q, y, lab,
                rate=rate, num_classes=num_classes, rng=rng,
                supersede_epoch=supersede_epoch, one_shot=one_shot,
                key_jitter=key_jitter, jitter_gen=jitter_gen,
                payload_mode=payload_mode, events=events,
                clean_rewrite_class=EVENT_CLEAN_REWRITE, hook=hook)
            if group is not None and group.superseded:
                truth = torch.where(hlab == group.pre_label,
                                    torch.full_like(hlab, group.post_label),
                                    hlab)

            stats = mem.get_stats()
            acc = (mem.forward(hq).argmax(dim=-1) == truth).float().mean()
            labeled = label_probes(mem, hq, truth, registry)
            ctx = {
                "vigilance_policy": "margin",
                "retrieval_floor_policy": "live-delta",
                "epoch": epoch, "contraction": 0.0,
                "rho_probe": round(labeled["aggregates"].get(
                    "mean_cross_class_sim", float("nan")), 6),
                "within_probe": round(labeled["aggregates"].get(
                    "mean_within_class_sim", float("nan")), 6),
                "offclass_weight_mean": round(labeled["aggregates"].get(
                    "mean_offclass_weight", float("nan")), 6),
                "acc_epoch": round(float(acc), 6),
                "sim_floor_active": round(float(stats.get(
                    "sim_floor_active", float("nan"))), 6),
                "floor_delta_ema": round(float(stats.get(
                    "floor_delta_ema", float("nan"))), 6),
                "arm": label,
                "injection_rate": rate if arm in ("contra", "mixed") else 0.0,
                "supersede_epoch": supersede_epoch if group is not None else -1,
            }
            total += append_rows(writer, ctx, labeled)
            slot_rows += slot_table_rows(mem, registry, epoch)
            topk_rows += topk_table_rows(labeled, epoch)
    _write_side_table(out_path.with_suffix(".per_slot.csv"),
                      SLOT_COLS, slot_rows, label)
    _write_side_table(out_path.with_suffix(".fork_events.csv"),
                      EVENT_COLS, events, label)
    _write_side_table(out_path.with_suffix(".topk.csv"),
                      TOPK_COLS, topk_rows, label)
    return total


# ---------------------------------------------------------------------------
# Cache-backed vision arm (PR-2b) — first exercised on gentoo.
# ---------------------------------------------------------------------------
def run_vision(arm: str, rate: float, epochs: int, out_path: Path, *,
               cache_path: str | None = None,
               classes: list[int] = (0, 8, 19, 33), attractor_class: int = 71,
               samples_per_class: int = 32, held_out_per_class: int = 64,
               contraction: float = 0.0, max_entries: int = 4096,
               blend_eps: float = 0.10, seed: int = 0,
               supersede_epoch: int = 6, one_shot: bool = False,
               key_jitter: float = 0.0, payload_mode: str = "onehot",
               govern: str = "none"):
    """Cache-backed clean/contra/stale arm over the DINOv2 ViT-L/14 manifold.

    PR-2b/PR-2c protocol. The stream is the verified ``vitl14_cifar100_train``
    cache through ``VisionDriftStream`` (the exact #87 A1 stream), held at a
    FIXED contraction (default 0.0, i.e. stationary). Stationarity is the
    point: the calibration sequence already characterizes confidence under
    drift (BLENDED); these arms ask whether an injected failure is visible
    to retrieval-time confidence on an otherwise-healthy manifold, so drift
    is deliberately excluded rather than crossed with injection.

    Per epoch: write the (same) train batch through ``learn_local``, then —
    contra arm only — re-write ``rate`` of the batch rows with permuted-label
    payloads via ``inject_contradictions`` (the PR-2a hallucinating-writer
    model). Forks therefore accumulate monotonically across epochs, giving a
    within-run dose ramp. Held-out probes are never written.

    Stale arm (PR-2c): the PR-2a supersession protocol, vision-backed. The
    first cached class (remapped label 0, pre-update label A) is the
    superseded group; the last cached class (remapped ``num_classes - 1``,
    label B) is the superseding ground truth — the synthetic-arm convention.
    Phase 1 (epochs < ``supersede_epoch``): the A rows of the batch are
    written K→A through ``write_phase1``; all other rows are clean writes.
    Phase 2 (epochs >= ``supersede_epoch``): the SAME keys are re-written
    K→B through ``write_phase2``; held-out ground truth for A-keys flips to
    B at that epoch. Whether phase 2 merges (EMA-freeze stale) or forks
    (co-resident stale) is decided by the engine, not the driver. STALE is
    kept distinct from CONTRADICTORY: phase-2 writes are never registered as
    contradictions, so any contra flag in this arm would be a labeling bug
    (pinned by test). No contra injections run in the stale arm.

    Returns ``(rows_written, summary_dict)``; the summary (injection outcome
    counts, per-epoch wrong/labeled rates, stale-selection counts) is also
    what the CLI dumps as JSON next to the CSV.
    """
    if arm not in ("clean", "contra", "stale", "mixed"):
        raise ValueError(f"vision mode wires clean/contra (PR-2b), stale "
                         f"(PR-2c) and mixed/variants (PR-3b) — got {arm!r}")
    stream = VisionDriftStream(
        categories=list(classes), attractor_category=attractor_class,
        samples_per_class=samples_per_class,
        held_out_per_class=held_out_per_class,
        seed=seed, cache_path=cache_path)
    (q, y, lab), (hq, hlab) = stream.batch(contraction)
    num_classes = stream.num_classes

    rng = torch.Generator().manual_seed(seed + 1)
    jitter_gen = torch.Generator().manual_seed(seed + 2)
    registry = FailureModeRegistry()
    hook = GovernanceHook(govern)  # PR-7 write-path seam (annotate=floor; q/r no-op)
    group = None
    if arm in ("stale", "mixed"):
        if not (0 < supersede_epoch < epochs):
            raise ValueError(
                f"stale/mixed arms need 0 < supersede_epoch < epochs so both "
                f"phases run — got supersede_epoch={supersede_epoch}, "
                f"epochs={epochs}")
        # Synthetic-arm convention: first remapped class is superseded by the
        # last remapped class. In cache-class terms: classes[0] → classes[-1].
        group = registry.new_stale_group(
            f"vision-class{classes[0]}", pre_label=0,
            post_label=num_classes - 1)
    mem = ContinuousCAM(key_dim=stream.dim, value_dim=num_classes,
                        max_entries=max_entries,
                        dynamic_vigilance=DynamicVigilance(),
                        retrieval_floor_policy=RetrievalFloorPolicy(),
                        track_provenance=True)
    label = arm_label_for(arm, one_shot, key_jitter, payload_mode)
    events: list = []
    slot_rows: list = []
    topk_rows: list = []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    epoch_log = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        writer.writeheader()
        for epoch in range(epochs):
            mem.reset_dynamic_vigilance_stats()
            truth = hlab.clone()
            run_epoch_writes(
                mem, registry, group, arm, epoch, q, y, lab,
                rate=rate, num_classes=num_classes, rng=rng,
                supersede_epoch=supersede_epoch, one_shot=one_shot,
                key_jitter=key_jitter, jitter_gen=jitter_gen,
                payload_mode=payload_mode, events=events,
                clean_rewrite_class=EVENT_DUPLICATE, hook=hook)
            if group is not None and group.superseded:
                truth = torch.where(hlab == group.pre_label,
                                    torch.full_like(hlab, group.post_label),
                                    hlab)

            stats = mem.get_stats()
            acc = (mem.forward(hq).argmax(dim=-1) == truth).float().mean()
            labeled = label_probes(mem, hq, truth, registry,
                                   blend_eps=blend_eps)
            ctx = {
                "vigilance_policy": "margin",
                "retrieval_floor_policy": "live-delta",
                "epoch": epoch, "contraction": round(contraction, 4),
                "rho_probe": round(labeled["aggregates"].get(
                    "mean_cross_class_sim", float("nan")), 6),
                "within_probe": round(labeled["aggregates"].get(
                    "mean_within_class_sim", float("nan")), 6),
                "offclass_weight_mean": round(labeled["aggregates"].get(
                    "mean_offclass_weight", float("nan")), 6),
                "acc_epoch": round(float(acc), 6),
                "sim_floor_active": round(float(stats.get(
                    "sim_floor_active", float("nan"))), 6),
                "floor_delta_ema": round(float(stats.get(
                    "floor_delta_ema", float("nan"))), 6),
                "arm": label,
                "injection_rate": rate if arm in ("contra", "mixed") else 0.0,
                "supersede_epoch": supersede_epoch if group is not None else -1,
            }
            total += append_rows(writer, ctx, labeled)
            slot_rows += slot_table_rows(mem, registry, epoch)
            topk_rows += topk_table_rows(labeled, epoch)
            V = labeled["n_rows"]
            wrong = (V - int(labeled["per_probe"]["vote_correct"].sum())
                     if V else 0)
            cs = int(labeled["contradictory_strict"].sum()) if V else 0
            cl = int(labeled["contradictory_lenient"].sum()) if V else 0
            n_fork_slots = len(registry.contra_fork_slots(mem))
            entry = {
                "epoch": epoch, "probes": V, "wrong": wrong,
                "contradictory_strict": cs, "contradictory_lenient": cl,
                "live_fork_slots": n_fork_slots,
                "acc_epoch": round(float(acc), 6),
            }
            if group is not None:
                ss = int(labeled["stale_strict"].sum()) if V else 0
                sl = int(labeled["stale_lenient"].sum()) if V else 0
                # Stale-selection accounting on the superseded keys: held-out
                # probes whose ORIGINAL class is A — after supersession their
                # ground truth is B; electing A is selecting the stale value.
                sup = (hlab[labeled["probe_index"]] == group.pre_label
                       if V and group.superseded
                       else torch.zeros(max(V, 0), dtype=torch.bool))
                pred = (labeled["per_probe"]["vote_pred_label"]
                        if V else torch.zeros(0, dtype=torch.long))
                sw = (labeled["stale_vote_weight"][sup]
                      if V else torch.zeros(0))
                entry.update({
                    "stale_strict": ss, "stale_lenient": sl,
                    "live_stale_slots": len(registry.stale_slots(mem)),
                    "superseded": bool(group.superseded),
                    "superseded_key_probes": int(sup.sum()),
                    "stale_selected": int(
                        ((pred == group.pre_label) & sup).sum()),
                    "updated_selected": int(
                        ((pred == group.post_label) & sup).sum()),
                    "stale_weight_median_superseded": (
                        round(float(sw.median()), 6) if sw.numel() else None),
                })
            epoch_log.append(entry)
            extra = (f" s_strict={entry.get('stale_strict')} "
                     f"s_lenient={entry.get('stale_lenient')} "
                     f"stale_slots={entry.get('live_stale_slots')} "
                     f"stale_sel={entry.get('stale_selected')}"
                     if group is not None else "")
            print(f"[{label}] epoch {epoch:3d} | acc={acc:.3f} probes={V} "
                  f"wrong={wrong} c_strict={cs} c_lenient={cl} "
                  f"fork_slots={n_fork_slots}{extra}")

    _write_side_table(out_path.with_suffix(".per_slot.csv"),
                      SLOT_COLS, slot_rows, label)
    _write_side_table(out_path.with_suffix(".fork_events.csv"),
                      EVENT_COLS, events, label)
    _write_side_table(out_path.with_suffix(".topk.csv"),
                      TOPK_COLS, topk_rows, label)

    outcomes = {}
    for meta in registry.contra.values():
        outcomes[meta["outcome"]] = outcomes.get(meta["outcome"], 0) + 1
    summary = {
        "arm": label, "base_arm": arm, "one_shot": one_shot,
        "key_jitter": key_jitter, "payload_mode": payload_mode,
        "cache_path": stream.cache_path,
        "classes": list(classes), "attractor_class": attractor_class,
        "samples_per_class": samples_per_class,
        "held_out_per_class": held_out_per_class,
        "contraction": contraction, "injection_rate": rate,
        "epochs": epochs, "seed": seed, "max_entries": max_entries,
        "dim": stream.dim,
        "n_injections": len(registry.contra),
        "injection_outcomes": outcomes,
        "rows_written": total,
        "per_epoch": epoch_log,
    }
    if group is not None:
        summary.update({
            "supersede_epoch": supersede_epoch,
            "stale_pre_label": group.pre_label,
            "stale_post_label": group.post_label,
            "stale_pre_cache_class": int(classes[0]),
            "stale_post_cache_class": int(classes[-1]),
            "n_phase1_writes": len(group.phase1_ids),
            "n_phase2_writes": len(group.phase2_ids),
            "fork_resolution": resolve_fork_outcome(epoch_log),
        })
    # PR-7: record the governance action only when a non-baseline action was
    # requested, so a `none`/no-flag run's summary stays byte-identical to the
    # pre-seam driver (PR7_DESIGN.md §11 test 1). `annotate` is active but the
    # null-action floor, so only this provenance block differs from baseline —
    # every emitted/scored artifact stays byte-identical (PR7_DESIGN.md §4).
    if hook.active:
        summary["govern"] = hook.provenance()
    return total, summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["clean", "contra", "stale", "mixed"],
                    required=True)
    ap.add_argument("--rate", type=float, default=0.15,
                    help="contradictory-injection rate (contra/mixed arms)")
    ap.add_argument("--supersede-epoch", type=int, default=3,
                    help="epoch at which K→B supersession begins "
                         "(stale/mixed arms)")
    ap.add_argument("--one-shot", action="store_true",
                    help="write the superseding fact exactly ONCE (PR-3b: "
                         "does the boundary tie regime persist?)")
    ap.add_argument("--key-jitter", type=float, default=0.0,
                    help="L2 noise scale on phase-2 keys before "
                         "renormalization (PR-3b: breaks the exact-tie "
                         "protocol artifact)")
    ap.add_argument("--payload-mode", choices=["onehot", "soft"],
                    default="onehot",
                    help="'soft' = non-orthogonal phase-2 targets (cosine "
                         "0.6 to the stored payload) forcing the EMA-merge "
                         "stale path (PR-3b)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true",
                    help="hermetic CPU mode (no caches, no GPU) — the only "
                         "mode exercised in PR-2a")
    # cache-backed vision mode (PR-2b) — mirrors calibration_probe --vision
    ap.add_argument("--vision", action="store_true",
                    help="drive the loop from the cached DINOv2 ViT-L/14 "
                         "feature manifold (clean/contra: PR-2b; stale: "
                         "PR-2c)")
    ap.add_argument("--vision-cache", type=str, default=None,
                    help="path to the *_train.pt feature cache; if unset, "
                         "resolve via the feature_cache_vitl14 symlink")
    ap.add_argument("--vision-classes", type=str, default="0,8,19,33")
    ap.add_argument("--vision-attractor-class", type=int, default=71)
    ap.add_argument("--samples-per-class", type=int, default=32)
    ap.add_argument("--held-out-per-class", type=int, default=64)
    ap.add_argument("--contraction", type=float, default=0.0,
                    help="FIXED contraction for the vision stream; PR-2b is "
                         "deliberately stationary (0.0) to decouple "
                         "contradiction from drift/BLENDED")
    ap.add_argument("--max-entries", type=int, default=4096)
    ap.add_argument("--govern", choices=list(GOVERN_ACTIONS), default="none",
                    help="PR-7 step-1 write-path governance seam (boundary "
                         "scaffold). EVERY action is a recorded NO-OP in this "
                         "step: writes route through the seam but make the "
                         "exact same decisions as the ungoverned baseline. No "
                         "governance behavior, write refusal, or quarantine is "
                         "implemented yet; the deployed retrieval path never "
                         "reaches this seam (PR7_DESIGN.md §13.1).")
    ap.add_argument("--out", type=str,
                    default="results/issue_failure_mode_blindness/"
                            "per_probe_injected.csv")
    args = ap.parse_args()

    if args.synthetic == args.vision:
        raise SystemExit(
            "pick exactly one mode: --synthetic (hermetic CPU, PR-2a) or "
            "--vision (cache-backed ViT-L/14 manifold, PR-2b).")
    if args.vision:
        classes = [int(c) for c in args.vision_classes.split(",") if c.strip()]
        n, summary = run_vision(
            args.arm, args.rate, args.epochs, Path(args.out),
            cache_path=args.vision_cache, classes=classes,
            attractor_class=args.vision_attractor_class,
            samples_per_class=args.samples_per_class,
            held_out_per_class=args.held_out_per_class,
            contraction=args.contraction, max_entries=args.max_entries,
            seed=args.seed, supersede_epoch=args.supersede_epoch,
            one_shot=args.one_shot, key_jitter=args.key_jitter,
            payload_mode=args.payload_mode, govern=args.govern)
        summary_path = Path(args.out).with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{args.arm}] wrote {n} labeled per-probe rows to {args.out}")
        print(f"[{args.arm}] summary -> {summary_path}")
        return
    n = run_synthetic(args.arm, args.rate, args.epochs, args.supersede_epoch,
                      Path(args.out), seed=args.seed, one_shot=args.one_shot,
                      key_jitter=args.key_jitter,
                      payload_mode=args.payload_mode, govern=args.govern)
    print(f"[{args.arm}] wrote {n} labeled per-probe rows to {args.out}")


if __name__ == "__main__":
    main()
