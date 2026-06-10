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

NEW_COLS = ["arm", "injection_rate", "supersede_epoch", "probe_index",
            "top1_slot", "failure_mode",
            "contradictory_strict", "contradictory_lenient",
            "stale_strict", "stale_lenient",
            "n_contra_topk", "n_stale_topk",
            "contra_vote_weight", "stale_vote_weight"]
OUT_COLS = ALL_COLS + NEW_COLS


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
                              rate: float, rng: torch.Generator):
        """Re-write ``rate`` of the batch rows with permuted-label payloads.

        Each injected row keeps its real query (key) but carries a one-hot
        target for a uniformly chosen WRONG class — the hallucinating-writer
        model. Writes go through the normal ``learn_local``; the outcome of
        each (forked / plain-miss / absorbed / dropped) is classified from the
        pre-write nearest sim, the effective vigilance, and provenance.
        """
        if mem.slot_records is None:
            raise RuntimeError(
                "FailureModeRegistry requires ContinuousCAM(track_provenance="
                "True): injection outcomes are classified from slot_records.")
        B = queries.size(0)
        n_inj = int(round(rate * B))
        if n_inj == 0:
            return []
        rows = torch.randperm(B, generator=rng)[:n_inj]
        inj_q = queries[rows]
        true = true_labels[rows]
        # Uniform wrong label: shift by 1..C-1 (mod C) — never the true class.
        shift = torch.randint(1, num_classes, (n_inj,), generator=rng)
        wrong = (true + shift) % num_classes
        inj_y = F.one_hot(wrong, num_classes).float()

        ids = self.next_ids("contra", n_inj)
        pre_best_slots, pre_best_sims = mem._get_nearest_batch(inj_q)
        thresholds = self._effective_vigilance(mem, inj_q, n_inj)

        mem.learn_local(inj_q, inj_y, record_ids=ids)

        owners = _owner_slots(mem, ids)
        for j, rid in enumerate(ids):
            slot = owners.get(rid)
            if slot is None:
                outcome = DROPPED
            elif mem.records_for(slot) == {rid}:
                outcome = (FORKED
                           if float(pre_best_sims[j]) >= float(thresholds[j])
                           else PLAIN_MISS)
            else:
                outcome = ABSORBED
            self.contra[rid] = {"label": int(wrong[j]), "outcome": outcome}
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
                  dim: int = 32) -> int:
    """End-to-end synthetic run for one arm; returns rows written.

    clean  — no injections (negative control: no contra/stale labels may fire)
    contra — permuted-label injections at ``rate`` per epoch
    stale  — class 0 superseded by ``num_classes - 1`` at ``supersede_epoch``;
             held-out class-0 ground truth flips at that epoch
    """
    batches, hq, hlab = build_synthetic_stream(
        num_classes=num_classes, dim=dim, epochs=epochs, seed=seed)
    rng = torch.Generator().manual_seed(seed + 1)
    registry = FailureModeRegistry()
    mem = ContinuousCAM(key_dim=dim, value_dim=num_classes, max_entries=1024,
                        dynamic_vigilance=DynamicVigilance(),
                        retrieval_floor_policy=RetrievalFloorPolicy(),
                        track_provenance=True)
    group = None
    if arm == "stale":
        group = registry.new_stale_group("synthetic-class0",
                                         pre_label=0,
                                         post_label=num_classes - 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        writer.writeheader()
        for epoch, (q, y, lab) in enumerate(batches):
            mem.reset_dynamic_vigilance_stats()
            truth = hlab.clone()
            if arm == "stale":
                g0 = lab == group.pre_label
                if epoch < supersede_epoch:
                    registry.write_phase1(mem, group, q[g0], y[g0])
                    ids = registry.next_ids("clean", int((~g0).sum()))
                    mem.learn_local(q[~g0], y[~g0], record_ids=ids)
                else:
                    if not group.superseded:
                        registry.supersede(group, epoch)
                    y2 = F.one_hot(
                        torch.full((int(g0.sum()),), group.post_label),
                        num_classes).float()
                    registry.write_phase2(mem, group, q[g0], y2)
                    ids = registry.next_ids("clean", int((~g0).sum()))
                    mem.learn_local(q[~g0], y[~g0], record_ids=ids)
                if group.superseded:
                    truth = torch.where(hlab == group.pre_label,
                                        torch.full_like(hlab, group.post_label),
                                        hlab)
            else:
                ids = registry.next_ids("clean", q.size(0))
                mem.learn_local(q, y, record_ids=ids)
                if arm == "contra":
                    registry.inject_contradictions(
                        mem, q, lab, num_classes, rate, rng)

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
                "arm": arm, "injection_rate": rate,
                "supersede_epoch": supersede_epoch if arm == "stale" else -1,
            }
            total += append_rows(writer, ctx, labeled)
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
               supersede_epoch: int = 6):
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
    if arm not in ("clean", "contra", "stale"):
        raise ValueError(f"vision mode wires clean/contra (PR-2b) and stale "
                         f"(PR-2c) — got {arm!r}")
    stream = VisionDriftStream(
        categories=list(classes), attractor_category=attractor_class,
        samples_per_class=samples_per_class,
        held_out_per_class=held_out_per_class,
        seed=seed, cache_path=cache_path)
    (q, y, lab), (hq, hlab) = stream.batch(contraction)
    num_classes = stream.num_classes

    rng = torch.Generator().manual_seed(seed + 1)
    registry = FailureModeRegistry()
    group = None
    if arm == "stale":
        if not (0 < supersede_epoch < epochs):
            raise ValueError(
                f"stale arm needs 0 < supersede_epoch < epochs so both "
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    epoch_log = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        writer.writeheader()
        for epoch in range(epochs):
            mem.reset_dynamic_vigilance_stats()
            truth = hlab.clone()
            if arm == "stale":
                g0 = lab == group.pre_label
                if epoch < supersede_epoch:
                    registry.write_phase1(mem, group, q[g0], y[g0])
                    ids = registry.next_ids("clean", int((~g0).sum()))
                    mem.learn_local(q[~g0], y[~g0], record_ids=ids)
                else:
                    if not group.superseded:
                        registry.supersede(group, epoch)
                    y2 = F.one_hot(
                        torch.full((int(g0.sum()),), group.post_label),
                        num_classes).float()
                    registry.write_phase2(mem, group, q[g0], y2)
                    ids = registry.next_ids("clean", int((~g0).sum()))
                    mem.learn_local(q[~g0], y[~g0], record_ids=ids)
                if group.superseded:
                    truth = torch.where(hlab == group.pre_label,
                                        torch.full_like(hlab, group.post_label),
                                        hlab)
            else:
                ids = registry.next_ids("clean", q.size(0))
                mem.learn_local(q, y, record_ids=ids)
                if arm == "contra":
                    registry.inject_contradictions(
                        mem, q, lab, num_classes, rate, rng)

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
                "arm": arm, "injection_rate": rate if arm == "contra" else 0.0,
                "supersede_epoch": supersede_epoch if arm == "stale" else -1,
            }
            total += append_rows(writer, ctx, labeled)
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
            if arm == "stale":
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
                entry.update({
                    "stale_strict": ss, "stale_lenient": sl,
                    "live_stale_slots": len(registry.stale_slots(mem)),
                    "superseded": bool(group.superseded),
                    "superseded_key_probes": int(sup.sum()),
                    "stale_selected": int(
                        ((pred == group.pre_label) & sup).sum()),
                    "updated_selected": int(
                        ((pred == group.post_label) & sup).sum()),
                })
            epoch_log.append(entry)
            extra = (f" s_strict={entry.get('stale_strict')} "
                     f"s_lenient={entry.get('stale_lenient')} "
                     f"stale_slots={entry.get('live_stale_slots')} "
                     f"stale_sel={entry.get('stale_selected')}"
                     if arm == "stale" else "")
            print(f"[{arm}] epoch {epoch:3d} | acc={acc:.3f} probes={V} "
                  f"wrong={wrong} c_strict={cs} c_lenient={cl} "
                  f"fork_slots={n_fork_slots}{extra}")

    outcomes = {}
    for meta in registry.contra.values():
        outcomes[meta["outcome"]] = outcomes.get(meta["outcome"], 0) + 1
    summary = {
        "arm": arm, "cache_path": stream.cache_path,
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
    if arm == "stale":
        summary.update({
            "supersede_epoch": supersede_epoch,
            "stale_pre_label": group.pre_label,
            "stale_post_label": group.post_label,
            "stale_pre_cache_class": int(classes[0]),
            "stale_post_cache_class": int(classes[-1]),
            "n_phase1_writes": len(group.phase1_ids),
        })
    return total, summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["clean", "contra", "stale"],
                    required=True)
    ap.add_argument("--rate", type=float, default=0.15,
                    help="contradictory-injection rate (contra arm)")
    ap.add_argument("--supersede-epoch", type=int, default=3,
                    help="epoch at which K→B supersession begins (stale arm)")
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
            seed=args.seed, supersede_epoch=args.supersede_epoch)
        summary_path = Path(args.out).with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{args.arm}] wrote {n} labeled per-probe rows to {args.out}")
        print(f"[{args.arm}] summary -> {summary_path}")
        return
    n = run_synthetic(args.arm, args.rate, args.epochs, args.supersede_epoch,
                      Path(args.out), seed=args.seed)
    print(f"[{args.arm}] wrote {n} labeled per-probe rows to {args.out}")


if __name__ == "__main__":
    main()
