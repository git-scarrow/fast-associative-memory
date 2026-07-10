#!/usr/bin/env python3
"""PR-13 consumer qualification gate — SYNTHETIC ONLY (decision gate).

    <venv>/bin/python -m harness.pr13_qualify_consumer --label baseline

Answers, for the exact sealed Qwen3-8B bf16 consumer at its registered
decoding configuration, whether the scoring host can actually run the
20,304-call replay:

  * module placement (how much of the model spills off the GPU)
  * peak VRAM and peak system RAM
  * sustained throughput, warm-up excluded, across prompt-length bands
  * malformed-output rate under the §8.4 parser
  * whether a model-backed crash/resume reconciles byte for byte

DELIBERATELY OUTSIDE harness/ctx/. The sealed scoring manifest hashes
every file under harness/ctx; a qualification script living there would
silently invalidate its `ctx_source_sha256` rollup. Nothing here changes
the consumer pin, the decoding configuration, the manifests, the
compiler, the sample, or the scoring rules.

NO §7 MATERIAL. Every prompt is built from registered *templates* filled
with invented values. The script refuses to touch the committed query
manifest, and asserts that refusal rather than relying on discipline.
This is what keeps the §8.1 consumer-motion option alive: the first §7
render is the moment a runtime change becomes a kill.

This is not the governed replay. It issues no verdict and scores no gate.
"""

import argparse
import json
import os
import resource
import statistics
import sys
import tempfile
import time

from harness.ctx import cells, replay
from harness.ctx.compile import load_policy
from harness.ctx.output_contract import MAX_NEW_TOKENS, parse_consumer_output

# The committed query manifest. Its digest is a tripwire, not an input.
COMMITTED_QUERY_MANIFEST = os.path.join("harness", "ctx", "manifests",
                                        "query_manifest_v1.json")

# Prompt-length bands, as *context-block* token targets under the pinned
# tokenizer. `none` is the floor arm (no block at all); `median` and
# `high` saturate the two registered budgets, so they bound the replay
# from above — a real block may be shorter, never longer.
BANDS = (("none", None), ("short", 150), ("median", 800), ("high", 1500))


def _synthetic_block(target_tokens, count_tokens, policy):
    """A governed-looking block built from the registered render template
    with invented slot/decode/rank/sim values. No §7 content."""
    template = policy["render_templates"]["ctx-fam-item-v1"]
    header = ("[governed context | shown:{n} caveated:0 unresolved:0 "
              "withheld:0 budget-withheld:0 | policy:1.0]")
    lines = []
    for i in range(2000):
        candidate = lines + ["- " + template.format(
            slot=1000 + i, decode=chr(65 + i % 26), rank=i,
            sim=f"0.{max(100, 999 - i):03d}")]
        block = "\n".join([header.format(n=len(candidate))] + candidate)
        if count_tokens(block) > target_tokens:
            break
        lines = candidate
    return "\n".join([header.format(n=len(lines))] + lines)


def _templated_tokens(consumer, prompt):
    tok = consumer._tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _out_tokens(consumer, raw):
    return len(consumer._tokenizer(raw, add_special_tokens=False)["input_ids"])


def _placement(model):
    counts = {}
    for module, where in getattr(model, "hf_device_map", {}).items():
        counts[str(where)] = counts.get(str(where), 0) + 1
    return counts


def _vram():
    import torch
    free, total = torch.cuda.mem_get_info()
    return {"free_bytes": free, "total_bytes": total,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def _peak_rss_bytes():
    # Linux ru_maxrss is KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _stats(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    n = len(xs)
    return {"n": n, "min": xs[0], "max": xs[-1], "mean": statistics.fmean(xs),
            "median": statistics.median(xs),
            "p10": xs[max(0, int(0.10 * (n - 1)))],
            "p90": xs[min(n - 1, int(0.90 * (n - 1)))],
            "stdev": statistics.stdev(xs) if n > 1 else 0.0}


# --- crash/resume, model-backed ---------------------------------------------

class _CrashAfter:
    """Wraps the real consumer; `is_real` stays True so the runner's
    real-render gate is exercised, not bypassed."""

    is_real = True

    def __init__(self, inner, after):
        self._inner = inner
        self._after = after
        self.pin_id = inner.pin_id
        self.calls = 0

    def count_tokens(self, text):
        return self._inner.count_tokens(text)

    def generate(self, prompt, max_new_tokens=MAX_NEW_TOKENS):
        if self.calls >= self._after:
            raise RuntimeError("simulated crash")
        self.calls += 1
        return self._inner.generate(prompt, max_new_tokens)


def _item(native_id, content, state="agent-readable", signals=None,
          candidate_set=None):
    import hashlib
    return {
        "item_id": f"qual:{native_id}", "source_id": "qual-v1",
        "content": content, "content_kind": "source-native",
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "event_time": None, "ingest_time": None,
        "evidence": [{"adapter_id": "qual-v1", "signal": s, "value": True,
                      "evidence_ptr": f"ptr:{native_id}:{s}",
                      "tier": "harness-heuristic"}
                     for s in (signals or ["source_identity"])],
        "relations": {"supersedes": [], "contradicts": [],
                      "candidate_set_id": candidate_set},
        "state": state, "policy_version": "1.0",
    }


def _bundle(items, text, turn=1):
    return {"items": items, "not_retrieved": [], "query_text": text,
            "turn_state": {"turn_index": turn, "prior_rendered": {}}}


def _qualification_sources():
    q = cells.FAM_QUERY_TEMPLATE.format(probe=999, epoch=99)
    served = _item("X", "the original synthetic fact.")
    superseded = _item("X", "the original synthetic fact.", state="superseded",
                       signals=["superseded_by"])
    fresh = _item("Y", "the replacement synthetic fact.")
    sources = {
        "qual:assert:q0": _bundle([_item("a", "slot 3 decodes to class A.")], q),
        "qual:dual:q0": _bundle(
            [_item("e", "slot 7 decodes to class E.", signals=["oneshot_tie"],
                   candidate_set="cs1"),
             _item("f", "slot 8 decodes to class F.", signals=["oneshot_tie"],
                   candidate_set="cs1")], q),
    }
    for turn, items in ((1, [served]), (2, [served]), (3, [superseded, fresh])):
        sources[f"qual:mt:session#t{turn}"] = _bundle(items, q, turn=turn)
    return sources


def _qualification_manifests(sources, consumer):
    qm = replay.seal_manifest({
        "manifest_id": "pr13-qualification-manifest",
        "version": "1.0",
        "note": "SYNTHETIC QUALIFICATION ONLY — not the committed query manifest",
        "arm_plan": [list(a) for a in replay.ARM_PLAN],
        "budgets": list(replay.BUDGETS),
        "queries": sorted(sources),
    })
    with open(COMMITTED_QUERY_MANIFEST, encoding="utf-8") as fh:
        committed = json.load(fh)["manifest_sha256"]
    if qm["manifest_sha256"] == committed:
        raise SystemExit("refusing to qualify against the committed manifest")
    sm = replay.seal_manifest({
        "manifest_id": "pr13-qualification-scoring-manifest",
        "note": "SYNTHETIC QUALIFICATION ONLY — not the committed scoring manifest",
        "query_manifest_sha256": qm["manifest_sha256"],
        "consumer": {"replay_ready": True, "pin_id": consumer.pin_id},
        "decoding": {"max_new_tokens": MAX_NEW_TOKENS},
    })
    return qm, sm, committed


def crash_resume_check(consumer, policy):
    sources = _qualification_sources()
    qm, sm, committed = _qualification_manifests(sources, consumer)
    workdir = tempfile.mkdtemp(prefix="pr13_qual_")
    resumed_path = os.path.join(workdir, "resumed.jsonl")
    clean_path = os.path.join(workdir, "clean.jsonl")

    crashed = False
    try:
        replay.run(qm, _CrashAfter(consumer, 11), sources, resumed_path, policy,
                   scoring_manifest=sm)
    except RuntimeError:
        crashed = True
    summary = replay.run(qm, consumer, sources, resumed_path, policy,
                         scoring_manifest=sm)
    replay.run(qm, consumer, sources, clean_path, policy, scoring_manifest=sm)

    with open(resumed_path, encoding="utf-8") as fh:
        a = fh.read()
    with open(clean_path, encoding="utf-8") as fh:
        b = fh.read()
    rows = [json.loads(line) for line in a.splitlines() if line.strip()]
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "crashed_as_designed": crashed,
        "committed_manifest_untouched": qm["manifest_sha256"] != committed,
        "rows": summary["rows"], "resumed": summary["resumed"],
        "executed_after_resume": summary["executed"],
        "evidence_set_pairs_checked": summary["evidence_set_pairs_checked"],
        "byte_reconcilable": a == b,
        "status_counts": statuses,
    }


def _install_placement_cap(gpu_gib, cpu_gib):
    """Cap the GPU share of `device_map="auto"` WITHOUT editing
    harness/ctx/consumer_qwen3.py, which the sealed scoring manifest
    hashes.

    `device_map="auto"` plans against the device's TOTAL memory, not its
    FREE memory, then OOMs while materializing tensors. The pin fixes
    precision, quantization, decoding, and mode; it says nothing about
    device placement, so a cap is legal. Patching the loader entry point
    keeps the frozen path — seal verification, bfloat16 dtype, greedy
    decoding, the pinned chat template — running exactly as registered.
    """
    import transformers
    original = transformers.AutoModelForCausalLM.from_pretrained

    def capped(*a, **kw):
        kw.setdefault("max_memory", {0: f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"})
        return original(*a, **kw)

    transformers.AutoModelForCausalLM.from_pretrained = capped
    return {"max_memory_gpu_gib": gpu_gib, "max_memory_cpu_gib": cpu_gib}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=10, help="timed gens per band")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--skip-resume", action="store_true")
    ap.add_argument("--max-gpu-gib", type=float, default=None,
                    help="cap the GPU share; omit to use the registered "
                         "device_map='auto' path unmodified")
    ap.add_argument("--cpu-gib", type=float, default=24)
    ap.add_argument("--auto-load-probe", action="store_true",
                    help="attempt only the registered load path and report "
                         "whether it succeeds; generate nothing")
    args = ap.parse_args()

    import torch
    from harness.ctx.consumer_qwen3 import AUTHORIZATION_TOKEN, Qwen3Consumer

    policy = load_policy()
    vram_before = _vram()

    override = None
    if args.max_gpu_gib is not None:
        override = _install_placement_cap(args.max_gpu_gib, args.cpu_gib)

    t0 = time.time()
    oom = (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError)
    try:
        consumer = Qwen3Consumer(authorize=AUTHORIZATION_TOKEN)
    except oom as exc:
        report = {"label": args.label, "loaded": False,
                  "placement_override": override,
                  "vram_before_load": vram_before,
                  "load_error": "CUDA OOM", "detail": str(exc).split("\n")[0]}
        text = json.dumps(report, indent=2, sort_keys=True)
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return 0
    load_seconds = time.time() - t0

    if args.auto_load_probe:
        report = {"label": args.label, "loaded": True,
                  "placement_override": override,
                  "load_seconds": load_seconds,
                  "module_placement": _placement(consumer._model),
                  "vram_before_load": vram_before,
                  "vram_after_load": _vram(),
                  "peak_system_rss_bytes": _peak_rss_bytes()}
        text = json.dumps(report, indent=2, sort_keys=True)
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return 0

    placement = _placement(consumer._model)
    vram_after_load = _vram()
    query = cells.FAM_QUERY_TEMPLATE.format(probe=999, epoch=99)

    prompts = {}
    for name, target in BANDS:
        if target is None:
            prompts[name] = replay.build_prompt("none", query, None)
        else:
            block = _synthetic_block(target, consumer.count_tokens, policy)
            prompts[name] = replay.build_prompt("governed", query, block)

    # Warm-up on the median band; discarded.
    warm = []
    for _ in range(args.warmup):
        t = time.time()
        consumer.generate(prompts["median"])
        warm.append(time.time() - t)

    # Peak counters are NOT reset: peak VRAM must span load and inference,
    # since that is the number the host has to satisfy.
    bands = {}
    for name, _target in BANDS:
        prompt = prompts[name]
        in_tokens = _templated_tokens(consumer, prompt)
        lat, outs, statuses, prose = [], [], {}, 0
        for _ in range(args.n):
            t = time.time()
            raw = consumer.generate(prompt)
            lat.append(time.time() - t)
            outs.append(_out_tokens(consumer, raw))
            parsed = parse_consumer_output(raw)
            statuses[parsed["status"]] = statuses.get(parsed["status"], 0) + 1
            prose += bool(parsed.get("extra_prose"))
        half = len(lat) // 2
        bands[name] = {
            "prompt_tokens": in_tokens,
            "output_tokens": _stats(outs),
            "latency_seconds": _stats(lat),
            "rows_per_hour": 3600.0 / statistics.fmean(lat),
            "drift_first_half_mean": statistics.fmean(lat[:half]) if half else None,
            "drift_second_half_mean": statistics.fmean(lat[half:]) if half else None,
            "status_counts": statuses,
            "extra_prose": prose,
            "malformed_rate": statuses.get("unparseable", 0) / len(lat),
        }

    report = {
        "label": args.label,
        "loaded": True,
        "placement_override": override,
        "consumer": {"pin_id": consumer.pin_id, "revision": consumer.revision},
        "runtime": consumer.runtime_versions(),
        "load_seconds": load_seconds,
        "module_placement": placement,
        "offloaded_modules": sum(v for k, v in placement.items() if k != "0"),
        "vram_before_load": vram_before,
        "vram_after_load": vram_after_load,
        "vram_after_bands": _vram(),
        "peak_system_rss_bytes": _peak_rss_bytes(),
        "warmup_seconds": warm,
        "bands": bands,
    }

    if not args.skip_resume:
        report["crash_resume"] = crash_resume_check(consumer, policy)
        report["peak_system_rss_bytes"] = _peak_rss_bytes()

    # §8.3 projection for the 20,304-row replay, bounded rather than
    # extrapolated from a point estimate. Rows: 3,384 queries × 6.
    queries = 3384
    mean = {b: bands[b]["latency_seconds"]["mean"] for b in bands}
    p10 = {b: bands[b]["latency_seconds"]["p10"] for b in bands}
    p90 = {b: bands[b]["latency_seconds"]["p90"] for b in bands}

    def total(pick, ctx_band):
        # 1 none row + 2 rows at B=800 + 3 rows at B=1500 per query
        return queries * (pick["none"] + 2 * pick[ctx_band[0]]
                          + 3 * pick[ctx_band[1]])

    report["projection_hours"] = {
        "note": ("upper bound assumes every context block saturates its "
                 "budget; lower bound assumes every context block sits at "
                 "the short band. The true distribution is unknown without "
                 "rendering §7, which this gate refuses to do."),
        "upper_bound_saturated_mean": total(mean, ("median", "high")) / 3600,
        "upper_bound_saturated_p90": total(p90, ("median", "high")) / 3600,
        "lower_bound_short_mean": total(mean, ("short", "short")) / 3600,
        "lower_bound_short_p10": total(p10, ("short", "short")) / 3600,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
