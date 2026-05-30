#!/usr/bin/env python3
"""G0/G1 endpoint-recovery feasibility probe (A2b, analysis-only).

Proves the invariant:
    cache row i  ->  ImageFolder sample train_indices[i]  ->  exact image path
                 ->  fresh DINOv2 ViT-L/14 embedding  ->  cosine(cached, fresh) > 0.999

This does NOT build ReembedDriftStream, run the subset experiment, or touch
forward()/associative_core.py/detector logic. It is a pure read-only probe over
the merged feature_cache_inr_vitl14 cache.

The split/order/transform/model are taken from the extractor module itself
(extract_imagenet_r_vitl14) as the source of truth -- nothing is reimplemented
from memory.
"""
import argparse
import json
import os
import socket
import sys

import torch
import torch.nn.functional as F

# Source of truth: the extractor that built the cache.
import extract_imagenet_r_vitl14 as ex
from torchvision.datasets import ImageFolder

CHOSEN_CLASSES = [166, 63, 77, 156]
ATTRACTOR_CLASS = 134
N_TARGET_SAMPLE = 32          # rows drawn from chosen+attractor classes
N_NEGATIVE = 5                # random rows OUTSIDE chosen+attractor classes
COS_THRESHOLD = 0.999
SEED = 0


def pick_rows(labels):
    """Deterministic selection: ~N_TARGET across chosen+attractor classes,
    plus N_NEGATIVE rows from other classes. Returns sorted list of cache rows."""
    g = torch.Generator().manual_seed(SEED)
    target_classes = CHOSEN_CLASSES + [ATTRACTOR_CLASS]
    per_class = max(1, N_TARGET_SAMPLE // len(target_classes))
    chosen = []
    for cls in target_classes:
        rows = (labels == cls).nonzero(as_tuple=True)[0]
        if len(rows) == 0:
            continue
        k = min(per_class, len(rows))
        perm = torch.randperm(len(rows), generator=g)[:k]
        chosen.extend(rows[perm].tolist())
    # negatives: rows whose label is not in target_classes
    mask = torch.ones(len(labels), dtype=torch.bool)
    for cls in target_classes:
        mask &= labels != cls
    neg_rows = mask.nonzero(as_tuple=True)[0]
    if len(neg_rows) > 0:
        k = min(N_NEGATIVE, len(neg_rows))
        perm = torch.randperm(len(neg_rows), generator=g)[:k]
        chosen.extend(neg_rows[perm].tolist())
    return sorted(set(chosen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", default="", help="verbatim command used (for the report)")
    ap.add_argument("--out", default="results/issue_input_reembed_fidelity/G0_G1_PROBE.md")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    args = ap.parse_args()

    host = socket.gethostname()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_path = os.path.join(
        ex.CACHE_DIR,
        f"imagenetr_dinov2_{args.split}.pt",
    )
    blob = torch.load(cache_path, map_location="cpu")
    cached_embeds = blob["embeds"]            # (N, D), as saved by extractor
    cached_labels = blob["labels"]            # (N,)
    n_rows, dim = cached_embeds.shape

    cache_norms = cached_embeds.norm(dim=-1)
    likely_normalized = bool(
        (cache_norms.min() > 0.999) and (cache_norms.max() < 1.001)
    )

    # --- G0: replay split using the EXTRACTOR'S OWN code ---
    full_ds = ImageFolder(root=ex.DATASET_ROOT, transform=ex.build_transform())
    train_sub, test_sub = ex.stratified_split(full_ds, ex.TRAIN_RATIO, ex.SEED)
    sub = train_sub if args.split == "train" else test_sub
    split_indices = sub.indices                # cache row i -> full_ds[split_indices[i]]

    split_len_ok = (len(split_indices) == n_rows)

    rows = pick_rows(cached_labels)

    # --- G1: fresh re-embed + endpoint identity ---
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model = model.to(device).eval()
    transform = ex.build_transform()

    records = []
    imgs = []
    meta = []
    label_matches = 0
    with torch.no_grad():
        for i in rows:
            ds_idx = split_indices[i]
            path, folder_label = full_ds.samples[ds_idx]
            cached_label = int(cached_labels[i])
            lm = (folder_label == cached_label)
            label_matches += int(lm)
            img = full_ds.loader(path)          # PIL via ImageFolder's own loader
            x = transform(img)
            imgs.append(x)
            meta.append((i, ds_idx, path, cached_label, folder_label, lm))

        batch = torch.stack(imgs).to(device)
        fresh = model(batch).cpu().float()      # (k, D) raw model output

    # cosine is norm-invariant; compares raw cached vs raw fresh directly
    cos = F.cosine_similarity(cached_embeds[rows].float(), fresh, dim=-1)

    for (i, ds_idx, path, cl, fl, lm), c in zip(meta, cos.tolist()):
        records.append({
            "cache_row": i, "ds_index": ds_idx, "path": path,
            "cached_label": cl, "recovered_label": fl,
            "label_match": lm, "cosine": round(c, 6),
        })

    cos_sorted = sorted(r["cosine"] for r in records)
    n = len(cos_sorted)
    cmin = cos_sorted[0]
    cmax = cos_sorted[-1]
    cmean = sum(cos_sorted) / n
    cmed = cos_sorted[n // 2] if n % 2 else (cos_sorted[n // 2 - 1] + cos_sorted[n // 2]) / 2
    n_pass = sum(1 for c in cos_sorted if c > COS_THRESHOLD)
    lm_rate = label_matches / n
    below = [r for r in records if r["cosine"] <= COS_THRESHOLD]
    label_mismatches = [r for r in records if not r["label_match"]]

    # --- verdict per decision rules ---
    if not split_len_ok:
        verdict = "FAIL"
        reason = (f"split replay length {len(split_indices)} != cache rows {n_rows}; "
                  "split/order mapping diverged.")
    elif label_mismatches:
        verdict = "FAIL"
        reason = (f"{len(label_mismatches)} label mismatch(es); "
                  "stop and diagnose split replay / order mapping.")
    elif n_pass == n:
        verdict = "PASS"
        reason = "all sampled endpoint cosines > 0.999 and all labels match."
    elif cmin > 0.99:
        verdict = "INCONCLUSIVE"
        reason = ("labels match but some cosines in (0.99, 0.999]; do NOT loosen "
                  "threshold -- diagnose model eval/dtype/device or normalization.")
    else:
        verdict = "FAIL"
        reason = ("labels match but cosine below threshold; diagnose "
                  "preprocessing/model/normalization mismatch.")

    summary = {
        "verdict": verdict, "reason": reason, "host": host,
        "cache_path": cache_path, "split": args.split,
        "n_rows_cache": n_rows, "dim": dim,
        "n_sampled": n, "n_pass_gt_0999": n_pass,
        "cos_min": round(cmin, 6), "cos_median": round(cmed, 6),
        "cos_mean": round(cmean, 6), "cos_max": round(cmax, 6),
        "label_match_rate": round(lm_rate, 6),
        "split_len_ok": split_len_ok,
        "cache_norm_min": round(float(cache_norms.min()), 6),
        "cache_norm_mean": round(float(cache_norms.mean()), 6),
        "cache_norm_max": round(float(cache_norms.max()), 6),
        "extractor_saved_normalized": likely_normalized,
        "device": str(device),
    }
    print(json.dumps(summary, indent=2))

    # --- write markdown report ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines = []
    lines.append("# A2b · G0/G1 endpoint-recovery feasibility probe\n")
    lines.append(f"**Verdict: {verdict}**\n")
    lines.append(f"_{reason}_\n")
    lines.append("Invariant tested: `cache row i -> ImageFolder sample "
                 "train_indices[i] -> exact image path -> fresh DINOv2 "
                 "embedding -> cosine(cached, fresh) > 0.999`.\n")
    lines.append("## Run metadata\n")
    lines.append(f"- Host: `{host}`")
    lines.append(f"- Command: `{args.command}`")
    lines.append(f"- Cache path: `{cache_path}` (split=`{args.split}`)")
    lines.append("- Model/checkpoint: `torch.hub facebookresearch/dinov2 :: dinov2_vitl14`")
    lines.append(f"- Device: `{device}`")
    lines.append("- Split/transform source of truth: `extract_imagenet_r_vitl14` "
                 f"(SEED={ex.SEED}, train_ratio={ex.TRAIN_RATIO})\n")
    lines.append("## G0 — split replay & label recovery\n")
    lines.append(f"- Cache rows: {n_rows}; replayed split length: "
                 f"{len(split_indices)} ({'OK' if split_len_ok else 'MISMATCH'})")
    lines.append(f"- Sampled rows: {n} "
                 f"(classes {CHOSEN_CLASSES} + attractor {ATTRACTOR_CLASS} + "
                 f"{N_NEGATIVE} negatives, seed={SEED})")
    lines.append(f"- Label-match rate: {lm_rate:.4f} "
                 f"({label_matches}/{n})\n")
    lines.append("## G1 — fresh re-embed endpoint identity\n")
    lines.append(f"- cosine(cached, reembedded): min={cmin:.6f} "
                 f"median={cmed:.6f} mean={cmean:.6f} max={cmax:.6f}")
    lines.append(f"- passing cos > {COS_THRESHOLD}: {n_pass}/{n}")
    lines.append(f"- cache feature L2 norm: min={float(cache_norms.min()):.6f} "
                 f"mean={float(cache_norms.mean()):.6f} "
                 f"max={float(cache_norms.max()):.6f}")
    lines.append(f"- extractor saved **normalized** features: "
                 f"**{likely_normalized}** "
                 f"(norms ~1.0 => yes; cosine is norm-invariant so this does "
                 f"not affect the G1 check, but it dictates how the re-embed "
                 f"arm must feed the engine)\n")
    if below:
        lines.append("## Rows at/below cosine threshold\n")
        lines.append("| cache_row | cosine | cached_label | recovered_label | path |")
        lines.append("|---|---|---|---|---|")
        for r in below:
            lines.append(f"| {r['cache_row']} | {r['cosine']:.6f} | "
                         f"{r['cached_label']} | {r['recovered_label']} | "
                         f"`{r['path']}` |")
        lines.append("")
    if label_mismatches:
        lines.append("## Label mismatches (STOP — diagnose split/order)\n")
        lines.append("| cache_row | ds_index | cached_label | recovered_label | path |")
        lines.append("|---|---|---|---|---|")
        for r in label_mismatches:
            lines.append(f"| {r['cache_row']} | {r['ds_index']} | "
                         f"{r['cached_label']} | {r['recovered_label']} | "
                         f"`{r['path']}` |")
        lines.append("")
    lines.append("## Next-step decision\n")
    if verdict == "PASS":
        lines.append("- ✅ Proceed to drafting `ReembedDriftStream` "
                     "(endpoints provably shared between cache-linear and "
                     "input-level arms).")
    elif verdict == "INCONCLUSIVE":
        lines.append("- ⏸ Do NOT loosen the threshold. Diagnose model "
                     "eval/dtype/device and raw-vs-unit normalization before "
                     "proceeding.")
    else:
        lines.append("- ⛔ Stop. Diagnose per the failure reason above; if "
                     "recovery cannot be made reliable, write a NEW "
                     "explicit-index/path cache in a follow-up (never edit the "
                     "merged cache).")
    lines.append("")
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
