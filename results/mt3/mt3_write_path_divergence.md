# MT-3: Write-Path Divergence Report

## Config
- **Commit**: `a21b666`
- **Seed**: 42
- **Device**: cuda
- **Dataset**: CIFAR-100 DINOv2 ViT-B/14 features (768-d)
- **Max entries**: 20,000
- **Batch size**: 256
- **Batches**: 100 (25,600 samples)
- **Eval subset**: 2000 test samples
- **Determinism**: fixed RNG seed, monkey-patched `time.time()` for deterministic `last_seen`
- **CUDA deterministic**: `torch.backends.cudnn.deterministic=True`

## Results

| Checkpoint (N) | Pair | Cos sim (mean) | Cos sim (min) | Cos sim (p5) | Occupancy agree | Acc left | Acc right | Acc delta | Eviction overlap | Evictions L/R |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | S1 vs S2 | 1.000000 | 1.000000 | 1.000000 | 1.0000 | 82.80% | 82.80% | 0.00pp | 1.0000 | 0/0 |
| 10 | S1 vs S3 | 0.058556 | -0.083916 | -0.020947 | 0.9997 | 82.80% | 83.80% | 1.00pp | 1.0000 | 0/0 |
| 50 | S1 vs S2 | 1.000000 | 1.000000 | 1.000000 | 1.0000 | 87.05% | 87.05% | 0.00pp | 1.0000 | 0/0 |
| 50 | S1 vs S3 | 0.059755 | -0.087029 | -0.018595 | 0.9992 | 87.05% | 87.75% | 0.70pp | 1.0000 | 0/0 |
| 100 | S1 vs S2 | 1.000000 | 1.000000 | 1.000000 | 1.0000 | 88.50% | 88.50% | 0.00pp | 1.0000 | 4252/4252 |
| 100 | S1 vs S3 | 0.059855 | -0.087029 | -0.018238 | 1.0000 | 88.50% | 88.65% | 0.15pp | 0.2487 | 4252/4265 |

## Nondeterminism Sources

- **`time.time()` in `learn_local()`**: Used for `last_seen` timestamps which feed into LFU-LRU eviction tiebreaking. Mitigated by monkey-patching to a deterministic counter (1s increments per call).
- **CUDA nondeterminism**: `atomicAdd` in scatter operations can produce floating-point non-associativity. Mitigated with `torch.backends.cudnn.deterministic=True`, but scatter_add on GPU may still vary.
- **`torch.unique()` ordering**: Used in `learn_local()` for scatter-mean over hit slots. Documented as stable on CUDA for identical inputs.
- **RNG state**: Fully captured via `torch.manual_seed` / `random.seed` / `np.random.seed`.

## Conclusion

**Same-order determinism**: CONFIRMED. S1 vs S2 are bitwise identical across all checkpoints.
**Order sensitivity**: CONFIRMED. The write path is **order-sensitive**. First detected at N=10 via cosine similarity (0.058556).

This is expected: EMA-based key drift, hit-count–adaptive alpha, and LFU-LRU eviction all depend on the order in which prototypes are created and updated. Reversed batch order changes which prototypes are allocated first, which accumulate more hits, and which get evicted.
