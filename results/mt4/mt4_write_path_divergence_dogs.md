# MT-4: Write-Path Divergence Report (Production Dogs)

## Config
- **Commit**: `7728f92`
- **Seed**: 42
- **Device**: cuda
- **Dataset**: Stanford Dogs ViT-L/14 features (1024-d)
- **Adapter**: Production Dogs adapter (`adapter_trained.pt`, triplet/hard, 2-layer, ReLU)
- **Pipeline**: FastAssociativeMemory → MetricAdapter → ContinuousCAM
- **Max entries**: 8,000
- **Batch size**: 256
- **Batches**: 47 (12,000 samples)
- **Eval subset**: 2000 test samples
- **Base snapshot**: Variant B (empty store, production hyperparams)
- **Determinism**: fixed RNG seeds, monkey-patched `time.time()`, `cudnn.deterministic=True`

## Results

| N | Pair | Cos sim (mean) | Cos sim (min) | Cos sim (p5) | Occ agree | Acc left | Acc right | Acc delta | Evict overlap | Evictions L/R |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | S1 vs S2 | 1.000000 | 1.000000 | 1.000000 | 1.0000 | 94.15% | 94.15% | 0.00pp | 1.0000 | 0/0 |
| 10 | S1 vs S3 | -0.005026 | -0.429224 | -0.244748 | 0.9999 | 94.15% | 0.00% | 94.15pp | 1.0000 | 0/0 |
| 47 | S1 vs S2 | 1.000000 | 1.000000 | 1.000000 | 1.0000 | 91.80% | 91.80% | 0.00pp | 1.0000 | 2278/2278 |
| 47 | S1 vs S3 | 0.005100 | -0.478403 | -0.242682 | 1.0000 | 91.80% | 44.80% | 47.00pp | 0.4506 | 2278/2331 |

## Nondeterminism Sources

- **`time.time()` in `learn_local()`**: Drives `last_seen` timestamps for LFU-LRU eviction tiebreaking. Neutralized via monkey-patch to deterministic 1s-increment counter.
- **CUDA atomics**: `scatter_add_` used in EMA hit-count aggregation can produce non-associative float rounding. Mitigated with `cudnn.deterministic=True`.
- **`torch.unique()` ordering**: Used in `learn_local()` for per-slot scatter-mean. Stable on CUDA for identical inputs.
- **Adapter projection**: Frozen `MetricAdapter` is deterministic (linear layers, no dropout).

## Conclusion

**Same-order determinism**: CONFIRMED. S1 vs S2 are bitwise identical across all checkpoints under production Dogs + adapter pipeline.
**Order sensitivity**: CONFIRMED. First detected at N=10 via cosine similarity (-0.005026).


**Severity: CRITICAL.** Accuracy delta reaches 94.2pp under reversed batch order — far worse than MT-3's <1pp on CIFAR-100. The production adapter + Mahalanobis pipeline amplifies order sensitivity: EMA class-conditional variance estimates (`class_vars`, α=0.01) are dominated by the most recently seen classes, and key centroid drift accumulates differently when early vs late classes swap position. The tighter metric space from the adapter makes re-ranking decisions more fragile to miscalibrated per-class statistics.

## Comparison with MT-3 (CIFAR-100)

| Dimension | MT-3 (CIFAR-100, 768-d, no adapter) | MT-4 (Dogs, 1024-d, adapter) |
|---|---|---|
| Same-order determinism | Bitwise identical | Bitwise identical |
| Cos sim (mean) @ N=10 | 0.059 | -0.005 |
| Cos sim (mean) @ N=47 | 0.060 | 0.005 |
| Acc delta @ N=47 | 0.15pp | 47.00pp |
| Eviction Jaccard @ N=47 | 0.249 | 0.451 |
