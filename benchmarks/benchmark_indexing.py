"""
benchmarks/benchmark_indexing.py — The Indexing Speed Gauntlet (G5)

Compares FAM's custom matmul-based cosine retrieval against FAISS brute-force
(IndexFlatIP) on GPU. If FAM is slower than raw FAISS by more than 1.5x,
the "Fast" claim is at risk.

Requires: pip install faiss-gpu  (or faiss-cpu for CPU comparison)
Requires: CUDA GPU for a fair comparison.
"""

import torch
import time
import numpy as np

try:
    import faiss
except ImportError:
    print("FAISS not installed. Run: pip install faiss-gpu")
    raise SystemExit(1)

from fast_associative_memory import FastAssociativeMemory


def run_speed_test():
    D = 1024
    N = 50000  # Full FAM capacity
    Q = 1000   # Query batch size

    print(f"\n\U0001f3ce\ufe0f  G5: INDEXING SPEED TEST  \U0001f3ce\ufe0f")
    print(f"N={N} prototypes, D={D}, Q={Q} queries, k=25")

    db = torch.nn.functional.normalize(torch.randn(N, D), dim=-1).cuda()
    queries = torch.nn.functional.normalize(torch.randn(Q, D), dim=-1).cuda()

    # --- FAISS Flat (Brute Force, GPU) ---
    res = faiss.StandardGpuResources()
    index_flat = faiss.GpuIndexFlatIP(res, D)
    index_flat.add(db.cpu().numpy())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, _ = index_flat.search(queries.cpu().numpy(), 25)
    torch.cuda.synchronize()
    faiss_ms = (time.perf_counter() - t0) * 1000
    faiss_us_per_q = faiss_ms * 1000 / Q
    print(f"FAISS IndexFlatIP (GPU) : {faiss_ms:.2f} ms total  |  {faiss_us_per_q:.3f} \u03bcs/query")

    # --- FAM (Custom Matmul, GPU) ---
    fam = FastAssociativeMemory(input_dim=D, value_dim=10, core_entries=N).cuda()
    fam.core_cam.keys[:] = db
    fam.core_cam.occupied[:] = True
    fam.core_cam._update_key_norm(torch.arange(N, device="cuda"))

    # Warmup
    with torch.no_grad():
        _ = fam(queries[:10])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = fam(queries)
    torch.cuda.synchronize()
    fam_ms = (time.perf_counter() - t0) * 1000
    fam_us_per_q = fam_ms * 1000 / Q
    print(f"FAM Forward (GPU)       : {fam_ms:.2f} ms total  |  {fam_us_per_q:.3f} \u03bcs/query")

    ratio = fam_ms / faiss_ms
    print(f"\nFAM / FAISS ratio: {ratio:.2f}x")
    if ratio < 1.5:
        print("\u2705  G5 PASS: FAM is within 1.5x of raw FAISS brute-force.")
    else:
        print(f"\u26a0\ufe0f  G5 WARNING: FAM is {ratio:.2f}x slower than FAISS — the \u2018Fast\u2019 claim needs qualification.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("G5 requires a CUDA GPU for a fair comparison. Exiting.")
        raise SystemExit(0)
    run_speed_test()
