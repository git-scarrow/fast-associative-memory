import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from fast_associative_memory import FastAssociativeMemory

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("FAISS not found.")
    print("  GPU:  conda install -c pytorch faiss-gpu")
    print("  CPU:  pip install faiss-cpu  (less representative, but runnable)")


def run_speed_test():
    if not torch.cuda.is_available():
        print("Skipping G5: GPU required for a fair comparison.")
        return

    if not HAS_FAISS:
        print("Skipping G5: FAISS not installed. See instructions above.")
        return

    D = 1024
    N = 50000
    Q = 1000
    DEVICE = "cuda"

    print(f"\n\U0001f3ce\ufe0f  SPEED GAUNTLET (G5) \U0001f3ce\ufe0f")
    print(f"N={N} prototypes, D={D}, Q={Q} queries, k=25")

    db = torch.nn.functional.normalize(torch.randn(N, D, device=DEVICE), dim=-1)
    queries = torch.nn.functional.normalize(torch.randn(Q, D, device=DEVICE), dim=-1)

    # --- FAISS ---
    # Attempt GPU index; fall back to CPU flat index gracefully
    try:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, D)
        faiss_mode = "GPU"
    except AttributeError:
        index = faiss.IndexFlatIP(D)
        faiss_mode = "CPU"
    index.add(db.cpu().numpy())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, _ = index.search(queries.cpu().numpy(), 25)
    torch.cuda.synchronize()
    t_faiss = (time.perf_counter() - t0) * 1000 / Q
    print(f"FAISS FlatIP ({faiss_mode}): {t_faiss:.3f} us/query")

    # --- FAM ---
    fam = FastAssociativeMemory(D, 10, core_entries=N).to(DEVICE)
    fam.core_cam.keys[:] = db
    fam.core_cam.occupied[:] = True
    fam.core_cam._update_key_norm(torch.arange(N, device=DEVICE))

    with torch.no_grad():
        _ = fam(queries[:10])  # warmup

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = fam(queries)
    torch.cuda.synchronize()
    t_fam = (time.perf_counter() - t0) * 1000 / Q
    print(f"FAM Forward (GPU):    {t_fam:.3f} us/query")

    ratio = t_fam / t_faiss
    print(f"FAM / FAISS ratio: {ratio:.2f}x")
    if ratio < 1.5:
        print("\u2705  G5 PASS: FAM within 1.5x of FAISS Flat.")
    else:
        print(f"\u26a0\ufe0f  G5 WARNING: FAM is {ratio:.2f}x slower \u2014 the 'Fast' claim needs qualification.")


if __name__ == "__main__":
    run_speed_test()
