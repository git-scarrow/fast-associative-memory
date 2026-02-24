import torch
import time
from fast_associative_memory import FastAssociativeMemory

try:
    import faiss
except ImportError:
    print("FAISS not installed. Install faiss-gpu or faiss-cpu to run G5.")
    exit()


def run_speed_test():
    if not torch.cuda.is_available():
        print("Skipping G5: GPU required for fair comparison.")
        return

    D = 1024
    N = 50000
    Q = 1000
    DEVICE = "cuda"
    
    print(f"\n\U0001f3ce\ufe0f  SPEED GAUNTLET (G5) \U0001f3ce\ufe0f")
    print(f"Database: {N} vectors (D={D}). Queries: {Q}")
    
    db = torch.nn.functional.normalize(torch.randn(N, D, device=DEVICE), dim=-1)
    queries = torch.nn.functional.normalize(torch.randn(Q, D, device=DEVICE), dim=-1)
    
    # 1. FAISS (Gold Standard)
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, D)
    index.add(db.cpu().numpy())
    
    torch.cuda.synchronize()
    t0 = time.time()
    _, _ = index.search(queries.cpu().numpy(), 25)
    torch.cuda.synchronize()
    t_faiss = (time.time() - t0) * 1000 / Q
    print(f"FAISS Flat (GPU): {t_faiss:.3f} us/query")
    
    # 2. FAM
    fam = FastAssociativeMemory(D, 10, core_entries=N).to(DEVICE)
    fam.core_cam.keys[:] = db
    fam.core_cam.occupied[:] = True
    fam.core_cam._update_key_norm(torch.arange(N, device=DEVICE))
    
    # Warmup
    with torch.no_grad():
        _ = fam(queries[:10])
    
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = fam(queries)
    torch.cuda.synchronize()
    t_fam = (time.time() - t0) * 1000 / Q
    print(f"FAM Forward:      {t_fam:.3f} us/query")
    
    ratio = t_fam / t_faiss
    print(f"FAM / FAISS ratio: {ratio:.2f}x")
    if t_fam < t_faiss * 1.5:
        print("\u2705  G5 PASS: Competitive latency (within 1.5x of FAISS Flat).")
    else:
        print(f"\u26a0\ufe0f  G5 WARNING: FAM is {ratio:.2f}x slower than FAISS \u2014 the \u2018Fast\u2019 claim needs qualification.")


if __name__ == "__main__":
    run_speed_test()
