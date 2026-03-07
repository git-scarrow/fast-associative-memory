# GPU Cost Model: Hyperbolic vs Dot-Product Attention

## A) Executive Summary
- **Asymptotic Parity:** Both methods require $\mathcal{O}(d N^2 H)$ FLOPs and $\mathcal{O}(N d H)$ memory traffic using kernel fusion. The fundamental compute/memory complexity is identical.
- **The Unfused Memory Wall:** Naive PyTorch implementations using element-wise operations for hyperbolic distance will cause a massive $\mathcal{O}(N^2 H)$ explosion in HBM reads/writes. This turns attention into a memory-bound disaster (estimated 5–10× overhead).
- **SIMT Bottlenecks in Fused Kernels:** Inside a fused kernel (like FlashAttention), computing `arcosh` and the Poincaré distance fraction hits CUDA cores (SIMT). SIMT throughput is ~15× slower than Tensor Cores (TC) for FP16, meaning the non-linear math artificially bottlenecks the fast matrix multiply.
- **The Exponential Cancellation Trick:** The composition of softmax with the negative distance score allows for exact algebraic collapse. When $\tau=1$, the term $\exp(-\text{arcosh}(x))$ simplifies perfectly to $\frac{1}{x + \sqrt{x^2 -1}}$. This entirely removes the need for expensive `log` and `exp` operations!
- **The Lorentz Trump Card:** Switching from the Poincaré ball to the Lorentz (hyperboloid) model transforms the distance calculation into a pseudo-inner product. This allows the core distance to be evaluated natively on Tensor Cores with standard GEMM, nearly eliminating all arithmetic overhead.

## B) Cost Model Table

Parameters: Batch $B=1$, dtype=FP16/BF16 (2 bytes). 
Assume a single layer. Memory traffic for the "Fused" model assumes a FlashAttention-style setup where $S_{ij}$ is kept in SRAM. 

| Setting ($N$, $d$, $H$) | Baseline FLOPs <br> *(Tensor Cores)* | SIMT FLOPs <br> *(Hyp Math)* | Unfused Bytes <br> *(QK Read/Write)* | Fused Kernel <br> *(Traffic)* | Fused Overhead <br> *(Expected)* | PyTorch Overhead <br> *(Naive)* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2K**, 64, 16 | 8.6 GFLOPs | ~1.3 GFLOPs | Read: 8.4 MB <br> **Write: 134 MB** | ~14 MB | **~1.15×** | **~4.0×** |
| **8K**, 128, 32 | 550 GFLOPs | ~43 GFLOPs | Read: 134 MB <br> **Write: 4.3 GB** | ~268 MB | **~1.20×** | **~8.0×** |
| **32K**, 128, 32 | 8.8 TFLOPs | ~687 GFLOPs | Read: 537 MB <br> **Write: 68.7 GB** | ~1.1 GB | **~1.20×** | **> 12.0×** |

*Note on Batch Scaling:* Setting $B > 1$ linearly scales FLOPs and linearly scales reading queries. Keys and Values are read only once per batch during decoding, but the $O(N^2)$ SIMT operations scale with $B$ during prefill. The structural overhead ratios remain constant.

## C) Practical Recommendations

To deploy this in production and keep overhead $< 1.5\times$, a custom CUDA/Triton kernel is strictly required.

1. **Switch to the Lorentz space model:**
   The Lorentz pseudo-inner product $\langle x, y \rangle_\mathcal{L} = -x_0 y_0 + x_1 y_1 + \dots$ can be computed by trivially negating the 0-th coordinate of Key vectors right before executing a standard Tensor Core GEMM. This reduces the score formulation to $s_{ij} = -\text{arcosh}(-\langle q_i, k_j \rangle_\mathcal{L})$. This approach completely bypasses the broadcasting, denominator norm divisions, and vector math intrinsic to the Poincaré model.
2. **Exploit the Softmax-Arcosh Algebraic Cancellation:**
   Do not compute `arcosh` and then `exp` for the softmax.
   Since weights $w \propto \exp(-\text{arcosh}(x))$, use the exact substitution: 
   $w \propto \frac{1}{x + \sqrt{x^2 - 1}}$ (for $x > 1$). 
   This condenses 19+ FLOPs of transcendental functions into 5 fast PTX instructions (`Mul`, `Sub`, `Sqrt`, `Add`, `Rcp`), shielding the CUDA cores from becoming a pipeline bottleneck.
3. **Use FP32 for intra-kernel Distance math:**
   Poincaré denominator terms like $(1 - \|u\|^2)$ and Lorentz identities approach 0 as points approach the boundary constraint. This leads to catastrophic cancellation in FP16/BF16. Ensure your Tensor Cores accumulate GEMM results in FP32, and keep the distance calculation in FP32 prior to the Softmax operation.
4. **Precompute norms vectorially:**
   If sticking with Poincaré, precompute $\|q\|^2$ and $\|k\|^2$ outside the attention loop/kernel. Load them into SRAM uniformly with $Q$ and $K$ blocks to avoid re-computing norms $O(N^2)$ times.
5. **Add Epsilon Clamping:**
   Clamp the domain of `sqrt` to $> \epsilon$ and ensure $x \geq 1 + 1e^{-6}$ to prevent NaNs on the kernel gradients or forward pass when $x \to 1$ (which happens when $q=k$).

## D) Implications for Attention at Scale

### Approximate Attention Compatibilities
- **Sparse, Sliding Window, & Block-Level:** Very compatible. These approaches skip dense segments of the $N \times N$ map entirely. Hyperbolic scoring modifies the values of the blocks evaluated but keeps structural sparsity intact.
- **Low-Rank Linear Attention (Performer/Linformer):** **Incompatible.** Linear attention relies on algebraically decoupling the query and key via matrix associativity (i.e., operating on $K^T V$ first to avoid the $N \times N$ matrix completely). Hyperbolic distance securely entangles queries and keys via complex nonlinearities ($1/(x+\sqrt{x^2-1})$), barring this optimization. 

### Origin Privilege & Numerical Stability
Hyperbolic origins mathematically grant equal distance weighting across unaligned representations (acting as a natural back-off to a uniform distribution). Utilizing the Lorentz space recommendation natively preserves exact isometric embedding distances, thus directly retaining "origin privilege" behaviors without paying the prohibitive element-wise mathematical cost of the Poincaré formulation on hardware.

### Arithmetic Derivations (Caveats)
- **Baseline TC FLOPs:** The standard dot product requires $d$ MACs (2 FLOPs) per element pair. Total Dense FLOPs = $2 \times d \times N^2 \times H$. 
  - For $N=8K, d=128, H=32$: $2 \times 128 \times (8192)^2 \times 32 = 549.7$ GFLOPs.
- **Hyperbolic Math (SIMT FLOPs):** Computing the explicit Poincaré term $x = 1 + \frac{2(U + V - 2M)}{(1-U)(1-V)}$ assuming $U, V, M$ are known and precomputed fractions takes ~5 PTX instructions. The naive `arcosh` + `exp` adds ~15-19 FLOPs. Total of roughly 20 FP32 SIMT FLOPs per cell.
  - For $N=8K, H=32$: $20 \times (8192)^2 \times 32 = 42.94$ GFLOPs.
- **Why Transcendentals Dominate:** Contemporary GPUs have imbalanced ratios of Tensor Cores (specialized for Matrix Multiply) versus CUDA cores (SIMT, used for log/sqrt). 
  - Nvidia H100 SXM5: **989 TFLOPs** (FP16 TC) vs **67 TFLOPs** (FP32 SIMT). A ratio of ~15:1. 
  - While SIMT FLOPs are dwarfed by TC FLOPs algebraically (43G vs 550G), 43G executed at 67 TFLOPs throughput takes $0.64$ ms. The larger 550G at 989 TFLOPs throughput takes $0.55$ ms. 
  - Because $0.64\text{ ms} > 0.55\text{ ms}$, the kernel flips from being Matrix-Multiply bound to entirely Element-Wise SIMT bound, establishing a hard performance ceiling (the ~1.20× fused overhead derived above).
