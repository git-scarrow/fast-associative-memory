# HCA-DS-9: bcachefs Structural Privilege Topology Mapping

## 1. Executive Summary

This document establishes a formal structural correspondence between the Hyperbolic Context Architecture (HCA) and the bcachefs filesystem. Both systems exhibit **hierarchical privilege** — a distinguished center (system token / root node) that receives preferential treatment via geometric or structural mechanisms. The mapping is not metaphorical: it identifies concrete functional correspondences between four subsystem pairs, derives formal relationships, and characterizes where the analogy holds tightly and where it breaks.

**Key results:**

1. **B-tree ↔ Radial position.** The B-tree depth-to-level mapping inverts naturally into a radial coordinate: $r(l) = R_{\max}(1 - l/L)$ where $l$ is the btree level and $L$ is tree depth. Root privilege arises from identical mechanisms — exponential fan-out in bcachefs (hundreds of children per node) mirrors exponential volume growth in hyperbolic space ($\text{vol}(B_r) \sim e^{(d-1)r}$). Both produce a single point (root/origin) that structurally dominates all others.

2. **Journal ↔ EMA tracking.** Journal sequence numbers provide a total order with crash-recovery guarantees; EMA tracking provides a continuous temporal signal with exponential decay. Both enforce "happens-before" relationships — the journal explicitly via replay ordering, the EMA implicitly via recency weighting. The journal's pin mechanism maps to the QoS monitor's state machine hysteresis.

3. **Allocation groups ↔ Attention heads.** Open buckets enable concurrent sequential writes across devices; attention heads enable parallel privilege channels across representation subspaces. Both use $H$ independent channels ($H$ open buckets / $H$ attention heads) with per-channel state and eventual consistency at the coordination layer.

4. **Tiering ↔ Per-layer curvature.** The foreground/background/promote target hierarchy maps to a radial shell structure with per-layer curvature controlling access speed. The promote-on-read mechanism is the Hebbian pull (DS-6 Section 5). The rebalance thread is the $\gamma$ controller (DS-7 Section 4). LRU cache eviction of demoted copies maps to prototype eviction in FAM.

---

## 2. Mapping 1: B-tree Hierarchy ↔ Radial Position

### 2.1 Structural Correspondence

| bcachefs B-tree | HCA Hyperbolic Space |
|---|---|
| Root node (level $L$) | System token at origin ($r = 0$) |
| Interior node (level $l$, $0 < l < L$) | Intermediate tokens ($0 < r < R_{\max}$) |
| Leaf node (level 0) | Context tokens at boundary ($r \approx R_{\max}$) |
| Node fanout $F$ ($\sim 100$–$1000$) | Exponential volume: $\text{vol}(B_r) \propto e^{(d-1)r}$ |
| Key count per level: $F^{L-l}$ | Token count at radius $r$: $\propto e^{(d-1)r}$ |
| B-tree depth $L$ (typically 2–4) | Radial extent $R_{\max} = \operatorname{arcosh}(R_{\text{float}}) / \sqrt{c}$ |

### 2.2 Depth-to-Radius Function

The natural mapping from btree level $l \in \{0, 1, \ldots, L\}$ to hyperbolic radius $r$ is:

$$r(l) = R_{\max} \cdot \left(1 - \frac{l}{L}\right)$$

- Root ($l = L$): $r = 0$ (origin, maximum privilege)
- Leaves ($l = 0$): $r = R_{\max}$ (boundary, minimum privilege)

**Why linear?** The key correspondence is between **exponential growth rates**, not between levels directly. In bcachefs, the number of nodes at level $l$ is $\sim F^{L-l}$, which grows exponentially as $l$ decreases. In hyperbolic space, the number of tokens that can fit at radius $r$ grows as $\sim e^{(d-1)r}$. Equating:

$$F^{L-l} \sim e^{(d-1) \cdot r(l)}$$

Taking logs: $r(l) = \frac{(L - l) \ln F}{d - 1}$. Setting $R_{\max} = \frac{L \ln F}{d - 1}$, we recover $r(l) = R_{\max}(1 - l/L)$. The linear mapping is the correct one because both systems have the same exponential growth law.

### 2.3 Privilege Mechanism Correspondence

**bcachefs root privilege** emerges from:
- **Structural centrality:** Every key lookup traverses the root. The root is on every access path.
- **Lock contention:** Root splits are the most expensive operation (create new root, grow tree depth by 1). The system avoids root splits by using large nodes (256 KiB) with high fanout.
- **Caching:** The root node is always hot in the page cache. Interior nodes near the root are cached with high probability.

**HCA origin privilege** emerges from:
- **Geometric centrality:** The origin has minimum distance to all points. The radial decay $\lambda(r) = \gamma \ln(\cosh(r))$ penalizes tokens proportionally to their distance from the origin.
- **Attention concentration:** At equilibrium, the system token receives $\alpha_s \geq 0.5$ of attention mass (DS-8 fixed point).
- **Curvature protection:** The QoS monitor (DS-7) actively maintains origin privilege via the $\gamma$ controller.

**Key difference:** bcachefs root privilege is *passive* (structural consequence of the tree), while HCA origin privilege is *active* (maintained by controllers). This is because bcachefs has a fixed topology (tree structure doesn't change without splits), while HCA has a learned geometry (token positions evolve during training).

### 2.4 Log-Structured Nodes ↔ Bset Geometry

A bcachefs node contains multiple **bsets** — sorted runs of keys arranged in a geometric size progression (8–16 KiB active bset, then progressively larger frozen bsets). This mirrors the **exponential volume shells** in hyperbolic space:

- Active bset (small, mutable) ↔ Tokens near the origin (few, high-privilege, frequently updated)
- Frozen bsets (large, read-only, with precomputed auxiliary search trees) ↔ Tokens at large radius (many, low-privilege, with cached attention patterns)

The auxiliary search trees precomputed over frozen bsets correspond to **cached attention indices** in sparse attention (DS-8 Section 7): once tokens at large radius are stable, their attention contributions can be precomputed and reused.

### 2.5 Node Splitting ↔ Curvature Expansion

When a bcachefs leaf node overflows, it splits — allocating new child nodes and potentially growing tree depth by 1 (new root). In HCA:

- **Leaf split** ↔ **Radial compression detection** (DS-7 Section 5.2: $x_{0,75}/x_{0,25} < 2$). When too many tokens crowd a radial shell, the curvature controller increases $c$, expanding $R_{\max}$ and giving tokens more room.
- **Root split** (depth increase) ↔ **No HCA equivalent.** The origin is always a single point. This is a fundamental asymmetry: bcachefs can grow its hierarchy, while HCA's radial structure is continuous and doesn't have discrete "levels" to add.

### 2.6 Locking ↔ Attention Exclusion

bcachefs uses SIX locks (Shared, Intent, Exclusive) with optimistic concurrency via lock sequence numbers. In HCA:

- **Shared lock** ↔ **Standard attention read** (multiple heads can read the same token simultaneously)
- **Intent lock** ↔ **QoS monitor observation** (reads $\alpha_s$ without modifying geometry, but signals intent to adjust)
- **Exclusive lock** ↔ **Geometry update** (occurs every $k = 100$ steps; during the update, the controller has exclusive write access to $\gamma_{\text{raw}}$)
- **Sequence numbers** for optimistic concurrency ↔ **EMA versioning** ($\hat{\alpha}_s$ is a versioned estimate; stale reads are acceptable because the EMA converges)

---

## 3. Mapping 2: Journal ↔ EMA Temporal Tracking

### 3.1 Ordering Mechanism Comparison

| Property | bcachefs Journal | HCA EMA Tracker |
|---|---|---|
| Ordering type | Total order (strict monotone $\text{seq}$) | Partial order (recency-weighted, no strict ordering) |
| Ordering primitive | 64-bit sequence number | Exponential moving average ($\rho = 0.99$) |
| Crash recovery | Replay journal entries; ignore bsets with $\text{seq}$ > last journal entry | No crash model; training restarts from checkpoint |
| Temporal resolution | Per-commit (batched, ~1s flush delay) | Per-geometry-update ($k = 100$ training steps) |
| Persistence | On-disk log region | In-memory state (lost on restart) |
| Ordering guarantee | If entry $A$ committed before $B$, $A.\text{seq} < B.\text{seq}$ | If update $A$ occurred before $B$, $\hat{\alpha}_s$ reflects $B$ more than $A$ (exponential decay) |

### 3.2 Journal Pins ↔ QoS State Machine Hysteresis

bcachefs **journal pins** prevent journal entries from being reclaimed while dependent operations are outstanding. A dirty btree node pins all journal entries containing keys that haven't been flushed to the btree yet.

In HCA, the QoS monitor's **state machine** (DS-7 Section 2.3) provides analogous temporal persistence:

| bcachefs Pin Concept | HCA QoS Analog |
|---|---|
| Dirty node pins journal entry | DEGRADED state pins corrective action (must observe recovery before transitioning to HEALTHY) |
| Pin flushing order (key cache before btree) | State transition order (must clear ALERT before DEGRADED → HEALTHY, 0.05 hysteresis) |
| Journal reclaim thread (background cleanup) | $\gamma$ relaxation path ($K_r = 0.1$, 5× slower than corrective path) |
| Journal watermark (throttle new writes when space low) | Emergency multiplier (3× at CRITICAL, throttle normal operation to restore privilege) |

### 3.3 Sequence Number ↔ Geometry Update Step

The journal's sequence number $\text{seq}$ maps directly to the geometry update step $n$ in HCA:

$$n = \lfloor t / k \rfloor$$

where $t$ is the training step and $k = 100$ is the update cadence. Both are strictly monotonic integers that define a total order over state mutations. The key difference:

- **Journal:** each $\text{seq}$ records a *set* of btree mutations (batched commit). Replay applies them atomically.
- **Geometry update:** each $n$ records a *single* controller output ($\Delta g_\gamma$, $\Delta g_c$). Application is incremental (gradient step), not atomic.

### 3.4 Coalescing ↔ EMA Smoothing

The bcachefs **write buffer** coalesces multiple journal entries for the same btree key — if a key is updated 10 times before flushing to the btree, only the final value is written. This is structurally identical to EMA smoothing:

$$\hat{\alpha}_s(n) = \rho^n \hat{\alpha}_s(0) + (1 - \rho) \sum_{i=0}^{n-1} \rho^{n-1-i} \alpha_s(i)$$

Both discard intermediate states in favor of the "current best estimate." The write buffer's coalescing is exact (last-writer-wins); the EMA's smoothing is approximate (exponential weighting). But the information-theoretic effect is the same: high-frequency fluctuations are suppressed, and only the persistent signal reaches the downstream consumer (btree / controller).

### 3.5 Idempotent Replay ↔ Controller Monotonicity

bcachefs journal replay is **idempotent** — re-inserting a journaled key into the btree is safe because btree updates are overwrites. This corresponds to the $\gamma$ controller's **monotone convergence** (DS-8 Section 5.1): the controller's corrective action always pushes $\gamma$ toward $\gamma^*$ regardless of the starting point, and re-applying the same correction doesn't cause oscillation or divergence.

---

## 4. Mapping 3: Allocation Groups ↔ Attention Heads

### 4.1 Parallel Channel Correspondence

| bcachefs Allocation | HCA Attention Heads |
|---|---|
| Open bucket (currently writing) | Attention head (currently computing) |
| $H_{\text{dev}}$ devices with independent allocators | $H$ attention heads with independent $\gamma^{(h)}$, $c^{(h)}$ |
| Bucket = fixed-size allocation unit | Attention head = fixed-dimension subspace ($d_{\text{head}} = d/H$) |
| Sequential write within bucket | Sequential token processing within head |
| Bucket sealing (full → read-only) | Head output (attention-weighted value → frozen until next forward pass) |
| Generation number (lazy invalidation) | EMA decay ($\rho^n$ naturally invalidates old observations) |

### 4.2 Parallelism Factor

bcachefs maintains one open bucket per (device, data type) pair. With $D$ devices and $T$ data types:

$$H_{\text{bcachefs}} = D \times T$$

HCA maintains $H$ attention heads per layer, each with independent geometric parameters. The parallelism factor is:

$$H_{\text{HCA}} = H \quad \text{(typically 8, 16, 32, or 64)}$$

Both systems achieve concurrent operation through **independence at the channel level** with **coordination at the commit/output level**:

- bcachefs: Independent bucket writes → coordinated btree update (write buffer flush is single-threaded)
- HCA: Independent head attention → coordinated output (concatenation + output projection $W_O$)

### 4.3 Generation Numbers ↔ EMA Decay

bcachefs bucket generation numbers enable **lazy invalidation**: when a bucket is reused, its generation number increments, automatically invalidating all stale pointers without scanning.

The HCA EMA tracker achieves the same effect through **exponential decay**: old observations of $\alpha_s$ are exponentially down-weighted with factor $\rho^{\Delta n}$, where $\Delta n$ is the number of updates since the observation. No explicit invalidation is needed — staleness decays naturally.

| Property | Generation Number | EMA Decay |
|---|---|---|
| Invalidation trigger | Explicit (generation increment on reuse) | Implicit (time passage, $\rho < 1$) |
| Invalidation scope | Per-bucket (atomic) | Per-observation (continuous) |
| Cost | O(1) — just increment counter | O(1) — just multiply by $\rho$ |
| Stale detection | Compare generation in pointer vs. bucket | No detection needed; stale values have negligible weight |

### 4.4 Stripe Allocation ↔ Multi-Head Coordination

bcachefs erasure coding uses **stripes**: one bucket per device, with parity computed across the stripe. All data buckets must fill before parity is written. This maps to **multi-head attention coordination**:

- Stripe = one forward pass across all heads
- Data bucket per device = attention computation per head
- Parity computation = output projection $W_O$ (combines head outputs)
- Stripe completion = forward pass completion (all heads must finish)
- Extra replica buckets (dropped after parity) = residual connection (original representation preserved alongside attention output)

### 4.5 Where the Analogy Breaks

**Buckets are homogeneous within a device.** All buckets on a device have the same performance characteristics. Attention heads are **heterogeneous** — different heads learn different functions (some attend locally, some globally, some to specific syntactic patterns). There is no bcachefs analog of "head specialization."

**Allocation is spatial; attention is semantic.** Bucket allocation decides *where* on disk to place data based on space availability. Head attention decides *how much weight* to give each token based on learned query-key similarity. The parallelism structure is analogous, but the decision criterion is fundamentally different.

---

## 5. Mapping 4: Device Tiering ↔ Per-Layer Curvature

### 5.1 Tier-to-Radial-Shell Mapping

| bcachefs Target | HCA Radial Shell | Access Characteristic |
|---|---|---|
| `foreground_target` (fast SSD) | Near-origin shell ($r < \delta/2$) | Lowest latency, highest privilege, limited capacity |
| `promote_target` (cache SSD) | Origin-adjacent cache ($r \approx 0$) | Temporary high-privilege position for recently accessed tokens |
| `background_target` (slow HDD) | Boundary shell ($r \approx R_{\max}$) | High capacity, high latency, low privilege |
| `metadata_target` (NVMe/Optane) | Origin itself ($r = 0$) | System token position; ultra-low latency for control metadata |

### 5.2 Promote ↔ Hebbian Pull

bcachefs **promote-on-read**: when data on the background device is read, a cached copy is created on the promote device (foreground). This is a **foreground operation** — it happens during the read path, not as background work.

HCA **Hebbian pull** (DS-6 Section 5): when a context token receives high attention from an important query, its radial position is pulled toward the origin. This is also a **foreground operation** — it happens during the forward pass (or at geometry update time).

| Property | bcachefs Promote | HCA Hebbian Pull |
|---|---|---|
| Trigger | Read miss on promote device | High attention weight ($\alpha > \alpha_{\text{threshold}}$) |
| Effect | Copy data to fast device, mark as cached | Decrease token radius (move toward origin) |
| Latency impact | Subsequent reads are fast (cache hit) | Subsequent attention is stronger (closer to origin → higher $\alpha_s$) |
| Persistence | Cached copy; evicted by LRU when space needed | Position update via $\exp$ map; can drift back if not reinforced |
| Cost | I/O bandwidth for copy | Gradient computation for position update |

### 5.3 Rebalance Thread ↔ $\gamma$ Controller

bcachefs **rebalance thread**: a kernel background thread that migrates data from foreground to background targets. Data on the foreground device is copied to the background device and then marked as cached (demoted).

HCA **$\gamma$ controller** (DS-7 Section 4): a background controller (runs every $k$ steps) that adjusts the decay strength $\gamma$. Increasing $\gamma$ "demotes" peripheral tokens (increases the attention penalty for large radius), effectively migrating attention weight from boundary tokens to origin.

| Property | bcachefs Rebalance | HCA $\gamma$ Controller |
|---|---|---|
| Runs | Background kernel thread | Every $k = 100$ training steps |
| Direction | Foreground → background (demotion) | Increases $\gamma$ → penalizes boundary (demotion) |
| Trigger | Space pressure on foreground device | $\hat{\alpha}_s < \alpha_{\text{safe}}$ (privilege loss) |
| Rate control | Configurable delay; respects I/O budget | $K_p = 0.5$, $\Delta g_{\max} = 1.0$; dead zone $\varepsilon = 0.05$ |
| Equilibrium | Steady-state: hot data on SSD, cold on HDD | Steady-state: $\gamma^* \approx 0.60$ (DS-8) |

### 5.4 LRU Cache Eviction ↔ Prototype Eviction

bcachefs cached copies (promoted data, demoted foreground copies) are evicted in **LRU order** when the cache device needs space. In FAM (the underlying memory system):

- **LRU eviction** (default mode): evict the least-recently-seen prototype (identical mechanism)
- **Coverage eviction** (`use_lfu=True`): evict the prototype with the nearest same-class neighbor (most replaceable)
- **Adaptive eviction** (`adaptive_eviction=True`): blend LRU and coverage based on class loss signal

The bcachefs model maps exactly to FAM's LRU mode. Coverage eviction has no bcachefs analog — bcachefs doesn't consider data similarity when evicting, only recency.

### 5.5 Durability ↔ Prototype Persistence

bcachefs `durability=0` means a device is pure cache — data there doesn't count toward replica requirements. `durability=2` means data counts as two replicas.

In HCA/FAM:
- **Ephemeral prototypes** (recently learned, not yet consolidated) ↔ `durability=0` cache copies
- **Consolidated prototypes** (high access count, validated by evaluation) ↔ `durability≥1` persistent data
- The FAM `access_count` field serves as a proxy for durability — frequently accessed prototypes are "more durable" (harder to evict under LRU).

### 5.6 Configurationless Tiering ↔ Learnable Curvature

bcachefs **members v2** auto-detects device performance (stores IOPS measurements in superblock) and automatically assigns foreground to the fastest device, background to the slowest.

HCA **learnable curvature** (DS-5): $c$ is a per-layer parameter that adapts during training. Layers that need tighter hierarchical control (more privilege concentration) learn higher $c$; layers that need flatter attention learn lower $c$.

Both systems achieve **self-organizing hierarchy** without manual configuration:
- bcachefs: hardware probing → automatic tier assignment
- HCA: gradient descent → automatic curvature setting

---

## 6. Cross-Cutting Structural Correspondence Table

| bcachefs Concept | HCA Concept | Correspondence Strength | Notes |
|---|---|---|---|
| Root node at max depth | System token at origin | **Strong** | Both derive privilege from geometric centrality |
| 256 KiB log-structured nodes | Exponential volume shells | **Strong** | Geometric bset sizes ↔ exponential volume growth |
| B-tree depth (2–4 levels) | Radial extent $R_{\max}$ | **Moderate** | Discrete vs. continuous; $r(l) = R_{\max}(1 - l/L)$ |
| Node splitting | Curvature expansion ($c$ increase) | **Moderate** | bcachefs grows depth; HCA expands $R_{\max}$ |
| Journal sequence number | Geometry update step $n$ | **Strong** | Both are monotonic total orders over state mutations |
| Journal pins | QoS state machine hysteresis | **Moderate** | Pins are explicit; hysteresis is threshold-based |
| Journal coalescing | EMA smoothing | **Strong** | Both suppress high-frequency noise, preserve persistent signal |
| Idempotent replay | Controller monotone convergence | **Strong** | Both are safe to re-execute without divergence |
| Open buckets ($H$ per device) | Attention heads ($H$ per layer) | **Strong** | Independent parallel channels with coordination at output |
| Bucket generation numbers | EMA exponential decay | **Moderate** | Explicit vs. implicit invalidation; same O(1) cost |
| Stripe parity | Output projection $W_O$ | **Weak** | Both combine parallel channel outputs; different operations |
| Foreground target (SSD) | Near-origin shell | **Strong** | Fast access, limited capacity, high privilege |
| Background target (HDD) | Boundary shell | **Strong** | Slow access, large capacity, low privilege |
| Promote-on-read | Hebbian pull | **Strong** | Foreground operation, read-triggered, caches at fast tier |
| Rebalance thread | $\gamma$ controller | **Strong** | Background demotion, rate-controlled, equilibrium-seeking |
| LRU cache eviction | FAM LRU eviction | **Exact** | Identical mechanism |
| `durability=0` cache | Ephemeral prototypes | **Moderate** | Both are expendable copies not counted for persistence |
| Configurationless tiering | Learnable curvature | **Strong** | Self-organizing hierarchy from performance measurement |
| SIX locks (optimistic) | EMA versioning (stale-tolerant) | **Moderate** | Both accept stale reads; differ in concurrency model |

---

## 7. Formal Mapping Summary

### 7.1 The Depth-Radius Isomorphism

For a bcachefs B+tree with fanout $F$ and depth $L$, and a hyperbolic space with dimension $d$ and maximum radius $R_{\max}$:

$$r: \{0, 1, \ldots, L\} \to [0, R_{\max}], \quad r(l) = R_{\max}\left(1 - \frac{l}{L}\right)$$

The capacity at each level matches:

$$\text{bcachefs nodes at level } l: \quad N(l) = F^{L-l}$$
$$\text{HCA tokens at radius } r: \quad n(r) \propto e^{(d-1)r}$$

Setting $F^{L-l} = e^{(d-1) \cdot R_{\max}(1 - l/L)}$ yields:

$$(L - l) \ln F = (d-1) R_{\max} \left(1 - \frac{l}{L}\right)$$

This holds for all $l$ when $R_{\max} = \frac{L \ln F}{d - 1}$.

**Example:** $F = 500$ (typical bcachefs fanout), $L = 3$, $d = 65$ (64-dim head + 1 temporal):

$$R_{\max} = \frac{3 \times 6.21}{64} = 0.291$$

In the Lorentz model with $c = 1$, this gives $x_0 = \cosh(0.291) \approx 1.043$ — a modest radial extent, consistent with most tokens being near the boundary.

### 7.2 The Temporal Ordering Homomorphism

The journal's total order and the EMA's partial order are related by a **forgetful functor**: the EMA preserves the ordering direction (later events dominate) but loses the exact sequence (old events blur together):

$$\text{Journal}: (\text{seq}_1 < \text{seq}_2) \implies (\text{entry}_1 \text{ applied before entry}_2)$$
$$\text{EMA}: (n_1 < n_2) \implies (\alpha_s(n_1) \text{ has weight } \rho^{n_2 - n_1} \text{ vs. } (1-\rho) \text{ for } \alpha_s(n_2))$$

The journal provides **exact recovery** (replay to any sequence number). The EMA provides **approximate tracking** (current estimate with exponential forgetting). The trade-off: the journal requires $O(\text{seq})$ storage; the EMA requires $O(1)$ storage.

### 7.3 The Parallelism Isomorphism

Both systems decompose a global operation into $H$ independent channels:

$$\text{bcachefs}: \quad \text{Global write} = \bigsqcup_{h=1}^{H} \text{Bucket}_h \quad \to \quad \text{Btree flush (single-threaded)}$$
$$\text{HCA}: \quad \text{Global attention} = \bigoplus_{h=1}^{H} \text{Head}_h \quad \to \quad W_O \text{ projection (single output)}$$

The structural parallel is that both systems have a **fan-out → independent operation → fan-in** architecture with a coordination bottleneck at the fan-in stage (write buffer flush / output projection).

---

## 8. Where the Mapping Breaks

### 8.1 Fundamental Asymmetries

| Property | bcachefs | HCA | Why It Matters |
|---|---|---|---|
| Topology | Discrete tree, mutable (splits) | Continuous manifold, fixed topology | bcachefs can grow depth; HCA cannot add "levels" |
| Privilege source | Passive (structural) | Active (controller-maintained) | HCA requires energy (gradient) to maintain privilege |
| Consistency model | Crash-consistent (journal + COW) | No crash model (training checkpoints) | bcachefs must handle partial writes; HCA doesn't |
| Data semantics | Opaque bytes | Learned embeddings with semantic structure | HCA tokens have meaningful distances; btree keys don't |
| Eviction criterion | LRU only | LRU, coverage, or adaptive | HCA has richer eviction policies than bcachefs |
| Channel specialization | Homogeneous buckets | Heterogeneous heads | Attention heads learn different functions; buckets don't |

### 8.2 Predictive Limitations

The mapping should **not** be used to:
- Import bcachefs's crash recovery model into HCA (HCA doesn't need it; training checkpoints serve a different purpose)
- Import HCA's semantic distance into bcachefs (btree keys are ordered lexicographically, not by "meaning")
- Assume bcachefs allocation strategies will work for attention weight distribution (different optimization landscapes)

---

## 9. Open Questions and DS-10 Handoff

### 9.1 Open Questions

**OQ-1: Backpointer ↔ Inverse attention.** bcachefs maintains a **backpointers btree** — a reverse index from disk extents to btree keys, enabling efficient data movement (copygc can find all references to a bucket without scanning the entire extents tree). Does HCA need an inverse attention map for targeted updates? If a token at large radius needs to be promoted, the current system must recompute attention; a backpointer index could enable O(1) lookup of which queries attend to it.

**OQ-2: Snapshot isolation ↔ Multi-task attention.** bcachefs snapshots use a **snapshot tree** where each snapshot ID partitions the key space. Multiple snapshots can coexist, sharing common data. Does this map to multi-task attention, where different tasks "see" different subsets of the context through task-specific masks? The snapshot ID in the bpos key would correspond to a task identifier in the attention pattern.

**OQ-3: Write buffer bottleneck ↔ Output projection bottleneck.** bcachefs's single-threaded write buffer flush is a known performance bottleneck. HCA's output projection $W_O$ is the analogous fan-in bottleneck. Can bcachefs's ongoing efforts to parallelize the write buffer (Kent Overstreet has discussed this) inform HCA's multi-head coordination?

### 9.2 DS-10 Handoff

DS-10 should address the **implementation boundary** — where the theoretical HCA design meets concrete PyTorch code:

1. **Lorentz-model attention kernel:** Custom CUDA kernel for hyperbolic distance computation with fp32 stability guarantees (per `hyperbolic_adapter_operations.md`).
2. **Radial position tracking:** Data structure for maintaining per-token radial coordinates across the forward pass, with Hebbian pull updates.
3. **QoS monitor integration:** Where in the training loop the geometry update ($k = 100$ cadence) executes, how $\gamma_{\text{raw}}$ and $c_{\text{raw}}$ gradients are accumulated.
4. **Benchmark against standard attention:** FLOP count, memory overhead, and accuracy comparison on a reference task.

---

## Appendix A: Notation Reference

| Symbol | Definition |
|--------|-----------|
| $l$ | B-tree level (0 = leaf, $L$ = root) |
| $L$ | B-tree depth (number of interior levels) |
| $F$ | B-tree fanout (children per interior node) |
| $r(l)$ | Radial coordinate corresponding to level $l$ |
| $R_{\max}$ | Maximum radial extent of the Poincaré ball |
| $H$ | Number of parallel channels (open buckets / attention heads) |
| $\text{seq}$ | Journal sequence number (monotonic 64-bit integer) |
| $n$ | Geometry update step ($n = \lfloor t/k \rfloor$) |
| $\rho$ | EMA decay factor (0.99) |
| $D$ | Number of storage devices |
| $T$ | Number of data types (btree IDs) |

## Appendix B: Summary of Mappings

| Deliverable | bcachefs Subsystem | HCA Subsystem | Correspondence |
|---|---|---|---|
| (a) Depth → radius | B-tree node hierarchy | Radial position | $r(l) = R_{\max}(1 - l/L)$; exponential capacity match |
| (b) Journal → EMA | Journal seq numbers + pins | EMA tracker + QoS state machine | Forgetful functor: total order → exponential weighting |
| (c) Allocation → heads | Open buckets per device | Attention heads per layer | Fan-out → independent ops → fan-in coordination |
| (d) Tiering → curvature | Foreground/background/promote | Radial shells + Hebbian pull + $\gamma$ controller | Promote = pull, rebalance = $\gamma$ control, LRU = LRU |
