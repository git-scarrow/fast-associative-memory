"""HCA-MT-1: DS-5 Adaptive-Curvature Kernel Verification

Synthetic experiments verifying the adaptive-curvature enforcement mechanism
before Phase 2 implementation. Pure numerical — no GPU, no model weights.

Experiments:
  1. Static N sweep — α_s verification table
  2. Dynamic N ramp — α_s heatmaps + EMA tracking
  3. N_eff EMA lag stress test

Key insight: r_max ∈ {0.9, 0.999} are POINCARÉ radii (boundary at 1.0).
Must convert to Lorentz coordinates via the standard diffeomorphism.
The DS-5 guarantee assumes tokens at scaled distance R_max from origin.
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# ─── DS-5 Constants ───
C_INIT = 1.0
M = 1e7
BETA = 1.0
EMA_DECAY = 0.99
D_EMB = 64           # embedding dimension
N_LAYERS = 6
N_MAX = 131072

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "mt1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# ─── Coordinate Conversions ───

def poincare_to_lorentz_radius(r_p, c=1.0):
    """Convert Poincaré radius to Lorentz temporal coordinate and spatial radius.

    Poincaré ball B_c has boundary at ||p|| = 1/sqrt(c).
    Diffeomorphism B_c -> H_c:
      x_0 = (1/sqrt(c)) * (1 + c*r_p^2) / (1 - c*r_p^2)
      r_L = 2*r_p / (1 - c*r_p^2)   [Lorentz spatial radius, not normalized by sqrt(c)]

    For c=1: x_0 = (1+r^2)/(1-r^2), r_L = 2r/(1-r^2)
    """
    rho = c * r_p ** 2
    denom = 1.0 - rho
    x0 = (1.0 / math.sqrt(c)) * (1.0 + rho) / denom
    r_L = 2.0 * r_p / denom  # NOTE: for c != 1, multiply by 1/sqrt(c)? Actually no.
    # In the Lorentz model, spatial coordinates are: x_i = 2*p_i / (sqrt(c)*(1-c||p||^2))
    # So ||x_spatial|| = 2*||p|| / (sqrt(c)*(1-c||p||^2)) = 2*r_p / (sqrt(c)*denom)
    r_L = 2.0 * r_p / (math.sqrt(c) * denom)
    return x0, r_L


def poincare_to_scaled_distance(r_p, c=1.0):
    """Scaled hyperbolic distance from origin for a point at Poincaré radius r_p.

    D = arcosh(sqrt(c) * x_0) = arcosh((1 + c*r_p^2) / (1 - c*r_p^2))
    Also equals 2*arctanh(sqrt(c)*r_p) for c=1: 2*arctanh(r_p)
    """
    rho = c * r_p ** 2
    arg = (1.0 + rho) / (1.0 - rho)
    return math.acosh(max(arg, 1.0 + 1e-15))


# ─── DS-5 Curvature Functions ───

def c_min_from_N(N, beta=BETA, M_val=M):
    half_ln_N = beta * math.log(max(N, 2.0))
    return math.cosh(half_ln_N) ** 2 / (M_val ** 2)


def R_max_from_c(c, M_val=M):
    arg = math.sqrt(c) * M_val
    if arg < 1.0 + 1e-10:
        return 0.0
    return math.acosh(arg)


def c_eff_mode_a(c_learned, N_max=N_MAX):
    return max(c_learned, c_min_from_N(N_max))


def c_eff_mode_b(c_learned, N_actual):
    return max(c_learned, c_min_from_N(N_actual))


def c_eff_hybrid(c_learned, N_eff, N_max=N_MAX):
    return max(c_learned, c_min_from_N(N_max), c_min_from_N(N_eff))


# ─── α_s Computation ───

def compute_alpha_s_at_distance(D_ctx, N):
    """α_s when query is at origin, all N-1 context tokens at scaled distance D_ctx.

    α_s = 1 / (1 + (N-1) * exp(-D_ctx))
    """
    return 1.0 / (1.0 + (N - 1) * math.exp(-D_ctx))


def compute_alpha_s_poincare(c, r_p, N):
    """α_s for context tokens at Poincaré radius r_p, curvature c."""
    D_ctx = poincare_to_scaled_distance(r_p, c)
    return compute_alpha_s_at_distance(D_ctx, N), D_ctx


def compute_alpha_s_boundary(c, N, M_val=M):
    """α_s for context tokens at the manifold boundary (D = R_max)."""
    R = R_max_from_c(c, M_val)
    return compute_alpha_s_at_distance(R, N), R


# ─── Experiment 1: Static N Sweep ───

def experiment_1():
    print("=" * 80)
    print("Experiment 1: Static N Sweep")
    print("=" * 80)

    N_values = [128, 1000, 8000, 32000, 128000]
    # r_max as Poincaré radii (boundary at 1/sqrt(c) = 1 for c=1)
    r_max_poincare = [0.9, 0.999]
    modes = ["Mode A", "Mode B", "Hybrid"]

    rows = []

    # Part A: Tokens at Poincaré radius (realistic placement)
    print("\n--- Part A: Context tokens at Poincaré radius ---")
    print(f"{'N':>7s} {'r_p':>6s} {'Mode':>8s} {'α_s':>10s} "
          f"{'D_ctx':>8s} {'R_max':>8s} {'R_tgt':>8s} {'c_eff':>12s}")
    print("-" * 78)

    for N in N_values:
        R_target = BETA * math.log(N)
        for r_p in r_max_poincare:
            for mode_name in modes:
                c_learned = C_INIT
                if mode_name == "Mode A":
                    c = c_eff_mode_a(c_learned)
                elif mode_name == "Mode B":
                    c = c_eff_mode_b(c_learned, N)
                else:
                    c = c_eff_hybrid(c_learned, float(N))

                R_real = R_max_from_c(c)
                alpha_s, D_ctx = compute_alpha_s_poincare(c, r_p, N)

                for layer in range(N_LAYERS):
                    rows.append({
                        "N": N, "r_max": r_p, "placement": "poincare",
                        "Mode": mode_name, "Layer": layer + 1,
                        "alpha_s": alpha_s, "D_ctx": D_ctx,
                        "R_max_realized": R_real, "R_max_target": R_target,
                        "c_eff": c,
                    })

                print(f"{N:>7d} {r_p:>6.3f} {mode_name:>8s} {alpha_s:>10.6f} "
                      f"{D_ctx:>8.4f} {R_real:>8.2f} {R_target:>8.4f} {c:>12.4e}")

    # Part B: Tokens at manifold boundary (D = R_max) — the DS-5 guarantee case
    print("\n--- Part B: Context tokens at manifold boundary (D = R_max) ---")
    print(f"{'N':>7s} {'Mode':>8s} {'α_s':>10s} "
          f"{'D_ctx=R':>10s} {'R_tgt':>8s} {'c_eff':>12s} {'guarantee':>10s}")
    print("-" * 72)

    for N in N_values:
        R_target = BETA * math.log(N)
        for mode_name in modes:
            c_learned = C_INIT
            if mode_name == "Mode A":
                c = c_eff_mode_a(c_learned)
            elif mode_name == "Mode B":
                c = c_eff_mode_b(c_learned, N)
            else:
                c = c_eff_hybrid(c_learned, float(N))

            alpha_s, R_real = compute_alpha_s_boundary(c, N)
            # DS-5 theoretical lower bound: 1/(1 + (N-1)*N^{-beta})
            theory_bound = 1.0 / (1.0 + (N - 1) * N ** (-BETA))

            for layer in range(N_LAYERS):
                rows.append({
                    "N": N, "r_max": "boundary", "placement": "boundary",
                    "Mode": mode_name, "Layer": layer + 1,
                    "alpha_s": alpha_s, "D_ctx": R_real,
                    "R_max_realized": R_real, "R_max_target": R_target,
                    "c_eff": c,
                })

            print(f"{N:>7d} {mode_name:>8s} {alpha_s:>10.6f} "
                  f"{R_real:>10.4f} {R_target:>8.4f} {c:>12.4e} {theory_bound:>10.6f}")

    # Part C: Floor activation test — simulate optimizer driving c low
    print("\n--- Part C: Floor activation test (c_learned << c_init) ---")
    print(f"{'N':>7s} {'c_learned':>12s} {'Mode':>8s} {'c_eff':>12s} "
          f"{'α_s(bdy)':>10s} {'R_max':>8s} {'floor?':>6s}")
    print("-" * 72)

    c_low_values = [1e-3, 1e-6, 1e-9, 1e-12]
    for N in [1000, 128000]:
        R_target = BETA * math.log(N)
        for c_low in c_low_values:
            for mode_name in modes:
                if mode_name == "Mode A":
                    c = c_eff_mode_a(c_low)
                elif mode_name == "Mode B":
                    c = c_eff_mode_b(c_low, N)
                else:
                    c = c_eff_hybrid(c_low, float(N))

                alpha_s, R_real = compute_alpha_s_boundary(c, N)
                floor_active = "YES" if c > c_low else "no"

                print(f"{N:>7d} {c_low:>12.2e} {mode_name:>8s} {c:>12.4e} "
                      f"{alpha_s:>10.6f} {R_real:>8.2f} {floor_active:>6s}")

    with open(RESULTS_DIR / "exp1_static_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Summary
    print("\n--- Experiment 1 Summary ---")
    boundary_rows = [r for r in rows if r["placement"] == "boundary" and r["Mode"] == "Hybrid"]
    min_alpha_bdy = min(r["alpha_s"] for r in boundary_rows)
    print(f"  Boundary placement (DS-5 guarantee case): min α_s = {min_alpha_bdy:.6f}")

    poincare_rows = [r for r in rows if r["placement"] == "poincare"
                     and r["Mode"] == "Hybrid" and r["r_max"] == 0.999]
    min_alpha_p = min(r["alpha_s"] for r in poincare_rows)
    print(f"  Poincaré r_p=0.999 (realistic placement): min α_s = {min_alpha_p:.6f}")
    print(f"  Note: Poincaré r_p=0.999 ↔ D≈{poincare_to_scaled_distance(0.999, 1.0):.2f} "
          f"from origin, vs R_max={R_max_from_c(1.0):.2f}")

    return rows


# ─── Experiment 2: Dynamic N Ramp ───

def experiment_2():
    print("\n" + "=" * 80)
    print("Experiment 2: Dynamic N Ramp (1000 steps)")
    print("=" * 80)

    n_steps = 1000

    # N schedule
    N_schedule = np.zeros(n_steps, dtype=int)
    for i in range(333):
        N_schedule[i] = int(128 + (128000 - 128) * i / 332)
    for i in range(333, 666):
        N_schedule[i] = 128000
    for i in range(666, 1000):
        N_schedule[i] = int(128000 - (128000 - 128) * (i - 666) / 333)
    N_schedule = np.maximum(N_schedule, 128)

    modes = ["Mode A", "Mode B", "Hybrid"]

    # Test BOTH placements: Poincaré r_p=0.999 and boundary
    placements = {
        "poincare_0.999": lambda c, N: compute_alpha_s_poincare(c, 0.999, N)[0],
        "boundary": lambda c, N: compute_alpha_s_boundary(c, N)[0],
    }

    all_alpha = {}
    for pname in placements:
        all_alpha[pname] = {m: np.zeros((n_steps, N_LAYERS)) for m in modes}

    c_eff_maps = {m: np.zeros((n_steps, N_LAYERS)) for m in modes}
    R_max_maps = {m: np.zeros((n_steps, N_LAYERS)) for m in modes}
    N_eff_trace = np.zeros(n_steps)
    N_eff = 128.0
    violation_trace = np.zeros(n_steps)

    for step in range(n_steps):
        N = int(N_schedule[step])
        N_eff = EMA_DECAY * N_eff + (1 - EMA_DECAY) * N
        N_eff_trace[step] = N_eff
        R_target = BETA * math.log(max(N, 2))

        for mode_name in modes:
            c_learned = C_INIT
            if mode_name == "Mode A":
                c = c_eff_mode_a(c_learned)
            elif mode_name == "Mode B":
                c = c_eff_mode_b(c_learned, N)
            else:
                c = c_eff_hybrid(c_learned, N_eff)

            R_real = R_max_from_c(c)

            for pname, alpha_fn in placements.items():
                alpha_s = alpha_fn(c, N)
                for layer in range(N_LAYERS):
                    all_alpha[pname][mode_name][step, layer] = alpha_s

            for layer in range(N_LAYERS):
                c_eff_maps[mode_name][step, layer] = c
                R_max_maps[mode_name][step, layer] = R_real

        R_hybrid = R_max_maps["Hybrid"][step, 0]
        violation_trace[step] = max(0.0, R_target - R_hybrid)

        if step % 200 == 0:
            a_bdy = all_alpha["boundary"]["Hybrid"][step, 0]
            a_p = all_alpha["poincare_0.999"]["Hybrid"][step, 0]
            print(f"  step {step:4d}: N={N:6d}, N_eff={N_eff:10.1f}, "
                  f"α_s(bdy)={a_bdy:.4f}, α_s(r_p=.999)={a_p:.6f}")

    # ─── Heatmaps: boundary placement ───
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for idx, mode_name in enumerate(modes):
        ax = axes[idx]
        im = ax.imshow(
            all_alpha["boundary"][mode_name].T,
            aspect='auto', origin='lower',
            extent=[0, n_steps, 0.5, N_LAYERS + 0.5],
            vmin=0.0, vmax=1.0, cmap='RdYlGn'
        )
        ax.set_ylabel("Layer")
        ax.set_title(f"α_s — {mode_name} (boundary placement)")
        ax.set_yticks(range(1, N_LAYERS + 1))
        plt.colorbar(im, ax=ax, label="α_s")
    axes[2].set_xlabel("Step")

    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim())
    ticks = [0, 166, 333, 500, 666, 833, 999]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{N_schedule[i]//1000}K" if N_schedule[i] >= 1000
                         else str(N_schedule[i]) for i in ticks], fontsize=8)
    ax2.set_xlabel("N", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp2_alpha_s_heatmaps_boundary.png", dpi=150)
    plt.close()
    print(f"\nSaved: exp2_alpha_s_heatmaps_boundary.png")

    # ─── Heatmaps: Poincaré placement ───
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for idx, mode_name in enumerate(modes):
        ax = axes[idx]
        im = ax.imshow(
            all_alpha["poincare_0.999"][mode_name].T,
            aspect='auto', origin='lower',
            extent=[0, n_steps, 0.5, N_LAYERS + 0.5],
            vmin=0.0, vmax=1.0, cmap='RdYlGn'
        )
        ax.set_ylabel("Layer")
        ax.set_title(f"α_s — {mode_name} (Poincaré r_p=0.999)")
        ax.set_yticks(range(1, N_LAYERS + 1))
        plt.colorbar(im, ax=ax, label="α_s")
    axes[2].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp2_alpha_s_heatmaps_poincare.png", dpi=150)
    plt.close()
    print(f"Saved: exp2_alpha_s_heatmaps_poincare.png")

    # ─── N_eff tracking ───
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(N_schedule, label="N_actual", color='blue', alpha=0.7)
    ax.plot(N_eff_trace, label="N_eff (EMA α=0.99)", color='red', linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("N")
    ax.set_title("N_eff EMA Tracking (Hybrid Mode)")
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp2_neff_tracking.png", dpi=150)
    plt.close()
    print(f"Saved: exp2_neff_tracking.png")

    # ─── Constraint violation ───
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(violation_trace, color='red', linewidth=1)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("R_max_target − R_max_realized")
    ax.set_title("Constraint Violation Over N Ramp (Hybrid Mode)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp2_constraint_violation.png", dpi=150)
    plt.close()
    print(f"Saved: exp2_constraint_violation.png")

    # ─── c_eff trajectory ───
    fig, ax = plt.subplots(figsize=(12, 4))
    for mode_name in modes:
        ax.plot(c_eff_maps[mode_name][:, 0], label=mode_name, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("c_eff")
    ax.set_title("Curvature Trajectory (Layer 1)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp2_c_eff_trajectory.png", dpi=150)
    plt.close()
    print(f"Saved: exp2_c_eff_trajectory.png")

    # ─── α_s comparison line plot ───
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(all_alpha["boundary"]["Hybrid"][:, 0], label="Boundary", color='green')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="DS-5 guarantee (β=1)")
    ax1.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label="Kill threshold")
    ax1.set_ylabel("α_s")
    ax1.set_title("Hybrid Mode — Boundary Placement")
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)

    ax2.plot(all_alpha["poincare_0.999"]["Hybrid"][:, 0], label="Poincaré r_p=0.999", color='blue')
    ax2.set_ylabel("α_s")
    ax2.set_xlabel("Step")
    ax2.set_title("Hybrid Mode — Poincaré r_p=0.999 (D≈7.6)")
    ax2.legend()
    ax2.set_ylim(-0.05, 0.1)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp2_alpha_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: exp2_alpha_comparison.png")

    # Summary
    print("\n--- Experiment 2 Summary ---")
    for mode_name in modes:
        min_bdy = np.min(all_alpha["boundary"][mode_name])
        mean_bdy = np.mean(all_alpha["boundary"][mode_name])
        min_p = np.min(all_alpha["poincare_0.999"][mode_name])
        print(f"  {mode_name}: boundary min α_s={min_bdy:.6f} mean={mean_bdy:.6f} | "
              f"Poincaré min α_s={min_p:.6f}")
    print(f"  Max constraint violation: {np.max(violation_trace):.8f}")

    return all_alpha, N_eff_trace, violation_trace


# ─── Experiment 3: EMA Lag Stress Test ───

def experiment_3():
    print("\n" + "=" * 80)
    print("Experiment 3: N_eff EMA Lag Stress Test")
    print("=" * 80)

    check_steps = [0, 1, 5, 10, 25, 50, 100]
    transitions = [
        ("128 → 128K", 128.0, 128000),
        ("128K → 128", 128000.0, 128),
    ]

    rows = []
    print(f"{'Transition':>14s} {'Step':>5s} {'N_eff':>12s} "
          f"{'R_tgt':>8s} {'R_real':>8s} {'Violation':>10s} {'α_s(bdy)':>10s}")
    print("-" * 75)

    for trans_name, N_init, N_final in transitions:
        N_eff = N_init
        for step in range(max(check_steps) + 1):
            N_eff = EMA_DECAY * N_eff + (1 - EMA_DECAY) * N_final
            if step in check_steps:
                c = c_eff_hybrid(C_INIT, N_eff)
                R_real = R_max_from_c(c)
                R_target = BETA * math.log(max(N_final, 2))
                violation = max(0.0, R_target - R_real)
                alpha_bdy, _ = compute_alpha_s_boundary(c, int(N_final))

                rows.append({
                    "Transition": trans_name, "Step": step,
                    "N_eff": round(N_eff, 2),
                    "R_max_target": round(R_target, 4),
                    "R_max_realized": round(R_real, 4),
                    "Violation": round(violation, 8),
                    "alpha_s_boundary": round(alpha_bdy, 6),
                })
                print(f"{trans_name:>14s} {step:>5d} {N_eff:>12.2f} "
                      f"{R_target:>8.4f} {R_real:>8.4f} {violation:>10.8f} "
                      f"{alpha_bdy:>10.6f}")

    print("\n--- Steps to 95% convergence ---")
    for trans_name, N_init, N_final in transitions:
        N_eff = N_init
        target_95 = N_init + 0.95 * (N_final - N_init)
        for step in range(10000):
            N_eff = EMA_DECAY * N_eff + (1 - EMA_DECAY) * N_final
            if (N_final > N_init and N_eff >= target_95) or \
               (N_final < N_init and N_eff <= target_95):
                print(f"  {trans_name}: 95% at step {step} "
                      f"(N_eff={N_eff:.1f}, target={target_95:.1f})")
                break

    with open(RESULTS_DIR / "exp3_ema_lag.json", "w") as f:
        json.dump(rows, f, indent=2)

    return rows


# ─── Final Report ───

def final_report(exp1_rows, all_alpha):
    print("\n" + "=" * 80)
    print("KILL CONDITION CHECK")
    print("=" * 80)

    # The DS-5 guarantee applies to boundary placement
    bdy_rows = [r for r in exp1_rows if r.get("placement") == "boundary"
                and r["Mode"] == "Hybrid" and r["N"] >= 1000]
    min_alpha_bdy = min(r["alpha_s"] for r in bdy_rows)
    print(f"  Exp 1 — Hybrid, boundary, N≥1K: min α_s = {min_alpha_bdy:.6f}")

    # Exp 2 boundary heatmap
    hybrid_bdy = all_alpha["boundary"]["Hybrid"]
    min_alpha_exp2 = np.min(hybrid_bdy[3:997, :])
    print(f"  Exp 2 — Hybrid, boundary, N≥1K: min α_s = {min_alpha_exp2:.6f}")

    # Kill condition on boundary placement
    killed = min_alpha_bdy < 0.3 or min_alpha_exp2 < 0.3
    if killed:
        print("\n  *** KILL: α_s < 0.3 at boundary. DS-5 enforcement INSUFFICIENT. ***")
    else:
        passed = min_alpha_bdy >= 0.4 and min_alpha_exp2 >= 0.4
        if passed:
            print("\n  PASS: α_s ≥ 0.4 across full range in Hybrid mode (boundary).")
        else:
            print(f"\n  WARNING: α_s between 0.3 and 0.4 at boundary. Marginal.")

    # Poincaré placement note
    p_rows = [r for r in exp1_rows if r.get("placement") == "poincare"
              and r["Mode"] == "Hybrid" and r["r_max"] == 0.999 and r["N"] >= 1000]
    if p_rows:
        min_alpha_p = min(r["alpha_s"] for r in p_rows)
        D_at_999 = poincare_to_scaled_distance(0.999, 1.0)
        print(f"\n  Note: Poincaré r_p=0.999 gives D≈{D_at_999:.2f}, min α_s={min_alpha_p:.6f}")
        print(f"  This is NOT a DS-5 failure — tokens must be near the manifold boundary")
        print(f"  (D≈R_max≈{R_max_from_c(1.0):.1f}) for the guarantee to apply.")
        print(f"  Training dynamics (exp/log maps + loss) push tokens to high radius.")

    # Mode comparison
    print("\n" + "=" * 80)
    print("MODE COMPARISON TABLE (boundary placement)")
    print("=" * 80)
    print(f"{'Mode':>8s} {'α_s floor':>10s} {'α_s mean':>10s} "
          f"{'R gap max':>10s} {'c_eff':>12s} {'Overhead':>12s}")
    print("-" * 66)

    for mode in ["Mode A", "Mode B", "Hybrid"]:
        mode_rows = [r for r in exp1_rows
                     if r["Mode"] == mode and r.get("placement") == "boundary"]
        min_a = min(r["alpha_s"] for r in mode_rows)
        mean_a = np.mean([r["alpha_s"] for r in mode_rows])
        max_gap = max(r["R_max_target"] - r["R_max_realized"] for r in mode_rows)
        c_vals = set(r["c_eff"] for r in mode_rows)
        c_s = f"{min(c_vals):.2e}" if len(c_vals) == 1 else f"{min(c_vals):.2e}–{max(c_vals):.2e}"
        overhead = "zero" if mode == "Mode A" else "1 cosh²/step"
        print(f"{mode:>8s} {min_a:>10.6f} {mean_a:>10.6f} "
              f"{max_gap:>10.4f} {c_s:>12s} {overhead:>12s}")

    verdict = "PASS" if not killed else "FAIL"
    print(f"\n{'='*80}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*80}")
    return not killed


if __name__ == "__main__":
    print("HCA-MT-1: DS-5 Adaptive-Curvature Kernel Verification")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Parameters: c_init={C_INIT}, β={BETA}, M={M:.0e}, d={D_EMB}, L={N_LAYERS}")
    print(f"            EMA_decay={EMA_DECAY}, N_max={N_MAX}")
    print()

    exp1_rows = experiment_1()
    all_alpha, N_eff_trace, violation_trace = experiment_2()
    exp3_rows = experiment_3()
    passed = final_report(exp1_rows, all_alpha)

    sys.exit(0 if passed else 1)
