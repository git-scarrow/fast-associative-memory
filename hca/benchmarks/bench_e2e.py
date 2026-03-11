"""HCA-MT-2: End-to-end training benchmark — fused vs unfused on WikiText-103.

Config: d_model=512, seq_len=1024, batch_size=8, n_heads=8, n_layers=4
Steps: 1000
Kill: final loss > 2.75 or any NaN/Inf
Stop: final loss <= 2.50 and speedup measured
"""

import json
import logging
import os
import time

import torch

from hca.training.harness import train_hca

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# -- Config (matches DS-11 baseline dimensions) --
# lr=3e-4: reduced from harness default (1e-3) to stabilize Lorentz
# geometry at d_model=512 (exp_map/distance overflow at higher LR)
CONFIG = dict(
    steps=1000,
    d_model=512,
    n_heads=8,
    n_layers=4,
    seq_len=1024,
    batch_size=8,
    lr=3e-4,
    device="cuda",
    use_wikitext=True,
    qos_cadence=100,
)


def run_one(label: str, use_fused: bool, amp_enabled: bool, seed: int = 42):
    """Run a single 1000-step training benchmark."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  {label}  (use_fused={use_fused}, amp={amp_enabled})")
    logger.info(f"{'='*70}")

    # Deterministic
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Reset VRAM tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    metrics = train_hca(**CONFIG, use_fused=use_fused, amp_enabled=amp_enabled)
    torch.cuda.synchronize()
    wall_time = time.perf_counter() - t0

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    # Check kill conditions
    killed = False
    kill_reason = None

    if "kill_reason" in metrics:
        killed = True
        kill_reason = metrics["kill_reason"]
    elif metrics.get("final_loss", float("inf")) > 2.75:
        killed = True
        kill_reason = f"final_loss={metrics['final_loss']:.4f} > 2.75"

    # Check for NaN/Inf in loss curve and gradients
    nan_inf = any(not torch.tensor(l).isfinite() for l in metrics.get("loss", []))
    nan_grad_steps = metrics.get("nan_grad_steps", [])
    if nan_inf:
        killed = True
        kill_reason = "NaN/Inf in loss curve"
    elif nan_grad_steps:
        # Any recorded NaN/Inf gradients should kill the run per benchmark spec
        killed = True
        kill_reason = f"NaN/Inf gradients detected at steps: {nan_grad_steps}"

    # Loss stability: check no sudden spikes > 3x running average
    losses = metrics.get("loss", [])
    loss_stable = True
    if len(losses) > 50:
        for i in range(50, len(losses)):
            window_avg = sum(losses[i-50:i]) / 50
            if losses[i] > 3 * window_avg and window_avg > 0.1:
                loss_stable = False
                break

    # Alpha_s stability (unfused only — fused returns default 0.5)
    alpha_s_data = metrics.get("alpha_s", [])
    alpha_s_stable = True
    alpha_s_note = ""
    if not use_fused and len(alpha_s_data) > 100:
        # Check all layer alpha_s values stay in [0.01, 0.99]
        for step_alphas in alpha_s_data[-100:]:
            for a in step_alphas:
                if a < 0.01 or a > 0.99:
                    alpha_s_stable = False
                    break
    elif use_fused:
        alpha_s_note = "fused path (approx via LSE not implemented, returns 0.5)"

    result = {
        "label": label,
        "use_fused": use_fused,
        "amp_enabled": amp_enabled,
        "wall_time_s": round(wall_time, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "final_loss": round(metrics.get("final_loss", float("nan")), 4),
        "initial_loss": round(losses[0], 4) if losses else None,
        "nan_inf": nan_inf,
        "nan_grad_steps": nan_grad_steps,
        "loss_stable": loss_stable,
        "alpha_s_stable": alpha_s_stable,
        "alpha_s_note": alpha_s_note,
        "killed": killed,
        "kill_reason": kill_reason,
        "loss_curve": [round(l, 4) for l in losses],
        "final_alpha_s": metrics.get("final_alpha_s"),
    }

    logger.info(f"\n--- {label} SUMMARY ---")
    logger.info(f"  Wall time:   {wall_time:.2f}s")
    logger.info(f"  Peak VRAM:   {peak_vram_gb:.3f} GB")
    logger.info(f"  Loss:        {losses[0]:.4f} -> {metrics.get('final_loss', 'N/A')}")
    logger.info(f"  NaN/Inf:     {nan_inf}")
    logger.info(f"  Stable:      {loss_stable}")
    if killed:
        logger.error(f"  KILLED: {kill_reason}")

    return result


def main():
    assert torch.cuda.is_available(), "CUDA required"

    print("=" * 70)
    print("HCA-MT-2: End-to-End Fused Training Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: d_model={CONFIG['d_model']}, seq_len={CONFIG['seq_len']}, "
          f"B={CONFIG['batch_size']}, H={CONFIG['n_heads']}, "
          f"n_layers={CONFIG['n_layers']}")
    print(f"Steps: {CONFIG['steps']}")
    print("=" * 70)

    results = {}

    # Run 1: Unfused baseline (DS-11 equivalent)
    results["unfused"] = run_one("Unfused (DS-11 baseline)", use_fused=False, amp_enabled=False)

    # Check if unfused was killed — if so, we have a bigger problem
    if results["unfused"]["killed"]:
        logger.error("Unfused baseline KILLED — cannot proceed with comparison")
    else:
        # Run 2: Fused (DS-12 fwd + DS-13 bwd)
        results["fused_fp32"] = run_one("Fused fp32 (DS-12+DS-13)", use_fused=True, amp_enabled=False)

        # Run 3: Fused + bf16 mixed precision
        results["fused_bf16"] = run_one("Fused bf16 (DS-12+DS-13)", use_fused=True, amp_enabled=True)

    # -- Comparison table --
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    header = f"{'Metric':<30} | {'Unfused (DS-11)':>18} | {'Fused fp32':>18} | {'Fused bf16':>18} | {'Notes'}"
    print(header)
    print("-" * len(header))

    u = results.get("unfused", {})
    f32 = results.get("fused_fp32", {})
    bf16 = results.get("fused_bf16", {})

    def _val(d, key, fmt=".2f"):
        v = d.get(key)
        if v is None:
            return "N/A"
        return f"{v:{fmt}}"

    # Wall time
    u_t = u.get("wall_time_s")
    f32_t = f32.get("wall_time_s")
    bf16_t = bf16.get("wall_time_s")
    speedup_f32 = f"{u_t/f32_t:.2f}x" if u_t and f32_t else "N/A"
    speedup_bf16 = f"{u_t/bf16_t:.2f}x" if u_t and bf16_t else "N/A"
    print(f"{'Wall time (s)':<30} | {_val(u, 'wall_time_s'):>18} | {_val(f32, 'wall_time_s'):>18} | {_val(bf16, 'wall_time_s'):>18} | speedup: fp32={speedup_f32}, bf16={speedup_bf16}")

    # Peak VRAM
    print(f"{'Peak VRAM (GB)':<30} | {_val(u, 'peak_vram_gb', '.3f'):>18} | {_val(f32, 'peak_vram_gb', '.3f'):>18} | {_val(bf16, 'peak_vram_gb', '.3f'):>18} |")

    # Loss
    print(f"{'Loss @ step 1000':<30} | {_val(u, 'final_loss', '.4f'):>18} | {_val(f32, 'final_loss', '.4f'):>18} | {_val(bf16, 'final_loss', '.4f'):>18} |")

    # NaN/Inf
    print(f"{'NaN/Inf observed':<30} | {str(u.get('nan_inf', 'N/A')):>18} | {str(f32.get('nan_inf', 'N/A')):>18} | {str(bf16.get('nan_inf', 'N/A')):>18} |")

    # Stability
    print(f"{'Loss stability':<30} | {str(u.get('loss_stable', 'N/A')):>18} | {str(f32.get('loss_stable', 'N/A')):>18} | {str(bf16.get('loss_stable', 'N/A')):>18} |")
    print(f"{'α_s tracking stability':<30} | {str(u.get('alpha_s_stable', 'N/A')):>18} | {str(f32.get('alpha_s_stable', 'N/A')):>18} | {str(bf16.get('alpha_s_stable', 'N/A')):>18} |")

    # -- Verdict --
    print("\n" + "=" * 70)

    # Use the best fused result for the verdict, preferring non-killed,
    # non-NaN/Inf runs with the lowest final loss.
    fused_candidates = []
    for label, run in (("fused_fp32", f32), ("fused_bf16", bf16)):
        if not run:
            continue
        killed = bool(run.get("killed"))
        nan_inf = bool(run.get("nan_inf"))
        loss = run.get("final_loss", float("inf"))
        fused_candidates.append(((killed, nan_inf, loss), label, run))

    if fused_candidates:
        # Select the best fused run based on (killed, nan_inf, final_loss)
        _, best_label, best_fused = min(fused_candidates, key=lambda x: x[0])
    else:
        # No fused runs available; fall back to empty result to keep logic robust
        best_label = "fused_fp32"
        best_fused = {}
    if u.get("killed"):
        print("VERDICT: INCONCLUSIVE — unfused baseline failed")
    elif best_fused.get("killed"):
        print(f"VERDICT: KILL — {best_label} failed: {best_fused.get('kill_reason')}")
    elif best_fused.get("nan_inf"):
        print(f"VERDICT: KILL — {best_label} produced NaN/Inf gradients")
    elif best_fused.get("final_loss", float("inf")) > 2.75:
        print(f"VERDICT: KILL — {best_label} final loss {best_fused['final_loss']:.4f} > 2.75")
    elif best_fused.get("final_loss", float("inf")) <= 2.50:
        print(f"VERDICT: PASS — {best_label} final loss {best_fused['final_loss']:.4f} <= 2.50, "
              f"speedup measured")
    else:
        print(f"VERDICT: INCONCLUSIVE — {best_label} final loss {best_fused.get('final_loss', 'N/A')} "
              f"between 2.50 and 2.75")

    print("=" * 70)

    # Save results
    os.makedirs("results/mt2", exist_ok=True)
    out_path = "results/mt2/e2e_benchmark.json"

    # Strip loss curves for the summary file (too large)
    summary = {}
    for k, v in results.items():
        summary[k] = {kk: vv for kk, vv in v.items() if kk != "loss_curve"}

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")

    # Save full loss curves separately
    curves_path = "results/mt2/loss_curves.json"
    curves = {k: v.get("loss_curve", []) for k, v in results.items()}
    with open(curves_path, "w") as f:
        json.dump(curves, f)
    print(f"Loss curves saved to {curves_path}")


if __name__ == "__main__":
    main()
