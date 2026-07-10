# PR-13 consumer qualification gate — evidence (2026-07-10)

Synthetic-only qualification of the exact sealed Qwen3-8B bf16 consumer
(`Qwen/Qwen3-8B` @ `b968826d`, greedy, non-thinking, `max_new_tokens=256`)
on the scoring host `gentoo` (RTX 4080 SUPER, 16.72 GB; 31 GB system RAM).

Produced by `harness/pr13_qualify_consumer.py`. **No §7 material was
rendered.** Every prompt is a registered template filled with invented
values; the script loads the committed query manifest only to assert its
own synthetic manifest is not that one. The consumer pin, decoding
configuration, manifests, compiler, sample, and scoring rules are
unchanged, and `harness/ctx/` was not touched — which is why the
qualification script lives outside it (the sealed scoring manifest hashes
every file under `harness/ctx`).

| file | what it shows |
|---|---|
| `01_baseline_services_running.json` | Ollama + Sunshine running. 13 of 40 modules offloaded. Bimodal latency (p90/p10 = 1.68). ~51 h projected. Required an explicit 12 GiB cap to load at all. |
| `02_services_stopped_registered_auto_path.json` | Services stopped, **registered `device_map="auto"` path, no cap**. 9 of 40 modules offloaded. Latency flat and tight. ~23 h projected. In-process crash/resume byte-reconcilable. |
| `03_registered_auto_path_clean_gpu.json` | The registered load path succeeds on a cleared GPU (31/9 placement, 12.82 GB peak) — it OOMs only when another process holds VRAM. |
| `04_cross_process_crash_resume.txt` | Crash, reload in a fresh interpreter, resume, and a clean run of the same manifest: byte-identical, 0 rows with differing raw output, identical `{GPU: 31, CPU: 9}` placement across all three loads. |

## Headline numbers

Weights are **16.38 GB** of bf16; torch reports the card's usable capacity
as **15.58 GiB**. The model therefore *cannot* be held entirely on this
GPU under the pin (which fixes bfloat16 and forbids quantization), and
some CPU offload is structural, not a tuning artifact.

| | services running | services stopped |
|---|---|---|
| modules offloaded to CPU | 13 / 40 | 9 / 40 |
| registered `auto` path loads | **no — CUDA OOM** | yes |
| latency, all bands | 5.9–9.8 s (p90/p10 1.68) | 4.05–4.12 s (p90/p10 1.01) |
| rows/hour | 368–608 | 874–889 |
| peak VRAM allocated | 12.78 GB | 14.33 GB |
| peak system RSS | 11.49 GB | 12.04 GB |
| malformed-output rate | 0.00 | 0.00 |
| projected 20,304 calls | 35.8–55.2 h | **23.1–23.3 h** |

Latency is **decode-bound, not prefill-bound**: a 1,610-token prompt costs
the same as a 109-token one, because each of the ~13 output tokens must
stream the CPU-resident layers. Duration therefore scales with *output*
tokens (~0.32 s each), and is nearly insensitive to context-block length —
which is what makes the projection a measurement rather than an
extrapolation.

## Operational preconditions established here

1. Ollama and Sunshine must be stopped for the duration. With them
   running, the registered load path OOMs, and even with a cap the run
   takes 2.2× longer with bimodal latency.
2. No other GPU consumer may start mid-run. `device_map="auto"` chooses
   placement from *free* VRAM at load time; a different placement on a
   post-crash reload is the one thing that could break G-C1's double-run
   byte-identity. On a cleared GPU the placement was reproducible across
   three independent loads, and the cross-process resume was byte-exact.
