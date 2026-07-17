# GPU acceleration (CPU + RAM + a single consumer GPU)

This branch (`gpu-accel`) makes the optional CUDA path actually **move the needle** on
a consumer GPU (tested target: a single **RTX 5090**, 32 GB, Blackwell `sm_120`), while
keeping the engine 100% functional with no GPU at all.

The engine is still the same three-tier design — **disk → RAM → VRAM**. The GPU is a
high-bandwidth cache + compute tier for the MoE experts (the dominant decode cost). Nothing
here changes model quality; it changes *where* the expert matmuls run.

---

## TL;DR — what was wrong, what changed

The CUDA backend already existed, but in every documented workflow the GPU did **almost
nothing** for the expensive part (the routed-expert matmuls). Three gaps, all fixed here:

| Gap (before) | Effect | Fix (this branch) |
|---|---|---|
| The batched expert kernel (`CUDA_EGROUP`) was **never enabled** by `coli`, `--auto-tier`, or `resource_plan.py` — it defaulted off. | VRAM-resident experts reached the GPU only through the per-tensor path: **3 synchronous PCIe round-trips per expert per token** (gate/up/down), with SiLU bouncing back to the CPU. For single-token decode that's overhead-bound and ties/loses to the CPU int4 kernels. | **`CUDA_EGROUP` defaults ON** whenever a VRAM expert tier exists. Resident experts now run as **one fused grouped `gate/up/down/SiLU` call per 64-expert block** (pinned async copies, a single sync). |
| `--gpu` without `--vram` set **no VRAM budget**, so nothing became resident. | The card was enabled but empty. | The engine **auto-sizes** the VRAM expert budget to *(free VRAM − reserve)* when you pass `--gpu` with no `--vram`. |
| The VRAM tier was filled **only** by `PRELOAD=1`, an explicit `PIN=`, or ≥5000 logged expert selections (`.coli_usage`). On a **fresh** model none of those hold. | `--gpu 0 --vram 28` on a new model uploaded **zero** experts → GPU idle → "engine runs on CPU+RAM only." | New **`cuda_vram_prefill`**: when the GPU tier is requested but nothing else filled it, the hottest experts are loaded straight into VRAM (evenly per layer, history-ordered within a layer) up to the budget, and their RAM copies are freed. |

Net result: on a fresh model, `--gpu 0` alone now fills the card and routes resident-expert
matmuls through the fast batched kernel. As `.coli_usage` warms up across a session, the
prefill places the *hottest* experts in VRAM, so GPU coverage of actual routes climbs.

Everything is still optional — build without `CUDA=1` and you get the exact original CPU
engine; build with CUDA but run without `--gpu` and it stays on the CPU path.

---

## Build on RunPod (Linux + NVIDIA)

```bash
cd c
make hy3 CUDA=1                 # links cudart; needs nvcc + NVIDIA driver
# optional extras:
make hy3 CUDA=1 IOURING=1       # + io_uring expert reads (PIPE=2)
```

`CUDA_ARCH` defaults to `native`. On a 5090 that resolves to `sm_120`. To build a binary
that runs on several card generations, set e.g. `make hy3 CUDA=1 CUDA_ARCH=all` (slower
compile) or a specific `CUDA_ARCH=sm_120`.

CPU-only (any machine, no GPU): `make hy3`.

---

## Run

```bash
# Simplest: enable GPU, auto-fill VRAM, auto-size everything.
COLI_MODEL=/path/to/hy3_i4 ./coli chat --gpu 0 --ram 40

# Explicit VRAM budget (recommended: leave ~2-4 GB headroom under the card size).
COLI_MODEL=/path/to/hy3_i4 ./coli chat --gpu 0 --vram 28 --ram 40 --verbose
```

At startup, with `--verbose`, confirm you see all of:

```
[CUDA] device 0: NVIDIA GeForce RTX 5090, 34.2 GB VRAM, sm_120
[CUDA] auto VRAM expert budget: 30.1 GB ...        (only if you did not pass --vram)
[CUDA] expert batching: egroup=1 stream=0
[CUDA] VRAM prefill: 3312 experts resident (29.8 GB), ~42/layer x 79 layers in 34s ...
```

and, after a turn:

```
CUDA expert tier: 3312 resident experts (29.80 GB) | 240118 calls served from VRAM
```

If `calls served from VRAM` is 0, the GPU isn't doing expert work — see Troubleshooting.

> The prefill does a **one-time** read of ~budget-worth of experts from disk at startup
> (e.g. ~30 GB). That's the cost of a full VRAM tier; decode is much faster afterwards.
> Lower `--vram` for a faster start, or `PRELOAD=1` to also fill RAM with the remainder.

### Getting the most GPU coverage

Only ~20% of a 142 GB model fits in 32 GB of VRAM, so the win scales with how concentrated
the routing is and how well VRAM covers the hot experts:

- **Run a warm-up session first.** `.coli_usage` records which experts fire. On the *next*
  run, the prefill loads the **hottest** experts into VRAM, so a much larger fraction of
  routes hit the card (uniform ~20% → often 50–80% once warmed).
- **RAM-limited pod?** `AUTOPIN=0` forces the VRAM-first prefill (it frees each expert's RAM
  copy after upload, so VRAM and RAM stay disjoint) instead of the history autopin (which
  keeps a RAM copy). Use it when system RAM is tight relative to VRAM.
- **Plenty of disk-load patience?** `PRELOAD=1` fills VRAM *and* RAM at startup; only the
  residual streams from disk.

---

## Prove the GPU makes a difference (A/B)

Use a fixed prompt, greedy decoding, and compare the **decode** tok/s. Warm the usage file
once first so residency reflects real routing.

```bash
M=/path/to/hy3_i4

# 0) warm .coli_usage once (routing history for the prefill to rank by)
SNAP=$M TEMP=0 PROMPT="Explain how a jet engine works." NGEN=64 ./hy3 64 4 8 >/dev/null

# 1) CPU baseline (GPU off)
SNAP=$M TEMP=0 PROMPT="Explain how a jet engine works." NGEN=64 \
    ./hy3 64 4 8

# 2) GPU on (auto budget + batched experts + prefill)
SNAP=$M TEMP=0 PROMPT="Explain how a jet engine works." NGEN=64 \
    COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 ./hy3 64 4 8
```

Compare the `... tok/s` line and the `PROFILE: ... expert-matmul ...` seconds between runs.

To isolate the **batched-kernel** win specifically (same residency, only the dispatch path
changes):

```bash
# per-tensor path (old behavior)
... COLI_CUDA=1 COLI_GPU=0 CUDA_EGROUP=0 ./hy3 64 4 8
# batched grouped path (this branch's default)
... COLI_CUDA=1 COLI_GPU=0 CUDA_EGROUP=1 ./hy3 64 4 8
```

Add `PERF=1` to either run for a periodic `[perf] attn=% disk=% expert_mm=% head=%`
breakdown, and `COLI_CUDA_PROFILE=1` to get the grouped `h2d / kernel / d2h` split.

---

## New / relevant tuning knobs

| Env / flag | Default | Effect |
|---|---|---|
| `--gpu 0` / `COLI_GPU=0` | off | Enable CUDA on device 0. |
| `--vram N` / `CUDA_EXPERT_GB=N` | **auto** (free − reserve) | VRAM budget for resident experts. Unset ⇒ auto-fill the card. |
| `CUDA_EGROUP` | **1** when a VRAM tier exists | Batched fused grouped-expert kernel. `0` = old per-tensor path. |
| `CUDA_VRAM_RESERVE_GB` | `2` | VRAM held back from the expert tier for dense/attention/scratch. |
| `CUDA_STREAM_EXPERTS` | `0` | JIT-stream *non-resident* int4 experts to the GPU. Helps **prefill / MTP** (many rows amortize the PCIe upload); usually a loss for single-token decode — leave off unless prefill-bound. |
| `CUDA_STREAM_BATCH` | `8` | Max experts per streamed sub-batch (bounds VRAM scratch). |
| `CUDA_DENSE` | `1` via `coli --gpu` | Keep dense + attention + shared-expert projections resident on the GPU. |
| `CUDA_ATTN` | `0` | Run GQA attention on the GPU (float KV only; do **not** combine with `KV_I8=1`). |
| `PRELOAD=1` | off | Fill VRAM **and** RAM at startup (whole-model resident minus the disk residual). |
| `AUTOPIN=0` | autopin on | Skip history-based RAM autopin; forces the VRAM-first prefill (good for RAM-limited pods). |

---

## Troubleshooting

- **`calls served from VRAM` is 0 / `hot expert tier: 0 experts`** — the VRAM tier is empty.
  Pass `--gpu 0` (so CUDA is actually enabled) and let auto-sizing run, or set an explicit
  `--vram N`. Confirm the `[CUDA] VRAM prefill:` line appears at startup.
- **Startup is slow** — the prefill reads ~`--vram` GB from disk once. Lower `--vram`, or
  put the model on fast local NVMe (not a slow network volume).
- **OOM on the card** — raise `CUDA_VRAM_RESERVE_GB` (e.g. `4`) or lower `--vram`. Dense
  (`CUDA_DENSE`) and GPU attention (`CUDA_ATTN`) also consume VRAM.
- **`COLI_CUDA_TC_INT4=1` produces garbage on a 5090** — the sub-byte tensor-core (s4 WMMA)
  path only compiles on `sm_75..sm_89` (Turing/Ampere/Ada). Blackwell removed that type, so
  the kernel is a no-op there. Leave `COLI_CUDA_TC_INT4` **off** on `sm_90+`; the default
  CUDA-core int4 GEMV is the correct path and is bandwidth-bound anyway.
- **Non-CUDA binary but you passed `--gpu`** — rebuild with `make hy3 CUDA=1`.
