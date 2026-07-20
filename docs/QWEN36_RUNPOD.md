# Running Qwen3.6-35B-A3B on RunPod (convert + test)

End-to-end runbook for the real model. The forward is validated bit-exact on the tiny
oracle (see [QWEN36_PORT.md](QWEN36_PORT.md)); **the real 35B run is a first bring-up** —
expect to sanity-check output, not trust it blindly. All commands assume the persistent
volume is mounted at `/workspace`.

## 0. What you need

| Resource | Value | Why |
|---|---|---|
| **Disk (volume)** | **≥ 150 GB NVMe** | 72 GB bf16 source + ~17 GB int4 container + HF cache |
| **RAM** | **≥ 32 GB** | conversion peaks ~8–16 GB; the int4 model (~17 GB) fits in RAM at run time |
| **GPU** | **optional** (12 GB+ helps decode) | engine is CPU-first; `--gpu` accelerates but isn't required |
| **Image** | RunPod **PyTorch** template (torch + CUDA + python) | conversion needs torch; CUDA build needs `nvcc` |

The int4 text container is only ~17 GB (35B is mostly the 256 tiny experts/layer at
`moe_intermediate=512`), so on a 32 GB pod it **fits in RAM** and runs compute-bound —
this is the 3B-active / 5–10 tok/s regime, not the disk-streaming regime Hy3 needs.

## 1. Get the code onto the pod + build

The Qwen3.6 changes are on the `gpu-accel` branch. Commit and push them from your dev box
first (`git add -A && git commit && git push`), then on the pod:

```bash
cd /workspace
git clone <your-fork-url> colibri && cd colibri/c
git checkout gpu-accel

# build toolchain (usually present on the pytorch image; install if missing)
command -v gcc || (apt-get update && apt-get install -y build-essential)

make hy3                 # CPU build  -> ./hy3
# make hy3 CUDA=1        # + GPU decode tier (needs nvcc)
# make hy3 CUDA=1 IOURING=1
```

## 2. Sanity-check the pod first (cheap, do this before the 72 GB download)

Reproduce the local 32/32 so you know the pod's toolchain + your checkout are good:

```bash
cd /workspace/colibri/c
pip install "transformers>=4.57.1" safetensors numpy     # torch already in the image
python3 tools/make_qwen36_oracle.py                       # -> qwen36_tiny/ + ref_qwen36.json
python3 tools/convert_qwen36.py --indir qwen36_tiny --outdir qwen36_tiny_f32 \
        --ebits 16 --io-bits 16 --xbits 16
SNAP=./qwen36_tiny_f32 TF=1 REF=ref_qwen36.json REF_FORCE=1 ./hy3 64 16 16
# EXPECT: PREFILL (teacher-forcing) C vs oracle: 32/32 positions
```

If that's not 32/32, stop — something in the checkout/toolchain differs; fix before scaling.

## 3. Download the real model (~72 GB, 26 shards)

```bash
pip install -U "huggingface_hub[hf_xet]"
hf download Qwen/Qwen3.6-35B-A3B --local-dir /workspace/qwen36
# (older CLI: huggingface-cli download Qwen/Qwen3.6-35B-A3B --local-dir /workspace/qwen36)
```

The vision-tower weights are interleaved in the shards, so you download them too; the
converter drops them.

## 4. Convert to a Colibri container

The converter strips the real model's `model.language_model.` prefix, skips `model.visual.*`,
unfuses the batched experts, and drops the MTP layer. Pick precision:

```bash
cd /workspace/colibri/c

# (a) int4 experts — the target: ~17 GB, fits in RAM, fastest. Start here.
python3 tools/convert_qwen36.py --indir /workspace/qwen36 --outdir /workspace/qwen36_i4 \
        --ebits 4 --io-bits 8 --xbits 4 --n-layers 40

# (b) int8 experts — higher fidelity (~30 GB) if int4 output looks degraded
# python3 tools/convert_qwen36.py --indir /workspace/qwen36 --outdir /workspace/qwen36_i8 \
#         --ebits 8 --io-bits 8 --xbits 8 --n-layers 40
```

Conversion is CPU+disk bound and unfuses ~10k expert tensors — **20–60 min**; run it in
`tmux`. It prints `converted 26 shard(s) -> …`. Peak RAM ~8–16 GB.

> **Quality note:** on the tiny *random* model, int4 experts flipped more tokens than int8
> because DeltaNet's recurrent state accumulates quant error over the sequence. On a *real
> trained* model int4 is usually fine, but if long generations drift, try the int8 container
> or measure with `tools/quant_ablation.py`.

## 5. Run & test

### Smoke test (does it produce coherent English?)

```bash
cd /workspace/colibri/c
SNAP=/workspace/qwen36_i4 TEMP=0 NGEN=64 RAM_GB=24 \
  PROMPT="Explain how a jet engine works." ./hy3 256 4 8
```

- `256` = expert-cache slots/layer (= all experts, so nothing streams — it's in RAM).
- `RAM_GB=24` sizes the budget; the model (~17 GB) fits, so decode is compute-bound.
- Watch the footer: `└─ N tok · X tok/s decode`. **This is your 5–10 tok/s number.**
- Add `QWEN_DEBUG=1` to trace per-layer values if the output is garbage (see §6).

### Perf breakdown

```bash
SNAP=/workspace/qwen36_i4 TEMP=0 NGEN=128 RAM_GB=24 PERF=1 \
  PROMPT="Write a short story about a lighthouse keeper." ./hy3 256 4 8
# PERF=1 prints [perf] attn=…% disk=…% expert_mm=…% head=…% every 100 tokens
```

### With the GPU (if you built `CUDA=1`)

```bash
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=12 \
SNAP=/workspace/qwen36_i4 TEMP=0 NGEN=128 RAM_GB=24 \
  PROMPT="Explain how a jet engine works." ./hy3 256 4 8
# confirm a line like: [CUDA] hot expert tier: N experts, VRAM … GB
```

### Interactive chat / OpenAI API (via coli)

`coli` now auto-routes `qwen3_5_moe` to the hy3 engine (or force with `COLI_ENGINE=hy3`):

```bash
COLI_MODEL=/workspace/qwen36_i4 ./coli chat --ram 24
COLI_MODEL=/workspace/qwen36_i4 ./coli chat --ram 24 --gpu 0 --vram 12   # with GPU
COLI_MODEL=/workspace/qwen36_i4 ./coli serve --ram 24 --host 127.0.0.1 --port 8000
```

## 6. Known limits (first bring-up) & troubleshooting

**Keep context modest for now.** Two things are deferred:
- `kv_alloc` still allocates a full KV cache for all 40 layers, incl. the 30 DeltaNet
  layers that only need a tiny recurrent state — wasteful at long context.
- `attention_qwen` caps scores at `sc[8192]`.

So **run with prompts + `NGEN` that keep total context well under ~4096** until the KV/state
split lands. Short prompts are fine.

**MTP is off** (converter drops the NEXTN layer) — you get plain autoregressive decode, no
speculative draft yet.

**mRoPE**: the full-attention layers assume text-only mRoPE collapses to standard partial
RoPE (validated on the tiny fixture). Very long contexts could drift if that assumption is
imperfect — sanity-check long-context output specifically.

**If output is garbage**, run with `QWEN_DEBUG=1` and check the `[qwen]` traces:
- `embed … x0=…` nonzero, `C afterL0/1/2/3 …` not all-zero and not NaN.
- all-zero → a norm or weight loaded wrong (cf. the `(1+weight)` bug in QWEN36_PORT.md).
- NaN/inf → check `A_log`/`dt_bias` decay or a missing tensor at load.
- coherent-but-wrong → likely quant; try the int8 container.

There's no cheap HF reference for the 35B (needs ~72 GB in torch); if you must diff
activations, cross-check against a llama.cpp Qwen3.6 GGUF instead.

## 7. Quick reference

```bash
# one-shot after clone + build + download:
cd /workspace/colibri/c
python3 tools/convert_qwen36.py --indir /workspace/qwen36 --outdir /workspace/qwen36_i4 --ebits 4 --io-bits 8 --xbits 4
SNAP=/workspace/qwen36_i4 TEMP=0 NGEN=64 RAM_GB=24 PROMPT="Hello, who are you?" ./hy3 256 4 8
```
