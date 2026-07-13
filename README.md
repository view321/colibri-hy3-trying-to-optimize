# colibrì × Tencent Hy3

**Run [Tencent Hy3](https://huggingface.co/tencent/Hy3) (295B MoE, 21B active) on a consumer PC** — pure C, disk-streaming experts, no GPU required.

This repository is a **Hy3 port** of [colibrì](https://github.com/JustVugg/colibri) (the engine that streams GLM-5.2 from disk). It adds:

- `c/hy3.c` — Hy3 (`hy_v3`) inference engine (GQA attention, sigmoid MoE router, expert streaming)
- `c/tools/convert_hy3.py` — FP8/BF16 → Colibri int4 container
- `coli` integration — chat, serve, convert, plan, doctor

Upstream GLM support in this tree is unchanged; Hy3 is auto-detected from `config.json` (`model_type: hy_v3`).

## Pre-converted weights (Hugging Face)

**https://huggingface.co/UnderstandLing/Hy3-colibri-int4**

~142 GB int4 Colibri container (107 main shards + 6 MTP shards). **Not** GGUF / AWQ / vLLM — only loads in this engine.

```bash
hf download UnderstandLing/Hy3-colibri-int4 --local-dir /path/to/hy3_i4
```

## Quick start

```bash
cd c
./setup.sh          # build hy3 + run hy3_tiny oracle self-test (32/32)

# download or convert model (see below), then:
COLI_MODEL=/path/to/hy3_i4 ./coli chat --ram 56
```

First reply on a **cold cache** is slow (minutes). Stay in the same session for faster follow-ups. Use `--verbose` to see engine stderr.

## Recommended flags (62 GB RAM machine)

```bash
COLI_MODEL=/path/to/hy3_i4 ./coli chat --ram 56 --temp 0.7
```

- **`--ram 56`** — larger expert LRU (vs 36 GB → more disk reads)
- Let **autopin** learn from `.coli_usage` (do not set `AUTOPIN=0` unless debugging)
- Model path on **native ext4** (WSL: use `/home/...`, not `/mnt/c`)

## Tiny oracle fixtures

These are **not** the full Hy3 model (~142 GB). They are a 5-layer random `hy_v3` checkpoint (~3 MB) used to validate `hy3.c` and `convert_hy3.py` without downloading Tencent weights.

| Path | In git? | Purpose |
|------|---------|---------|
| `c/hy3_tiny/` | **yes** | fp32 teacher-forcing oracle — `setup.sh` and CI expect **32/32** |
| `c/ref_hy3.json` | **yes** | expected prompt/full token IDs and TF logits from the HF reference run |
| `c/hy3_tiny_i4/` | **no** | int4 Colibri container derived from `hy3_tiny`; tests the converter + int4 load path |

`hy3_tiny_i4` is intentionally **not** committed (regenerable in seconds). Add `c/hy3_tiny_i4/` to your local `.gitignore` if you like.

### Regenerate `hy3_tiny` (+ `ref_hy3.json`)

Run from `c/` if the fixture is missing or you need to refresh the oracle:

```bash
cd c
pip install 'transformers>=5.6.0' torch safetensors
python3 tools/make_hy3_oracle.py   # writes hy3_tiny/ and ref_hy3.json
make hy3
SNAP=./hy3_tiny TF=1 ./hy3 64 16 16    # expect 32/32 positions
```

Greedy generate (no teacher forcing):

```bash
SNAP=./hy3_tiny ./hy3 64 16 16         # expect 20/20 new tokens vs ref_hy3.json
```

Use **`64 16 16`** on fp32 `hy3_tiny`. Running `./hy3 64 4 4` quantizes fp32 weights on load and **breaks** the oracle.

### Regenerate `hy3_tiny_i4` (local only)

After `hy3_tiny` exists:

```bash
cd c
./coli convert --repo ./hy3_tiny --model ./hy3_tiny_i4
# or: python3 tools/convert_hy3.py --indir hy3_tiny --outdir hy3_tiny_i4 --ebits 4

SNAP=./hy3_tiny_i4 TF=1 ./hy3 64 4 8    # expect 32/32 positions
```

Use **`64 4 8`** for the int4 tiny fixture (4-bit weights, 8-bit scales).

## Convert Hy3 yourself

From Hugging Face FP8 (resumable, never needs full checkpoint on disk):

```bash
pip install torch safetensors huggingface_hub numpy
./coli convert --repo tencent/Hy3-FP8 --model /path/to/hy3_i4
```

To smoke-test conversion on the tiny fixture before a full Hy3 run, see [Tiny oracle fixtures](#tiny-oracle-fixtures) (`hy3_tiny` → `hy3_tiny_i4`).

## Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 16 GB (4 cache slots/layer, slow) | **32–62 GB** (30–64 slots/layer) |
| **Disk** | ~150 GB model + fast NVMe | ext4, not WSL 9p `/mnt/c` |
| **CPU** | gcc, OpenMP, **AVX2** | more cores help matmul after cache warms |
| **GPU** | optional CUDA expert tier (see below) | 16 GB VRAM speeds hot experts |

Check your machine:

```bash
COLI_MODEL=/path/to/hy3_i4 ./coli plan --ram 56
COLI_MODEL=/path/to/hy3_i4 ./coli doctor --ram 56
```

**Dense resident RAM** is ~4–5 GB. Peak RSS = dense + KV + expert cache (set by `--ram`).

## Performance (honest)

Hy3 cold decode reads **~5–6 GB of experts per token** from disk (79 MoE layers × 8 experts). Expect:

| Phase | Typical speed |
|-------|----------------|
| First prompt (cold) | ~0.05–0.15 tok/s |
| Warm cache + 56 GB RAM | ~0.2–0.5 tok/s |
| Same chat session, turn 2+ | often several× faster |

Tuning: higher `--ram`, autopin/PIN from `.coli_usage`, `DIRECT=1` on NVMe, optional `PIPE=1`.

### CUDA expert tier (optional)

Same three-tier model as GLM: **disk → RAM → VRAM**. Hot pinned experts can run on GPU for faster matmul.

```bash
cd c
make hy3 CUDA=1                    # requires nvcc + NVIDIA driver
COLI_MODEL=/path/to/hy3_i4 ./coli chat --ram 56 --gpu 0 --vram 16
```

Or set env vars directly: `COLI_CUDA=1`, `CUDA_EXPERT_GB=16`, `PIN` / autopin from `.coli_usage`.
Use `coli plan --auto-tier` and `coli doctor` to size RAM/VRAM tiers. Optional `CUDA_DENSE=1` uploads resident dense weights to GPU; `REPIN=N` swaps cold VRAM pins for hot experts during chat.

## OpenAI-compatible API

```bash
COLI_MODEL=/path/to/hy3_i4 ./coli serve --ram 56 --host 127.0.0.1 --port 8000
```

Model id: `hy3-colibri`.

## What we contribute (vs upstream Hy3)

| Piece | Description |
|-------|-------------|
| **Colibri int4 format** | `out-*.safetensors` + per-row `.qs` scales — incompatible with GGUF/AWQ |
| **`convert_hy3.py`** | Hy3-FP8 per-tensor `weight_scale` dequant, `shared_mlp` → `shared_experts` naming |
| **`hy3.c`** | GQA, sigmoid+bias router, route norm, batch-union MoE, LRU/pin cache |
| **Chat template** | Hunyuan `｜` delimiters, `coli chat` + `coli serve` |

Base model: [tencent/Hy3](https://huggingface.co/tencent/Hy3) / [Hy3-FP8](https://huggingface.co/tencent/Hy3-FP8) (Apache 2.0).  
Same idea as [GLM-5.2-colibri-int4](https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4), applied to Hy3.

## Architecture (Hy3 vs GLM in colibrì)

| | GLM-5.2 | Hy3 |
|--|---------|-----|
| Attention | MLA (compressed KV) | **GQA** (64Q / 8KV) |
| Experts | 256 × 75 layers | **192 × 79 layers** |
| Expert size (int4) | ~19 MB | **~9 MB** |
| Dense prefix | 3 layers | **1 layer** |
| Disk container | ~370 GB | **~142 GB** |

## Status & roadmap

**Working today**

- FP8 → int4 conversion (full 80-layer model)
- `hy3_tiny` oracle **32/32** (committed); `hy3_tiny_i4` oracle **32/32** (regenerate locally)
- Full-model chat (coherent output with `IDOT=0`, fixed serve KV alloc)
- `coli chat` / `coli serve` / `coli convert` / `coli plan` / `coli doctor`

**Not yet**

- MTP speculative decode (weights converted; runtime not wired)
- KV disk persistence (GQA layout)
- int8 IDOT fast path (disabled by default until oracle-clean on all shapes)

## Build

```bash
cd c && make hy3
# optional CUDA build (Linux + nvcc):
make hy3 CUDA=1
# optional self-test (requires hy3_tiny/ — see Tiny oracle fixtures):
SNAP=./hy3_tiny TF=1 ./hy3 64 16 16
```

## Provenance & license

- Engine fork: colibrì ([MIT](https://github.com/JustVugg/colibri))
- Hy3 weights: [Tencent Hy3](https://huggingface.co/tencent/Hy3) (Apache 2.0)
- This port's converted weights: [UnderstandLing/Hy3-colibri-int4](https://huggingface.co/UnderstandLing/Hy3-colibri-int4) (Apache 2.0 derivative)

## Citation

Credit [Tencent Hy3](https://huggingface.co/tencent/Hy3) for the base model and [colibrì](https://github.com/JustVugg/colibri) for the streaming engine.
