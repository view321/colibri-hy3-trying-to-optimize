# Porting the engine to Qwen3.6-35B-A3B (Gated-DeltaNet hybrid MoE)

Status: **in progress**. This is a *port*, not the softmax-router adapter first scoped for
plain-GQA Qwen3. Qwen3.6 (`model_type: qwen3_5_moe`) is a **hybrid linear/full-attention**
model — 30 of 40 layers are Gated DeltaNet (recurrent linear attention, no KV cache), 10 are
gated full attention. The engine (`c/hy3.c`) currently has exactly one attention path (GQA +
growing KV cache). Adding DeltaNet is a new subsystem; the MoE side is the easy part.

Target hardware / goal unchanged: **5–10 tok/s on 32 GB RAM + NVMe (+ optional 12 GB VRAM)**.
DeltaNet actually *helps* here: its per-layer state is a small **constant-size** matrix (no KV
growth), so long-context RAM stays flat and only the MoE experts stream from disk.

---

## 1. Target architecture (from the real `config.json`)

`Qwen/Qwen3.6-35B-A3B` — 35B total / ~3B active, multimodal (`Qwen3_5MoeForConditionalGeneration`;
we run **text-only** and skip the vision tower). Text config:

| Field | Value | Meaning |
|---|---|---|
| `num_hidden_layers` | 40 | every layer is `attention → MoE` |
| `layer_types` | `3×linear, 1×full` ×10 | `full_attention_interval: 4` |
| `hidden_size` | 2048 | |
| **Full-attn** `num_attention_heads` / `num_key_value_heads` / `head_dim` | 16 / 2 / 256 | GQA, **head_dim 256** |
| `attn_output_gate` | true | q_proj emits `head_dim*2`; `attn_out *= sigmoid(gate)` |
| `partial_rotary_factor` | 0.25 | rotate only first `0.25*256 = 64` dims |
| `rope_parameters` | mRoPE, `mrope_section [11,11,10]`, θ=1e7 | multimodal RoPE; text-only collapses |
| **DeltaNet** `linear_num_key_heads` / `linear_key_head_dim` | 16 / 128 | key_dim = 2048 |
| `linear_num_value_heads` / `linear_value_head_dim` | 32 / 128 | value_dim = 4096 |
| `linear_conv_kernel_dim` | 4 | depthwise causal conv, silu |
| `mamba_ssm_dtype` | float32 | recurrent state precision |
| **MoE** `num_experts` / `num_experts_per_tok` | 256 / 8 | + 1 shared |
| `moe_intermediate_size` / `shared_expert_intermediate_size` | 512 / 512 | tiny experts (~1.5 MB int4 each) |
| router | softmax + `norm_topk_prob` | **no** expert bias (unlike Hy3's sigmoid+bias) |
| `mtp_num_hidden_layers` | 1 | one NEXTN speculative layer |
| `vocab_size` | 248320 | large (multimodal vocab) |
| `tie_word_embeddings` | false | |

Recurrent-state size per DeltaNet layer: `32 × 128 × 128` fp32 ≈ **2 MB**, plus a `8192 × 3`
conv ring ≈ 96 KB — **independent of context length**.

---

## 2. The math to implement (HF reference = ground truth)

References: HF `transformers/models/qwen3_next/modular_qwen3_next.py`, NVlabs/GatedDeltaNet
(ICLR 2025), `ggml-org/llama.cpp` `src/models/delta-net-base.cpp` (C++ port to cross-check),
Raschka LLMs-from-scratch ch04/08 (clean pedagogical version). **The oracle (§4) is the final
arbiter** — implement to match its logits, not to match prose here.

### 2a. Gated DeltaNet layer (30 of 40)

```
# projections
q,k,v,z   = split(in_proj_qkvz(h))      # q,k ∈ key_dim=2048 ; v,z ∈ value_dim=4096
beta,a    = split(in_proj_ba(h))        # per value-head (32 each)
# short causal conv (depthwise, kernel 4, silu) over concat[q,k,v]  (conv_dim = 8192)
q,k,v     = silu(causal_conv1d([q,k,v], conv_state))
# per-head reshape: q,k → [16 heads,128] ; v,z → [32 heads,128]   (16 k/q heads shared across 32 v heads)
q,k       = l2norm(q,eps=1e-6), l2norm(k,eps=1e-6)
g         = exp( -exp(A_log) * softplus(a + dt_bias) )     # decay ∈ (0,1), per v-head
# per value-head h, state S_h ∈ [k_head_dim=128, v_head_dim=128], per step:
S_h      *= g_h
kv        = Σ_j S_h[j,:] * k_h[j]        # → [v_head_dim]
delta     = (v_h - kv) * beta_h          # → [v_head_dim]
S_h[j,:] += k_h[j] * delta               # outer product update
o_h       = Σ_j S_h[j,:] * q_h[j]        # → [v_head_dim]
# output: concat o over 32 heads → [4096]
out       = RMSNorm(o, eps=1e-6) * silu(z)     # RMSNormGated
out       = out_proj(out)                       # 4096 → 2048
```

Two code paths: **autoregressive single-step** (decode; the recurrence above literally) and a
**chunked scan** (prefill; llama.cpp's `build_delta_net_chunking` — cumsum-decay masked form for
speed). Correctness first with the simple scan, optimize later. **Open item:** confirm the exact
16→32 head grouping (`v_head // 2`?) against the oracle before trusting either path.

### 2b. Gated full attention (10 of 40)

Standard GQA (16Q/2KV, head_dim 256) **plus**: q_proj emits `num_heads*head_dim*2`, chunk into
`query` and `gate`; per-head QK-RMSNorm (engine already has this); **partial** RoPE on the first
`64` dims only (rest pass through); mRoPE (for pure text all three `mrope_section` positions equal
the token index → reduces to standard RoPE on the rotated dims — **verify against oracle**); after
attention, `attn_out *= sigmoid(gate)`; then o_proj.

### 2c. MoE (all 40 layers)

256 experts, top-8, **softmax** router with `norm_topk_prob` renormalize, **+1 shared expert**
(`shared_expert_intermediate_size`), **no** router bias. This is olmoe.c's router grafted onto
hy3.c's batching/streaming (see §3.5).

---

## 3. Engine integration map (`gpu-accel` working tree, file:line)

Call chain: `main → generate → step/step_all → layers_forward → layer_forward → attention + (moe|dense_mlp)`.

### 3.1 Dispatch point
`layer_forward` **hy3.c:2181-2190**; the unconditional `attention(m,l,li,nrm,S,pos_base,tmp)` at
**2184** is where we branch `if (l->attn_type==LINEAR) gated_deltanet(...) else attention(...)`.

### 3.2 Config → struct
`Cfg` struct **111-117**, parsed in `load_cfg` **1040-1066** (helpers `gi`/`gf`, JSON via `cfg_root`
1030; arrays via `json_get(...)->kids[i]->str`, `->len`). Add: `layer_types[]`,
`full_attention_interval`, `linear_{key,value}_head_dim`, `linear_num_{key,value}_heads`,
`linear_conv_kernel_dim`, `shared_expert_intermediate_size`, `router_softmax`, `partial_rotary_factor`.
Note the tiny config already carries an (unparsed) `mlp_layer_types` and `moe_router_use_sigmoid` —
precedent for the new keys.

### 3.3 The KV-cache assumption to break (the core work)
`kv_alloc` **1658-1687** allocates a full growing `[n_kv_heads·max_t·head_dim]` K and V for **every**
layer (`NR = n_layers+has_mtp`). `kv_pool_bytes` **2572-2580** budgets RAM the same way; the CUDA-attn
guard **1742** and `kv_start[]` **162/1678/1684** assume it too. Plan: keep K/V/kv_start indexed by
layer but allocate the growing buffer **only for the 10 full-attn layers**; add parallel
`float **recur_state; float **conv_state;` (Model fields near **159-162**) allocated **only for the 30
linear layers** (fixed size, context-independent). Every one of the 14 sites in §5 branches on
`attn_type`.

### 3.4 Attention internals to generalize
`attention()` **1701-1811** (GQA only). `rope_rotate_half` **1019-1028** rotates the full head_dim with
a `float tmp[512]` cap — add `partial_rotary_factor`. Per-head QK-norm at 1712-1719 is reusable.

### 3.5 MoE router (graft)
`moe` **1991-2179**. Replace the sigmoid+bias at **2000** with `softmax(logit,E)` (drop `router_bias`),
per olmoe.c **267-296** (`softmax_row` → top-k → `norm_topk` renormalize, flag parsed at olmoe.c:134).
Keep batching **2063-2153**, shared expert **2158-2175**, LRU/pin/stream **2066-2152** unchanged.

### 3.6 Tensor load / quant
Resident dense loaded in `model_init` **1491-1656** (names via `P(...)` macro **1527-1557**); experts
streamed by name in `expert_load` **1153-1155**. Quant packing inferred by byte size (`.qs` sibling
scales) in `qt_from_disk` **1068-1083** / `expert_finalize` **1140-1144** — attention-agnostic, reuse.
Load small DeltaNet params (conv, A_log, dt_bias, norms) **f32 resident** via `ld()` like `qn`/`kn`
(1534-1535).

### 3.7 Oracle gate
`make_hy3_oracle.py` → `ref_hy3.json` = `{prompt_ids, full_ids, tf_pred}`. C gate: `main`
**3548-3572** reads the ref, runs `forward_all` **2203-2217** (teacher-forced prefill, argmax/pos),
prints `PREFILL (teacher-forcing) C vs oracle: N/N positions`. Honors `REF=path` (3542) and
`REF_FORCE=1` (bypass the tiny-vocab guard, 3553). We replicate this for the hybrid fixture.

### 3.8 Converter
`convert_hy3.py` (175 lines): `classify` **56-82** / `rename_out` **51-53** / `dequant` **32-48** /
`convert_shard` **85-104**; `config.json` copied verbatim **118-119** (so new config fields flow to
`load_cfg`). Fork → `convert_qwen36.py` for the new tensor names; source is **bf16** (not FP8).

---

## 4. Correctness strategy (non-negotiable — the repo's discipline)

Build the oracle first, match it, never claim a step works until the gate is green:

1. **`make_qwen36_oracle.py`** → tiny random hybrid fixture (`qwen36_tiny/`) + `ref_qwen36.json`
   (same 3 fields). A handful of layers using the real `layer_types` pattern (e.g.
   `[linear,linear,linear,full]`), small dims, small MoE with a shared expert, tiny vocab. Runs on
   **RunPod** (needs `torch` + `transformers>=4.57.1`; the dev box is Windows/py3.14 with no torch).
2. **Per-subsystem gates** so failures localize:
   - full-attn only fixture → `forward_all` N/N (M4),
   - deltanet-only fixture → N/N (M5),
   - full hybrid at **fp32** N/N, then **int4** (measure argmax flips, cf. `hy3_tiny_i4`) (M6).
3. Only then the real 35B (M7).

Dev workflow: **edit + build (`make hy3`) + run the tiny oracle locally** on Windows/MSYS2 (gcc 15.2,
`hy3.exe` already builds; CPU-only, no CUDA here). **Generate oracles + run the real model on RunPod.**

---

## 5. Everything hardwired to "GQA + KV-for-every-layer" (generalization checklist)

1. `Layer` struct **135-143** — add `attn_type` + DeltaNet weight fields.
2. `Cfg` + `load_cfg` **111-117 / 1040-1066** — new fields (§3.2).
3. `layer_forward` **2184** — the dispatch branch.
4. `attention()` **1701-1811** — GQA-only; DeltaNet is a separate fn.
5. `rope_rotate_half` **1019-1028** — add `partial_rotary_factor`.
6. `kv_alloc` **1658-1687** — growing KV only for full-attn layers.
7. `kv_pool_bytes` **2572-2580** — RAM budget wrong once 30/40 are O(1)-state.
8. KV indexing in `attention` **1721-1727 / 1773-1801** (+ int8 `Kq/Vq` variant).
9. `kv_start[]` **162 / 1678 / 1684 / 1743 / 1761** — overload/extend for linear layers.
10. CUDA GQA-attn path **1732-1755** + `backend_loader.c:55-61` — exclude linear layers.
11. Model KV fields **159-162** — add `recur_state`/`conv_state` arrays.
12. `sparse` predicate **1536** — Qwen3.6 has two independent per-layer axes (attn-type and
    MoE-vs-dense); `first_k_dense_replace` can't express the interleave — use parsed `layer_types[]`.
13. `predict_prefetch` **1947-1988** — router math must match the softmax variant.
14. MTP head **1600-1638** + `mtp_one_forward`/`mtp_absorb` **2265-2389** — inherits single-attn
    assumption; the NEXTN layer's own type must be honored.

**Reusable as-is:** MoE batching/expert/LRU/streaming, `.qs` quant inference, `st.h` loader,
`matmul_qt` dispatch — all attention-agnostic.

---

## 6. Milestones (see task list)

1. Oracle generator + tiny fixture (foundation).
2. `convert_qwen36.py`.
3. Config parsing + per-layer attention dispatch + KV/state split (scaffolding compiles).
4. Gated full-attention path → full-attn oracle N/N.
5. Gated DeltaNet path → deltanet oracle N/N.
6. Softmax+shared MoE router + MTP → full hybrid oracle N/N (fp32) then int4.
7. Real 35B bring-up on RunPod + perf (CUDA/surgery/PREDICT), measure vs 5–10 tok/s.

## 6b. Implementation status — VALIDATED (32/32 bit-exact on the tiny oracle)

The full text forward path is implemented AND validated locally (uv venv, CPU torch — no
RunPod needed): the tiny hybrid oracle passes **32/32 f32** (bit-exact vs HF) and **29/32
int8-experts** (the 3 flips are int8 expert-quant noise compounding through the DeltaNet
recurrent state — 2 are sub-0.02-logit close-calls, the 3rd is the last position; f32
experts give 32/32). Hy3 stays 32/32 throughout. Traces: `QWEN_DEBUG=1`.

Done in `hy3.c`:
- Config parsing (`load_cfg`): `layer_types`, `linear_*`, `partial_rotary_factor`,
  softmax router, `norm_topk`, `text_config` nesting. Defaults preserve Hy3.
- Dispatch (`layer_forward`): `attn_type==1 → gated_deltanet`, else `is_qwen36 →
  attention_qwen`, else `attention`.
- `attention_qwen()`: q_proj [query|gate] split, per-head QK-norm, partial RoPE,
  GQA over float KV, `sigmoid(gate)` output gate, o_proj.
- `gated_deltanet()`: in_proj splits, causal depthwise conv+silu with persistent
  `conv_state`, L2-norm q/k, per-step gated delta rule with persistent
  `recur_state`, gate-first `RMSNormGated`, out_proj.
- MoE softmax router (`moe` + `predict_prefetch`): softmax, no bias, `route_norm`=
  `norm_topk`; `router_bias` load skipped under softmax.
- Loader: Qwen attn (`qo*2`) + DeltaNet weights; `convert_qwen36.py`; `make_qwen36_oracle.py`.

### RunPod bring-up checklist (in order)
1. Run `make_qwen36_oracle.py`; reconcile the printed **state_dict names** against the
   loader `P("linear_attn.*")` / `self_attn.*` and `convert_qwen36.py::classify`.
2. Convert the tiny fixture; run `SNAP=./qwen36_tiny TF=1 REF=ref_qwen36.json REF_FORCE=1
   QWEN_DEBUG=1 ./hy3 64 16 16`; drive toward N/N, fixing the verify points below.
3. Only then scale to the real 35B (M7).

### Bugs found & fixed during bring-up (the oracle earned its keep)
Against the real state_dict + HF per-layer activations, six things were wrong; all fixed:
1. **DeltaNet has 4 SEPARATE projections** (`in_proj_qkv|z|b|a`), not fused `qkvz`/`ba`.
2. **MoE experts are FUSED/batched** (`experts.gate_up_proj [E,2I,D]` + `down_proj [E,D,I]`) —
   `convert_qwen36.py` unfuses to per-expert `gate/up/down_proj` (gate = first I rows).
3. **`shared_expert_gate [1,D]` exists** — multiply the shared expert by `sigmoid(x·w)`.
4. **Standard RMSNorm is zero-centered**: `x*r*(1+weight)` (`Qwen3_5MoeRMSNorm`; weights
   store 0). Missing this **zeroed the entire forward**. The gated `linear_attn.norm` is plain.
5. **RMSNormGated order is norm-FIRST**: normalize → weight → `×silu(z)` (I'd done gate-first).
6. **DeltaNet query scale** `query *= 1/sqrt(k_head_dim)` after l2norm — NOT washed by the
   RMSNorm here because `core` is tiny (~1e-3) so `eps` matters; missing it made `core` 4× big.

Matched HF on the first try (no fix needed): `beta=sigmoid(b)`,
`g=-exp(A_log)·softplus(a+dt_bias)`, `decay=exp(g)`, q/k L2-norm, `repeat_interleave` grouping
`hv/grp`, causal conv tap order, softmax router + norm_topk, partial RoPE, mRoPE→partial text collapse.

### Still open (not blocking the base gate)
- **MTP-for-Qwen** NEXTN head (tiny oracle uses `mtp=0`; the real model has `mtp=1`).
- **State vs spec-decode**: `recur_state`/`conv_state` reset only at `pos_base==0`; the
  spec-decode re-forward may need explicit save/restore.
- **Perf, deferred**: KV/state split in `kv_alloc` (full KV still allocated for linear layers —
  wasteful at long ctx); `sc[8192]` score cap; chunked-scan prefill; int4 experts on the real
  model (int8 already shows recurrent-noise sensitivity — measure quality with `quant_ablation`).

### Local dev loop (no RunPod)
```
uv venv --python 3.12 <venv>; uv pip install --python <venv> torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python <venv> "transformers>=4.57.1" safetensors numpy
<venv>/python tools/make_qwen36_oracle.py                        # -> qwen36_tiny/ + ref_qwen36.json
<venv>/python tools/convert_qwen36.py --indir qwen36_tiny --outdir qwen36_tiny_f32 --ebits 16 --io-bits 16 --xbits 16
make hy3 && SNAP=./qwen36_tiny_f32 TF=1 REF=ref_qwen36.json REF_FORCE=1 ./hy3.exe 64 16 16   # 32/32
tools/dump_qwen36_hidden.py diffs HF per-layer activations when a stage is wrong.
```

## 7. Open risks

- **DeltaNet head grouping** (16 k/q ↔ 32 v) and the exact decay/beta application — verify against
  the oracle, not prose.
- **mRoPE for text** — confirm it collapses to standard partial RoPE with no image tokens.
- **Chunked-scan prefill** numerics vs the simple recurrence — start with the slow-but-obvious scan.
- **int4 on DeltaNet** — conv/gate/A_log/dt kept f32; only expert (and maybe q/k/v/o proj) weights
  quantize. Measure quality like the existing int4 ablation.
- **transformers class/version** for the tiny fixture — pin the exact `qwen3_5_moe` support version
  on first RunPod run.
