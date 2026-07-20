#!/usr/bin/env python3
"""Tiny random Qwen3.6 (qwen3_5_moe, Gated-DeltaNet hybrid) oracle for the C engine port.

Mirrors make_hy3_oracle.py: seed -> randomize -> greedy generate -> teacher-force argmax ->
dump {prompt_ids, full_ids, tf_pred}. The C engine's TF gate (hy3.c:3548) reaches N/N against
this. See docs/QWEN36_PORT.md.

RUN ON RUNPOD (needs torch + transformers>=4.57.1; the dev box is Windows/py3.14 with no torch):

    pip install 'transformers>=4.57.1' torch safetensors
    python3 tools/make_qwen36_oracle.py        # writes qwen36_tiny/ and ref_qwen36.json

UNVALIDATED until first RunPod run — the exact class/config resolution for this brand-new arch
must be confirmed there. The state_dict dump below is what we use to map tensor names into
convert_qwen36.py and the loader. If AutoModel fails to resolve the arch, the except block lists
the Qwen classes your installed transformers actually exposes so we can adjust model_type/class.
"""
import json
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

OUT_DIR = "qwen36_tiny"
REF = "ref_qwen36.json"

# Tiny but architecturally faithful: 3:1 linear/full (full_attention_interval=4), a small MoE
# with a shared expert, partial+mRoPE, DeltaNet with grouped heads (2 k/q -> 4 v). Field names are
# copied verbatim from the real config.json so the registered config class parses them.
ROT = int(32 * 0.25)  # rotary_dim = head_dim * partial_rotary_factor = 8 ; half = 4 = sum(mrope_section)
cfg_dict = {
    # text-only; if the installed transformers only registers the multimodal wrapper "qwen3_5_moe",
    # switch to that + a text_config block (verify on RunPod).
    "model_type": "qwen3_5_moe_text",
    "architectures": ["Qwen3_5MoeTextForCausalLM"],
    "vocab_size": 512,
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "full_attention_interval": 4,
    # full attention (the 1 full layer)
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "attn_output_gate": True,
    "attention_bias": False,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000000,
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],  # sums to ROT//2 = 4
        "partial_rotary_factor": 0.25,
    },
    # gated deltanet (the 3 linear layers)
    "linear_key_head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_value_head_dim": 16,
    "linear_num_value_heads": 4,
    "linear_conv_kernel_dim": 4,
    "mamba_ssm_dtype": "float32",
    # moe (every layer)
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 4096,
    "mtp_num_hidden_layers": 0,  # base oracle; MTP gets its own fixture in M6
    "tie_word_embeddings": False,
}

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(cfg_dict, open(f"{OUT_DIR}/config.json", "w"), indent=2)

torch.manual_seed(5678)

try:
    cfg = AutoConfig.from_pretrained(OUT_DIR)
    cfg._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(cfg).eval()
except Exception as e:  # noqa: BLE001 — diagnostic: show what the installed transformers exposes
    import transformers

    qwen = sorted(n for n in dir(transformers) if "wen3" in n.lower() or "qwen" in n.lower())
    print(f"\nAutoModel failed to build the tiny qwen3_5_moe fixture: {e}\n", file=sys.stderr)
    print("Qwen-related classes in this transformers build:", file=sys.stderr)
    for n in qwen:
        print("   ", n, file=sys.stderr)
    print(
        "\nAdjust cfg_dict['model_type'] / the class above to match, then rerun. "
        "The real model needs transformers>=4.57.1.",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

with torch.no_grad():
    for _, p in model.named_parameters():
        if p.dim() >= 2:  # leave 1-D params (norms, A_log, dt_bias) at their meaningful inits
            p.normal_(0, 0.05)

print("=== qwen3_5_moe tiny state_dict (drives convert_qwen36.py + the loader) ===")
for n, p in model.state_dict().items():
    print(f"  {n:64s} {tuple(p.shape)}")

prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
ids = torch.tensor([prompt])
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
full = out[0].tolist()
print("\nprompt:", prompt)
print("full  :", full)

with torch.no_grad():
    lg = model(torch.tensor([full]), use_cache=False).logits[0]
tf_pred = lg.argmax(-1).tolist()
print("tf_pred:", tf_pred)

model.save_pretrained(OUT_DIR, safe_serialization=True)
json.dump(cfg.to_dict(), open(f"{OUT_DIR}/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred}, open(REF, "w"))
print(f"\nsaved: {OUT_DIR}/ and {REF}")
print("next: SNAP=./qwen36_tiny TF=1 REF=ref_qwen36.json REF_FORCE=1 ./hy3 64 16 16   (once the loader lands)")
