#!/usr/bin/env python3
"""Dump HF per-layer hidden states for the tiny fixture, to diff against the C engine
and localize where the Qwen3.6 forward diverges. Run in the same venv as the oracle."""
import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM

OUT = "qwen36_tiny"
POS = 0  # which teacher-forcing position to inspect

cfg = AutoConfig.from_pretrained(OUT)
cfg._attn_implementation = "eager"
model = AutoModelForCausalLM.from_pretrained(OUT).eval()
full = json.load(open("ref_qwen36.json"))["full_ids"]
ids = torch.tensor([full])

# hook layer-0 (deltanet) + layer-3 (full attn) sub-modules to split attn vs mlp
caps = {}
def mk(name):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        caps[name] = o.detach()
    return hook
L = model.model.layers
L[0].input_layernorm.register_forward_hook(mk("L0.in_ln"))
L[0].linear_attn.register_forward_hook(mk("L0.attn"))
L[0].mlp.register_forward_hook(mk("L0.mlp"))
L[3].self_attn.register_forward_hook(mk("L3.attn"))
L[3].mlp.register_forward_hook(mk("L3.mlp"))
def norm_pre(mod, args, kwargs):
    caps["L0.norm.core"] = args[0].detach()
    caps["L0.norm.z"] = (args[1] if len(args) > 1 else kwargs["gate"]).detach()
L[0].linear_attn.norm.register_forward_pre_hook(norm_pre, with_kwargs=True)
L[0].linear_attn.norm.register_forward_hook(mk("L0.norm.out"))
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
hs = out.hidden_states  # tuple: [embed, after L0, after L1, ..., after L_{n-1}]
print(f"positions={len(full)} layers(hidden_states)={len(hs)} pos={POS}")
for L, h in enumerate(hs):
    v = h[0, POS, :5].tolist()
    tag = "embed" if L == 0 else f"afterL{L-1}"
    print(f"HF {tag:8s} pos{POS}: " + " ".join(f"{x:+.5f}" for x in v) + f"   |norm={h[0,POS].norm().item():.5f}")
print("--- sub-layer outputs (pre-residual) at pos0 ---")
for k in ("L0.in_ln", "L0.attn", "L0.mlp", "L3.attn", "L3.mlp"):
    if k in caps:
        v = caps[k][0, POS, :5].tolist()
        print(f"HF {k:9s}: " + " ".join(f"{x:+.5f}" for x in v))
# deltanet norm internals at pos0, v-head0 (row 0 of the [B*S*num_v_heads, head_v_dim] reshape)
for k in ("L0.norm.core", "L0.norm.z", "L0.norm.out"):
    if k in caps:
        v = caps[k][0, :5].tolist()
        print(f"HF {k:12s}: " + " ".join(f"{x:+.5f}" for x in v))
# final logits argmax at POS
lg = out.logits[0, POS]
print(f"HF logits pos{POS}: argmax={int(lg.argmax())} top5={lg.topk(5).indices.tolist()}")
