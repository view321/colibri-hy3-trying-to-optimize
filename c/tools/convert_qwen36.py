#!/usr/bin/env python3
"""Convert Qwen/Qwen3.6-35B-A3B (qwen3_5_moe, Gated-DeltaNet hybrid, bf16)
-> Colibri container. Fork of convert_hy3.py. See docs/QWEN36_PORT.md.

Reconciled against the real state_dict (make_qwen36_oracle.py dump):
  - MoE experts are FUSED/batched: mlp.experts.gate_up_proj [E,2I,D] +
    mlp.experts.down_proj [E,D,I]. We UNFUSE into per-expert tensors the engine
    streams: experts.{j}.gate_proj.weight [I,D], .up_proj.weight [I,D],
    .down_proj.weight [D,I]. (gate = first I rows of gate_up, up = next I.)
  - DeltaNet has 4 separate projections: in_proj_qkv|z|b|a (not fused qkvz/ba).
  - shared_expert.* (renamed -> shared_experts.*) + shared_expert_gate.weight.
  - softmax router mlp.gate.weight, no expert bias.

Quantization: bits>8 (e.g. 16) writes f32 (no .qs) — the loader then keeps it f32
at dbits=16. Use --ebits 16 --io-bits 16 --xbits 8 for a math-isolating run
(only MoE experts quantized). Use 4/4/4 for a real int4 container.

Usage:
  python3 tools/convert_qwen36.py --indir qwen36_tiny --outdir qwen36_tiny_c \
      --ebits 16 --io-bits 16 --xbits 8
"""
import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import convert_hy3 as base  # noqa: E402  (dequant reused)
from convert_fp8_to_int4 import quant_int2, quant_int4, quant_int8, layer_idx  # noqa: E402

SHARED_RE = re.compile(r"\.mlp\.shared_expert\.")


def rename_out(name):
    return SHARED_RE.sub(".mlp.shared_experts.", name)


def remap_mtp(name, n_layers):
    """Map the NEXTN/MTP head (mtp.*) onto the engine's MTP slots at layer n_layers."""
    L = "model.layers.%d." % n_layers
    name = name.replace("mtp.fc.weight", L + "eh_proj.weight")
    name = name.replace("mtp.pre_fc_norm_embedding.weight", L + "enorm.weight")
    name = name.replace("mtp.pre_fc_norm_hidden.weight", L + "hnorm.weight")
    name = name.replace("mtp.norm.weight", L + "shared_head.norm.weight")
    name = name.replace("mtp.layers.0.", L)   # the MTP decoder layer (full-attn + MoE)
    return name


def emit(out, name, w, bits):
    """Write w to out under name, quantized to `bits` (bits>8 or non-2D => f32)."""
    w = w.astype(np.float32)
    if bits > 8 or w.ndim != 2:
        out[name] = w
        return
    q, s = (quant_int2(w, bits) if bits <= 2 else
            quant_int4(w, bits) if bits <= 4 else quant_int8(w, bits))
    out[name] = q
    out[name + ".qs"] = s


def classify(name):
    if name.endswith("_scale") or name.endswith("_scale_inv"):
        return "consumed"
    if (name.endswith("norm.weight") or name == "model.norm.weight"
            or name.endswith("q_norm.weight") or name.endswith("k_norm.weight")
            or name.endswith("mlp.gate.weight")
            or name.endswith("shared_expert_gate.weight")
            or name.endswith(".A_log") or name.endswith(".dt_bias")
            or name.endswith("conv1d.weight") or name.endswith("conv1d.bias")):
        return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "io"
    if name.endswith(".weight"):
        return "q"  # q/k/v/o_proj, in_proj_qkv/z/b/a, out_proj, shared_expert.*_proj
    return "f32"


def convert_shard(path, out_dict, n_layers, ebits, io_bits, xbits, keep_mtp=False):
    from safetensors import safe_open
    import torch
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            # skip the vision tower — the engine runs text-only
            if name.startswith("model.visual") or ".visual." in name:
                continue
            # NEXTN/MTP head: keep only with --mtp, remapped onto layer n_layers
            if name.startswith("mtp"):
                if not keep_mtp:
                    continue
                oname = remap_mtp(name, n_layers)
            else:
                # the real multimodal model nests the LM under model.language_model.* ;
                # strip to model.* so the container matches the loader (no-op for the
                # text-only tiny fixture; lm_head.weight is already top-level).
                oname = name.replace("model.language_model.", "model.", 1)
            li = layer_idx(oname)
            if li is not None and li >= 0 and li >= n_layers and not keep_mtp:
                continue   # drops the MTP layer (index n_layers) for now
            # --- fused experts -> per-expert (the big reconciliation) ---
            if oname.endswith("mlp.experts.gate_up_proj"):
                w = f.get_tensor(name).to(torch.float32).numpy()   # [E, 2I, D]
                E, twoI, _ = w.shape
                I = twoI // 2
                pre = oname[:-len("gate_up_proj")]                 # ...mlp.experts.
                for j in range(E):
                    emit(out_dict, f"{pre}{j}.gate_proj.weight", w[j, :I, :], xbits)
                    emit(out_dict, f"{pre}{j}.up_proj.weight",   w[j, I:, :], xbits)
                continue
            if oname.endswith("mlp.experts.down_proj"):
                w = f.get_tensor(name).to(torch.float32).numpy()   # [E, D, I]
                E = w.shape[0]
                pre = oname[:-len("down_proj")]
                for j in range(E):
                    emit(out_dict, f"{pre}{j}.down_proj.weight", w[j], xbits)
                continue
            # --- everything else ---
            kind = classify(oname)
            if kind == "consumed":
                continue
            w = base.dequant(f, name)
            oname = rename_out(oname)
            bits = io_bits if kind == "io" else (ebits if kind == "q" else 99)  # f32 => 99
            emit(out_dict, oname, w, bits)


def main():
    ap = argparse.ArgumentParser(description="Qwen3.6 -> Colibri container")
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ebits", type=int, default=4, help="proj weights; >8 => f32")
    ap.add_argument("--io-bits", type=int, default=8, help="embed/lm_head; >8 => f32")
    ap.add_argument("--xbits", type=int, default=4, help="MoE experts")
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--mtp", action="store_true", help="include the NEXTN/MTP head for speculative decode")
    a = ap.parse_args()

    from safetensors.numpy import save_file
    shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))
    os.makedirs(a.outdir, exist_ok=True)
    for i, sp in enumerate(shards):
        out = {}
        convert_shard(sp, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_mtp=a.mtp)
        save_file(out, os.path.join(a.outdir, f"out-{i:05d}.safetensors"))
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json",
               "generation_config.json", "chat_template.jinja",
               "vocab.json", "merges.txt", "special_tokens_map.json"):
        src = os.path.join(a.indir, fn)
        if os.path.exists(src):
            shutil.copy(src, a.outdir)
    print(f"converted {len(shards)} shard(s) -> {a.outdir}")


if __name__ == "__main__":
    main()
