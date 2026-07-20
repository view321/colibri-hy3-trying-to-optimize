#!/usr/bin/env python3
"""Convert Qwen/Qwen3.6-35B-A3B (qwen3_5_moe, Gated-DeltaNet hybrid, bf16)
-> Colibri int4 container. Fork of convert_hy3.py. See docs/QWEN36_PORT.md.

Reuses convert_hy3's shard/quant/download machinery and only overrides how tensors
are classified + renamed. Source is **bf16** (not FP8), which the shared dequant()
already passes through to f32.

Tensor kinds (per the HF qwen3_5_moe naming; UNVERIFIED — reconcile against the
state_dict dump printed by make_qwen36_oracle.py before a full run):
  full-attn layers : self_attn.{q,k,v,o}_proj.weight, self_attn.{q,k}_norm.weight
  deltanet layers  : linear_attn.{in_proj_qkvz,in_proj_ba,out_proj}.weight,
                     linear_attn.conv1d.weight (+bias), linear_attn.A_log,
                     linear_attn.dt_bias, linear_attn.norm.weight
  MoE (every layer): mlp.experts.{j}.{gate,up,down}_proj.weight, mlp.gate.weight
                     (softmax router), mlp.shared_expert.{gate,up,down}_proj.weight,
                     mlp.shared_expert_gate.weight
  MTP              : layer n_layers (--mtp)

Usage:
  python3 tools/convert_qwen36.py --indir qwen36_tiny --outdir qwen36_tiny_i4 --ebits 4
  # real model: hf download Qwen/Qwen3.6-35B-A3B --local-dir qwen36 && \
  #   python3 tools/convert_qwen36.py --indir qwen36 --outdir /path/qwen36_i4 --ebits 4
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import convert_hy3 as base  # noqa: E402  reuse convert_shard/convert_local/dequant/quant
from convert_fp8_to_int4 import layer_idx  # noqa: E402

# Qwen uses mlp.shared_expert.* (singular); the Colibri loader convention is
# mlp.shared_experts.* (plural), same as Hy3/GLM.
SHARED_RE = re.compile(r"\.mlp\.shared_expert\.")


def rename_out(name):
    return SHARED_RE.sub(".mlp.shared_experts.", name)


def classify(name, n_layers, keep_mtp=False):
    if name.endswith("_scale_inv") or name.endswith("_scale"):
        return "consumed"
    li = layer_idx(name)
    if keep_mtp:
        if li != n_layers:
            return "skip"
    else:
        if li >= n_layers:
            return "skip"
    # keep f32: all norms, softmax router gate, shared-expert gate, and the small
    # DeltaNet decay/conv params (per-row int quant would wreck the recurrence).
    if (name.endswith("norm.weight") or name == "model.norm.weight"
            or name.endswith("q_norm.weight") or name.endswith("k_norm.weight")
            or name.endswith("mlp.gate.weight")
            or name.endswith("shared_expert_gate.weight")
            or name.endswith(".A_log") or name.endswith(".dt_bias")
            or name.endswith("conv1d.weight") or name.endswith("conv1d.bias")):
        return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "io"
    if ".mlp.experts." in name and name.endswith(".weight"):
        return "x"
    if name.endswith(".weight"):
        return "q"  # q/k/v/o_proj, in_proj_qkvz/ba, out_proj, shared_expert.*_proj
    return "f32"


def main():
    ap = argparse.ArgumentParser(description="Qwen3.6 -> Colibri int4 container")
    ap.add_argument("--repo", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=False)
    ap.add_argument("--ebits", type=int, default=4)
    ap.add_argument("--io-bits", type=int, default=8)
    ap.add_argument("--xbits", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--mtp", action="store_true")
    a = ap.parse_args()
    if a.xbits is None:
        a.xbits = a.ebits

    # Graft the Qwen classify/rename onto the shared shard machinery. convert_shard
    # (defined in convert_hy3) looks these names up at call time, so reassigning the
    # module globals is enough; dequant() already handles bf16 -> f32.
    base.classify = classify
    base.rename_out = rename_out

    if a.indir:
        if not a.outdir:
            sys.exit("--outdir required with --indir")
        base.convert_local(a.indir, a.outdir, a.n_layers, a.ebits, a.io_bits, a.xbits)
        return

    # --repo streaming path: convert_hy3.main delegates to the FP8 converter's remote
    # loop with classify/dequant patched. For the 35B bring-up (M7) the simplest route
    # is `hf download ... --local-dir qwen36` then --indir, so that's the supported path
    # for now; wire the remote loop here once the tensor names are locked.
    sys.exit("for now: hf download the repo to a dir, then rerun with --indir <dir>")


if __name__ == "__main__":
    main()
