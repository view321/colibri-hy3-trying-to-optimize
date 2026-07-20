#!/usr/bin/env python3
"""Checkpoint surgery on a Colibri Hy3 int4/int8 container (see docs/SURGERY.md).

Three independent, composable operations on an ALREADY-CONVERTED container
(the output of convert_hy3.py — out-*.safetensors with .qs per-row scales):

  1. PRUNE      --keep N | --keep-frac F
     Drop the least-salient routed experts per layer (uniform count across
     layers), slice the router gate/bias rows to match, renumber the kept
     experts 0..N-1, and remap .coli_usage/.coli_saliency to the new ids.
     Criterion (--criterion): saliency (engine-collected gate mass, best),
     usage (selection counts), wnorm (weight Frobenius norm, computable
     offline from the container alone), auto = saliency -> usage -> wnorm
     per layer.

  2. MIXED PRECISION   --cold-frac F
     Re-encode the coldest F fraction of the KEPT experts (by the same
     criterion) from int4 to int2 (engine fmt 3 — kernels already exist on
     CPU and CUDA; the loader infers the format per tensor from its byte
     size, so mixed containers need no engine flag). Halves the disk bytes
     of exactly the experts that cause cache misses. Per-row int2 is
     aggressive: keep the fraction moderate and A/B with SCORE mode.

  3. REORDER    --reorder freq|coact|none
     Rewrite each layer's experts physically contiguous and ordered hot-first
     (freq) or by co-activation clustering (coact, needs a ROUTE_TRACE file).
     Ids do NOT change unless pruning also runs: without pruning this is a
     pure placement change and the output is bit-identical to the source.
     Each expert's gate/up/down weights are written back-to-back, which also
     re-enables the engine's single-pread contiguous fast path.

The output container is written shard-per-layer:
  out-00000.safetensors           dense stack (embed, head, attn, norms,
                                  routers, shared experts, dense layer MLP)
  out-{L+1:05d}.safetensors       layer L's routed experts (weights first,
                                  in placement order, then all .qs scales)
  out-mtp-00000.safetensors       MTP layer (when present and not --mtp drop)

Only numpy is required (safetensors files are parsed/written directly, so
tensor placement inside the shard is exactly what this tool decides).

Typical use (on the machine that holds the container):

  # plan only — prints per-layer decisions and projected sizes
  python3 tools/surgery_hy3.py --model /data/hy3_i4 --out /data/hy3_s \
      --keep-frac 0.75 --cold-frac 0.5 --reorder coact --dry-run

  # execute (resumable: finished shards are skipped on re-run)
  python3 tools/surgery_hy3.py --model /data/hy3_i4 --out /data/hy3_s \
      --keep-frac 0.75 --cold-frac 0.5 --reorder coact
"""
import argparse
import glob
import json
import os
import re
import shutil
import struct
import sys

import numpy as np

EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight(\.qs)?$")
PROJS = ("gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------- container IO
class Shards:
    """Index over the safetensors shards of a container. Manual parse: 8-byte
    little-endian header length + JSON header + raw data (the same layout st.h
    reads). get() returns raw bytes as uint8 or decoded f32."""

    def __init__(self, indir):
        self.indir = indir
        self.t = {}     # name -> (path, dtype, shape, abs_off, nbytes)
        self._fh = {}   # path -> open handle (reads are hot: cache them)
        files = sorted(glob.glob(os.path.join(indir, "*.safetensors")))
        if not files:
            sys.exit(f"no *.safetensors in {indir}")
        for path in files:
            with open(path, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(hlen))
                base = 8 + hlen
                for name, meta in hdr.items():
                    if name == "__metadata__":
                        continue
                    a, b = meta["data_offsets"]
                    self.t[name] = (path, meta["dtype"], tuple(meta["shape"]),
                                    base + a, b - a)

    def has(self, name):
        return name in self.t

    def nbytes(self, name):
        return self.t[name][4]

    def _read(self, path, off, nb):
        f = self._fh.get(path)
        if f is None:
            f = self._fh[path] = open(path, "rb")
        f.seek(off)
        return f.read(nb)

    def raw(self, name):
        path, _, _, off, nb = self.t[name]
        return np.frombuffer(self._read(path, off, nb), dtype=np.uint8)

    def f32(self, name):
        path, dt, shape, off, nb = self.t[name]
        if dt != "F32":
            sys.exit(f"{name}: expected F32, found {dt}")
        return np.frombuffer(self._read(path, off, nb), dtype=np.float32).reshape(shape).copy()


DTYPE_SIZE = {"F32": 4, "U8": 1, "I8": 1}


def write_shard(path, entries, src, force=False):
    """entries: list of dicts {name, dtype, shape, src:("copy",src_name)|("mem",arr)}.
    Data is laid out exactly in entry order (this is the whole point: expert
    placement on disk is the reorder feature). Returns total bytes written."""
    if os.path.exists(path) and not force:
        return os.path.getsize(path)
    hdr, off = {}, 0
    for e in entries:
        kind, ref = e["src"]
        nb = src.nbytes(ref) if kind == "copy" else ref.nbytes
        e["_nb"] = nb
        hdr[e["name"]] = {"dtype": e["dtype"], "shape": list(e["shape"]),
                          "data_offsets": [off, off + nb]}
        off += nb
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * (-(8 + len(hj)) % 8)          # pad so data starts 8-aligned
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for e in entries:
            kind, ref = e["src"]
            buf = src.raw(ref) if kind == "copy" else np.ascontiguousarray(ref)
            if buf.nbytes != e["_nb"]:
                sys.exit(f"{e['name']}: size changed during write")
            f.write(buf.tobytes())
    os.replace(tmp, path)
    return 8 + len(hj) + off


# ------------------------------------------------------------- quant / requant
# Exact mirrors of c/tools/convert_fp8_to_int4.py (per-row symmetric absmax)
# and of hy3.c's unpack conventions: int4 = offset +8, low nibble first;
# int2 = offset +2, 4 values/byte, element i at bits (i&3)*2.
def infer_fmt(nbytes, O, I):
    if nbytes == O * I:
        return 8
    if nbytes == O * ((I + 1) // 2):
        return 4
    if nbytes == O * ((I + 3) // 4):
        return 2
    sys.exit(f"cannot infer packing: {nbytes} bytes for [{O},{I}]")


def unpack_codes(q, O, I, fmt):
    """packed U8 -> signed integer codes [O,I] (no scale applied)."""
    if fmt == 8:
        return q.view(np.int8).reshape(O, I).astype(np.int16)
    if fmt == 4:
        b = q.reshape(O, (I + 1) // 2)
        codes = np.empty((O, 2 * b.shape[1]), np.int16)
        codes[:, 0::2] = (b & 0xF).astype(np.int16) - 8
        codes[:, 1::2] = (b >> 4).astype(np.int16) - 8
        return codes[:, :I]
    b = q.reshape(O, (I + 3) // 4)
    codes = np.empty((O, 4 * b.shape[1]), np.int16)
    for k in range(4):
        codes[:, k::4] = ((b >> (2 * k)) & 3).astype(np.int16) - 2
    return codes[:, :I]


def quant_int2(w):
    O, I = w.shape
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / 1.0, 1e-8)
    q = np.clip(np.rint(w / s), -2, 1).astype(np.int32)
    rb = (I + 3) // 4
    out = np.zeros((O, rb), np.uint8)
    for k in range(4):
        vk = q[:, k::4]
        out[:, :vk.shape[1]] |= ((vk + 2).astype(np.uint8) << (k * 2))
    return out.reshape(-1), s[:, 0].astype(np.float32)


def quant_int4(w):
    O, I = w.shape
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / 7.0, 1e-8)
    q = np.clip(np.rint(w / s), -8, 7).astype(np.int32)
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, 0].astype(np.float32)


def requant(sh, wname, O, I, target_bits):
    """dequant a stored expert matrix and re-encode at target_bits.
    Second quantization on top of the stored grid: fine for COLD experts,
    which is the only place surgery applies it."""
    q = sh.raw(wname)
    s = sh.f32(wname + ".qs").reshape(-1)
    fmt = infer_fmt(q.nbytes, O, I)
    w = unpack_codes(q, O, I, fmt).astype(np.float32) * s[:, None]
    return quant_int2(w) if target_bits == 2 else quant_int4(w)


# ----------------------------------------------------------------- criterion
def load_stats(path):
    st = {}
    if not os.path.exists(path):
        return st
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) != 3:
                continue
            try:
                l, e, v = int(p[0]), int(p[1]), float(p[2])
            except ValueError:
                continue
            st[(l, e)] = st.get((l, e), 0.0) + v
    return st


def expert_names(l, e):
    base = f"model.layers.{l}.mlp.experts.{e}."
    return [base + p + ".weight" for p in PROJS]


def wnorm_layer(sh, cfg, l):
    """Frobenius norm of each expert's three matrices, straight from the
    container codes+scales — the always-available offline criterion."""
    E, I, D = cfg["num_experts"], cfg["moe_intermediate_size"], cfg["hidden_size"]
    dims = [(I, D), (I, D), (D, I)]
    out = np.zeros(E)
    for e in range(E):
        acc = 0.0
        for wn, (O, II) in zip(expert_names(l, e), dims):
            q = sh.raw(wn)
            s = sh.f32(wn + ".qs").reshape(-1).astype(np.float64)
            codes = unpack_codes(q, O, II, infer_fmt(q.nbytes, O, II)).astype(np.int64)
            acc += float(((codes * codes).sum(axis=1) * s * s).sum())
        out[e] = np.sqrt(acc)
    return out


def layer_scores(sh, cfg, l, criterion, sal, usage, warned):
    E = cfg["num_experts"]

    def from_stats(st):
        v = np.array([st.get((l, e), 0.0) for e in range(E)])
        return v if v.sum() > 0 else None

    if criterion in ("auto", "saliency"):
        v = from_stats(sal)
        if v is not None:
            return v, "saliency"
        if criterion == "saliency":
            sys.exit(f"layer {l}: no .coli_saliency data (run calibration first, "
                     f"or use --criterion auto)")
    if criterion in ("auto", "usage"):
        v = from_stats(usage)
        if v is not None:
            return v, "usage"
        if criterion == "usage":
            sys.exit(f"layer {l}: no .coli_usage data")
    if criterion == "auto" and l not in warned:
        warned.add(l)
        print(f"  layer {l}: no calibration data — falling back to weight-norm")
    return wnorm_layer(sh, cfg, l), "wnorm"


# ------------------------------------------------------------------- reorder
def load_trace(path, E):
    """ROUTE_TRACE lines: '<layer> <e1> ... <ek>'. Same-line experts were
    routed together for one token — that is the co-activation signal."""
    co = {}
    if not path or not os.path.exists(path):
        return co
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) < 2:
                continue
            try:
                l = int(p[0]); es = [int(x) for x in p[1:]]
            except ValueError:
                continue
            if any(e < 0 or e >= E for e in es):
                continue
            C = co.get(l)
            if C is None:
                C = co[l] = np.zeros((E, E), np.float64)
            for i in range(len(es)):
                for j in range(i + 1, len(es)):
                    C[es[i], es[j]] += 1.0
                    C[es[j], es[i]] += 1.0
    return co


def order_coact(kept, C, score, window=4):
    """Greedy chain: start from the hottest kept expert, repeatedly append the
    kept expert with the strongest co-activation to the last `window` placed
    (ties broken by score). Co-activated experts end up disk-adjacent, so one
    token's fetches for a layer coalesce and readahead warms likely partners."""
    kept = list(kept)
    placed = [max(kept, key=lambda e: (score[e], -e))]
    rest = set(kept) - {placed[0]}
    while rest:
        tail = placed[-window:]
        best = max(rest, key=lambda e: (sum(C[t, e] for t in tail), score[e], -e))
        placed.append(best)
        rest.remove(best)
    return placed


# ----------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description="Colibri Hy3 container surgery")
    ap.add_argument("--model", required=True, help="source container dir")
    ap.add_argument("--out", required=True, help="output container dir")
    ap.add_argument("--keep", type=int, default=0, help="experts kept per layer")
    ap.add_argument("--keep-frac", type=float, default=0.0,
                    help="fraction of experts kept per layer (alternative to --keep)")
    ap.add_argument("--criterion", choices=["auto", "saliency", "usage", "wnorm"],
                    default="auto")
    ap.add_argument("--cold-frac", type=float, default=0.0,
                    help="coldest fraction of KEPT experts re-encoded at int2")
    ap.add_argument("--reorder", choices=["freq", "coact", "none"], default="freq")
    ap.add_argument("--trace", default=None,
                    help="ROUTE_TRACE file for --reorder coact "
                         "(default: <model>/.coli_route_trace)")
    ap.add_argument("--mtp", choices=["prune", "keep", "drop"], default=None,
                    help="MTP layer: prune like the main layers (default when "
                         "pruning), keep untouched (only valid without pruning), "
                         "or drop the MTP head entirely")
    ap.add_argument("--mtp-ebits", type=int, choices=[0, 4], default=0,
                    help="4: re-encode MTP experts int8->int4 (drafts only — "
                         "verified by the main model, so quality risk is bounded "
                         "to acceptance rate)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rewrite output shards even if they already exist")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    sh = Shards(a.model)
    E, NL = cfg["num_experts"], cfg["num_hidden_layers"]
    topk = cfg["num_experts_per_tok"]
    first_dense = cfg.get("first_k_dense_replace", 0)
    D, I = cfg["hidden_size"], cfg["moe_intermediate_size"]
    sparse_layers = [l for l in range(first_dense, NL)
                     if sh.has(f"model.layers.{l}.mlp.experts.0.gate_proj.weight")]
    has_mtp = sh.has(f"model.layers.{NL}.mlp.experts.0.gate_proj.weight")

    keepN = a.keep or (round(a.keep_frac * E) if a.keep_frac else 0) or E
    if keepN > E:
        sys.exit(f"--keep {keepN} > num_experts {E}")
    if keepN < topk:
        sys.exit(f"--keep {keepN} < num_experts_per_tok {topk}: the router "
                 f"cannot select top-{topk} of {keepN}")
    if keepN < 2 * topk:
        print(f"WARNING: keep={keepN} < 2*topk={2*topk} — routing diversity "
              f"collapses; expect real quality loss")
    pruning = keepN < E
    mtp_mode = a.mtp or ("prune" if pruning else "keep")
    if has_mtp and pruning and mtp_mode == "keep":
        sys.exit("--mtp keep is invalid when pruning: the MTP layer must have "
                 "the same num_experts as config.json (use prune or drop)")
    if not 0.0 <= a.cold_frac <= 1.0:
        sys.exit("--cold-frac must be in [0,1]")

    sal = load_stats(os.path.join(a.model, ".coli_saliency"))
    usage = load_stats(os.path.join(a.model, ".coli_usage"))
    trace_path = a.trace or os.path.join(a.model, ".coli_route_trace")
    co = load_trace(trace_path if a.reorder == "coact" else None, E)
    if a.reorder == "coact" and not co:
        print(f"NOTE: no usable trace at {trace_path} — --reorder coact falls "
              f"back to freq per layer")

    # -------- plan
    surgery_layers = list(sparse_layers)
    if has_mtp and mtp_mode == "prune":
        surgery_layers.append(NL)
    plan, warned = {}, set()
    tot_before = tot_after = 0.0
    exp_bytes_tok_before = exp_bytes_tok_after = 0.0
    for l in surgery_layers:
        score, crit = layer_scores(sh, cfg, l, a.criterion, sal, usage, warned)
        order = np.lexsort((np.arange(E), -score))         # score desc, id asc
        kept = sorted(order[:keepN].tolist())
        if a.reorder == "freq" or (a.reorder == "coact" and l not in co):
            placed = sorted(kept, key=lambda e: (-score[e], e))
        elif a.reorder == "coact":
            placed = order_coact(kept, co[l], score)
        else:
            placed = kept                                   # original id order
        ncold = int(a.cold_frac * keepN)
        cold = set(sorted(kept, key=lambda e: (score[e], -e))[:ncold])
        if l == NL:
            cold = set()          # MTP experts: flat --mtp-ebits requant only, never int2 zones
        # per-expert byte sizes (before / after)
        sz_b, sz_a = {}, {}
        for e in range(E):
            nb = sum(sh.nbytes(w) + sh.nbytes(w + ".qs") for w in expert_names(l, e))
            sz_b[e] = nb
            if e in cold:
                sz_a[e] = (I * ((D + 3) // 4) * 2 + D * ((I + 3) // 4)) + (2 * I + D) * 4
            elif l == NL and a.mtp_ebits == 4:
                sz_a[e] = (I * ((D + 1) // 2) * 2 + D * ((I + 1) // 2)) + (2 * I + D) * 4
            else:
                sz_a[e] = nb
        tot_before += sum(sz_b.values())
        tot_after += sum(sz_a[e] for e in kept)
        if l != NL:
            p = score / max(score.sum(), 1e-12)
            exp_bytes_tok_before += topk * float(sum(p[e] * sz_b[e] for e in range(E)))
            pk = np.array([score[e] for e in kept]); pk = pk / max(pk.sum(), 1e-12)
            exp_bytes_tok_after += topk * float(sum(pk[i] * sz_a[e] for i, e in enumerate(kept)))
        plan[l] = {"criterion": crit, "kept": placed, "cold": sorted(cold),
                   "score": [float(score[e]) for e in placed]}
        tag = " [MTP]" if l == NL else ""
        print(f"  layer {l:3d}{tag}: keep {keepN}/{E} by {crit}"
              f"{f' | {len(cold)} coldest -> int2' if cold else ''}"
              f" | order {a.reorder}")

    print(f"\nexperts on disk: {tot_before/1e9:.2f} GB -> {tot_after/1e9:.2f} GB"
          f"  ({100*(1-tot_after/max(tot_before,1)):.0f}% smaller)")
    if exp_bytes_tok_before:
        print(f"projected cold expert reads/token (criterion-weighted): "
              f"{exp_bytes_tok_before/1e9:.2f} GB -> {exp_bytes_tok_after/1e9:.2f} GB")
    if a.dry_run:
        print("\n--dry-run: nothing written")
        return

    os.makedirs(a.out, exist_ok=True)

    def remap_for(l):
        """old id -> new id (identity when not pruning: ids only change when
        the expert space is compacted)."""
        if not pruning:
            return {e: e for e in plan[l]["kept"]}
        return {e: i for i, e in enumerate(plan[l]["kept"])}

    def layer_entries(l):
        rm = remap_for(l)
        weights, scales = [], []
        for e_old in plan[l]["kept"]:
            e_new = rm[e_old]
            dims = [(I, D), (I, D), (D, I)]
            for wn_old, p, (O, II) in zip(expert_names(l, e_old), PROJS, dims):
                wn_new = f"model.layers.{l}.mlp.experts.{e_new}.{p}.weight"
                to_bits = 2 if e_old in plan[l]["cold"] else \
                          (4 if (l == NL and a.mtp_ebits == 4) else 0)
                stored = infer_fmt(sh.nbytes(wn_old), O, II)
                if to_bits and stored != to_bits and stored > to_bits:
                    q, s = requant(sh, wn_old, O, II, to_bits)
                    weights.append({"name": wn_new, "dtype": "U8",
                                    "shape": [q.size], "src": ("mem", q)})
                    scales.append({"name": wn_new + ".qs", "dtype": "F32",
                                   "shape": [s.size], "src": ("mem", s)})
                else:
                    weights.append({"name": wn_new, "dtype": "U8",
                                    "shape": [sh.nbytes(wn_old)], "src": ("copy", wn_old)})
                    scales.append({"name": wn_new + ".qs", "dtype": "F32",
                                   "shape": [sh.nbytes(wn_old + ".qs") // 4],
                                   "src": ("copy", wn_old + ".qs")})
        return weights + scales      # all weights first: dense readahead region

    def router_sliced(l, name, arr2d):
        kept = plan[l]["kept"]
        if arr2d.shape[0] != E:
            sys.exit(f"{name}: expected {E} router rows, found {arr2d.shape[0]}")
        return arr2d[np.array(kept)]

    # -------- dense shard (everything that is not a routed expert)
    dense_entries = []
    for name in sh.t:
        mm = EXPERT_RE.match(name)
        if mm and (int(mm.group(1)) in plan or int(mm.group(1)) == NL):
            continue        # routed experts: rewritten by the layer/MTP shards (or dropped)
        lm = re.match(r"^model\.layers\.(\d+)\.", name)
        l = int(lm.group(1)) if lm else -1
        if l == NL and mtp_mode == "drop":
            continue
        if pruning and l in plan and (
                name.endswith("mlp.gate.weight") or name.endswith("mlp.router.gate.weight")):
            arr = router_sliced(l, name, sh.f32(name))
            ent = {"name": name, "dtype": "F32", "shape": list(arr.shape),
                   "src": ("mem", arr)}
        elif pruning and l in plan and (
                name.endswith("mlp.expert_bias") or name.endswith("e_score_correction_bias")):
            arr = sh.f32(name).reshape(-1)[np.array(plan[l]["kept"])]
            ent = {"name": name, "dtype": "F32", "shape": [arr.size], "src": ("mem", arr)}
        else:
            path, dt, shape, _, _ = sh.t[name]
            ent = {"name": name, "dtype": dt, "shape": list(shape), "src": ("copy", name)}
        if l == NL:
            ent["_mtp"] = True
        dense_entries.append(ent)
    mtp_extra = [e for e in dense_entries if e.get("_mtp")]
    dense_entries = [e for e in dense_entries if not e.get("_mtp")]

    written = 0
    written += write_shard(os.path.join(a.out, "out-00000.safetensors"),
                           dense_entries, sh, a.force)
    print(f"out-00000.safetensors (dense stack)  {written/1e9:.2f} GB")
    for l in sparse_layers:
        p = os.path.join(a.out, f"out-{l+1:05d}.safetensors")
        skip = os.path.exists(p) and not a.force
        nb = write_shard(p, layer_entries(l), sh, a.force)
        written += nb
        print(f"out-{l+1:05d}.safetensors (layer {l}) {nb/1e9:6.2f} GB"
              f"{' [existing, skipped]' if skip else ''}")
    if has_mtp and mtp_mode != "drop":
        ents = mtp_extra + (layer_entries(NL) if NL in plan else [])
        if NL not in plan:      # mtp keep (no pruning): experts copied verbatim
            for e_old in range(E):
                for wn, (O, II) in zip(expert_names(NL, e_old), [(I, D), (I, D), (D, I)]):
                    ents.append({"name": wn, "dtype": "U8",
                                 "shape": [sh.nbytes(wn)], "src": ("copy", wn)})
                    ents.append({"name": wn + ".qs", "dtype": "F32",
                                 "shape": [sh.nbytes(wn + ".qs") // 4],
                                 "src": ("copy", wn + ".qs")})
        nb = write_shard(os.path.join(a.out, "out-mtp-00000.safetensors"), ents, sh, a.force)
        written += nb
        print(f"out-mtp-00000.safetensors            {nb/1e9:6.2f} GB")
    elif has_mtp:
        print("MTP head dropped (--mtp drop): engine will run without native drafts")

    # -------- config + sidecars
    cfg_out = dict(cfg)
    if pruning:
        cfg_out["num_experts"] = keepN
    json.dump(cfg_out, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
               "chat_template.jinja"):
        s = os.path.join(a.model, fn)
        if os.path.exists(s):
            shutil.copy(s, a.out)

    def remap_stats(src_name, fmt):
        src_p = os.path.join(a.model, src_name)
        if not os.path.exists(src_p):
            return
        out_lines = []
        with open(src_p) as f:
            for ln in f:
                p = ln.split()
                if len(p) != 3:
                    continue
                try:
                    l, e = int(p[0]), int(p[1])
                except ValueError:
                    continue
                if l in plan:
                    rm = remap_for(l)
                    if e in rm:
                        out_lines.append(f"{l} {rm[e]} {p[2]}\n")
                elif l == NL and (not has_mtp or mtp_mode == "drop"):
                    continue
                elif e < cfg_out["num_experts"]:
                    out_lines.append(ln if ln.endswith("\n") else ln + "\n")
        with open(os.path.join(a.out, src_name), "w") as f:
            f.writelines(out_lines)

    remap_stats(".coli_usage", "u")
    remap_stats(".coli_saliency", "s")

    prov = {"tool": "surgery_hy3.py", "source": os.path.abspath(a.model),
            "args": {k: v for k, v in vars(a).items()},
            "num_experts": {"before": E, "after": cfg_out["num_experts"]},
            "layers": {str(l): plan[l] for l in plan},
            "bytes_written": written}
    json.dump(prov, open(os.path.join(a.out, "surgery.json"), "w"), indent=1)
    print(f"\ndone: {written/1e9:.2f} GB -> {a.out}")
    print("next: point SNAP/COLI_MODEL at the new container; "
          "usage/saliency history was remapped and carries over")


if __name__ == "__main__":
    main()
