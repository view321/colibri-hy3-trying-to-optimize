# Checkpoint surgery — making the model fit the engine

The engine streams routed experts from disk, so warm-decode speed is governed by
one line:

> **disk bytes/token = MoE layers × experts fetched × expert size × miss rate**

For stock Hy3 that is 79 × 8 × ~9 MB ≈ **5.7 GB/token cold**. Everything here
attacks one of those four factors *on the model side* — the container itself is
rewritten so there is less to read and what remains caches better. The source
container is never modified; every variant is a standalone copy you can A/B and
delete.

| Lever | Tool / flag | What it does | Speed effect | Quality risk |
|---|---|---|---|---|
| **Prune** | `surgery_hy3.py --keep-frac F` | Drop the least-salient experts per layer, slice the router, renumber | Container 142 → ~107 GB (F=.75) / ~75 GB (F=.50): far better cache hit rate; at F=.50 the model nearly fits a 62 GB box's RAM | ~25% usually mild, 50% real — **measure** |
| **Mixed precision** | `--cold-frac F` | Re-encode the coldest F of kept experts int4 → int2 | Misses are by definition cold ⇒ miss bytes drop up to ~F/2; footprint shrinks too | Per-row int2 is aggressive; keep F ≤ 0.5 and gate on SCORE |
| **Reorder** | `--reorder coact\|freq` | Physically place co-activated / hot experts adjacent; gate/up/down of each expert contiguous | Re-enables the single-`pread` fast path (6 reads → 4 per expert); page-cache readahead warms likely co-routed neighbors; autopin's startup reads turn sequential | **None** — placement only, output bit-identical |
| **Adaptive k** | env `GATE_TAU=0.2..0.4` | At run time, drop routed experts whose gate < τ × strongest, renormalize | Directly cuts experts fetched/token (readable in the `experts loaded/token` line) | Small at low τ, grows with τ; **not** bit-identical, off by default |

The levers compose: a pruned + cold-int2 + reordered container run with a small
`GATE_TAU` stacks all four.

## Where the signals come from

Pruning and zoning need to know which experts matter **for your workload**:

- **`.coli_usage`** — selection counts (the engine has always written this).
- **`.coli_saliency`** — NEW: the sum of *applied post-norm gate weights* per
  (layer, expert). This is the default pruning criterion — frequency cannot
  tell a dominant expert from a perennial 8th place, gate mass can
  (REAP-flavored: the router-weight part of `E[g·‖f(x)‖]`, without the
  activation-norm factor). Written automatically on every run (`SALIENCY=0`
  disables).
- **`ROUTE_TRACE=<file>`** — NEW, opt-in: one line per (token, layer) with the
  routed expert ids. Only needed for `--reorder coact`; grows a few KB/token,
  delete it after surgery.
- **weight norms** — always available fallback computed from the container
  itself (`--criterion wnorm`); used automatically for layers with no
  calibration data (including the MTP layer on old usage files).

Both stats files accumulate across sessions and are **remapped and carried
over** into surgery output, so autopin starts warm on the new container.

## End-to-end on RunPod

```bash
cd c && make hy3            # (CUDA=1 / IOURING=1 as usual)

# 1. calibrate — ~24 diverse prompts with the observers on (also warms usage)
SNAP=/workspace/hy3_i4 ./scripts/surgery_calibrate.sh

# 2. plan — prints per-layer decisions, projected sizes and GB/token, writes nothing
python3 tools/surgery_hy3.py --model /workspace/hy3_i4 --out /workspace/hy3_p75c50 \
    --keep-frac 0.75 --cold-frac 0.5 --reorder coact --dry-run

# 3. apply — the default ladder (p75, p75c50, p50c50); resumable, source untouched
SRC=/workspace/hy3_i4 DST_BASE=/workspace ./scripts/surgery_apply.sh

# 4. quality gate material (once): teacher-forced windows from any plain text
pip install tokenizers
python3 tools/make_score_file.py --model /workspace/hy3_i4 \
    --text corpus.txt --out score_req.txt --ctx 512 --cont 128 --n 32

# 5. benchmark speed + quality, baseline vs variants, each also with GATE_TAU
BASELINE=/workspace/hy3_i4 \
CONTAINERS="/workspace/hy3_i4_p75 /workspace/hy3_i4_p75c50 /workspace/hy3_i4_p50c50" \
SCORE_FILE=score_req.txt EXTRA="RAM_GB=48 PIPE=2" ./scripts/surgery_bench.sh
```

Adopting a variant is just `COLI_MODEL=/workspace/hy3_i4_p75c50` (or `SNAP=`).
Rolling back is deleting the directory.

### Reading the bench output

- **decode tok/s** and `experts loaded/token` — the speed story.
- **SCORE `mean logprob/token`** — the quality story. Same windows, same
  harness, same machine as the baseline run, so the delta *is* the surgery
  cost (the same protocol argument as `tools/quant_ablation.py`). There is no
  universal threshold, but as a rule of thumb a drop under ~0.02 nats/token is
  hard to feel in chat; a drop of 0.1+ is a different model.
- `GATE_TAU:` line — average experts actually kept per layer at that τ.

### Recommended ladder

1. **`p75` first** (keep 75%, reorder): the conservative baseline of the
   surgery world — meaningful footprint cut, quality usually barely moves.
2. **`p75c50`** — the balanced pick; misses now read mostly int2 bytes.
3. **`p50c50`** only if SCORE stays acceptable — at 50% keep the 142 GB
   container drops to ~75 GB, which on a 62 GB "comfortable" box means the
   model *nearly fits in RAM* and warm decode can jump several ×.
4. Sweep `GATE_TAU` 0.2 → 0.4 on whichever container won.

## Sharp edges (read before trusting a variant)

- **Calibration is workload-specific.** Saliency collected on English chat will
  prune experts a Chinese-code workload needed. Calibrate on what you actually
  run, or keep more experts.
- **int2 is per-row symmetric** — cheap but blunt (the repo's own ablation data
  shows per-row int4 already costs real points on hard tasks). That is why only
  *cold* experts get it and why the bench script insists on SCORE.
- **Requant is int4 → int2** (double rounding). Re-encoding from the FP8
  source would be slightly better; not implemented — the int4 container is
  what's on disk and on HF.
- **`--keep` below 2×topk** collapses routing diversity; the tool warns and you
  should listen.
- **MTP**: pruned with the same keep-count by default (`--mtp drop` removes the
  draft head entirely; `--mtp-ebits 4` halves MTP expert reads at some
  acceptance-rate cost — drafts are verified by the main model, so this can
  only cost speed, not correctness).
- **CUDA**: the backend has int2 kernels (upload, matmul, grouped-expert), so a
  VRAM tier over a mixed container is expected to work — but it has only been
  gate-tested on CPU; bench with your GPU before adopting. The JIT-stream path
  (`CUDA_STREAM_EXPERTS=1`) intentionally skips non-int4 experts.
- **Route traces go stale after pruning** (ids are renumbered): recollect on
  the new container if you want to re-run `--reorder coact` later.
- **`GATE_TAU` breaks bit-exactness by design** — it is a quality/speed dial,
  not a free optimization like PREDICT. Keep it 0 for oracle/CI runs.
- **Disk budgeting**: each variant is a full copy (source 142 GB → variant
  75–110 GB). `surgery_apply.sh` checks free space before each build; builds
  are resumable (finished shards are skipped).

## Output container anatomy

```
out-00000.safetensors        dense stack: embed, lm_head, norms, attention,
                             routers (+bias, sliced when pruned), shared experts
out-000{L+1}.safetensors     layer L's experts, placement-ordered; per expert
                             gate/up/down weights back-to-back (single-pread
                             fast path), all .qs scales bundled after the weights
out-mtp-00000.safetensors    MTP layer (absent with --mtp drop)
config.json                  num_experts patched when pruned
.coli_usage / .coli_saliency remapped to the new expert ids
surgery.json                 full provenance: args, per-layer keep lists in
                             placement order, criterion used, cold sets
```

The engine needs **no flag** to run a surgery container: expert count comes
from `config.json`, and the loader infers each tensor's packing (int8/int4/int2)
from its byte size — mixed-precision containers are self-describing.
