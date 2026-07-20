#!/usr/bin/env bash
# surgery_apply.sh — build surgery variants of a Hy3 container (docs/SURGERY.md).
#
# Variants are NAME:ARGS pairs; the defaults are the recommended ladder:
#   p75        keep 75% of experts               (safest, ~25% smaller)
#   p75c50     + coldest half of the kept at int2 (the balanced pick)
#   p50c50     keep 50% + cold int2               (aggressive: measure quality!)
# All use --reorder coact when a route trace exists (surgery falls back to
# freq per layer automatically when it does not).
#
# The SOURCE CONTAINER IS NEVER MODIFIED — every variant is a full standalone
# copy, so budget disk: each variant costs its own printed size (source 142 GB
# -> ~75-107 GB per variant). Free space is checked before each build.
#
# Usage on RunPod:
#   SRC=/workspace/hy3_i4 DST_BASE=/workspace ./scripts/surgery_apply.sh
# Env: SRC (required) | DST_BASE (required) | VARIANTS override, e.g.
#   VARIANTS="mine:--keep 160 --cold-frac 0.3 --reorder freq"
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${SRC:?SRC=<source container> required}"
DST_BASE="${DST_BASE:?DST_BASE=<output parent dir> required}"
PY="${PY:-python3}"

DEFAULT_VARIANTS=(
  "p75:--keep-frac 0.75 --reorder coact"
  "p75c50:--keep-frac 0.75 --cold-frac 0.5 --reorder coact"
  "p50c50:--keep-frac 0.50 --cold-frac 0.5 --reorder coact"
)
if [ -n "${VARIANTS:-}" ]; then
  IFS='|' read -r -a DEFAULT_VARIANTS <<< "$VARIANTS"
fi

src_gb=$(du -s --block-size=1G "$SRC" 2>/dev/null | cut -f1)
echo "source: $SRC (${src_gb:-?} GB)"
[ -f "$SRC/.coli_saliency" ] || \
  echo "NOTE: no .coli_saliency — run scripts/surgery_calibrate.sh first for the best pruning criterion (falls back to usage/weight-norm)"

for v in "${DEFAULT_VARIANTS[@]}"; do
  name="${v%%:*}"; args="${v#*:}"
  dst="$DST_BASE/$(basename "$SRC")_$name"
  echo
  echo "=== variant $name: $args"
  echo "    -> $dst"
  free_gb=$(df --output=avail --block-size=1G "$DST_BASE" | tail -1 | tr -d ' ')
  if [ "${free_gb:-0}" -lt "${src_gb:-0}" ]; then
    echo "    SKIP: only ${free_gb} GB free on $DST_BASE (need up to ~${src_gb} GB)"
    continue
  fi
  $PY tools/surgery_hy3.py --model "$SRC" --out "$dst" $args --dry-run
  $PY tools/surgery_hy3.py --model "$SRC" --out "$dst" $args
done

echo
echo "all variants done. benchmark with:"
echo "  BASELINE=$SRC CONTAINERS=\"$DST_BASE/$(basename "$SRC")_p75 ...\" ./scripts/surgery_bench.sh"
