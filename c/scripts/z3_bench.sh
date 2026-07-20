#!/usr/bin/env bash
# A/B benchmark matrix for GCP Z3 (or any little-RAM + fast-NVMe box).
# Usage: ./scripts/z3_bench.sh /mnt/lssd/hy3_i4 [NGEN]
#
# Two cache strategies are compared, each with prediction on/off, plus an MTP
# draft A/B (speculative decode multiplies DISK traffic per forward — on
# disk-bound configs DRAFT=0 can win even though it wins on RAM-resident boxes):
#
#   pagecache : small engine LRU (RAM_GB=10), buffered reads — the OS page
#               cache holds the hot experts; PREDICT uses fadvise hints.
#   lru-direct: big engine LRU (RAM_GB=T-12), O_DIRECT reads — all RAM belongs
#               to the engine (no double caching); PREDICT uses speculative
#               loads (PREDICT_LOAD, auto under DIRECT=1).
#
# COLD=0 skips the page-cache drop between runs (needs sudo otherwise).
# FRESH=1 deletes the model's .coli_usage first (disables autopin warm start,
# fully reproducible cold numbers; default keeps it — closer to real use).
set -euo pipefail
cd "$(dirname "$0")/.."

SNAP=${1:?usage: z3_bench.sh /path/to/hy3_i4 [NGEN]}
NGEN=${2:-96}
PROMPT=${PROMPT:-"Explain, step by step, how a turbofan jet engine produces thrust, and where its efficiency losses come from."}
TOTAL_GB=$(awk '/MemTotal/{printf "%.0f",$2/1e6}' /proc/meminfo)
BIG=$((TOTAL_GB-12))
OUT=z3_bench_$(date +%Y%m%d_%H%M%S).log
[ "${FRESH:-0}" = 1 ] && rm -f "$SNAP/.coli_usage"

drop_caches(){ [ "${COLD:-1}" = 1 ] && sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' && sleep 1 || true; }

run(){ # name, env...
    local name=$1; shift
    echo "───── $name ─────" | tee -a "$OUT"
    drop_caches
    env SNAP="$SNAP" TEMP=0 PROMPT="$PROMPT" NGEN="$NGEN" PERF=1 KV_I8=1 "$@" \
        ./hy3 64 4 8 2>&1 | tee /tmp/zb_run.txt \
        | grep -E "tok/s\)|experts loaded/token|speculation|PROFILE|PREDICT|^\[perf\]|^\[PREDICT" || true
    grep -E "tok/s\)|experts loaded|PROFILE|PREDICT" /tmp/zb_run.txt >> "$OUT" || true
    echo | tee -a "$OUT"
}

echo "Machine: $(nproc) vCPU, ${TOTAL_GB} GB RAM | NGEN=$NGEN | log: $OUT" | tee "$OUT"
echo "Model:   $SNAP" | tee -a "$OUT"

# 1. page-cache strategy
run "A1 pagecache PREDICT=4"        RAM_GB=10   PIPE=2 PREDICT=4
run "A2 pagecache PREDICT=0"        RAM_GB=10   PIPE=2 PREDICT=0

# 2. LRU + O_DIRECT strategy (PREDICT_LOAD auto-on)
run "B1 lru-direct PREDICT=4"       RAM_GB=$BIG DIRECT=1 PIPE=2 PREDICT=4 SPEC_WORKERS=8
run "B2 lru-direct PREDICT=0"       RAM_GB=$BIG DIRECT=1 PIPE=2 PREDICT=0

# 3. LRU + buffered + drop-behind (no O_DIRECT, same single-cache idea)
run "C1 lru-drop PREDICT_LOAD=1"    RAM_GB=$BIG DROP=1 PIPE=2 PREDICT=4 PREDICT_LOAD=1 SPEC_WORKERS=8

# 4. MTP draft A/B on the winner-shaped config (disk-bound: drafts may hurt)
run "D1 lru-direct DRAFT=0"         RAM_GB=$BIG DIRECT=1 PIPE=2 PREDICT=4 DRAFT=0
run "D2 lru-direct DRAFT=3"         RAM_GB=$BIG DIRECT=1 PIPE=2 PREDICT=4 DRAFT=3

echo "Summary written to $OUT"
echo "Note: .coli_usage accumulates across runs; from the 2nd run autopin pins hot"
echo "experts at startup (higher hit rate). Re-run the matrix once warmed for the"
echo "steady-state numbers, or FRESH=1 for reproducible cold numbers."
