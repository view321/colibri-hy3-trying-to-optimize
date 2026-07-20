#!/usr/bin/env bash
# surgery_bench.sh — A/B a baseline Hy3 container against surgery variants:
# decode tok/s + expert traffic on a fixed greedy prompt, and (optionally) a
# SCORE-mode quality gate (teacher-forced mean log-probability on the SAME
# requests — the honest quality delta, no benchmark-protocol confounds).
#
# Each container also gets a GATE_TAU pass so the runtime knob is measured on
# the same footing as the container-level surgery.
#
# Usage on RunPod:
#   BASELINE=/workspace/hy3_i4 \
#   CONTAINERS="/workspace/hy3_i4_p75 /workspace/hy3_i4_p75c50" \
#   ./scripts/surgery_bench.sh
# Env: BASELINE (required) | CONTAINERS (space-separated variants)
#      NGEN=128 | TAU=0.35 (extra GATE_TAU pass; TAU=0 skips)
#      SCORE_FILE=requests.txt (from tools/make_score_file.py; optional)
#      COLD=1 drop page cache between runs (needs sudo; COLD=0 to skip)
#      EXTRA="RAM_GB=48 DIRECT=1 PIPE=2" engine env passed to every run
set -euo pipefail
cd "$(dirname "$0")/.."
BASELINE="${BASELINE:?BASELINE=<container> required}"
NGEN="${NGEN:-128}"
TAU="${TAU:-0.35}"
PROMPT="${PROMPT:-Explain how a jet engine works, step by step.}"
BIN=./hy3; [ -x "$BIN" ] || BIN=./hy3.exe
OUT="surgery_bench_$(date +%Y%m%d_%H%M%S).log"

drop_caches(){ [ "${COLD:-1}" = 1 ] && sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' && sleep 1 || true; }

bench_one(){ # $1 snap  $2 label  $3 extra env
  drop_caches
  echo "--- $2  ($1)" | tee -a "$OUT"
  env SNAP="$1" TEMP=0 NGEN="$NGEN" PROMPT="$PROMPT" ${EXTRA:-} $3 "$BIN" 64 4 8 2>/dev/null \
    | grep -E "tokens in|experts loaded/token|speculation|GATE_TAU:|PROFILE|PREDICT:" \
    | sed 's/^/    /' | tee -a "$OUT"
}

score_one(){ # $1 snap  $2 label
  [ -n "${SCORE_FILE:-}" ] || return 0
  drop_caches
  local res
  res=$(env SNAP="$1" SCORE="$SCORE_FILE" ${EXTRA:-} "$BIN" 64 4 8 2>/dev/null \
        | awk '{lp+=$1; n+=$2; g+=$3; r++} END{if(n>0) printf "mean logprob/token %.4f | greedy-exact %d/%d", lp/n, g, r}')
  echo "    SCORE: $res" | tee -a "$OUT"
}

echo "prompt: $PROMPT | NGEN=$NGEN | extra: ${EXTRA:-none}" | tee "$OUT"
echo | tee -a "$OUT"

bench_one "$BASELINE" "baseline" ""
score_one "$BASELINE" "baseline"
[ "$TAU" != "0" ] && bench_one "$BASELINE" "baseline + GATE_TAU=$TAU" "GATE_TAU=$TAU"

for c in ${CONTAINERS:-}; do
  name=$(basename "$c")
  bench_one "$c" "$name" ""
  score_one "$c" "$name"
  [ "$TAU" != "0" ] && bench_one "$c" "$name + GATE_TAU=$TAU" "GATE_TAU=$TAU"
done

echo | tee -a "$OUT"
echo "summary written to $OUT" | tee -a "$OUT"
echo "read it as: decode tok/s up = surgery worked; SCORE mean logprob down vs baseline = the quality you paid."
