#!/usr/bin/env bash
# surgery_calibrate.sh — collect the calibration data checkpoint surgery needs:
#   <SNAP>/.coli_usage        expert selection counts   (existing engine feature)
#   <SNAP>/.coli_saliency     applied gate-weight mass  (pruning criterion)
#   <SNAP>/.coli_route_trace  per-(token,layer) routed sets (co-activation reorder)
#
# Runs the engine over a diverse prompt set with the observers on. Routing is
# untouched (GATE_TAU stays 0), so this doubles as cache warmup: .coli_usage
# grows and autopin gets smarter with every run.
#
# Every generated/prefilled token contributes n_sparse_layers*topk selections
# (~632 for Hy3), so the default 24 prompts x ~160 tokens ≈ 2.4M selections —
# comfortably past the >=50k the pruning criterion wants. Add your own REAL
# workload on top (chat sessions, SCORE runs): the criterion generalizes to
# what you actually route.
#
# Usage on RunPod:
#   SNAP=/workspace/hy3_i4 ./scripts/surgery_calibrate.sh
# Env: SNAP (required) | NGEN=96 | PROMPTS_FILE=one-prompt-per-line override
#      EXTRA="RAM_GB=48 PIPE=2 DIRECT=1" extra engine env
set -euo pipefail
cd "$(dirname "$0")/.."
SNAP="${SNAP:?SNAP=<container dir> required}"
NGEN="${NGEN:-96}"
BIN=./hy3; [ -x "$BIN" ] || BIN=./hy3.exe

PROMPTS=(
  "Explain how a jet engine works, step by step."
  "Write a Python function that parses an ISO-8601 timestamp without libraries."
  "Prove that the square root of 2 is irrational."
  "Summarize the causes of the French Revolution in one paragraph."
  "Translate to French: 'The cache hit rate depends on the working set size.'"
  "Write a SQL query that finds the top 3 customers by revenue per region."
  "What is the difference between TCP and UDP? When would you pick each?"
  "Draft a polite email declining a meeting invitation."
  "Explain the Monty Hall problem and why switching wins."
  "Write a C function that reverses a singly linked list in place."
  "Describe the water cycle for a ten-year-old."
  "What are the tradeoffs between microservices and a monolith?"
  "Solve: a train leaves at 60 km/h, another at 90 km/h one hour later. When does it catch up?"
  "Write a haiku about winter, then explain its structure."
  "Explain gradient descent to someone who knows high-school math."
  "List the steps to safely replace a car tire."
  "What does the CAP theorem say, and what do real databases choose?"
  "Write a bash one-liner that finds the 10 largest files under /var."
  "Explain why the sky is blue and sunsets are red."
  "Compare renewable energy storage options for a national grid."
  "Write a JSON schema for a books API with authors and reviews."
  "Explain the difference between stack and heap memory."
  "Give a beginner's overview of how DNS resolution works."
  "Describe photosynthesis at the level of a biology undergraduate."
)
if [ -n "${PROMPTS_FILE:-}" ]; then
  mapfile -t PROMPTS < "$PROMPTS_FILE"
fi

echo "calibrating on ${#PROMPTS[@]} prompts, NGEN=$NGEN -> $SNAP/.coli_{usage,saliency,route_trace}"
i=0
for p in "${PROMPTS[@]}"; do
  i=$((i+1))
  echo "--- [$i/${#PROMPTS[@]}] $p"
  env SNAP="$SNAP" TEMP=0 NGEN="$NGEN" PROMPT="$p" \
      ROUTE_TRACE="$SNAP/.coli_route_trace" ${EXTRA:-} \
      "$BIN" 64 4 8 >/dev/null 2>&1 || echo "    (run failed — continuing)"
  sel=$(awk '{s+=$3} END{print s+0}' "$SNAP/.coli_usage" 2>/dev/null)
  echo "    total selections so far: ${sel:-0}"
done
echo
echo "done. files:"
wc -l "$SNAP/.coli_usage" "$SNAP/.coli_saliency" "$SNAP/.coli_route_trace" 2>/dev/null || true
echo "next: python3 tools/surgery_hy3.py --model $SNAP --out <dst> --keep-frac 0.75 --cold-frac 0.5 --reorder coact --dry-run"
