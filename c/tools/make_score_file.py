#!/usr/bin/env python3
"""Build a SCORE-mode request file for hy3's log-likelihood gate.

Slices a plain-text corpus into (context, continuation) windows and tokenizes
them with the container's own tokenizer.json, emitting the engine's SCORE
format — one request per line: "<ctx_len> <cont_len> <ids...>".

Teacher-forced mean logprob on the SAME windows is the cleanest way to compare
a surgery variant against its baseline: same tokens, same harness, same
machine — the delta IS the surgery cost (same reasoning as quant_ablation.py).

  pip install tokenizers
  python3 tools/make_score_file.py --model /workspace/hy3_i4 \
      --text wiki_sample.txt --out score_req.txt --ctx 512 --cont 128 --n 32

  SNAP=/workspace/hy3_i4 SCORE=score_req.txt ./hy3 64 4 8
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="container dir (for tokenizer.json)")
    ap.add_argument("--text", required=True, help="plain-text corpus file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--n", type=int, default=32, help="number of windows")
    a = ap.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("pip install tokenizers")
    tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    ids = tok.encode(open(a.text, encoding="utf-8", errors="replace").read()).ids
    win = a.ctx + a.cont
    if len(ids) < win + 1:
        sys.exit(f"corpus too small: {len(ids)} tokens < one {win}-token window")
    stride = max(1, (len(ids) - win) // max(1, a.n - 1))
    wrote = 0
    with open(a.out, "w") as f:
        for i in range(0, len(ids) - win + 1, stride):
            if wrote >= a.n:
                break
            w = ids[i:i + win]
            f.write(f"{a.ctx} {a.cont} " + " ".join(map(str, w)) + "\n")
            wrote += 1
    print(f"{wrote} requests ({a.ctx}+{a.cont} tokens each) -> {a.out}")


if __name__ == "__main__":
    main()
