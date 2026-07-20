# Running Hy3 on GCP Z3 (little RAM, fast disk)

Z3 is Google Cloud's storage-optimized family: Sapphire Rapids CPUs with big
**Titanium SSDs bundled into the machine price**. It is the opposite experiment
of a memory-optimized box (whole 142 GB container resident): here the model
lives on NVMe and the engine streams experts, with RAM acting only as cache.

The free-trial-compatible shape:

| | z3-highmem-8-highlssd |
|--|--|
| vCPU | 8 (4 physical Sapphire Rapids cores, 2.2/3.0 GHz, AVX-512 VNNI + AMX) |
| RAM | 64 GB |
| Titanium SSD | 1 × 3,000 GiB, **3,000 MiB/s read, 750K read IOPS** (single NVMe namespace, no RAID needed) |
| Price (us-central1, on-demand) | **$1.15/h** (SSD included) → ~260 h on the $300 trial credit |
| Spot | supported (compute part ≈ $0.12/h; SSD data dies on preemption) |

Bigger shapes (need an upgraded account, credit still applies): 16 vCPU/128 GB/6 TiB
@ 5.9 GiB/s ($2.30/h), 22 vCPU/176 GB/9 TiB @ 8.8 GiB/s ($3.25/h — fits the
whole container in RAM after warmup). SSD throughput scales at 3,000 MiB/s per
3-TiB disk.

## Expected performance (do the math before paying)

Hy3 numbers (int4 container): 79 sparse layers × top-8 → **632 expert visits /
decode token × 9.47 MB/expert ≈ 6.0 GB** of expert reads per fully-cold token.
The expert pool is 79 × 192 × 9.47 MB ≈ **143 GB**; the dense stack (~5–7 GB)
is always resident.

On z3-highmem-8-highlssd (64 GB RAM):

- **Cache**: ~64 − 7 (dense+runtime) − ~3 (KV/scratch/OS) ≈ **~50 GB of experts
  cached ≈ 35% of the pool**. Hy3's sigmoid+bias router balances experts fairly
  uniformly long-run, so the usage-weighted hit rate lands near the cache
  fraction, improving to ~45–55% with autopin (`.coli_usage`) + session locality.
- **Disk per token**: 6.0 GB × (0.45–0.65 miss) ≈ **2.7–3.9 GB → 0.9–1.3 s/token**
  at 3.0 GiB/s. This is the bottleneck. (Reads are 9 MB — throughput-bound;
  750K IOPS is never the limit.)
- **CPU per token**: ~13 GB of weight+KV traffic through 4 cores (~25–35 GB/s
  sustainable) ≈ 0.4–0.5 s — overlapped with disk by PIPE + PREDICT.
- **Prefill**: each forward unions most experts per layer → reads ≈ the whole
  143 GB once ≈ **~50 s + compute for the first prompt**, then amortizes.

**Bottom line: expect ≈ 0.7–1.0 tok/s decode steady-state** (first tokens
slower while caches warm), vs ~4–5 tok/s on a 32-vCPU / 256 GB all-in-RAM box.
The disk does ~3 GB/s where RAM does ~50: a single Titanium SSD replaces ~75%
of the RAM at ~1/16 of the bandwidth, and prediction/pipelining hides latency,
not bytes. For >1 tok/s you need more cache or more disks:
z3-highmem-16-highlssd (~1.5–2 tok/s), z3-highmem-22-highlssd (~3–4.5 tok/s,
effectively RAM-resident after warmup).

## Engine configuration for Z3

Two coherent strategies (the bench script A/Bs both):

1. **pagecache** — `RAM_GB=10 PIPE=2 PREDICT=4`: tiny engine LRU; the OS page
   cache holds ~45 GB of hot experts; PREDICT issues fadvise readahead.
2. **lru-direct** (recommended) — `RAM_GB=52 DIRECT=1 PIPE=2 PREDICT=4`:
   all RAM belongs to the engine's own LRU + autopin, reads are O_DIRECT (no
   page-cache double-buffering of the same bytes, no copy overhead), and
   PREDICT automatically switches to **`PREDICT_LOAD`**: speculative
   `expert_load()`s executed by a worker pool straight into the target layer's
   LRU slot. This branch adds that mode — plain `fadvise` hints physically
   cannot work under O_DIRECT, and get dropped by the kernel exactly when the
   LRU owns most of RAM.

Also use: `KV_I8=1` (int8 KV: saves ~2 GB/token of memory traffic at 4k ctx),
`ARCH=native` build (unlocks avx512-vnni int8 kernels on Sapphire Rapids),
`IOURING=1` build + `PIPE=2` (async expert loads at queue depth without
burning threads), and A/B `DRAFT=0` vs `DRAFT=3`: MTP speculative decode
multiplies **disk** traffic per forward (~30 unique experts/layer instead of
8) for ~2 tokens/forward — a win in RAM, often a loss when disk-bound.

## Deploying (free trial account)

### 0. Push this branch

The VM clones your repo; commit and push `gpu-accel` (or your fork's branch)
first. Alternative: `gcloud compute scp --recurse` the tree to the VM.

### 1. One-time project setup (local machine or Cloud Shell)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable compute.googleapis.com

# Z3 draws from a per-VM-family quota (not the generic CPUS quota). The most
# reliable check is the console: IAM & Admin -> Quotas & system limits ->
# filter "Z3" -> the CPUs-per-family value for your region must be >= 8.
# Legacy regional quotas (incl. Local SSD GB) are visible via:
gcloud compute regions describe us-central1 --format=json | grep -iE '"metric".*(CPUS|SSD)' -A1 | head -20
```

Free-trial gotchas (from Google's ToS/docs):
- Hard cap of **8 concurrent vCPUs** per trial → `z3-highmem-8-highlssd` is the
  *only* Z3 shape that fits, and nothing else can run at the same time.
- **No quota-increase requests** on trial. If the Z3-family quota shows 0 in
  your region, try another Z3 region (us-east1/4/5, us-west1, us-south1) or
  **upgrade the account** (Console → Billing → "Activate full account"): your
  remaining $300 stays usable until the original 90-day expiry, the 8-core cap
  lifts, and quota requests unlock.

### 2. Create the instance

```bash
gcloud compute instances create hy3-z3 \
  --zone=us-central1-a \
  --machine-type=z3-highmem-8-highlssd \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB --boot-disk-type=pd-balanced
```

Notes: the 3-TiB Titanium SSD attaches automatically (**no `--local-ssd`
flags**); gVNIC is the default NIC on Z3; Ubuntu 24.04 has the drivers.
Optional spot variant (≈5–8× cheaper, can be preempted; SSD wiped on preempt):
add `--provisioning-model=SPOT --instance-termination-action=DELETE`.

If creation fails with a quota error, see the gotchas above.

### 3. Bootstrap on the VM

```bash
gcloud compute ssh hy3-z3 --zone=us-central1-a

# on the VM:
git clone -b gpu-accel https://github.com/YOUR_FORK/colibri-hy3.git
cd colibri-hy3/c
bash scripts/z3_setup.sh --download  # deps, format+mount SSD, build, 32/32 selftest,
                                     # ~142 GB model pull (10-30 min at GCP ingress speeds),
                                     # iobench sanity check (expect ~2.9-3.1 GB/s)
```

`z3_setup.sh` handles: ext4 on the Titanium SSD (RAID0 automatically on
multi-disk shapes), `make hy3 iobench ARCH=native IOURING=1`, the tiny-oracle
self-test, the Hugging Face download, and an `iobench` run against a real
shard in both O_DIRECT and buffered modes so you know the disk delivers its
rated 3 GB/s *before* burning time on model runs.

### 4. Benchmark

```bash
bash scripts/z3_bench.sh /mnt/lssd/hy3_i4          # full matrix, ~30-45 min
NGEN=192 bash scripts/z3_bench.sh /mnt/lssd/hy3_i4 # longer steady-state runs
```

What to read in the output of each run:

- `(X tok/s)` — decode throughput for that config.
- `expert hit %` + `experts loaded/token` — cache effectiveness (632 visits/token
  max; `loaded/token × 9.47 MB / tok-time` should ≈ your disk bandwidth when
  disk-bound).
- `PROFILE: expert-disk … expert-matmul … attention …` — where the time went;
  disk-bound means expert-disk dominates.
- `PREDICT: … hit %` and `PREDICT_LOAD: N speculative loads | dropped | waited` —
  prediction quality and whether speculative loads kept up.
- Run the matrix twice: the second pass starts with `.coli_usage` history →
  autopin pre-pins hot experts → higher hit rate ≈ steady-state numbers.

Interactive chat / API with the winning config:

```bash
COLI_MODEL=/mnt/lssd/hy3_i4 DIRECT=1 PIPE=2 PREDICT=4 KV_I8=1 ./coli chat --ram 52
COLI_MODEL=/mnt/lssd/hy3_i4 DIRECT=1 PIPE=2 PREDICT=4 KV_I8=1 ./coli serve --ram 52 --port 8000
```

### 5. Cost control

- `gcloud compute instances stop hy3-z3 --zone=us-central1-a` stops vCPU/RAM
  billing but **wipes the Titanium SSD** (model re-download on next start; the
  boot disk with the repo and `.coli_usage` survives if you copy it home first).
- `gcloud compute instances delete hy3-z3 --zone=us-central1-a` when done.
- At $1.15/h the $300 credit ≈ 260 machine-hours; a full benchmark session
  (create → download → matrix × 2 → chat) fits in ~2-3 h ≈ **$3-4**.
- Optional persistence between sessions: snapshot the model onto a Hyperdisk
  Balanced volume (~$0.08/GiB·mo ≈ $12/mo for 150 GiB, restore at 800 MiB/s on
  this shape) — usually not worth it vs a 15-min re-download.

## AWS alternative (new-account credits)

AWS's post-July-2025 free tier gives new accounts **$100 at signup + $100 for
five click-through activities** (credits valid 12 months; usable on any EC2
including storage-optimized, if you pick the **paid plan** at signup — the
"free plan" has an undocumented service scope and no extra credit). One catch:
new accounts start with a **5-vCPU quota**, so file a Service Quotas increase
for "Running On-Demand Standard instances" (L-1216C47A) to **16** right after
signup — unlike GCP's trial, increases are allowed and small bumps usually
approve in minutes-to-hours. Same for Spot (L-34B43A08) if you want ~70-85%
off (spot interruption wipes the NVMe, like GCP).

Candidates (us-east-1/2; NVMe included in price; throughputs are
community-measured — AWS only documents 4K IOPS for these):

| Instance | HW | $/h od / spot | Disk read | Expected decode |
|---|---|---|---|---|
| i7i.2xlarge | 8 vCPU EMR x86, 64 GB, 1.9 TB | $0.755 / ~$0.29 | ~2.1 GB/s | ~0.5–0.75 tok/s |
| i8g.2xlarge | 8 vCPU Graviton4 ARM, 64 GB, 1.9 TB | $0.686 / ~$0.19 | ~2.1 GB/s | ~0.5–0.75 tok/s |
| i4i.4xlarge | 16 vCPU ICL x86, 128 GB, 3.75 TB | $1.373 / ~$0.66 | ~3.1 GB/s | **~1.7–2.3 tok/s** |
| i8g.4xlarge | 16 vCPU Graviton4, 128 GB, 3.75 TB | $1.373 / n/a | ~4.3 GB/s | **~2–3 tok/s** |

The 4xlarge rows beat GCP's trial-limited z3-highmem-8 (~0.7–1.0 tok/s)
mainly because 128 GB RAM caches ~80% of the expert pool. Graviton (i8g) needs
the aarch64 build — the engine's NEON/idot path already runs on ARM (see the
GB10 row in the README table); i7i/i4i are the zero-risk x86 route
(Emerald/Ice Lake, avx512-vnni).

Deploy: launch Ubuntu 24.04 (arm64 AMI for i8g), then the same
`bash scripts/z3_setup.sh --download` — it auto-detects AWS instance-store
NVMe (`nvme-Amazon_EC2_NVMe_Instance_Storage-*`) as well as GCP Titanium SSD.
Stop/terminate/interruption erases the NVMe → re-download per session.

## Other providers checked (2026-07)

- **Scaleway** — no standing free trial (only periodic ~€100 promo vouchers,
  payment method required; vouchers typically **exclude Apple Silicon** and
  never cover Elastic Metal commitment fees — hourly EM usage is fine). But it
  is the cheapest cash venue: **Elastic Metal EM-B230E-NVMe** (8C/16T Zen4,
  64 GB DDR5, 2×1.02 TB NVMe, **€0.33/h**, hourly, no commitment) is real bare
  metal — native io_uring/O_DIRECT, RAID0 across both drives. NVMe SKUs are
  undocumented → run `iobench` in the first hour. Install with custom
  partitioning (default layout soft-RAIDs the OS across both drives); then
  `LSSD_DEVS=/dev/nvme1n1 bash scripts/z3_setup.sh --download`. Also
  **Mac mini M4 Pro-XL** (14-core M4 Pro, 64 GB unified, 2.05 TB internal SSD
  measured ~6.8 GB/s, €0.49/h, **24 h minimum**): the engine runs CPU-only
  there (NEON), `DIRECT=1` maps to F_NOCACHE, and `PREDICT_LOAD` supplies the
  prefetch that fadvise (a macOS no-op) never did — likely ~1.5–2 tok/s in the
  same 64 GB cache regime, the best small-shape number of any provider.
- **Modal** — genuine **$30/month recurring** free compute (Starter plan), but
  unfit for this experiment: containers run under **gVisor** (io_uring
  disabled/limited, O_DIRECT semantics not passed through, fadvise no-ops),
  Volumes are network storage capped ~1–2.5 GB/s, and 8 cores + 64 GB costs
  ~$0.89/h ≈ 33 h/month of credit. Fine for demos/CI, not for disk-path
  benchmarking.

## Results template

Please record: shape, config line, cold + warm `tok/s`, hit %, `PROFILE` split,
`PREDICT`/`PREDICT_LOAD` lines, and the two `iobench` numbers. That feeds the
README performance table.
