#!/usr/bin/env bash
# Storage-optimized cloud VM bootstrap for the Hy3 engine (local-NVMe streaming).
# Targets fresh Ubuntu 24.04 LTS on:
#   - GCP Z3 (Titanium SSD, e.g. z3-highmem-8-highlssd)
#   - AWS storage-optimized (instance-store NVMe: i4i / i7ie / i3en / i8g ...)
# Multiple local SSDs are striped into one md RAID0. Run as a normal user
# with sudo rights:
#
#   ./scripts/z3_setup.sh                 # deps + SSD mount + build + selftest
#   ./scripts/z3_setup.sh --download      # + fetch UnderstandLing/Hy3-colibri-int4 (~142 GB)
#
# Titanium SSD is EPHEMERAL: instance stop / spot preemption destroys the model
# copy — re-run with --download after every fresh boot of a new instance.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> c/

MNT=${MNT:-/mnt/lssd}
MODEL_DIR=${MODEL_DIR:-$MNT/hy3_i4}
HF_REPO=${HF_REPO:-UnderstandLing/Hy3-colibri-int4}
DO_DOWNLOAD=0
[ "${1:-}" = "--download" ] && DO_DOWNLOAD=1

echo "== [1/5] packages =="
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential liburing-dev mdadm python3-venv python3-pip fio >/dev/null
gcc --version | head -1

echo "== [2/5] Titanium SSD -> $MNT =="
if mountpoint -q "$MNT"; then
    echo "   already mounted:"; df -h "$MNT" | tail -1
else
    # GCP: /dev/disk/by-id/google-local-nvme-ssd-N ; AWS: instance-store NVMe
    # devices carry the model string "Amazon EC2 NVMe Instance Storage".
    # LSSD_DEVS overrides autodetection — required on bare metal (e.g. Scaleway
    # Elastic Metal), where the OS boots FROM one of the NVMe drives and blind
    # globbing could grab the root disk: LSSD_DEVS="/dev/nvme1n1" ./z3_setup.sh
    collect(){ ls "$@" 2>/dev/null | grep -v part | xargs -r -n1 readlink -f | sort -u; }
    if [ -n "${LSSD_DEVS:-}" ]; then
        read -r -a SSDS <<< "$LSSD_DEVS"
    else
        mapfile -t SSDS < <(collect /dev/disk/by-id/google-local-nvme-ssd-*)
        if [ ${#SSDS[@]} -eq 0 ]; then
            mapfile -t SSDS < <(collect /dev/disk/by-id/nvme-Amazon_EC2_NVMe_Instance_Storage-*)
        fi
    fi
    [ ${#SSDS[@]} -ge 1 ] || { echo "no local SSD found (GCP Z3 / AWS instance store?). On bare metal pass LSSD_DEVS=/dev/nvmeXn1 explicitly."; exit 1; }
    for d in "${SSDS[@]}"; do
        mounted=$(lsblk -no MOUNTPOINTS "$d" 2>/dev/null | grep -v '^$' || true)
        [ -z "$mounted" ] || { echo "refusing to format $d — it has mounted partitions ($mounted)"; exit 1; }
    done
    echo "   found ${#SSDS[@]} local SSD device(s): ${SSDS[*]}"
    if [ ${#SSDS[@]} -eq 1 ]; then
        DEV=${SSDS[0]}
    else
        DEV=/dev/md0
        yes | sudo mdadm --create "$DEV" --level=0 --raid-devices=${#SSDS[@]} -c 512K "${SSDS[@]}"
    fi
    # ext4 without lazy init: full speed from the first read, no background zeroing
    sudo mkfs.ext4 -q -F -E lazy_itable_init=0,lazy_journal_init=0,discard "$DEV"
    sudo mkdir -p "$MNT"
    sudo mount -o defaults,noatime "$DEV" "$MNT"
    sudo chown "$(id -u):$(id -g)" "$MNT"
    df -h "$MNT" | tail -1
fi

echo "== [3/5] build (ARCH=native unlocks avx512-vnni on Sapphire Rapids; IOURING=1 for PIPE=2) =="
make -s hy3 iobench ARCH=native IOURING=1
if gcc -march=native -dM -E - </dev/null | grep -qi AVX512VNNI; then
    echo "   avx512-vnni: available (idot int8 kernels active)"
else
    echo "   avx512-vnni: NOT reported by gcc (unexpected on Z3 — check machine type)"
fi

echo "== [4/5] self-test =="
if [ -d hy3_tiny ]; then
    SNAP=./hy3_tiny TF=1 ./hy3 64 16 16 2>/dev/null | grep positions   # expect 32/32
else
    echo "   hy3_tiny/ missing (clone incomplete?) — skipping oracle"
fi

if [ "$DO_DOWNLOAD" = 1 ]; then
    echo "== [5/5] model download: $HF_REPO -> $MODEL_DIR (~142 GB) =="
    if [ ! -d "$HOME/hfenv" ]; then python3 -m venv "$HOME/hfenv"; "$HOME/hfenv/bin/pip" -q install -U "huggingface_hub[cli]" hf_transfer; fi
    HF_HUB_ENABLE_HF_TRANSFER=1 "$HOME/hfenv/bin/hf" download "$HF_REPO" --local-dir "$MODEL_DIR" --max-workers 16
    du -sh "$MODEL_DIR"
else
    echo "== [5/5] skipped model download (pass --download) =="
fi

echo
echo "== disk sanity check (expect ~2.9-3.1 GB/s per Titanium SSD, random 9 MB reads) =="
SHARD=$(ls "$MODEL_DIR"/out-*.safetensors 2>/dev/null | head -1 || true)
if [ -n "$SHARD" ]; then
    ./iobench "$SHARD" 9 256 8 1     # O_DIRECT
    ./iobench "$SHARD" 9 256 8 0     # buffered (page-cache path)
else
    echo "   (model not downloaded yet — re-run after --download)"
fi

echo
echo "Done. Benchmark with:   bash scripts/z3_bench.sh $MODEL_DIR"
echo "Chat with (Z3 config):  SNAP=$MODEL_DIR RAM_GB=52 DIRECT=1 PIPE=2 PREDICT=4 KV_I8=1 \\"
echo "                        TEMP=0.7 PROMPT='Hello!' NGEN=200 ./hy3 64 4 8"
echo "or via the wrapper:     COLI_MODEL=$MODEL_DIR DIRECT=1 PIPE=2 PREDICT=4 KV_I8=1 ./coli chat --ram 52"
