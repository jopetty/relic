#!/bin/bash

set -euo pipefail

# defaults
HOURS=2  # integer hours
MEM=32
GPUS=2
GPU_TYPE=""

usage() {
	echo "usage: $0 [-h hours] [-m memory (in GB)] [-g gpus] [-a gpu_type]" >&2
  exit 1
}

# parse flags
while getopts ":h:m:g:a:" opt; do
  case "$opt" in
    h) HOURS="$OPTARG" ;;
    m) MEM="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    a) GPU_TYPE="$OPTARG" ;;
    *) usage ;;
  esac
done

# validate hours
[[ "$HOURS" =~ ^[0-9]+$ ]] || { echo "hours must be an integer" >&2; exit 1; }

TIME=$(printf "%02d:00:00" "$HOURS")

GRES="gpu:${GPUS}"

echo "requesting ${GPUS} gpu(s)${GPU_TYPE:+ (constraint: $GPU_TYPE)} | ${MEM}g mem | ${HOURS}h"

srun --gres="$GRES" \
     ${GPU_TYPE:+--constraint="$GPU_TYPE"} \
     --cpus-per-task="$GPUS" \
     --mem="${MEM}G" \
     --time="$TIME" \
     --pty /bin/bash

