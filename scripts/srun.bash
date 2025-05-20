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

# build slurm options
declare -a SRUN_OPTS
SRUN_OPTS+=(--mem="${MEM}G" --time="$TIME")

if (( GPUS == 0 )); then
  SRUN_OPTS+=(--cpus-per-task=1)
  echo "requesting 1 cpu | ${MEM}g mem | ${HOURS}h"
else
  GRES="gpu:${GPUS}"
  [[ -n "$GPU_TYPE" ]] && { GRES="gpu:${GPU_TYPE}:${GPUS}"; SRUN_OPTS+=(--constraint="$GPU_TYPE"); }
  SRUN_OPTS+=(--gres="$GRES" --cpus-per-task="$GPUS")
  echo "requesting ${GPUS} gpu(s)${GPU_TYPE:+ (constraint: $GPU_TYPE)} | ${MEM}g mem | ${HOURS}h"
fi

srun "${SRUN_OPTS[@]}" --pty /bin/bash
