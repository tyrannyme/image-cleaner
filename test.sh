#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export TMPDIR="$PWD/.tools/tmp" MPLCONFIGDIR="$PWD/.tools/cache/matplotlib" HF_HOME="$PWD/.tools/cache/huggingface"
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset REVERSE_SYNTHID_DIR WATERMARKS_SYNTHID_SCORER_URL
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$HF_HOME"
exec .tools/noai-watermark/.venv/bin/python test.py "$@"
