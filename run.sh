#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python="$PWD/.tools/noai-watermark/.venv/bin/python"
if [[ ! -x "$python" || ! -f .tools/watermarks-remover/service/scripts/image_meta.py ]]; then
    echo "Run ./setup.sh first." >&2
    exit 1
fi
mkdir -p in out .tools/tmp .tools/cache/huggingface .tools/cache/torch .tools/cache/matplotlib
export TMPDIR="$PWD/.tools/tmp"
export HF_HOME="$PWD/.tools/cache/huggingface" TORCH_HOME="$PWD/.tools/cache/torch"
export MPLCONFIGDIR="$PWD/.tools/cache/matplotlib" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# The local spectral score disagreed with the real detector on our test image.
unset REVERSE_SYNTHID_DIR WATERMARKS_SYNTHID_SCORER_URL

exec "$python" clean.py
