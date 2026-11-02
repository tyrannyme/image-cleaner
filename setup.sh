#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python="${PYTHON:-python3.11}"
for tool in git nvidia-smi "$python"; do
    command -v "$tool" >/dev/null || { echo "Missing required command: $tool" >&2; exit 1; }
done
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11 (set PYTHON to its executable)"'
nvidia-smi >/dev/null
mkdir -p .tools/tmp .tools/cache/pip .tools/cache/huggingface .tools/cache/torch .tools/cache/matplotlib in out
export TMPDIR="$PWD/.tools/tmp" PIP_CACHE_DIR="$PWD/.tools/cache/pip"
export HF_HOME="$PWD/.tools/cache/huggingface" TORCH_HOME="$PWD/.tools/cache/torch"
export MPLCONFIGDIR="$PWD/.tools/cache/matplotlib" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

checkout() {
    local repo="$1" destination="$2" revision="$3"
    if [[ ! -d "$destination" ]]; then
        git clone --depth 1 --filter=blob:none --sparse "https://github.com/$repo.git" "$destination"
        git -C "$destination" fetch --depth 1 origin "$revision"
        git -C "$destination" checkout --detach "$revision"
    fi
    if [[ "$(git -C "$destination" rev-parse HEAD)" != "$revision" ]]; then
        echo "Unexpected revision in $destination; expected $revision. Leaving it untouched." >&2
        exit 1
    fi
}

checkout guillaumemeyer/watermarks-remover .tools/watermarks-remover 5a62be6338d2812372b1da440e55c82922f97d53
git -C .tools/watermarks-remover sparse-checkout set service config
checkout mertizci/noai-watermark .tools/noai-watermark b642ae45d20eded52c96d570985eb4e3e427aac8
git -C .tools/noai-watermark sparse-checkout set src

venv="$PWD/.tools/noai-watermark/.venv"
if [[ ! -x "$venv/bin/python" ]]; then
    "$python" -m venv "$venv"
fi
"$venv/bin/python" -m pip install 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu128
"$venv/bin/python" -m pip install -r .tools/watermarks-remover/service/scripts/requirements-ctrlregen.txt
PYTHONPATH="$PWD/.tools/noai-watermark/src" "$venv/bin/python" - <<'PY'
import torch
from ctrlregen.engine import is_ctrlregen_available

if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Check your NVIDIA driver (CUDA 12.8 wheels are installed).')
if not is_ctrlregen_available():
    raise SystemExit('CtrlRegen dependencies failed to load.')
print(f'Ready on {torch.cuda.get_device_name(0)}. Put images in in/ and run ./run.sh.')
print('Model weights download into .tools/cache/ on the first run.')
PY
