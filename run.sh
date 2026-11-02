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
export WATERMARKS_CTRLREGEN_RLIMIT_AS=137438953472 WATERMARKS_CTRLREGEN_RLIMIT_FSIZE=17179869184
export NOAI_WATERMARK_DIR="$PWD/.tools/noai-watermark"
# The local spectral score disagreed with the real detector on our test image.
unset REVERSE_SYNTHID_DIR WATERMARKS_SYNTHID_SCORER_URL

exec "$python" - <<'PY'
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path('.tools/watermarks-remover/service/scripts').resolve()))
from image_meta import clean_image

extensions = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.heic', '.heif', '.bmp', '.gif', '.tif', '.tiff'}
sources = sorted(p for p in Path('in').rglob('*') if p.is_file() and p.suffix.lower() in extensions)
if not sources:
    print('No images found in in/.')
    raise SystemExit(0)

def output_path(source):
    relative = source.relative_to('in')
    name = relative.name if relative.suffix.lower() == '.png' else relative.name + '.png'
    return Path('out') / relative.parent / name

# Detect ambiguous names before doing any GPU work or skipping an existing result.
destinations = {}
for source in sources:
    destination = output_path(source)
    if destination in destinations:
        raise SystemExit(f'Output collision: {destinations[destination]} and {source}. Rename one input.')
    destinations[destination] = source

completed = failed = 0
for index, source in enumerate(sources, 1):
    # Preserve the original extension when converting, avoiding photo.jpg / photo.png collisions.
    destination = output_path(source)
    if destination.exists():
        print(f'[{index}/{len(sources)}] Skipped existing: {destination}', flush=True)
        continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'[{index}/{len(sources)}] Processing: {source}', flush=True)
    fd, temporary = tempfile.mkstemp(prefix='.removing-', suffix='.png', dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        result = clean_image(
            source.resolve(), temporary.resolve(), remove_pixel='ctrlregen',
            ctrlregen_intensity=0.25, ctrlregen_steps=50, ctrlregen_device='cuda',
        )
        removal = result.get('pixel_removal') or {}
        if not removal.get('available'):
            raise RuntimeError(removal.get('error', 'Pixel removal did not run.'))
        if result.get('still_has_c2pa') or result.get('still_has_ai_metadata'):
            raise RuntimeError('Provenance metadata remains after processing.')
        # Publish only a successful result; failures never leave an apparent output.
        temporary.replace(destination)
        completed += 1
        print(f'  Saved: {destination}', flush=True)
    except Exception as error:
        failed += 1
        print(f'  FAILED: {source}: {error}', file=sys.stderr, flush=True)
    finally:
        temporary.unlink(missing_ok=True)

skipped = len(sources) - completed - failed
print(f'Done: {completed} processed, {skipped} skipped, {failed} failed.', flush=True)
raise SystemExit(1 if failed else 0)
PY
