"""Process in/ into out/ with one CtrlRegen engine. Run through ./run.sh."""

import math
import os
import secrets
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.tools/watermarks-remover/service/scripts'))
from image_meta import clean_image, inspect_image  # noqa: E402

sys.path.insert(0, str(ROOT / '.tools/noai-watermark/src'))
warnings.simplefilter('ignore')  # Dependency warnings buried real errors; failures still print tracebacks.
from diffusers.utils import logging as diffusers_logging  # noqa: E402
from ctrlregen import engine as engine_module  # noqa: E402
from ctrlregen import tiling  # noqa: E402
from ctrlregen.engine import TILE_SIZE, CtrlRegenEngine  # noqa: E402

STRENGTH = 0.25
STEPS = 50
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.heic', '.heif', '.bmp', '.gif', '.tif', '.tiff'}
_upstream_blend_weight = tiling.make_blend_weight
_upstream_color_match = engine_module.color_match


def positive_blend_weight(h, w, overlap):
    """Keep the overlap ramps, but give the outer canvas edge weight to normalize."""
    # The floor squared stays above the 1e-8 clamp in run_tiled, including at corners.
    return np.maximum(_upstream_blend_weight(h, w, overlap), 1e-3)


def guarded_color_match(reference, source):
    """Use color matching only when it helps; on flat images it can shift whole backgrounds."""
    target = np.asarray(reference, dtype=float)
    matched = _upstream_color_match(reference=reference, source=source)
    return min((matched, source), key=lambda image: np.abs(np.asarray(image, dtype=float) - target).mean())


tiling.make_blend_weight = positive_blend_weight
engine_module.color_match = guarded_color_match


def prepare(image):
    """Return a canvas the engine neither crops nor resizes, and the image's box in it."""
    width, height = image.size
    if width <= TILE_SIZE and height <= TILE_SIZE:
        # The single-pass path works at 512px. Fit the whole image rather than cropping it.
        scale = TILE_SIZE / max(width, height)
        size = max(1, round(width * scale)), max(1, round(height * scale))
        image = image.resize(size, Image.LANCZOS)
    width, height = image.size
    canvas_width = max(TILE_SIZE, math.ceil(width / 8) * 8)
    canvas_height = max(TILE_SIZE, math.ceil(height / 8) * 8)
    left, top = (canvas_width - width) // 2, (canvas_height - height) // 2
    # Mirroring adds no hard edge for the edge detector, even when padding exceeds the image.
    canvas = np.pad(
        np.asarray(image),
        ((top, canvas_height - height - top), (left, canvas_width - width - left), (0, 0)),
        mode='symmetric',
    )
    return Image.fromarray(canvas), (left, top, left + width, top + height)


def regenerate(engine, image, seed):
    """Regenerate an RGB image and return a result with its exact original dimensions."""
    canvas, box = prepare(image)
    result = engine.run(canvas, strength=STRENGTH, num_inference_steps=STEPS, seed=seed).crop(box)
    return result if result.size == image.size else result.resize(image.size, Image.LANCZOS)


def output_path(source):
    relative = source.relative_to('in')
    # Preserve the original extension when converting, avoiding photo.jpg / photo.png collisions.
    name = relative.name if relative.suffix.lower() == '.png' else relative.name + '.png'
    return Path('out') / relative.parent / name


def clean_file(engine, source, destination, seed):
    fd, temporary = tempfile.mkstemp(prefix='.removing-', suffix='.png', dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary).resolve()
    try:
        clean_image(source.resolve(), temporary)  # Validates the format and strips metadata.
        with Image.open(temporary) as stripped:
            image = stripped.convert('RGB')
        regenerate(engine, image, seed).save(temporary, format='PNG')
        report = inspect_image(temporary, run_synthid=False, include_exiftool=False)
        if report.has_c2pa or report.has_ai_metadata:
            raise RuntimeError('Provenance metadata remains after processing.')
        # Publish only a successful result; failures never leave an apparent output.
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def reporter():
    """Print each stage once, and denoising progress at most every ten seconds."""
    last = {'time': 0.0}

    def report(message):
        if '█' in message or '░' in message:
            if time.monotonic() - last['time'] < 10:
                return
            last['time'] = time.monotonic()
        elif ' on cuda' in message:  # Repeated animation text from the backend's updater thread.
            return
        print(f'  {message}', flush=True)

    return report


def load_engine():
    diffusers_logging.disable_progress_bar()
    engine = CtrlRegenEngine(device='cuda', progress_callback=reporter())
    engine.load()
    engine._pipeline.set_progress_bar_config(disable=True)
    return engine


def main():
    fixed_seed = os.environ.get('SEED')
    fixed_seed = int(fixed_seed) if fixed_seed else None
    sources = sorted(p for p in Path('in').rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not sources:
        print('No images found in in/.')
        return 0

    # Detect ambiguous names before doing any GPU work or skipping an existing result.
    destinations = {}
    for source in sources:
        destination = output_path(source)
        if destination in destinations:
            raise SystemExit(f'Output collision: {destinations[destination]} and {source}. Rename one input.')
        destinations[destination] = source

    engine = None
    completed = 0
    failures = []
    for index, source in enumerate(sources, 1):
        destination = output_path(source)
        if destination.exists():
            print(f'[{index}/{len(sources)}] Skipped existing: {destination}', flush=True)
            continue
        if engine is None:
            print('Loading the model once for this batch...', flush=True)
            engine = load_engine()
        seed = fixed_seed if fixed_seed is not None else secrets.randbelow(2**31)
        print(f'[{index}/{len(sources)}] Processing: {source} (seed {seed})', flush=True)
        started = time.monotonic()
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            clean_file(engine, source, destination, seed)
            completed += 1
            print(f'  Saved: {destination} ({time.monotonic() - started:.0f}s)', flush=True)
        except Exception as error:
            traceback.print_exc()
            failures.append(f'{source}: {type(error).__name__}: {error}')
            print(f'  FAILED: {failures[-1]}', file=sys.stderr, flush=True)

    skipped = len(sources) - completed - len(failures)
    print(f'Done: {completed} processed, {skipped} skipped, {len(failures)} failed.', flush=True)
    for failure in failures:
        print(f'FAILED: {failure}', file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
