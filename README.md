# image-cleaner

![image-cleaner: 100% SynthID detected before, 0% SynthID detected after](docs/banner.png)

![Local processing](https://shieldcn.dev/badge/processing-local-blue.svg)
![NVIDIA CUDA](https://shieldcn.dev/badge/GPU-NVIDIA%20CUDA-green.svg)

A local image-cleaning tool by [Tyranny](https://github.com/tyrannyme). Batch-process images from `in/` into `out/` with conservative CtrlRegen settings: **strength 0.25, 50 steps**.

Uses [watermarks-remover](https://github.com/guillaumemeyer/watermarks-remover) and its [noai-watermark](https://github.com/mertizci/noai-watermark) backend at pinned commits. Images are processed locally. There is no HTTP server, background daemon, or global configuration change.

## Setup

Requires Linux, Bash, Git, Python **3.11** with `venv`/`pip`, and an NVIDIA GPU with a driver compatible with CUDA 12.8. The tested machine has a 16 GB RTX 4070 Ti SUPER. Allow roughly 25 GB of free disk space for dependencies, model weights, and installation temporary files.

```bash
git clone https://github.com/tyrannyme/image-cleaner.git
cd image-cleaner
./setup.sh
```

If Python 3.11 has a different executable path:

```bash
PYTHON=/path/to/python3.11 ./setup.sh
```

Setup installs PyTorch and the backend dependencies in `.tools/noai-watermark/.venv/`. Re-running setup reuses the installation. All downloads, caches, and temporary files stay under `.tools/`; model weights download on the first processing run and are reused afterward. No separate CUDA toolkit installation is needed.

## Run

Place images in `in/`, then run:

```bash
./run.sh
```

Scripts work from any current directory. Input subfolders are preserved:

```text
in/portrait.png       -> out/portrait.png
in/album/photo.jpg    -> out/album/photo.jpg.png
```

- Originals stay untouched. Outputs are PNGs at the original dimensions.
- Existing outputs are skipped. Delete an output to process that input again.
- If two inputs would map to the same output name, the batch stops before processing and asks you to rename one.
- PNG, JPEG, WebP, AVIF, HEIC/HEIF, BMP, GIF, and TIFF extensions are discovered, case-insensitively. Decoding depends on the installed Pillow codecs; unsupported codecs produce a reported failure. Animated or multipage images produce a single still image. Transparency is flattened by the RGB regeneration backend.
- Failed files are reported and the batch continues. The command exits with status `1` if any file fails; failed processing never publishes a metadata-only copy as a successful output.
- `in/`, `out/`, `.tools/`, and the initial experiment PNGs are ignored by Git.

## What the result means

The conservative pass passed the real detector for our test image. This is one verified example, not a guarantee for every input. Regeneration can change fine details and colors; inspect outputs before using them.

The unofficial local spectral scorer still flagged that successful output, so these scripts do not install or run it. The batch reports processing success, not a verified watermark verdict. Verify outputs with the real detector when that distinction matters.

Randomness follows the backend default, as in the successful pass, so repeat runs can differ. Raising strength to 0.5 produced severe artifacts in our test and is deliberately not exposed as a default option.
