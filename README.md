# image-cleaner

![image-cleaner: 100% SynthID detected before, 0% SynthID detected after](.github/banner.png)

![Local processing](https://shieldcn.dev/badge/processing-local-blue.svg)
![NVIDIA CUDA](https://shieldcn.dev/badge/GPU-NVIDIA%20CUDA-green.svg)

Clean a folder of images on your own GPU. Put images in `in/`, run one command, get results in `out/`.

Made by [Tyranny](https://github.com/tyrannyme).

## Quick start

```bash
git clone https://github.com/tyrannyme/image-cleaner.git
cd image-cleaner
./setup.sh      # once
./run.sh        # every time you have new images in in/
```

## Requirements

- Linux with Bash and Git
- Python 3.11 with `venv` and `pip`
- NVIDIA GPU with a driver that supports CUDA 12.8 (no CUDA toolkit needed)
- About 25 GB of free disk space

Tested on an RTX 4070 Ti SUPER (16 GB).

If Python 3.11 is not on your `PATH` as `python3.11`:

```bash
PYTHON=/path/to/python3.11 ./setup.sh
```

## How it works

- Runs CtrlRegen at **strength 0.25, 50 steps**, a deliberately gentle setting.
- Uses [watermarks-remover](https://github.com/guillaumemeyer/watermarks-remover) and [noai-watermark](https://github.com/mertizci/noai-watermark), pinned to fixed commits.
- Loads the model once per batch. Inputs are padded, never cropped, so the whole image is processed.
- Everything stays in this folder. Dependencies, model weights, caches, and temp files go in `.tools/`.
- Nothing runs in the background. No server, no daemon, no global config changes.
- Model weights download on the first run and are reused after that.

## Input and output

Folders are mirrored. Outputs are always PNG:

```text
in/portrait.png       -> out/portrait.png
in/album/photo.jpg    -> out/album/photo.jpg.png
```

- **Originals are never changed.** Outputs keep the original dimensions.
- **Already-processed files are skipped.** Delete an output to redo it.
- **Supported formats:** PNG, JPEG, WebP, AVIF, HEIC/HEIF, BMP, GIF, TIFF. Some need extra Pillow codecs; if one is missing, that file fails and is reported.
- **Animated or multipage files** become a single still image.
- **Transparency** is lost; outputs are flattened to RGB.
- **Model textures:** do not put UV atlases or other model data textures through this as routine cleanup. Regenerating one damaged a character's hair in practice. Review any intentional texture edit on the actual model.
- **Name clashes:** if two inputs would produce the same output name, nothing runs until you rename one.
- **Progress:** each file prints its stages, elapsed time, and seed.
- **Failures:** a failed file prints its full error and the batch continues. Failures are listed again at the end, and the command exits with `1`. A failed file never produces an output. A hard crash, such as a driver fault, stops the batch; rerun to continue where it stopped.

`in/`, `out/`, and `.tools/` are ignored by Git.

## Tests

`./test.sh` checks sizing, blending, and failure reporting on the CPU with a stand-in for the model. It does not check image quality or watermark removal.

## Limits

- **Not guaranteed.** Our test image passed the real detector after processing. That is one example, not a promise for every image.
- **"Done" does not mean "verified."** The script only reports whether processing succeeded. Check important outputs with the real detector.
- **No local score.** A local spectral scorer still flagged our successful output, so it is not included.
- **Details can shift.** Fine detail may change, and flat backgrounds can pick up faint blotches from the model. Look at your outputs before using them.
- **Reject any bad output.** Delete it and run again for a new random seed. Use `SEED=123 ./run.sh` to reuse a printed seed.
- **Strength stays at 0.25.** Strength 0.5 caused severe artifacts in testing, so it is not offered.
