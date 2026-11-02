"""CPU-only checks with an identity model. Run through ./test.sh."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

import clean
from ctrlregen.tiling import tile_positions

SIZES = [
    (1, 1), (64, 64), (220, 190), (190, 220), (511, 300), (300, 511), (512, 512), (512, 513), (513, 512),
    (520, 512), (432, 540), (540, 432), (1400, 228), (228, 1400), (973, 171), (171, 973), (1200, 639),
]


def identity_engine():
    engine = clean.CtrlRegenEngine(device='cpu')
    engine._pipeline = lambda **kwargs: SimpleNamespace(images=[kwargs['image'][0]])
    engine._canny_detector = lambda image, **kwargs: image
    return engine


def strips(size):
    """A green image with a red left edge, blue right edge, and white top and bottom edges."""
    width, height = size
    pixels = np.zeros((height, width, 3), np.uint8)
    pixels[:] = 0, 255, 0
    band_x, band_y = max(1, width // 20), max(1, height // 20)
    pixels[:band_y] = pixels[-band_y:] = 255, 255, 255
    pixels[:, :band_x] = 255, 0, 0
    pixels[:, -band_x:] = 0, 0, 255
    return Image.fromarray(pixels)


class Canvas(unittest.TestCase):
    def test_canvas_is_exact_for_the_engine(self):
        for size in SIZES:
            with self.subTest(size=size):
                canvas, (left, top, right, bottom) = clean.prepare(Image.new('RGB', size))
                width, height = canvas.size
                self.assertGreaterEqual(min(width, height), 512)
                self.assertEqual((width % 8, height % 8), (0, 0))
                self.assertTrue(0 <= left < right <= width and 0 <= top < bottom <= height)
                if max(size) > 512:  # Tiled images keep their native pixels.
                    self.assertEqual((right - left, bottom - top), size)
                else:  # Small images fill the 512px pass and keep their proportions.
                    self.assertEqual(max(right - left, bottom - top), 512)
                    self.assertAlmostEqual((right - left) / (bottom - top), size[0] / size[1], delta=0.01 * 512)

    def test_tiled_pixels_are_not_resampled(self):
        image = strips((1203, 639))
        canvas, box = clean.prepare(image)
        self.assertEqual(canvas.crop(box).tobytes(), image.tobytes())


class Blending(unittest.TestCase):
    def test_every_canvas_pixel_has_weight(self):
        for total in (513, 520, 832, 1400, 2048):
            weight = np.zeros(total)
            for start in tile_positions(total, 512, 192):
                weight[start:start + 512] += clean.positive_blend_weight(512, 512, 192)[256]
            self.assertTrue((weight > 0).all())
        self.assertGreater(clean.positive_blend_weight(512, 512, 192).min(), 1e-8)

    def test_overlap_ramps_are_retained(self):
        self.assertIs(clean.tiling.make_blend_weight, clean.positive_blend_weight)
        center = clean.positive_blend_weight(512, 512, 192)[256]
        self.assertEqual(center[0], 1e-3)
        self.assertLess(center[50], 0.5)
        self.assertEqual(center[192], 1)


class ColorMatch(unittest.TestCase):
    def test_keeps_only_the_closer_result(self):
        reference = Image.new('RGB', (8, 8), (100, 100, 100))
        source = Image.new('RGB', (8, 8), (110, 110, 110))
        for matched_level, expected in ((130, 110), (101, 101)):
            matched = Image.new('RGB', (8, 8), (matched_level,) * 3)
            with mock.patch('clean._upstream_color_match', return_value=matched):
                self.assertEqual(clean.guarded_color_match(reference, source).getpixel((0, 0))[0], expected)


class Regeneration(unittest.TestCase):
    def test_whole_image_survives_at_its_original_size(self):
        engine = identity_engine()
        for size in SIZES:
            if min(size) < 20:
                continue
            with self.subTest(size=size):
                result = np.asarray(clean.regenerate(engine, strips(size), seed=1), dtype=int)
                self.assertEqual(result.shape[:2], size[::-1])
                height, width = size[1], size[0]
                # Sample inside each band: left and right strips, top and bottom edges, center.
                self.assertGreater(result[height // 2, 0, 0], 200)
                self.assertGreater(result[height // 2, -1, 2], 200)
                self.assertTrue((result[0, width // 2] > 200).all())
                self.assertTrue((result[-1, width // 2] > 200).all())
                self.assertGreater(result[height // 2, width // 2, 1], 200)

    def test_constant_image_gets_no_dark_border(self):
        engine = identity_engine()
        with mock.patch('ctrlregen.engine.color_match', lambda reference, source: source):
            for size in [(1024, 768), (1400, 228), (432, 540), (220, 190)]:
                with self.subTest(size=size):
                    result = np.asarray(clean.regenerate(engine, Image.new('RGB', size, (200, 120, 40)), seed=1))
                    self.assertLessEqual(np.abs(result.astype(int) - (200, 120, 40)).max(), 1)


class Batch(unittest.TestCase):
    def run_batch(self, folder, engine):
        os.chdir(folder)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch('clean.load_engine', return_value=engine) as self.load_engine:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = clean.main()
        return code, out.getvalue(), err.getvalue()

    def setUp(self):
        self.addCleanup(os.chdir, os.getcwd())
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        (self.folder / 'in').mkdir()
        (self.folder / 'out').mkdir()
        strips((220, 190)).save(self.folder / 'in/a.png')
        strips((64, 64)).save(self.folder / 'in/b.webp')

    def test_batch_loads_once_and_keeps_inputs(self):
        before = (self.folder / 'in/a.png').read_bytes()
        code, out, _ = self.run_batch(self.folder, identity_engine())
        self.assertEqual(code, 0)
        self.load_engine.assert_called_once()
        self.assertIn('Done: 2 processed, 0 skipped, 0 failed.', out)
        self.assertIn('seed', out)
        with Image.open(self.folder / 'out/a.png') as a, Image.open(self.folder / 'out/b.webp.png') as b:
            self.assertEqual((a.size, b.size), ((220, 190), (64, 64)))
        self.assertEqual((self.folder / 'in/a.png').read_bytes(), before)

    def test_failure_reports_cause_and_batch_continues(self):
        engine = identity_engine()
        real_run = engine.run
        calls = iter([ValueError('the actual cause')])

        def run(image, **kwargs):
            error = next(calls, None)
            if error:
                raise error
            return real_run(image, **kwargs)

        engine.run = run
        code, out, err = self.run_batch(self.folder, engine)
        self.assertEqual(code, 1)
        self.assertIn('Done: 1 processed, 0 skipped, 1 failed.', out)
        self.assertIn('ValueError: the actual cause', err.splitlines()[-1])
        self.assertIn('Traceback', err)
        self.assertFalse((self.folder / 'out/a.png').exists())
        self.assertTrue((self.folder / 'out/b.webp.png').exists())
        self.assertEqual([p.name for p in (self.folder / 'out').iterdir() if p.name.startswith('.removing')], [])


if __name__ == '__main__':
    unittest.main()
