"""PNG colour metadata must not change the bytes packed into a UNORM texture.

Blender, GIMP, Krita and Paint.NET tag their PNGs with an sRGB chunk and/or
gAMA 0.45455. texconv's WIC loader reads such a file as *_UNORM_SRGB and, when
the target codec is plain UNORM, gamma-decodes it on the way in - a UI icon
rendered in Blender came out ~3.5x darker in game (mean 65 -> 18). cobra's own
extraction writes raw values, so the only consistent rule is: UNORM stores the
PNG's bytes verbatim, whatever the file claims about its gamma.
"""
import os
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from modules.formats.utils.dds_conversion import dds_to_png, png_to_dds
from utils.shared import BinaryNotAvailableError, argv_for_binary


def _texconv_available():
	try:
		argv_for_binary("texconv")
		return True
	except BinaryNotAvailableError:
		return False


pytestmark = pytest.mark.skipif(not _texconv_available(), reason="texconv not available")


def _insert_chunk(path, ctype, body):
	"""Insert a PNG chunk directly after IHDR."""
	data = open(path, "rb").read()
	ihdr_end = 8 + 8 + 13 + 4
	chunk = struct.pack(">I", len(body)) + ctype + body
	chunk += struct.pack(">I", zlib.crc32(ctype + body) & 0xffffffff)
	with open(path, "wb") as f:
		f.write(data[:ihdr_end] + chunk + data[ihdr_end:])


def _write_png(path, tag):
	# smooth mid-range gradient, fully opaque: compresses cleanly in every BC codec
	y, x = np.mgrid[0:64, 0:64]
	rgba = np.zeros((64, 64, 4), dtype=np.uint8)
	rgba[..., 0] = 40 + x * 2
	rgba[..., 1] = 40 + y * 2
	rgba[..., 2] = 100
	rgba[..., 3] = 255
	Image.fromarray(rgba, "RGBA").save(path)
	if tag == "srgb":
		_insert_chunk(path, b"sRGB", b"\x00")
	elif tag == "gama":
		_insert_chunk(path, b"gAMA", struct.pack(">I", 45455))
	elif tag == "blender":
		_insert_chunk(path, b"gAMA", struct.pack(">I", 45455))
		_insert_chunk(path, b"sRGB", b"\x00")
	return rgba


def _round_trip(tmp_path, tag, codec):
	png = os.path.join(tmp_path, f"in_{tag}_{codec}.png")
	src = _write_png(png, tag)
	out_dir = os.path.join(tmp_path, "out")
	os.makedirs(out_dir, exist_ok=True)
	dds = png_to_dds(png, out_dir, codec=codec, num_mips=1, dds_use_gpu=False)
	back = np.asarray(Image.open(dds_to_png(dds, codec)).convert("RGBA")).astype(float)
	return src[..., :3].astype(float), back[..., :3]


@pytest.mark.parametrize("codec", ["BC7_UNORM", "BC3_UNORM", "BC1_UNORM"])
@pytest.mark.parametrize("tag", ["none", "srgb", "gama", "blender"])
def test_unorm_stores_png_bytes_verbatim(tmp_path, tag, codec):
	src, back = _round_trip(str(tmp_path), tag, codec)
	# BC1 is 5:6:5, so allow a few levels of quantisation; linearising costs ~60
	assert abs(back.mean() - src.mean()) < 3.0, f"{codec} {tag}: mean {src.mean():.1f} -> {back.mean():.1f}"
	assert np.abs(back - src).mean() < 4.0


@pytest.mark.parametrize("tag", ["none", "blender"])
def test_srgb_codec_round_trips(tmp_path, tag):
	src, back = _round_trip(str(tmp_path), tag, "BC7_UNORM_SRGB")
	assert abs(back.mean() - src.mean()) < 3.0
