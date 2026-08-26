import math
import os

import pytest
import numpy as np

from source.formats.manis.acl import decode_file
from source.formats.manis.acl_patch import (
	CONSTANT,
	fnv1a32,
	parse_transform_layout,
	patch_constant,
	read_animated_range,
	read_constant,
	scale_animated_range,
	sub_track_types,
	validate_layout,
)
from source.formats.manis.database import rekey_database_clip, verify_bulk
from source.formats.manis.splice import list_clip_blobs, read_blob_header


MANIS = (r"D:\JWE2 Stuff\Personal Mods\JWE3\Images and Models\Dinosaurs\Land (Base)"
		 r"\Deinosuchus\Female\notmotionextracted.manisetdc504a07.manis")
pytestmark = pytest.mark.skipif(not os.path.isfile(MANIS), reason="reference manis not present")


def standidle_blob():
	data = open(MANIS, "rb").read()
	transforms = [(offset, size) for offset, size in list_clip_blobs(data)
				  if read_blob_header(data, offset)["track_type"] == 12]
	offset, size = transforms[19]
	return data[offset:offset + size]


def rest01_blob():
	data = open(MANIS, "rb").read()
	transforms = [(offset, size) for offset, size in list_clip_blobs(data)
				  if read_blob_header(data, offset)["track_type"] == 12]
	offset, size = transforms[10]
	return data[offset:offset + size]


def test_standidle_layout_matches_header_counts():
	blob = standidle_blob()
	layout = parse_transform_layout(blob)
	validate_layout(blob, layout)
	assert layout.num_tracks == 170
	assert not layout.has_scale
	assert sub_track_types(blob, layout, "rotation").count(CONSTANT) == 10


def test_constant_root_yaw_is_size_preserving_and_rehashed():
	blob = standidle_blob()
	before = read_constant(blob, "rotation", 0)
	half = math.radians(30.0)
	spin = (0.0, math.sin(half), 0.0, math.cos(half))
	x1, y1, z1, w1 = spin
	x2, y2, z2, w2 = before
	after = (
		w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
		w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
		w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
		w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
	)
	patched = patch_constant(blob, "rotation", 0, after)
	assert len(patched) == len(blob)
	assert patched[:4] == blob[:4]
	assert int.from_bytes(patched[4:8], "little") == fnv1a32(patched[8:])
	assert read_constant(patched, "rotation", 0) == pytest.approx(after, abs=1.0e-6)


def test_rejects_animated_head_rotation():
	with pytest.raises(ValueError, match="animated"):
		patch_constant(standidle_blob(), "rotation", 76, (0.0, 0.0, 0.0, 1.0))


def test_rejects_scale_when_clip_has_no_scale_payload():
	with pytest.raises(ValueError, match="no scale"):
		patch_constant(standidle_blob(), "scale", 68, (1.1, 1.1, 1.1))


def test_animated_head_scale_range_is_size_preserving_and_rehashed():
	blob = rest01_blob()
	minimum, extent = read_animated_range(blob, "scale", 81)
	patched = scale_animated_range(blob, "scale", 81, (1.5, 1.5, 1.5))
	patched_minimum, patched_extent = read_animated_range(patched, "scale", 81)
	assert len(patched) == len(blob)
	assert int.from_bytes(patched[4:8], "little") == fnv1a32(patched[8:])
	assert patched_minimum == pytest.approx(tuple(value * 1.5 for value in minimum))
	assert patched_extent == pytest.approx(tuple(value * 1.5 for value in extent))


def test_database_rekey_preserves_every_other_decoded_value(tmp_path):
	data = open(MANIS, "rb").read()
	transforms = [(index, offset, size)
				  for index, (offset, size) in enumerate(list_clip_blobs(data))
				  if read_blob_header(data, offset)["track_type"] == 12]
	global_index, offset, size = transforms[19]
	blob = data[offset:offset + size]
	before = read_constant(blob, "rotation", 0)
	half = math.radians(30.0)
	spin = (0.0, math.sin(half), 0.0, math.cos(half))
	x1, y1, z1, w1 = spin
	x2, y2, z2, w2 = before
	after = (
		w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
		w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
		w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
		w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
	)
	patched_blob = patch_constant(blob, "rotation", 0, after)
	patched_data = data[:offset] + patched_blob + data[offset + size:]
	patched_data, stats = rekey_database_clip(
		patched_data, patched_blob, read_blob_header(blob)["hash"],
		read_blob_header(patched_blob)["hash"])
	assert len(patched_data) == len(data)
	assert stats["clip_index"] == 19
	assert stats["medium_segments"] == 15
	assert stats["low_segments"] == 15

	patched_path = tmp_path / "database_rekey.manis"
	patched_path.write_bytes(patched_data)
	ok, message = verify_bulk(str(patched_path))
	assert ok, message
	original_streams = decode_file(MANIS)
	patched_streams = decode_file(str(patched_path))
	assert len(patched_streams) == len(original_streams)
	for stream_index, (original, modified) in enumerate(
			zip(original_streams, patched_streams)):
		if stream_index != global_index:
			assert np.array_equal(modified.values, original.values, equal_nan=True)
			continue
		assert not np.array_equal(modified.values[:, 0, :4],
							  original.values[:, 0, :4])
		expected = original.values.copy()
		expected[:, 0, :4] = modified.values[:, 0, :4]
		assert np.array_equal(modified.values, expected, equal_nan=True)
