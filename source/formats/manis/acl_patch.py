"""Size-preserving edits to constant sub-tracks in an ACL transform blob.

This deliberately keeps the compressed clip's topology, database links, segment
headers, bit rates, and byte size unchanged.  Frontier's runtime rejects blobs
produced by stock ACL even when their decoded motion and observable metadata match;
editing the existing constant-data payload avoids invoking the encoder at all.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from source.formats.manis.splice import QVVF_TRACK_TYPE, read_blob_header


DEFAULT = 0
CONSTANT = 1
ANIMATED = 2
TRANSFORM_HEADER_OFFSET = 32
DROP_W_FORMATS = (2, 3)
SCALAR_TRACK_TYPE = 0
SCALAR_HEADER_OFFSET = 32
BIT_RATE_BITS_V9 = (0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 32)
BIT_RATE_BITS = tuple(range(24)) + (32,)


@dataclass(frozen=True)
class TransformLayout:
	num_tracks: int
	num_segments: int
	num_animated_rotation: int
	num_animated_translation: int
	num_animated_scale: int
	num_constant_rotation: int
	num_constant_translation: int
	num_constant_scale: int
	sub_track_types_offset: int
	constant_track_data_offset: int
	clip_range_data_offset: int
	rotation_format: int
	translation_format: int
	scale_format: int
	has_scale: bool


@dataclass(frozen=True)
class ScalarLayout:
	num_tracks: int
	version: int
	num_bits_per_frame: int
	metadata_offset: int
	constant_values_offset: int
	range_values_offset: int
	animated_values_offset: int


def fnv1a32(data: bytes) -> int:
	value = 2166136261
	for byte in data:
		value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
	return value


def parse_scalar_layout(blob: bytes) -> ScalarLayout:
	"""Parse ACL 2.x float1 track storage used by MANIS scalar channels.

	Offsets in ``scalar_tracks_header`` are relative to that header at byte 32.
	The layout is documented by ACL 2.1's ``compressed_headers.h`` and
	``write_track_data_impl.h``; keeping this separate from transform parsing
	prevents the two unrelated headers from being conflated.
	"""
	header = read_blob_header(blob)
	if header["track_type"] != SCALAR_TRACK_TYPE:
		raise ValueError("ACL blob does not contain float1 scalar tracks")
	if len(blob) != header["size"]:
		raise ValueError(f"ACL blob is {len(blob)} bytes but declares {header['size']}")
	fields = struct.unpack_from("<5I", blob, SCALAR_HEADER_OFFSET)
	layout = ScalarLayout(
		num_tracks=header["num_tracks"],
		version=header["version"],
		num_bits_per_frame=fields[0],
		metadata_offset=fields[1],
		constant_values_offset=fields[2],
		range_values_offset=fields[3],
		animated_values_offset=fields[4],
	)
	offsets = (layout.metadata_offset, layout.constant_values_offset,
			   layout.range_values_offset, layout.animated_values_offset)
	if offsets != tuple(sorted(offsets)):
		raise ValueError(f"scalar data offsets are not monotonic: {offsets}")
	if SCALAR_HEADER_OFFSET + layout.metadata_offset + layout.num_tracks > len(blob):
		raise ValueError("scalar track metadata points outside ACL blob")
	return layout


def scalar_bit_rates(blob: bytes, layout: ScalarLayout | None = None) -> tuple[int, ...]:
	if layout is None:
		layout = parse_scalar_layout(blob)
	base = SCALAR_HEADER_OFFSET + layout.metadata_offset
	rates = tuple(blob[base:base + layout.num_tracks])
	bits_table = BIT_RATE_BITS_V9 if layout.version == 9 else BIT_RATE_BITS
	if any(rate >= len(bits_table) for rate in rates):
		raise ValueError(f"invalid scalar bit rate in {rates}")
	return rates


def _scalar_num_bits(layout: ScalarLayout, bit_rate: int) -> int:
	table = BIT_RATE_BITS_V9 if layout.version == 9 else BIT_RATE_BITS
	return table[bit_rate]


def _scalar_value_offset(blob: bytes, layout: ScalarLayout,
						 track_index: int) -> tuple[str, int]:
	if not 0 <= track_index < layout.num_tracks:
		raise IndexError(f"track {track_index} outside 0..{layout.num_tracks - 1}")
	rates = scalar_bit_rates(blob, layout)
	rate = rates[track_index]
	bits = _scalar_num_bits(layout, rate)
	if bits == 0:
		ordinal = sum(_scalar_num_bits(layout, item) == 0 for item in rates[:track_index])
		return "constant", SCALAR_HEADER_OFFSET + layout.constant_values_offset + ordinal * 4
	if bits == 32:
		raise ValueError(f"scalar track {track_index} uses raw samples and has no range")
	ordinal = sum(0 < _scalar_num_bits(layout, item) < 32 for item in rates[:track_index])
	return "range", SCALAR_HEADER_OFFSET + layout.range_values_offset + ordinal * 8


def read_scalar(blob: bytes, track_index: int) -> tuple[float, ...]:
	"""Return a constant value or an animated track's (minimum, extent)."""
	layout = parse_scalar_layout(blob)
	kind, offset = _scalar_value_offset(blob, layout, track_index)
	if kind == "constant":
		return (struct.unpack_from("<f", blob, offset)[0],)
	return struct.unpack_from("<2f", blob, offset)


def patch_scalar_constant(blob: bytes, track_index: int, value: float) -> bytes:
	if not math.isfinite(value):
		raise ValueError("scalar value must be finite")
	layout = parse_scalar_layout(blob)
	kind, offset = _scalar_value_offset(blob, layout, track_index)
	if kind != "constant":
		raise ValueError(f"scalar track {track_index} is animated, not constant")
	out = bytearray(blob)
	struct.pack_into("<f", out, offset, value)
	_rehash(out)
	return bytes(out)


def scale_scalar_range(blob: bytes, track_index: int, factor: float,
					   pivot: float = 0.0) -> bytes:
	"""Multiply one quantized float1 curve through its track-wide min/extent."""
	if not math.isfinite(factor) or factor <= 0.0:
		raise ValueError("scalar range factor must be finite and positive")
	if not math.isfinite(pivot):
		raise ValueError("scalar range pivot must be finite")
	layout = parse_scalar_layout(blob)
	kind, offset = _scalar_value_offset(blob, layout, track_index)
	if kind != "range":
		raise ValueError(f"scalar track {track_index} is constant, not animated")
	minimum, extent = struct.unpack_from("<2f", blob, offset)
	out = bytearray(blob)
	struct.pack_into("<2f", out, offset,
				 pivot + (minimum - pivot) * factor, extent * factor)
	_rehash(out)
	return bytes(out)


def parse_transform_layout(blob: bytes) -> TransformLayout:
	header = read_blob_header(blob)
	if header["track_type"] != QVVF_TRACK_TYPE:
		raise ValueError("ACL blob does not contain QVV transform tracks")
	if len(blob) != header["size"]:
		raise ValueError(f"ACL blob is {len(blob)} bytes but declares {header['size']}")
	fields = struct.unpack_from("<13I", blob, TRANSFORM_HEADER_OFFSET)
	layout = TransformLayout(
		num_tracks=header["num_tracks"],
		num_segments=fields[0],
		num_animated_rotation=fields[2],
		num_animated_translation=fields[3],
		num_animated_scale=fields[4],
		num_constant_rotation=fields[5],
		num_constant_translation=fields[6],
		num_constant_scale=fields[7],
		sub_track_types_offset=fields[10],
		constant_track_data_offset=fields[11],
		clip_range_data_offset=fields[12],
		rotation_format=header["rotation_format"],
		translation_format=(header["misc_packed"] >> 3) & 1,
		scale_format=(header["misc_packed"] >> 2) & 1,
		has_scale=header["has_scale"],
	)
	if layout.rotation_format not in (0, *DROP_W_FORMATS):
		raise ValueError(f"unsupported ACL rotation format {layout.rotation_format}")
	return layout


def _packed_entry_count(num_tracks: int) -> int:
	return (num_tracks + 15) // 16


def sub_track_types(blob: bytes, layout: TransformLayout, kind: str) -> tuple[int, ...]:
	kinds = ("rotation", "translation") + (("scale",) if layout.has_scale else ())
	if kind not in kinds:
		if kind == "scale":
			raise ValueError("ACL clip has no scale payload")
		raise ValueError(f"unknown sub-track kind {kind!r}")
	entry_count = _packed_entry_count(layout.num_tracks)
	kind_index = kinds.index(kind)
	base = (TRANSFORM_HEADER_OFFSET + layout.sub_track_types_offset
			+ kind_index * entry_count * 4)
	out = []
	for entry in range(entry_count):
		word, = struct.unpack_from("<I", blob, base + entry * 4)
		out.extend((word >> ((15 - lane) * 2)) & 3 for lane in range(16))
	result = tuple(out[:layout.num_tracks])
	if any(value > ANIMATED for value in result):
		raise ValueError(f"invalid packed {kind} sub-track type")
	return result


def validate_layout(blob: bytes, layout: TransformLayout) -> None:
	for kind, expected_constant, expected_animated in (
			("rotation", layout.num_constant_rotation, layout.num_animated_rotation),
			("translation", layout.num_constant_translation, layout.num_animated_translation),
			("scale", layout.num_constant_scale, layout.num_animated_scale)):
		if kind == "scale" and not layout.has_scale:
			if expected_constant or expected_animated:
				raise ValueError("scale counts are non-zero while has_scale is false")
			continue
		types = sub_track_types(blob, layout, kind)
		if types.count(CONSTANT) != expected_constant:
			raise ValueError(
				f"{kind} constant count {types.count(CONSTANT)} != {expected_constant}")
		if types.count(ANIMATED) != expected_animated:
			raise ValueError(
				f"{kind} animated count {types.count(ANIMATED)} != {expected_animated}")


def _rotation_sample_size(layout: TransformLayout) -> int:
	return 16 if layout.rotation_format == 0 else 12


def _constant_data_base(layout: TransformLayout) -> int:
	return TRANSFORM_HEADER_OFFSET + layout.constant_track_data_offset


def _constant_vector_offset(blob: bytes, layout: TransformLayout,
							kind: str, track_index: int) -> int:
	if not 0 <= track_index < layout.num_tracks:
		raise IndexError(f"track {track_index} outside 0..{layout.num_tracks - 1}")
	types = sub_track_types(blob, layout, kind)
	if types[track_index] != CONSTANT:
		labels = {DEFAULT: "default", CONSTANT: "constant", ANIMATED: "animated"}
		raise ValueError(
			f"{kind} track {track_index} is {labels.get(types[track_index], 'invalid')}, "
			"not constant")
	ordinal = types[:track_index].count(CONSTANT)
	base = _constant_data_base(layout)
	base += layout.num_constant_rotation * _rotation_sample_size(layout)
	if kind == "translation":
		return base + ordinal * 12
	if kind == "scale":
		base += layout.num_constant_translation * 12
		return base + ordinal * 12
	raise ValueError(f"{kind} is not a vector constant kind")


def _constant_rotation_offsets(blob: bytes, layout: TransformLayout,
								track_index: int) -> tuple[int, ...]:
	if not 0 <= track_index < layout.num_tracks:
		raise IndexError(f"track {track_index} outside 0..{layout.num_tracks - 1}")
	types = sub_track_types(blob, layout, "rotation")
	if types[track_index] != CONSTANT:
		labels = {DEFAULT: "default", CONSTANT: "constant", ANIMATED: "animated"}
		raise ValueError(
			f"rotation track {track_index} is {labels.get(types[track_index], 'invalid')}, "
			"not constant")
	ordinal = types[:track_index].count(CONSTANT)
	base = _constant_data_base(layout)
	if layout.rotation_format == 0:
		start = base + ordinal * 16
		return (start, start + 4, start + 8, start + 12)
	group_start = (ordinal // 4) * 4
	group_size = min(4, layout.num_constant_rotation - group_start)
	lane = ordinal - group_start
	group_base = base + (group_start // 4) * 48
	return tuple(group_base + component * group_size * 4 + lane * 4
				 for component in range(3))


def read_constant(blob: bytes, kind: str, track_index: int) -> tuple[float, ...]:
	layout = parse_transform_layout(blob)
	validate_layout(blob, layout)
	if kind == "rotation":
		offsets = _constant_rotation_offsets(blob, layout, track_index)
		stored = tuple(struct.unpack_from("<f", blob, offset)[0] for offset in offsets)
		if len(stored) == 4:
			return stored
		x, y, z = stored
		w = math.sqrt(max(0.0, 1.0 - x * x - y * y - z * z))
		return x, y, z, w
	offset = _constant_vector_offset(blob, layout, kind, track_index)
	return struct.unpack_from("<3f", blob, offset)


def _rehash(blob: bytearray) -> None:
	struct.pack_into("<I", blob, 4, fnv1a32(blob[8:]))


def patch_constant(blob: bytes, kind: str, track_index: int,
					value: tuple[float, ...]) -> bytes:
	"""Return a same-size blob with one existing constant sample replaced."""
	layout = parse_transform_layout(blob)
	validate_layout(blob, layout)
	out = bytearray(blob)
	if kind == "rotation":
		if len(value) != 4:
			raise ValueError("rotation value must be xyzw")
		length = math.sqrt(sum(component * component for component in value))
		if not math.isfinite(length) or length < 1.0e-8:
			raise ValueError("rotation must be a finite non-zero quaternion")
		value = tuple(component / length for component in value)
		if layout.rotation_format in DROP_W_FORMATS and value[3] < 0.0:
			value = tuple(-component for component in value)
		offsets = _constant_rotation_offsets(blob, layout, track_index)
		for offset, component in zip(offsets, value):
			struct.pack_into("<f", out, offset, component)
	else:
		if len(value) != 3 or not all(math.isfinite(component) for component in value):
			raise ValueError(f"{kind} value must contain three finite floats")
		offset = _constant_vector_offset(blob, layout, kind, track_index)
		struct.pack_into("<3f", out, offset, *value)
	_rehash(out)
	return bytes(out)


def _animated_vector_range_offset(blob: bytes, layout: TransformLayout,
								  kind: str, track_index: int) -> int:
	if kind not in ("translation", "scale"):
		raise ValueError("animated range patch supports translation or scale vectors")
	if kind == "scale" and not layout.has_scale:
		raise ValueError("ACL clip has no scale payload")
	format_value = (layout.translation_format if kind == "translation"
					else layout.scale_format)
	if format_value != 1:
		raise ValueError(f"{kind} format is not variable and has no clip range")
	if not 0 <= track_index < layout.num_tracks:
		raise IndexError(f"track {track_index} outside 0..{layout.num_tracks - 1}")
	types = sub_track_types(blob, layout, kind)
	if types[track_index] != ANIMATED:
		labels = {DEFAULT: "default", CONSTANT: "constant", ANIMATED: "animated"}
		raise ValueError(
			f"{kind} track {track_index} is {labels.get(types[track_index], 'invalid')}, "
			"not animated")
	ordinal = types[:track_index].count(ANIMATED)
	base = TRANSFORM_HEADER_OFFSET + layout.clip_range_data_offset
	if layout.rotation_format == 3:  # quatf_drop_w_variable
		base += layout.num_animated_rotation * 24
	if kind == "scale" and layout.translation_format == 1:
		base += layout.num_animated_translation * 24
	offset = base + ordinal * 24
	if offset + 24 > len(blob):
		raise ValueError(f"{kind} clip range points outside ACL blob")
	return offset


def read_animated_range(blob: bytes, kind: str,
						track_index: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
	"""Return the clip-wide (minimum, extent) for an animated vector track."""
	layout = parse_transform_layout(blob)
	validate_layout(blob, layout)
	offset = _animated_vector_range_offset(blob, layout, kind, track_index)
	return (struct.unpack_from("<3f", blob, offset),
			struct.unpack_from("<3f", blob, offset + 12))


def scale_animated_range(blob: bytes, kind: str, track_index: int,
						 factor: tuple[float, ...],
						 pivot: tuple[float, ...] = (0.0, 0.0, 0.0)) -> bytes:
	"""Multiply an animated vector curve without changing its normalized samples."""
	if len(factor) != 3 or not all(math.isfinite(value) and value > 0.0
								  for value in factor):
		raise ValueError("range factor must contain three finite positive floats")
	if len(pivot) != 3 or not all(math.isfinite(value) for value in pivot):
		raise ValueError("range pivot must contain three finite floats")
	layout = parse_transform_layout(blob)
	validate_layout(blob, layout)
	offset = _animated_vector_range_offset(blob, layout, kind, track_index)
	minimum, extent = read_animated_range(blob, kind, track_index)
	out = bytearray(blob)
	struct.pack_into("<3f", out, offset,
				 *(origin + (value - origin) * multiplier
				   for value, multiplier, origin in zip(minimum, factor, pivot)))
	struct.pack_into("<3f", out, offset + 12,
				 *(value * multiplier for value, multiplier in zip(extent, factor)))
	_rehash(out)
	return bytes(out)
