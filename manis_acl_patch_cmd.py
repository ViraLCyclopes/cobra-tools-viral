"""Patch an existing constant or animated range without re-encoding its ACL clip.

The output keeps every blob offset, size, segment, bit rate, database link, and
external sample bits unchanged. Animated vector curves can be multiplied through
their clip-wide range metadata while preserving Frontier's normalized samples.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys

from generated.formats.manis import ManisFile
from generated.formats.ms2 import Ms2File
from source.formats.manis.acl import decode_blob, decode_file
from source.formats.manis.acl_patch import (
	patch_constant,
	read_animated_range,
	read_constant,
	scale_animated_range,
)
from source.formats.manis.database import rekey_database_clip, verify_bulk
from source.formats.manis.splice import list_clip_blobs, read_blob_header


QVVF = 12


def multiply_quat(a, b):
	x1, y1, z1, w1 = a
	x2, y2, z2, w2 = b
	return (
		w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
		w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
		w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
		w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
	)


def resolve_clip(manis, wanted):
	matches = [(index, str(info.name)) for index, info in enumerate(manis.mani_infos)
			   if str(info.name) == wanted or str(info.name).endswith(wanted)]
	if len(matches) != 1:
		raise ValueError(f"clip {wanted!r} resolved to {matches}")
	return matches[0]


def resolve_track(ms2_path, track, bone):
	if track is not None:
		return track, None
	ms2 = Ms2File()
	ms2.load(ms2_path)
	names = [str(item.name) for item in ms2.model_infos[0].bone_info.bones]
	matches = [index for index, name in enumerate(names)
			   if name == bone or name.endswith(bone)]
	if len(matches) != 1:
		raise ValueError(f"bone {bone!r} resolved to {matches}")
	return matches[0], names[matches[0]]


def main():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("manis")
	ap.add_argument("--out", required=True)
	ap.add_argument("--clip", required=True, help="full clip name or unique suffix")
	ap.add_argument("--kind", choices=("rotation", "translation", "scale"),
					required=True)
	group = ap.add_mutually_exclusive_group(required=True)
	group.add_argument("--track", type=int)
	group.add_argument("--bone")
	ap.add_argument("--ms2", help="required with --bone")
	ap.add_argument("--yaw-deg", type=float,
					help="pre-multiply an existing constant rotation around local Y")
	ap.add_argument("--value", type=float, nargs="+",
					help="replacement xyzw rotation or xyz translation/scale")
	ap.add_argument("--range-scale", type=float, nargs="+",
					help="multiply an animated translation/scale curve's xyz range")
	args = ap.parse_args()
	if args.bone and not args.ms2:
		ap.error("--bone requires --ms2")
	if args.yaw_deg is not None and args.kind != "rotation":
		ap.error("--yaw-deg only applies to rotations")
	provided = sum(value is not None
				   for value in (args.yaw_deg, args.value, args.range_scale))
	if provided != 1:
		ap.error("provide exactly one of --yaw-deg, --value, or --range-scale")
	if args.range_scale is not None:
		if args.kind not in ("translation", "scale"):
			ap.error("--range-scale only applies to translation or scale")
		if len(args.range_scale) != 3:
			ap.error("--range-scale requires three xyz factors")

	logging.disable(logging.CRITICAL)
	data = open(args.manis, "rb").read()
	manis = ManisFile()
	manis.load(args.manis)
	clip_index, clip_name = resolve_clip(manis, args.clip)
	track, bone_name = resolve_track(args.ms2, args.track, args.bone)
	blobs = list_clip_blobs(data)
	transforms = [(global_index, offset, size)
				  for global_index, (offset, size) in enumerate(blobs)
				  if read_blob_header(data, offset)["track_type"] == QVVF]
	if clip_index >= len(transforms):
		raise ValueError(f"clip {clip_index} has no transform blob")
	global_index, offset, size = transforms[clip_index]
	blob = data[offset:offset + size]
	if args.range_scale is not None:
		before = read_animated_range(blob, args.kind, track)
		patched = scale_animated_range(
			blob, args.kind, track, tuple(args.range_scale))
		after = read_animated_range(patched, args.kind, track)
	elif args.yaw_deg is not None:
		before = read_constant(blob, args.kind, track)
		half = math.radians(args.yaw_deg) / 2.0
		spin = (0.0, math.sin(half), 0.0, math.cos(half))
		after = multiply_quat(spin, before)
		patched = patch_constant(blob, args.kind, track, after)
	else:
		before = read_constant(blob, args.kind, track)
		after = tuple(args.value)
		patched = patch_constant(blob, args.kind, track, after)
	if len(patched) != len(blob):
		raise AssertionError("constant patch changed ACL blob size")
	out = data[:offset] + patched + data[offset + size:]
	out, database_patch = rekey_database_clip(
		out, patched, read_blob_header(blob)["hash"], read_blob_header(patched)["hash"])
	if len(out) != len(data):
		raise AssertionError("constant patch changed MANIS size")

	# Official ACL must accept the modified hash/payload and expose the requested value.
	decoded = decode_blob(patched)
	component = {"rotation": (0, 4), "translation": (4, 7), "scale": (7, 10)}[args.kind]
	decoded_value = tuple(float(x) for x in decoded.values[0, track, component[0]:component[1]])
	with open(args.out, "wb") as stream:
		stream.write(out)
	ok, message = verify_bulk(args.out)
	if not ok:
		raise RuntimeError(f"database bulk verification failed: {message}")
	# This exercises the database context and both external streamed tiers, not just
	# the standalone compressed_tracks payload.
	decode_file(args.out)
	changed = sum(a != b for a, b in zip(data, out))
	print(f"clip {clip_index} {clip_name}; ACL blob {global_index} @{offset} ({size} bytes)")
	print(f"track {track}{f' {bone_name}' if bone_name else ''} {args.kind}: {before} -> {after}")
	print(f"decoded first sample: {decoded_value}")
	print(f"database clip {database_patch['clip_index']} runtime "
		  f"@{database_patch['clip_header_offset']}: "
		  f"{database_patch['medium_segments']} medium + "
		  f"{database_patch['low_segments']} low segment headers re-keyed")
	print(f"wrote {args.out}: same size {len(out)}, {changed} changed bytes; {message}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
