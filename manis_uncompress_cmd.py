"""Rewrite a JWE3 .manis as UNCOMPRESSED key arrays (dtype compression 0).

JWE3 still reads the pre-ACL format JWE2 used and ships one asset in it
(`hatcheryexitcamera` is dtype 0: no ACL blobs, no database, no limb structures).
Converting a dinosaur bundle to it sidesteps ACL entirely - and the game accepts the
result: the Indoraptor spawns, lives and renders.

The buffers are assembled byte by byte rather than through `ManisFile.save()`, because
the writer drifts on rebuilt blocks - block starts wander by tens of bytes per clip, so
every clip after the first is read from the wrong offset and its channel maps come back
as garbage.

Layout per ManiBlock, verified against the shipped camera bundle:

    names (uint32 into the name buffer) | channel_to_bone (uint8) | bone_to_channel (uint8)
    pad to 4 | PosBones f32[frames][pos][3] | OriBones i16[frames][ori][4] (value * 16384)
    ShrBones f32[frames][scl][2] | SclBones f32[frames][scl][3] | Floats f32[frames][flo]
    pad to 8

`--omit-shear` is the JWE3 scale-layout experiment. Runtime disassembly shows that
the crashing reader consumes a scale float (`0x3F800000`) as a channel-table index,
proving the block walk is misaligned. ShrBones is only evidenced in DLA and PZ; this
option keeps SclBones but leaves ShrBones out.

Blocks are aligned to 16 relative to the start of the keys buffer.

  python manis_uncompress_cmd.py IN.manis --ms2 models.ms2 --out OUT.manis
      [--yaw-clip N --yaw-track N --yaw-deg D]

Then inject with --update.
"""
import argparse
import math
import os
import struct
import sys

import numpy as np

from generated.formats.manis import ManisFile
from generated.formats.manis.acl import decode_file
from modules.helpers import as_bytes
from source.formats.manis.bindpose import read_ms2_bind
from source.formats.manis.splice import list_clip_blobs, read_blob_header

QVVF = 12
MANI_INFO_SIZE = 304
ORI_SCALE = 16384.0
DTYPE_OFFSET = 8            # dtype sits 8 bytes into a ManiInfo
ORI_RELATED_OFFSET = 288    # JWE3 ubyte BoneIndex fields in the 304-byte ManiInfo
ORI_REPEAT_OFFSET = 289
SCL_RELATED_OFFSET = 290
SCL_REPEAT_OFFSET = 291
COMPRESSION_BIT = 1 << 4
HAS_LIST_MASK = 3 << 5


def pad_to(size, alignment):
	return (-size) % alignment


# A qvv row with nothing applied: identity rotation, no translation, unit scale.
IDENTITY_QVV = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)


def extend_bind(bind, tracks):
	"""Grow a bind pose to cover clips authored on a LARGER skeleton.

	`target_bone_count` names the rig a clip was authored on, and it is routinely
	not the species' own: 4 of Acrocanthosaurus' 22 idle-bundle clips declare 172
	bones against its 170-bone rig, and Deinosuchus ships 24 clips built on
	Dimetrodon's 212-bone skeleton. Those extra tracks address bones this species
	does not have, so there is no bind value to substitute - and the engine drops
	them at runtime anyway, because it retargets by bone name.

	Identity is the honest filler: it applies nothing. Clamping the index instead
	would silently graft one bone's rest pose onto another.
	"""
	if tracks <= bind.shape[0]:
		return bind
	padding = np.tile(np.array(IDENTITY_QVV, dtype=bind.dtype),
					  (tracks - bind.shape[0], 1))
	return np.concatenate((bind, padding[:, :bind.shape[1]]))


def close_wrap(values, frame_count, what=""):
	"""Restore the final frame of a wrap-optimised clip.

	A looping clip stores `frame_count - 1` samples because its last frame is
	identical to its first; ACL records this in the blob header as
	`wrap_optimized` and the runtime reinstates the frame. **dtype 0 has no such
	flag.** Writing the short sample array under a header that still declares
	`frame_count` hands the engine one frame less animation than it schedules
	against, so the clip ends early and its state can leave by an exit it would
	never normally reach - an Acrocanthosaurus charging after a preen, as seen in
	Test AG where the ONLY change was this conversion.

	11 of the 22 clips in Acro's idle bundle are wrap-optimised, `standpreen` and
	`standidle01` among them, so this is the common case rather than an edge one.
	"""
	if values is None or frame_count is None:
		return values
	have = values.shape[0]
	if have == frame_count:
		return values
	if have == frame_count - 1:
		return np.concatenate((values, values[:1]))
	# Any other mismatch is not a wrap and must not be papered over.
	logging.warning(
		f"{what or 'clip'} has {have} samples for {frame_count} frames; "
		"not a wrap-optimised off-by-one, leaving as is")
	return values


def fill_defaults(values, bind):
	"""Substitute the bind pose wherever the decoder marked a stripped sub-track."""
	out = values.copy()
	bind = extend_bind(bind, out.shape[1])
	for track in range(out.shape[1]):
		for low, high in ((0, 4), (4, 7), (7, 10)):
			mask = np.isnan(out[:, track, low:high])
			if mask.any():
				out[:, track, low:high][mask] = np.broadcast_to(
					bind[track, low:high], (out.shape[0], high - low))[mask]
	return out


def yaw(values, track, degrees):
	"""Rotate one bone, so a deliberate change can be proven in game."""
	half = math.radians(degrees) / 2.0
	spin = np.array([0.0, math.sin(half), 0.0, math.cos(half)])

	def multiply(a, b):
		x1, y1, z1, w1 = a
		x2, y2, z2, w2 = b
		return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
						 w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
						 w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
						 w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])

	for sample in range(values.shape[0]):
		values[sample, track, 0:4] = multiply(spin, values[sample, track, 0:4].astype(float))
	return values


def resample_frames(values, frame_count):
	"""Linearly resample a frame-major array, preserving its trailing dimensions."""
	values = np.asarray(values)
	if len(values) == frame_count:
		return values.copy()
	old_t = np.linspace(0.0, 1.0, len(values))
	new_t = np.linspace(0.0, 1.0, frame_count)
	flat = values.reshape(len(values), -1)
	out = np.empty((frame_count, flat.shape[1]), dtype=float)
	for column in range(flat.shape[1]):
		out[:, column] = np.interp(new_t, old_t, flat[:, column])
	return out.reshape((frame_count,) + values.shape[1:])


def overlay_blender_clip(samples, scalars, source_mi, overlay_mi):
	"""Put Blender-exported channels into the original full-track sample arrays."""
	frames = int(overlay_mi.frame_count)
	samples = resample_frames(samples, frames)
	keys = overlay_mi.keys
	for data_attr, map_attr, low, high in (
			("ori_bones", "ori_channel_to_bone", 0, 4),
			("pos_bones", "pos_channel_to_bone", 4, 7),
			("scl_bones", "scl_channel_to_bone", 7, 10)):
		data = np.asarray(getattr(keys, data_attr))
		for channel, track in enumerate(getattr(keys, map_attr)):
			samples[:, int(track), low:high] = data[:, channel]

	if scalars is not None:
		scalars = resample_frames(scalars, frames)
		source_names = [str(name) for name in source_mi.keys.floats_names]
		overlay_names = [str(name) for name in keys.floats_names]
		for overlay_channel, name in enumerate(overlay_names):
			if name in source_names:
				scalars[:, source_names.index(name), 0] = keys.floats[:, overlay_channel]
	return samples, scalars


def build_block(mani_info, keys, names_lut, samples, scalars, drop_scale=False,
				frame_count=None, omit_shear=False, scale_remap=None):
	"""Serialise one uncompressed ManiBlock."""
	frames = int(mani_info.frame_count if frame_count is None else frame_count)
	pos_n, ori_n = int(mani_info.pos_bone_count), int(mani_info.ori_bone_count)
	scl_n = 0 if drop_scale else int(mani_info.scl_bone_count)
	flo_n = int(mani_info.float_count)
	out = bytearray()

	name_attrs = ["pos_bones_names", "ori_bones_names"]
	if not drop_scale:
		name_attrs.append("scl_bones_names")
	name_attrs.append("floats_names")
	for attr in name_attrs:
		names = [str(name) for name in getattr(keys, attr)]
		if attr == "scl_bones_names" and scale_remap:
			for channel, (_, name) in scale_remap.items():
				names[channel] = name
		for name in names:
			out += struct.pack("<I", names_lut[str(name)])

	pos_map = [int(x) for x in keys.pos_channel_to_bone]
	ori_map = [int(x) for x in keys.ori_channel_to_bone]
	original_scl_map = [int(x) for x in keys.scl_channel_to_bone]
	scl_map = [] if drop_scale else original_scl_map.copy()
	if scale_remap and not drop_scale:
		for channel, (track, _) in scale_remap.items():
			scl_map[channel] = track
	for values in (pos_map, ori_map, scl_map):
		out += bytes(values)

	bone_to_channel_fields = [
			("pos_bone_to_channel", mani_info.pos_bone_min, mani_info.pos_bone_max),
			("ori_bone_to_channel", mani_info.ori_bone_min, mani_info.ori_bone_max)]
	if not drop_scale:
		bone_to_channel_fields.append(
			("scl_bone_to_channel", mani_info.scl_bone_min, mani_info.scl_bone_max))
	for attr, low, high in bone_to_channel_fields:
		if int(low) <= int(high):
			values = [int(x) & 0xFF for x in getattr(keys, attr)]
			if attr == "scl_bone_to_channel" and scale_remap:
				for channel, (track, _) in scale_remap.items():
					old_track = original_scl_map[channel]
					if not int(low) <= track <= int(high):
						raise ValueError(
							f"remapped scale track {track} is outside "
							f"ManiInfo range {int(low)}..{int(high)}")
					values[old_track - int(low)] = 255
					values[track - int(low)] = channel
			out += bytes(values)
	out += b"\x00" * pad_to(len(out), 4)

	usable = min(frames, samples.shape[0])
	pos = np.zeros((frames, pos_n, 3), dtype="<f4")
	ori = np.zeros((frames, ori_n, 4), dtype="<i2")
	scl = np.zeros((frames, scl_n, 3), dtype="<f4")
	shr = np.ones((frames, scl_n, 2), dtype="<f4")
	flo = np.zeros((frames, flo_n), dtype="<f4")

	for channel, track in enumerate(pos_map):
		pos[:usable, channel] = samples[:usable, track, 4:7]
	for channel, track in enumerate(ori_map):
		quantised = np.clip(samples[:usable, track, 0:4] * ORI_SCALE, -32768, 32767)
		ori[:usable, channel] = np.rint(quantised).astype("<i2")
	for channel, track in enumerate(scl_map):
		scl[:usable, channel] = samples[:usable, track, 7:10]
	# ACL can omit the redundant endpoint while ManiInfo keeps
	# frame_count = duration * rate + 1. Leaving the extra dtype-0 frame at
	# zero collapses the skeleton for one rendered frame at the clip boundary.
	# The compressed runtime clamps past its final sample, so do the same.
	if usable and usable < frames:
		pos[usable:] = pos[usable - 1]
		ori[usable:] = ori[usable - 1]
		scl[usable:] = scl[usable - 1]
	if scalars is not None and flo_n:
		n = min(flo_n, scalars.shape[1])
		float_usable = min(frames, scalars.shape[0])
		flo[:float_usable, :n] = scalars[:float_usable, :n, 0]
		if float_usable and float_usable < frames:
			flo[float_usable:] = flo[float_usable - 1]

	out += pos.tobytes() + ori.tobytes()
	if not omit_shear:
		out += shr.tobytes()
	out += scl.tobytes() + flo.tobytes()
	out += b"\x00" * pad_to(len(out), 8)
	return bytes(out)


def main():
	ap = argparse.ArgumentParser(description=__doc__,
								 formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("manis")
	ap.add_argument("--out", required=True)
	ap.add_argument("--ms2", required=True)
	ap.add_argument("--yaw-clip", type=int)
	ap.add_argument("--yaw-track", type=int)
	ap.add_argument("--yaw-deg", type=float, default=60.0)
	ap.add_argument("--drop-scale", action="store_true",
					help="experimental control: omit scale tables using shipped no-scale conventions")
	ap.add_argument("--omit-shear", action="store_true",
					help="JWE3 experiment: retain SclBones but omit the DLA/PZ ShrBones block")
	ap.add_argument("--keep-stream", action="store_true",
					help="Keep the external stream name so the keys buffer stays in its "
						 "Anim_L* stream instead of becoming resident in STATIC")
	ap.add_argument("--overlay",
					help="Blender-exported dtype-0 MANIS whose named clips replace source samples")
	ap.add_argument("--overlay-clip", action="append", default=[],
					help="clip name (or unique suffix) to take from --overlay; repeatable")
	ap.add_argument("--scale-clip", action="append", default=[],
					help="clip name (or suffix) whose decoded scale samples are multiplied; repeatable")
	ap.add_argument("--scale-track", action="append", type=int, default=[],
					help="skeleton track index to scale in --scale-clip; repeatable")
	ap.add_argument("--scale-factor", type=float, default=1.0,
					help="multiplier applied to selected decoded scale samples")
	ap.add_argument("--scale-value", type=float,
					help="set selected scale samples to this constant instead of multiplying")
	ap.add_argument("--remap-scale-channel", action="append", default=[],
					metavar="CHANNEL:TRACK:NAME",
					help="in --scale-clip, retarget one existing scale channel")
	ap.add_argument("--promote-scale-related", action="store_true",
					help="JWE3 experiment: for --scale-clip, copy the active orientation "
						 "related/repeat metadata into the scale fields")
	args = ap.parse_args()
	if args.drop_scale and args.omit_shear:
		ap.error("--drop-scale and --omit-shear are mutually exclusive")
	if bool(args.scale_clip) != bool(args.scale_track):
		ap.error("--scale-clip and --scale-track must be supplied together")
	scale_remap = {}
	for value in args.remap_scale_channel:
		try:
			channel_text, track_text, name = value.split(":", 2)
			channel, track = int(channel_text), int(track_text)
		except ValueError:
			ap.error(f"invalid --remap-scale-channel {value!r}; expected CHANNEL:TRACK:NAME")
		scale_remap[channel] = (track, name)
	if scale_remap and not args.scale_clip:
		ap.error("--remap-scale-channel requires --scale-clip")
	if args.promote_scale_related and not args.scale_clip:
		ap.error("--promote-scale-related requires --scale-clip")

	raw = open(args.manis, "rb").read()
	headers = [read_blob_header(raw, offset) for offset, _ in list_clip_blobs(raw)]
	transforms = [i for i, h in enumerate(headers) if h["track_type"] == QVVF]
	streams = decode_file(args.manis)
	parents, bind = read_ms2_bind(args.ms2)

	manis = ManisFile()
	manis.load(args.manis)
	overlay_lut = {}
	if args.overlay:
		overlay = ManisFile()
		overlay.load(args.overlay)
		for overlay_mi in overlay.mani_infos:
			name = str(overlay_mi.name)
			if not args.overlay_clip or any(
					name == wanted or name.endswith(wanted) for wanted in args.overlay_clip):
				overlay_lut[name] = overlay_mi
		if args.overlay_clip and len(overlay_lut) != len(args.overlay_clip):
			raise ValueError(
				f"resolved {len(overlay_lut)} overlay clips for {len(args.overlay_clip)} requests")
	count = len(manis.mani_infos)
	print(f"{os.path.basename(args.manis)}: {count} clips, "
		  f"{len(transforms)} transform streams")

	# Preamble, as ManisLoader.extract writes it. The stream name decides where
	# MANI._buffer_layout puts the keys buffer: an empty name makes everything
	# resident in STATIC, while keeping it leaves the clips in their Anim_L*
	# stream the way vanilla ships them. Dropping it also strands the stream's
	# old ACL bulk, which nothing then updates - a suspect for the spurious
	# state exits seen after a conversion (Test AG).
	preamble = struct.pack("<HHI", manis.version, manis.context.mani_version, count)
	stream_name = str(manis.stream or "")
	if args.keep_stream and stream_name:
		preamble += as_bytes(stream_name)
		print(f"  keeping external stream name {stream_name!r}")
	else:
		preamble += b"\x00"
	for name in manis.names:
		preamble += as_bytes(str(name))
	root = as_bytes(manis.header)
	preamble += root

	# buffer 0: the ManiInfo array with compression and has_list cleared, no database
	original_preamble = 8 + len(as_bytes(str(manis.stream or "")))
	for name in manis.names:
		original_preamble += len(as_bytes(str(name)))
	original_preamble += len(root)
	infos = bytearray(raw[original_preamble:original_preamble + count * MANI_INFO_SIZE])
	for index in range(count):
		at = index * MANI_INFO_SIZE + DTYPE_OFFSET
		dtype, = struct.unpack_from("<I", infos, at)
		struct.pack_into("<I", infos, at, dtype & ~(COMPRESSION_BIT | HAS_LIST_MASK))
		overlay_mi = overlay_lut.get(str(manis.mani_infos[index].name))
		if overlay_mi is not None:
			struct.pack_into("<f", infos, index * MANI_INFO_SIZE, float(overlay_mi.duration))
			struct.pack_into("<I", infos, index * MANI_INFO_SIZE + 4,
						 int(overlay_mi.frame_count))
		if args.drop_scale:
			# Shipped no-scale ManiInfos use count=0, min=255, max=0. The dtype-0
			# scale layout has never been game-verified; this control isolates it.
			infos[index * MANI_INFO_SIZE + 28] = 0
			infos[index * MANI_INFO_SIZE + 284] = 255
			infos[index * MANI_INFO_SIZE + 285] = 0
		if args.promote_scale_related and any(
				str(manis.mani_infos[index].name) == wanted
				or str(manis.mani_infos[index].name).endswith(wanted)
				for wanted in args.scale_clip):
			base = index * MANI_INFO_SIZE
			infos[base + SCL_RELATED_OFFSET] = infos[base + ORI_RELATED_OFFSET]
			infos[base + SCL_REPEAT_OFFSET] = infos[base + ORI_REPEAT_OFFSET]
			print(f"  metadata: {manis.mani_infos[index].name} scale related/repeat "
				  f"promoted to {infos[base + SCL_RELATED_OFFSET]}/"
				  f"{infos[base + SCL_REPEAT_OFFSET]}")
	buffer0 = bytes(infos)

	# buffer 1: the name/hash table, untouched
	names_lut = {str(name): i for i, name in enumerate(manis.name_buffer.target_names)}
	buffer1 = as_bytes(manis.name_buffer)

	# buffer 2: the blocks
	buffer2 = bytearray()
	for clip, mani_info in enumerate(manis.mani_infos):
		blob = transforms[clip]
		samples = fill_defaults(streams[blob].values, bind)
		# Do this before any edit below, so everything downstream sees the clip
		# at its declared length.
		samples = close_wrap(samples, int(mani_info.frame_count), str(mani_info.name))
		if args.scale_clip and any(
				str(mani_info.name) == wanted or str(mani_info.name).endswith(wanted)
				for wanted in args.scale_clip):
			clip_scale_remap = scale_remap
			for track in args.scale_track:
				if not 0 <= track < samples.shape[1]:
					raise ValueError(f"scale track {track} is outside 0..{samples.shape[1] - 1}")
				if args.scale_value is None:
					samples[:, track, 7:10] *= args.scale_factor
				else:
					samples[:, track, 7:10] = args.scale_value
			operation = (f"set to {args.scale_value:g}" if args.scale_value is not None
						 else f"multiplied by {args.scale_factor:g}")
			print(f"  edit: {mani_info.name} scale tracks {args.scale_track} {operation}")
		else:
			clip_scale_remap = None
		if args.yaw_clip == clip and args.yaw_track is not None:
			samples = yaw(samples, args.yaw_track, args.yaw_deg)
			print(f"  edit: clip {clip} track {args.yaw_track} "
				  f"rotated {args.yaw_deg:g} deg")
		scalars = None
		if blob + 1 < len(streams) and streams[blob + 1].track_type != QVVF:
			# The scalar stream wraps on the same clip, so it is short by one too.
			scalars = close_wrap(streams[blob + 1].values, int(mani_info.frame_count),
								 f"{mani_info.name} scalars")
		overlay_mi = overlay_lut.get(str(mani_info.name))
		frame_count = None
		if overlay_mi is not None:
			samples, scalars = overlay_blender_clip(
				samples, scalars, mani_info, overlay_mi)
			frame_count = int(overlay_mi.frame_count)
			print(f"  overlay: {mani_info.name} ({frame_count} frames)")
		buffer2 += b"\x00" * pad_to(len(buffer2), 16)
		buffer2 += build_block(
			mani_info, mani_info.keys, names_lut, samples, scalars,
			drop_scale=args.drop_scale, frame_count=frame_count,
			omit_shear=args.omit_shear, scale_remap=clip_scale_remap)
	buffer2 += b"\x00" * pad_to(len(buffer2), 16)

	with open(args.out, "wb") as fh:
		fh.write(preamble + buffer0 + buffer1 + bytes(buffer2))
	print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes; "
		  f"b0={len(buffer0)} b1={len(buffer1)} b2={len(buffer2)})")

	if args.omit_shear:
		# Cobra's provisional reader still expects ShrBones, so a cobra round trip is
		# not a valid gate for this deliberately different layout. The direct builder
		# has already consumed every source clip and emitted one aligned block per clip.
		# The game is the falsifier for this experiment.
		print(f"verify: {count}/{count} blocks emitted; cobra parse intentionally skipped "
			  "because its schema still expects ShrBones")
		return 0

	check = ManisFile()
	check.load(args.out)
	parsed = sum(1 for mani_info in check.mani_infos
				 if getattr(mani_info, "keys", None) is not None)
	dtypes = sorted({int(mani_info.dtype) for mani_info in check.mani_infos})
	bad = sum(1 for mani_info in check.mani_infos
			  if getattr(mani_info, "keys", None) is not None
			  for name in mani_info.keys.ori_bones_names if str(name) == "bad_name")
	maps_ok = all(
		[int(x) for x in check.mani_infos[i].keys.ori_channel_to_bone]
		== [int(x) for x in manis.mani_infos[i].keys.ori_channel_to_bone]
		for i in range(count) if getattr(check.mani_infos[i], "keys", None) is not None)
	blobs = len(list_clip_blobs(open(args.out, "rb").read()))
	print(f"verify: {parsed}/{count} clips parse, dtypes {dtypes}, ACL blobs {blobs}, "
		  f"bad names {bad}, channel maps preserved {maps_ok}")
	return 0 if (parsed == count and bad == 0 and maps_ok) else 1


if __name__ == "__main__":
	sys.exit(main())
