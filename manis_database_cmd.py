"""Rebuild a JWE3 .manis with a real ACL database (Approach E).

The game rejects self-contained clips. Re-encoding every clip with
has_database = false crashes on spawn whether the original database is kept or removed,
while the untouched bundle and a no-op re-inject both load, so clips have to stay
database-backed. This compresses every transform clip, hands the set to ACL's
build_database(), splits the bulk out of line with split_database_bulk_data(), and
reassembles the bundle in JWE3's layout:

    [preamble][buffer 0: ManiInfo array + compressed_database][buffer 1][buffer 2: clips]
    [low tier bulk][medium tier bulk]

which is what ManisLoader.extract() writes and what MANI.py splits back into the STATIC
entry plus the _L0 (low) and _L1 (medium) data entries.

  python manis_database_cmd.py IN.manis --ms2 models.ms2 --out OUT.manis

Then inject it with --update; an ordinary inject reallocates pools and data entries and
the game rejects the result.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

from generated.formats.manis.acl import decode_file, encode_tracks, decode_blob, _write_jacl
from source.formats.manis.bindpose import (
	bind_bytes, clip_defaults, extend_bind_pose, read_ms2_bind, read_ms2_bone_names,
	write_jbind)
from source.formats.manis.database import (
	check_name_buffer, find_database, read_bulk_info, locate_bulk)
from source.formats.manis.limbs import block_layout, buffer_residue, rebuild_block
from source.formats.manis.selfcontained import count_parsed_maniblocks
from source.formats.manis.splice import list_clip_blobs, read_blob_header

QVVF = 12
BULK_ALIGNMENT = 16
MAX_ROTATION_DEGREES = 0.5


def builder_path():
	override = os.environ.get("COBRA_ACL_DATABASE")
	if override:
		return override
	here = os.path.dirname(os.path.abspath(__file__))
	return os.path.join(here, "bin", "jwe3_acl_database.exe")


def pad_to(data, alignment=BULK_ALIGNMENT):
	over = len(data) % alignment
	return data if not over else data + b"\x00" * (alignment - over)


def build_database(streams, headers, bind, work_dir, parents, bind_values,
				   precision=None, medium_proportion=None, low_proportion=None,
				   dense=()):
	"""Run the ACL database builder over every transform clip.

	Returns (bound_blobs_by_stream_index, database_bytes, low_bulk, medium_bulk).
	"""
	manifest_lines = []
	order = []
	for index, stream in enumerate(streams):
		if stream.track_type != QVVF:
			continue
		slot = len(order)
		jacl = os.path.join(work_dir, f"clip.{slot}.jacl")
		_write_jacl(jacl, stream.values, stream.track_type, stream.sample_rate)
		# Defaults are per clip, not per bundle: they decide which sub-tracks ACL
		# strips, and a stripped sub-track is filled in by the game rather than by
		# the file. clip_defaults() derives them from the vanilla decode so the
		# stripped set comes out identical to what Frontier shipped.
		clip_bind = os.path.join(work_dir, f"clip.{slot}.jbind")
		tracks = stream.values.shape[1]
		clip_parents = parents[:tracks]
		write_jbind(clip_bind, clip_parents,
					clip_defaults(stream.values, bind_values, keep_all=index in dense))
		manifest_lines.append(
			f"{jacl}|{1 if headers[index]['wrap_optimized'] else 0}|{clip_bind}")
		order.append(index)
	if not order:
		sys.exit("no transform clips in this bundle; nothing to build a database from")

	manifest = os.path.join(work_dir, "manifest.txt")
	with open(manifest, "w", encoding="utf-8") as fh:
		fh.write("\n".join(manifest_lines))
	bind_file = os.path.join(work_dir, "bind.jbind")
	with open(bind_file, "wb") as fh:
		fh.write(bind)

	out_dir = os.path.join(work_dir, "out")
	os.makedirs(out_dir, exist_ok=True)
	builder = builder_path()
	if not os.path.isfile(builder):
		sys.exit(f"ACL database builder not found at '{builder}'. "
				 f"Build acl_decoder/build_database.cmd or set COBRA_ACL_DATABASE.")
	command = [builder, manifest, out_dir, "--bind", bind_file]
	if precision is not None:
		command += ["--precision", repr(precision)]
	if medium_proportion is not None:
		command += ["--medium", repr(medium_proportion)]
	if low_proportion is not None:
		command += ["--low", repr(low_proportion)]
	result = subprocess.run(command, check=False, capture_output=True, text=True)
	if result.returncode != 0:
		sys.exit(f"ACL database build failed: {result.stderr.strip() or result.stdout.strip()}")
	print(result.stdout.rstrip())

	bound = {}
	for position, index in enumerate(order):
		with open(os.path.join(out_dir, f"clip.{position}.blob"), "rb") as fh:
			bound[index] = fh.read()
	with open(os.path.join(out_dir, "database.bin"), "rb") as fh:
		database = fh.read()
	with open(os.path.join(out_dir, "bulk_low.bin"), "rb") as fh:
		low = fh.read()
	with open(os.path.join(out_dir, "bulk_medium.bin"), "rb") as fh:
		medium = fh.read()
	return bound, database, low, medium


def apply_yaw(streams, headers, bind, clip, track, degrees):
	"""Rotate one bone on one clip, so a deliberate change can be proven in game.

	A default sub-track decodes to NaN, so the bind value is substituted first -
	otherwise the rotation is applied to nothing and the clip comes back unchanged.
	"""
	import math

	transforms = [i for i, h in enumerate(headers) if h["track_type"] == QVVF]
	if not 0 <= clip < len(transforms):
		sys.exit(f"clip {clip} out of range (bundle has {len(transforms)})")
	index = transforms[clip]
	values = streams[index].values.copy()
	if not 0 <= track < values.shape[1]:
		sys.exit(f"track {track} out of range (clip has {values.shape[1]})")

	missing = np.isnan(values[:, track, 0:4])
	if missing.any():
		values[:, track, 0:4][missing] = np.broadcast_to(
			bind[track, 0:4], (values.shape[0], 4))[missing]

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
	streams[index].values = values
	print(f"  edit: clip {clip} (blob {index}) track {track} rotated {degrees:g} deg "
		  f"on {values.shape[0]} samples")
	return streams


def scale_motion_track(manis_path, streams, headers, clip_name, factor):
	"""Multiply one clip's 'Z Motion Track' scalar, its baked world displacement.

	Locomotion clips are motion-extracted: the root motion is not in the transform
	tracks, it is this scalar curve, and the engine advances the entity along it.
	That is why playing a clip faster does not make the animal travel further - the
	declared distance is unchanged, so the entity is moved the same amount and
	re-synchronised.

	Scaling is applied as a delta from the curve's first sample, so the clip still
	starts where it started and only the distance covered changes.
	"""
	from generated.formats.manis import ManisFile

	manis = ManisFile()
	manis.load(manis_path)
	transforms = [i for i, h in enumerate(headers) if h["track_type"] == QVVF]
	for order, info in enumerate(manis.mani_infos):
		if info.name != clip_name:
			continue
		names = [str(n) for n in getattr(info.keys, "floats_names", [])]
		if "Z Motion Track" not in names:
			sys.exit(f"clip '{clip_name}' has no 'Z Motion Track' channel; it is not "
					 f"motion-extracted. Channels: {names}")
		channel = names.index("Z Motion Track")
		# the scalar stream sits immediately after its clip's transform stream
		index = transforms[order] + 1
		if index >= len(streams) or streams[index].track_type == QVVF:
			sys.exit(f"clip '{clip_name}' has no scalar stream to edit")
		values = streams[index].values.copy()
		column = values[:, channel, 0]
		finite = np.isfinite(column)
		if not finite.any():
			sys.exit(f"clip '{clip_name}' Z Motion Track is empty")
		origin = column[finite][0]
		before = float(column[finite][-1] - origin)
		values[:, channel, 0] = np.where(
			finite, origin + (column - origin) * factor, column)
		streams[index].values = values
		after = before * factor
		print(f"  motion: '{clip_name}' travel {before:.3f} -> {after:.3f} "
			  f"over {info.duration:.4f}s  ({before / info.duration:.2f} -> "
			  f"{after / info.duration:.2f} m/s)")
		print(f"  set SpeciesAnimation.WalkSpeed/RunSpeed to {after / info.duration:.2f} "
			  f"to keep the engine's reference in step")
		return streams
	sys.exit(f"no clip named '{clip_name}' in this bundle")


def resample_clip(manis_path, streams, headers, clip_name, factor):
	"""Genuinely shorten a clip by dropping frames, not by lying about its timing.

	Every other way of speeding a clip up desynchronises two fields that must
	agree. `ManiInfo.duration` is what the state machine schedules against;
	`sample_rate` is how ACL maps time to sample index. Editing duration alone
	changes nothing but the transition; editing sample_rate makes the decoder run
	off the end of the clip and the entity twitches even while the game is PAUSED.
	The motiongraph `speed` field works but leaves duration stale, and the state
	then exits by an edge it should never reach - the Acrocanthosaurus charge.

	Resampling avoids all of it: keep sample_rate, keep the authored motion, just
	store fewer frames. The caller must then set ManiInfo frame_count and duration
	to match, so num_samples / sample_rate == duration and nothing is inconsistent.

	factor 0.5 halves the frames, so the clip plays twice as fast.
	"""
	from generated.formats.manis import ManisFile

	if not 0.05 <= factor <= 1.0:
		sys.exit(f"resample factor {factor} outside 0.05..1.0; only shortening is supported")
	manis = ManisFile()
	manis.game = "Jurassic World Evolution 3"
	manis.load(manis_path)
	transforms = [i for i, h in enumerate(headers) if h["track_type"] == QVVF]
	names = [str(i.name) for i in manis.mani_infos]
	if clip_name not in names:
		sys.exit(f"no clip named '{clip_name}' in this bundle")
	order = names.index(clip_name)
	first = transforms[order]
	targets = [first]
	if first + 1 < len(headers) and headers[first + 1]["track_type"] != QVVF:
		targets.append(first + 1)

	before = streams[first].values.shape[0]
	keep = np.unique(np.rint(np.linspace(0, before - 1,
										 max(2, int(round(before * factor))))).astype(int))
	for index in targets:
		streams[index].values = streams[index].values[keep]
	rate = streams[first].sample_rate
	samples = len(keep)

	# Measured across all 22 clips of Acro's idle bundle, exactly:
	#   wrap_optimized : frame_count = samples + 1, duration = samples / rate
	#   not wrapped    : frame_count = samples,     duration = (samples - 1) / rate
	# A wrap-optimised clip's final frame is implicit and equal to its first, which
	# is why it declares one more frame than it stores.
	wrap = headers[first]["wrap_optimized"]
	frame_count = samples + 1 if wrap else samples
	duration = (samples if wrap else samples - 1) / rate
	print(f"  resample: '{clip_name}' {before} -> {samples} samples "
		  f"at an unchanged {rate:.4f} fps  (wrap_optimized={wrap})")
	return streams, frame_count, duration


def retime_mani_info(data, clip_name, frames, duration):
	"""Write a resampled clip's new frame_count and duration into its ManiInfo.

	ManiInfo is 304 bytes: duration is a float at +0, frame_count a uint32 at +4.
	Both have to match the resampled blob, because they are read by different
	consumers - duration by the state machine, the sample count by ACL - and the
	whole point of resampling is that nothing disagrees.
	"""
	import struct

	from generated.formats.manis import ManisFile
	from modules.helpers import as_bytes

	# ManisFile has no bytes loader, so round-trip through a scratch file to get the
	# parsed preamble size; the byte offsets are then checked against what it decoded
	with tempfile.NamedTemporaryFile(suffix=".manis", delete=False) as tmp:
		tmp.write(data)
		scratch = tmp.name
	try:
		manis = ManisFile()
		manis.game = "Jurassic World Evolution 3"
		manis.load(scratch)
		names = [str(i.name) for i in manis.mani_infos]
		if clip_name not in names:
			sys.exit(f"cannot retime '{clip_name}': not in the rebuilt bundle")
		index = names.index(clip_name)
		size = 8 + len(as_bytes(str(manis.stream or "")))
		for name in manis.names:
			size += len(as_bytes(str(name)))
		size += len(as_bytes(manis.header))
		base = size + index * 304
		out = bytearray(data)
		was_duration = struct.unpack_from("<f", out, base)[0]
		was_frames = struct.unpack_from("<I", out, base + 4)[0]
		if abs(was_duration - float(manis.mani_infos[index].duration)) > 1e-6:
			sys.exit(f"ManiInfo offset check failed for {clip_name}: bytes say "
					 f"{was_duration}, parser says {manis.mani_infos[index].duration}")
		struct.pack_into("<f", out, base, duration)
		struct.pack_into("<I", out, base + 4, frames)
		print(f"  ManiInfo: '{clip_name}' duration {was_duration:.4f} -> {duration:.4f}s, "
			  f"frame_count {was_frames} -> {frames}")
		return bytes(out)
	finally:
		os.unlink(scratch)


def read_jacl(path):
	"""Read a JACL sample array - the interchange the Blender exporter writes."""
	import struct as _struct

	with open(path, "rb") as fh:
		raw = fh.read()
	if len(raw) < 28 or raw[:4] != b"JACL":
		sys.exit(f"{path} is not a JACL file")
	_version, track_type, tracks, samples, comps = _struct.unpack_from("<IIIII", raw, 4)
	rate, = _struct.unpack_from("<f", raw, 24)
	need = samples * tracks * comps
	values = np.frombuffer(raw, dtype="<f4", count=need, offset=28)
	return values.reshape(samples, tracks, comps).copy(), track_type, rate


def resolve_tracks(spec, bone_names):
	"""Turn a comma-separated list of bone names and/or track indices into indices."""
	out = []
	for token in spec.split(","):
		token = token.strip()
		if not token:
			continue
		if token.lstrip("-").isdigit():
			out.append(int(token))
			continue
		if token not in bone_names:
			sys.exit(f"--hold-track: no bone named '{token}'. The skeleton has "
					 f"{len(bone_names)} bones.")
		out.append(bone_names.index(token))
	return sorted(set(out))


def replace_clip_samples(manis_path, streams, headers, clip_name, jacl_path,
						 hold=()):
	"""Swap one clip's animation for samples authored outside the game.

	This is the Blender export path. The bundle is NOT rebuilt from scratch -
	`ManisFile.save()` would drop the ACL database, the compressed blobs and the
	limb data, and it can only write uncompressed dtype 0, which the charge bug and
	the 9x size blowup make unusable. Instead the vanilla bundle stays the template
	and one clip's samples are replaced inside it, so every ManiInfo, channel map,
	name table and limb structure is carried through untouched.

	The track count must match: a clip's channel list lives in the resident
	ManiBlock, and growing it is a different (unsolved) problem. Sample COUNT may
	differ - that is a retime, and the caller fixes frame_count and duration.
	"""
	from generated.formats.manis import ManisFile

	manis = ManisFile()
	manis.game = "Jurassic World Evolution 3"
	manis.load(manis_path)
	names = [str(i.name) for i in manis.mani_infos]
	if clip_name not in names:
		sys.exit(f"no clip named '{clip_name}' in this bundle. Available: {names}")
	order = names.index(clip_name)
	transforms = [i for i, h in enumerate(headers) if h["track_type"] == QVVF]
	index = transforms[order]

	values, track_type, rate = read_jacl(jacl_path)
	wrap = headers[index]["wrap_optimized"]
	if track_type != QVVF:
		sys.exit(f"{jacl_path} holds track type {track_type}, expected {QVVF} (qvvf)")
	have = streams[index].values.shape[1]
	if values.shape[1] != have:
		sys.exit(f"'{clip_name}' has {have} tracks, {os.path.basename(jacl_path)} has "
				 f"{values.shape[1]}. The channel list is fixed by the resident "
				 f"ManiBlock; export against this clip's own skeleton.")
	before = streams[index].values.shape[0]

	# A wrap-optimised clip's final frame is IMPLICIT and equal to its first, so it
	# declares one more frame than it stores. Blender imports frame_count keys, so an
	# export of such a clip comes back with exactly one sample too many. Drop it here
	# rather than let it read as a retime and silently lengthen the clip. Only the
	# exact off-by-one is treated this way; any other change is a real retime.
	if wrap and values.shape[0] == before + 1:
		head, tail = values[0], values[-1]
		finite = np.isfinite(head) & np.isfinite(tail)
		drift = float(np.abs(head[finite] - tail[finite]).max()) if finite.any() else 0.0
		values = values[:-1]
		print(f"           wrap-optimised: dropped the implicit last frame "
			  f"(it differs from the first by {drift:.4f})")

	# Blender has no concept of a stripped sub-track: every bone always has a pose,
	# so a .jacl exported from it is fully dense. Storing all of that would change
	# the stripped set, which is a CONTRACT with the runtime - the game supplies
	# stripped values itself, and storing more than vanilla did is what makes
	# animals crush and stretch (see ACL_REENCODE_2026-08-29.md section 3).
	# So re-apply the template's NaN mask: whatever Frontier left to the game stays
	# left to the game. The consequence is that a bone vanilla never stored cannot
	# be animated by this path.
	template = streams[index].values
	incoming = values
	if incoming.shape[0] == template.shape[0]:
		mask = np.isnan(template)
	else:
		# retimed: resample the mask along time, nearest sample
		src = np.rint(np.linspace(0, template.shape[0] - 1, incoming.shape[0])).astype(int)
		mask = np.isnan(template)[src]
	dropped = int((mask & ~np.isnan(incoming)).sum())
	incoming = np.where(mask, np.float32("nan"), incoming)
	if dropped:
		print(f"           {dropped} components discarded to preserve the vanilla "
			  f"stripped set ({mask.mean():.1%} of the clip is game-supplied)")

	# The exporter writes NaN for a bone it could not author - one whose Blender
	# fcurves hold a single identity key because the rig reproduces it through a
	# constraint rather than through keys. Taking those samples would overwrite a
	# real curve with a frozen identity pose, which is what happened to
	# def_rearLegUpr_joint.L: vanilla stores that track, so the stripped-set mask
	# above does NOT catch it. Fall back to the template wherever the export
	# declined to speak, but only where the template has something to say.
	unauthored = np.isnan(incoming) & ~mask
	if unauthored.any():
		incoming = np.where(unauthored, template if incoming.shape[0] == template.shape[0]
							else template[np.rint(np.linspace(
								0, template.shape[0] - 1, incoming.shape[0])).astype(int)],
							incoming)
		bones = int(np.unique(np.where(unauthored)[1]).size)
		print(f"           {bones} bone(s) not authored in Blender kept their "
			  f"template animation")
	# Some bones cannot survive a Blender round trip. `srb` and the LOD nodes
	# (Deinosuchus_Female_L0..L5 and their equivalents) are PARENTLESS bones whose
	# imported rest matrix already carries the ManisCorrector's -90 degrees about X,
	# where def_c_root_joint - also parentless - sits at identity. Exporting them
	# applies the correction a second time, so they come back a constant 90 degrees
	# out on every frame. Vanilla leaves all of them stripped; they only became
	# writable at all because --dense-clip un-strips everything, and srb is the
	# motion-extraction node, so authoring it from a pose is never intended.
	# Holding them at the template is therefore the correct answer, not a patch over
	# an exporter bug: 42 of the 49 identity-rotation bones round-trip cleanly and
	# only these seven do not.
	for track in hold:
		if not 0 <= track < template.shape[1]:
			sys.exit(f"--hold-track: track {track} is out of range "
					 f"(clip has {template.shape[1]})")
		if incoming.shape[0] == template.shape[0]:
			incoming[:, track] = template[:, track]
		else:
			src = np.rint(np.linspace(0, template.shape[0] - 1,
									  incoming.shape[0])).astype(int)
			incoming[:, track] = template[src, track]
	if hold:
		print(f"           held {len(hold)} track(s) at the template: "
			  f"{', '.join(str(t) for t in hold)}")
	streams[index].values = incoming
	samples = values.shape[0]
	frame_count = samples + 1 if wrap else samples
	duration = (samples if wrap else samples - 1) / streams[index].sample_rate
	print(f"  replace: '{clip_name}' {before} -> {samples} samples from "
		  f"{os.path.basename(jacl_path)}")
	if samples == before:
		# Same length, so the clip's declared timing is already correct. Rewriting it
		# from samples/rate would round differently and change bytes for no reason;
		# a pure sample swap must leave every other field untouched.
		return streams, None, None
	print(f"           sample count changed, so ManiInfo follows: "
		  f"frame_count {frame_count}, duration {duration:.4f}s")
	return streams, frame_count, duration


def main():
	ap = argparse.ArgumentParser(description=__doc__,
								 formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("manis")
	ap.add_argument("--out", required=True)
	ap.add_argument("--ms2", required=True,
					help="skeleton supplying the bind pose used as ACL defaults")
	ap.add_argument("--precision", type=float,
					help="ACL error threshold in cm; looser values avoid the raw bit rate "
						 "(index 31 in the stream) that vanilla never uses")
	ap.add_argument("--medium", type=float,
					help="fraction of keyframes moved to the medium-importance tier "
						 "(ACL default 0.35). The tiers are the game's LOD streams - low "
						 "goes to Anim_L0 and medium to Anim_L1 - so a clip whose keys are "
						 "mostly streamed reconstructs badly at distance. These clips animate "
						 "bone TRANSLATION, so that shows up as stretched limbs, not just a "
						 "soft pose. Lower both to keep more keyframes resident.")
	ap.add_argument("--low", type=float,
					help="fraction of keyframes moved to the lowest-importance tier "
						 "(ACL default 0.5); see --medium")
	ap.add_argument("--yaw-clip", type=int,
					help="clip index to rotate, for proving an edit reaches the game")
	ap.add_argument("--yaw-track", type=int, help="track (== .ms2 bone index) to rotate")
	ap.add_argument("--yaw-deg", type=float, default=60.0, help="degrees to rotate")
	ap.add_argument("--motion-clip", type=str,
					help="clip NAME whose 'Z Motion Track' scalar channel to scale, e.g. "
						 "acrocanthosaurus$walk. That channel is how far the clip carries "
						 "the animal, which is the number SpeciesAnimation.WalkSpeed is "
						 "meant to agree with - Acrocanthosaurus' walk clip measures "
						 "4.33 m/s against a declared 4.32.")
	ap.add_argument("--motion-factor", type=float, default=2.0,
					help="multiplier for --motion-clip (default 2.0)")
	ap.add_argument("--resample-clip", type=str,
					help="clip NAME to shorten by dropping frames - the only way to "
						 "speed a clip up that keeps every timing field consistent")
	ap.add_argument("--resample-factor", type=float, default=0.5,
					help="fraction of frames to keep (default 0.5 = twice as fast)")
	ap.add_argument("--replace-clip", type=str,
					help="clip NAME whose samples to replace from --jacl (Blender export)")
	ap.add_argument("--jacl", type=str,
					help="JACL sample file to read for --replace-clip")
	ap.add_argument("--hold-track", type=str, default="srb",
					help="comma-separated bone names or track indices to keep at the "
						 "template's values instead of taking them from --jacl. "
						 "Defaults to 'srb'; the LOD nodes are added automatically. "
						 "Pass an empty string to hold nothing.")
	ap.add_argument("--dense-clip", type=str,
					help="clip NAME to store in full - every sub-track kept, so every "
						 "bone becomes editable from Blender instead of only the ones "
						 "vanilla animated. Bigger, and unverified in game.")
	ap.add_argument("--store-sub", action="append", default=[], metavar="CLIP:BONE:GROUP",
					help="store ONE sub-track vanilla stripped, keeping the vanilla "
						 "stripped set everywhere else. GROUP is ori, pos or scl. "
						 "Repeatable. This is the one-bone-at-a-time version of "
						 "--dense-clip, which flips all 1,700 components at once: a "
						 "whole-clip failure says nothing about whether an individual "
						 "bone is safe, which is why the crush/stretch result does not "
						 "settle the stripped-set contract. "
						 "e.g. 'species$rest03:def_c_jaw_joint:pos'")
	ap.add_argument("--store-animate", action="store_true",
					help="make --store-offset sweep over the clip (0 -> full -> 0) "
						 "instead of holding a constant. ACL classifies a sub-track as "
						 "default, constant or ANIMATED, and an added constant may be "
						 "treated differently from an added animated track - three "
						 "added constants on three unrelated bones were all ignored in "
						 "game, which is what makes this the variable left to test.")
	ap.add_argument("--store-offset", type=str, default="",
					help="comma-separated offset added to the bind value --store-sub "
						 "writes, e.g. '0,-1.5,0'. Storing the plain bind is a POSITIVE "
						 "CONTROL PROBLEM: if the runtime substitutes the same bind for "
						 "a stripped sub-track, storing it explicitly is invisible, and "
						 "'looks normal' cannot distinguish 'the game read our value' "
						 "from 'the game ignored it'. An offset the eye cannot miss "
						 "makes the run interpretable in both directions.")
	args = ap.parse_args()

	with open(args.manis, "rb") as fh:
		data = fh.read()
	streams = decode_file(args.manis)
	blobs = list_clip_blobs(data)
	if len(blobs) != len(streams):
		sys.exit(f"{len(blobs)} ACL blobs but {len(streams)} decoded streams")
	headers = [read_blob_header(data, offset) for offset, _ in blobs]

	original = read_bulk_info(data)
	if original is None:
		sys.exit("this bundle has no database to rebuild; it is not a JWE3 stripped bundle")
	found = locate_bulk(data)
	print(f"{os.path.basename(args.manis)}: {len(streams)} streams, "
		  f"vanilla database {original['db_size']} bytes, "
		  f"bulk low={original['low_size']} medium={original['medium_size']}")

	parents, bind_values = read_ms2_bind(args.ms2)
	print(f"  bind pose: {len(parents)} bones from {os.path.basename(args.ms2)}")
	# One bind serves every clip in the bundle, so it has to cover the longest.
	# Clips authored on a bigger rig are normal here - see extend_bind_pose.
	widest = max((stream.values.shape[1] for stream in streams
				  if stream.track_type == QVVF), default=0)
	if widest > len(parents):
		parents, bind_values = extend_bind_pose(parents, bind_values, widest)
		print(f"  padded bind to {widest} tracks for clips authored on a larger rig")
	bind = bind_bytes(parents, bind_values)

	if args.yaw_clip is not None and args.yaw_track is not None:
		streams = apply_yaw(streams, headers, bind_values, args.yaw_clip,
							args.yaw_track, args.yaw_deg)

	if args.motion_clip:
		streams = scale_motion_track(args.manis, streams, headers,
									 args.motion_clip, args.motion_factor)

	dense_indices = set()
	if args.dense_clip:
		from generated.formats.manis import ManisFile as _MF
		_m = _MF(); _m.game = "Jurassic World Evolution 3"; _m.load(args.manis)
		_names = [str(i.name) for i in _m.mani_infos]
		if args.dense_clip not in _names:
			sys.exit(f"no clip named '{args.dense_clip}'")
		_tf = [i for i, h in enumerate(headers) if h["track_type"] == QVVF]
		_i = _tf[_names.index(args.dense_clip)]
		dense_indices = {_i}
		# A stripped sub-track has no samples - it is all NaN, and the encoder fills
		# NaN with the default, so it would be constant-equal-to-default and get
		# stripped straight back. To store it we must give it real samples, and the
		# only honest value is the bind pose: exactly what the game would have
		# supplied. The clip is then unchanged to look at, but every bone is present
		# and can be edited.
		_v = streams[_i].values
		_n = _v.shape[1]
		_fill = np.broadcast_to(bind_values[None, :_n, :], _v.shape)
		_was = int(np.isnan(_v).all(axis=0).all(axis=-1).sum())
		streams[_i].values = np.where(np.isnan(_v), _fill, _v)
		_now = int(np.isnan(streams[_i].values).all(axis=0).all(axis=-1).sum())
		print(f"  dense: '{args.dense_clip}' filled from the bind pose - "
			  f"fully stripped bones {_was} -> {_now}, every sub-track will be stored")

	if args.store_sub:
		# Same mechanism as --dense-clip but scoped to one (clip, bone, group).
		# Filling the samples is both necessary AND sufficient: once a sub-track
		# holds real values it is no longer all-NaN, so clip_defaults sees it as
		# "vanilla stored this", moves the default away from it, and ACL keeps it.
		# No change to clip_defaults is needed, and every other sub-track in the
		# bundle keeps the vanilla stripped set by construction.
		from generated.formats.manis import ManisFile as _MF2
		_m2 = _MF2(); _m2.game = "Jurassic World Evolution 3"; _m2.load(args.manis)
		_names2 = [str(i.name) for i in _m2.mani_infos]
		_tf2 = [i for i, h in enumerate(headers) if h["track_type"] == QVVF]
		_bones = read_ms2_bone_names(args.ms2)
		_groups = {"ori": (0, 4), "pos": (4, 7), "scl": (7, 10)}
		for spec in args.store_sub:
			parts = spec.split(":")
			if len(parts) != 3:
				sys.exit(f"--store-sub wants CLIP:BONE:GROUP, got '{spec}'")
			clip_name, bone_name, group = parts[0], parts[1], parts[2].lower()
			if group not in _groups:
				sys.exit(f"--store-sub group must be ori, pos or scl, got '{group}'")
			if clip_name not in _names2:
				sys.exit(f"--store-sub: no clip named '{clip_name}'")
			si = _tf2[_names2.index(clip_name)]
			lo, hi = _groups[group]
			v = streams[si].values
			if bone_name == "*":
				# Every sub-track vanilla stripped in this group. A shotgun: it loses
				# per-bone attribution but answers "is ANY added sub-track ever read?"
				# without having to guess which bone is visible from the camera.
				tracks = [t for t in range(v.shape[1])
						  if np.isnan(v[:, t, lo:hi]).all()]
				print(f"  store-sub: '*' matched {len(tracks)} stripped {group} "
					  f"sub-tracks in {clip_name}")
			elif bone_name.lstrip("-").isdigit():
				tracks = [int(bone_name)]
			elif bone_name in _bones:
				tracks = [_bones.index(bone_name)]
			else:
				sys.exit(f"--store-sub: no bone named '{bone_name}' in the {len(_bones)} "
						 f"bone skeleton")
			for track in tracks:
				if track >= v.shape[1]:
					sys.exit(f"--store-sub: track {track} is past this clip's "
							 f"{v.shape[1]} tracks")
				if not np.isnan(v[:, track, lo:hi]).all():
					sys.exit(f"--store-sub: {clip_name} {bone_name} {group} is ALREADY "
							 f"stored by vanilla - forcing it would change nothing and "
							 f"the test would read a no-op as a pass. Pick a sub-track "
							 f"that is actually stripped.")
				value = np.array(bind_values[track, lo:hi], dtype="<f4", copy=True)
				if args.store_offset:
					delta = [float(x) for x in args.store_offset.split(",") if x.strip()]
					if group == "ori":
						# For a rotation the offset is euler XYZ DEGREES composed onto
						# the bind quaternion - adding to quaternion components would
						# leave the unit sphere and mean nothing.
						if len(delta) != 3:
							sys.exit("--store-offset for group 'ori' wants 3 euler "
									 f"degrees, got {len(delta)}")
						rx, ry, rz = (np.radians(d) / 2.0 for d in delta)
						cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry),
												  np.sin(ry), np.cos(rz), np.sin(rz))
						q = np.array([sx * cy * cz - cx * sy * sz,
									  cx * sy * cz + sx * cy * sz,
									  cx * cy * sz - sx * sy * cz,
									  cx * cy * cz + sx * sy * sz], dtype="<f8")
						x1, y1, z1, w1 = q
						x2, y2, z2, w2 = value.astype("<f8")
						value = np.array([
							w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
							w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
							w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
							w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], dtype="<f8")
						value = (value / np.linalg.norm(value)).astype("<f4")
					else:
						if len(delta) != hi - lo:
							sys.exit(f"--store-offset needs {hi - lo} numbers for group "
									 f"'{group}', got {len(delta)}")
						value = value + np.array(delta, dtype="<f4")
				if args.store_animate and args.store_offset:
					# Sweep bind -> full offset -> bind across the clip so ACL cannot
					# collapse the sub-track to a constant.
					n = v.shape[0]
					w = np.sin(np.linspace(0.0, np.pi, n)).astype("<f8")
					base = bind_values[track, lo:hi].astype("<f8")
					if group == "ori":
						tgt = value.astype("<f8")
						if float(base @ tgt) < 0.0:
							tgt = -tgt
						th = np.arccos(np.clip(float(base @ tgt), -1.0, 1.0))
						if th < 1e-6:
							frames = np.broadcast_to(base, (n, 4)).copy()
						else:
							frames = ((np.sin((1 - w)[:, None] * th) * base
									   + np.sin(w[:, None] * th) * tgt) / np.sin(th))
						frames = frames / np.linalg.norm(frames, axis=1, keepdims=True)
					else:
						frames = base + w[:, None] * (value.astype("<f8") - base)
					v[:, track, lo:hi] = frames.astype("<f4")
					mode = "ANIMATED bind -> offset -> bind"
				else:
					v[:, track, lo:hi] = value
					mode = "constant"
				if len(tracks) <= 4:
					label = _bones[track] if track < len(_bones) else str(track)
					print(f"  store-sub: {clip_name} track {track} ({label}) {group} "
						  f"{mode} - now STORED")
			if len(tracks) > 4:
				print(f"  store-sub: {clip_name} {group} - {len(tracks)} sub-tracks "
					  f"now STORED ({'animated' if args.store_animate else 'constant'})")

	resampled = None
	if args.replace_clip:
		if not args.jacl:
			sys.exit("--replace-clip needs --jacl")
		bone_names = read_ms2_bone_names(args.ms2)
		hold = resolve_tracks(args.hold_track or "", bone_names)
		# The LOD nodes share srb's parentless, pre-rotated rest matrix and break the
		# same way, but their names carry the species, so match on shape rather than
		# making every caller spell out six bones.
		hold += [i for i, n in enumerate(bone_names)
				 if re.fullmatch(r".+_L\d+", n) and i not in hold]
		hold = sorted(set(hold))
		streams, resampled, new_duration = replace_clip_samples(
			args.manis, streams, headers, args.replace_clip, args.jacl, hold)

	if args.resample_clip:
		streams, resampled, new_duration = resample_clip(
			args.manis, streams, headers, args.resample_clip, args.resample_factor)

	with tempfile.TemporaryDirectory(prefix="jwe3_acl_db_") as work_dir:
		bound, database, low, medium = build_database(
			streams, headers, bind, work_dir, parents, bind_values,
			precision=args.precision,
			medium_proportion=args.medium, low_proportion=args.low,
			dense=dense_indices)

	# Work on the keys region alone and append the bulk once it is settled, so
	# nothing below can shift the bulk out from under locate_bulk().
	keys = data[:found["low_offset"]]
	original_blobs = list_clip_blobs(keys)
	base = buffer_residue(original_blobs)
	blocks = block_layout(keys, original_blobs, headers)

	# Encode everything first; the file is only touched afterwards.
	new_blobs = []
	worst = 0.0
	for index, stream in enumerate(streams):
		if stream.track_type == QVVF:
			blob = bound[index]
			if not read_blob_header(blob).get("has_database"):
				sys.exit(f"clip {index} came back without a database; build_database did not bind it")
		else:
			blob = encode_tracks(stream.values, stream.track_type, stream.sample_rate,
								 wrap=headers[index]["wrap_optimized"])
			got = decode_blob(blob).values
			finite = np.isfinite(stream.values) & np.isfinite(got)
			if finite.any():
				worst = max(worst, float(np.abs(stream.values[finite] - got[finite]).max()))
		new_blobs.append(blob)
		print(f"  stream {index:>3}: {original_blobs[index][1]:>7} -> {len(blob):>7} bytes"
			  f"{'  database-backed' if stream.track_type == QVVF else '  scalar'}")

	# Splice a whole ManiBlock at a time, back to front so the offsets of the blocks
	# not yet visited stay valid. Per-blob replacement cannot work here: the padding
	# after a block's last blob rounds to 8, not 16, and getting that wrong silently
	# eats the head of the limb structure and crashes the game on spawn at
	# JWE3.exe+0x1697FBD. See source/formats/manis/limbs.py.
	cursor = len(new_blobs)
	with_limbs = 0
	for block in reversed(blocks):
		cursor -= len(block["blobs"])
		keys = rebuild_block(keys, block, base, new_blobs[cursor:cursor + len(block["blobs"])])
		with_limbs += bool(block["limb"])
	print(f"  rebuilt {len(blocks)} ManiBlocks, {with_limbs} carrying limb data")

	# The database sits at the tail of buffer 0, immediately after the ManiInfo array.
	#
	# Replace its PADDED extent, not its declared size. CompressedHeaderReader consumes
	# `size + (-size % 16)` - the blob plus the padding up to 16 that belongs to buffer 0
	# - so the region the reader treats as "the database" is the padded one. Splicing a
	# new blob over only `db_size` bytes leaves the old padding behind: appending a clip
	# grows the database by ~8 bytes per clip (312 -> 320 for Deinosuchus 29 -> 30), the
	# reader consumes 320, lands on the 8 orphaned padding bytes, and parses them as two
	# extra zero entries at the head of the name buffer's hash array.
	#
	# Everything then shifts by two: target_names[0] swallows the last two hashes and
	# comes out as '<8 junk bytes>def_l_rearLegUpr_joint', so that bone's name matches
	# nothing. Buffer1's own docstring - "the game verifies that hash and target name
	# match; if they don't, the target won't be animated" - makes that a live bug, not
	# just a parsing nuisance. Shipped bundles pair 164/164 at offset 0; the bundle this
	# tool produced paired 0/163 until this fix.
	db_offset, db_size = find_database(keys)
	old_extent = db_size + (-db_size % BULK_ALIGNMENT)
	new_database = pad_to(database)
	keys = keys[:db_offset] + new_database + keys[db_offset + old_extent:]

	data = keys + pad_to(low) + pad_to(medium)

	if resampled is not None:
		data = retime_mani_info(data, args.resample_clip or args.replace_clip,
								resampled, new_duration)

	with open(args.out, "wb") as fh:
		fh.write(data)

	# Gates
	rebuilt_info = read_bulk_info(data)
	if rebuilt_info is None:
		sys.exit("Gate C FAILED: the rebuilt database header is not readable")
	if locate_bulk(data) is None:
		sys.exit("Gate C FAILED: the rebuilt bulk does not hash-match its header")
	print(f"Gate C  OK: database {rebuilt_info['db_size']} bytes, "
		  f"bulk low={rebuilt_info['low_size']} medium={rebuilt_info['medium_size']}, hashes match")

	name_ok, name_msg = check_name_buffer(args.out)
	if not name_ok:
		sys.exit(f"Gate N FAILED: {name_msg}")
	print(f"Gate N  OK: {name_msg}")

	parsed, total = count_parsed_maniblocks(args.out)
	print(f"Gate B  {parsed}/{total} ManiBlocks parse")
	if parsed != total:
		sys.exit(f"Gate B FAILED: only {parsed} of {total} ManiBlocks parse")

	headers_out = [read_blob_header(data, off) for off, _ in list_clip_blobs(data)]
	transforms = [h for h in headers_out if h["track_type"] == QVVF]
	if not all(h["has_database"] for h in transforms):
		sys.exit("Gate F FAILED: some transform clips are not database-backed")
	print(f"Gate F  OK: {len(transforms)}/{len(transforms)} transform clips have has_database=1")
	if worst:
		print(f"Gate A  scalar streams max abs error {worst:.2e}")
	print(f"wrote {args.out} ({len(data)} bytes)")
	return 0


if __name__ == "__main__":
	sys.exit(main())

