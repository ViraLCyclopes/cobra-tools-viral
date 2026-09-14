"""Sample a Blender action into the .jacl interchange the JWE3 ACL encoder reads.

This is the export half of compressed animation editing. It deliberately does NOT
build a .manis: `ManisFile.save()` can only write uncompressed dtype 0, which
costs ~10x the size, loses animated scale and brings the charge bug. Instead the
vanilla bundle stays the template and one clip's SAMPLES are replaced:

    blender action -> .jacl -> manis_database_cmd.py --replace-clip NAME --jacl F
                            -> ovl_tool_cmd.py inject --update --update-aux

Every ManiInfo, channel map, name table and limb structure is carried through
untouched, and the result stays compressed and database-backed.

Conventions, all verified against the ACL decode of a shipped clip:

* Track index == the .ms2 bone index == the armature's bone order. No remapping.
* Rotation is xyzw (w LAST) and translation is xyz - both exactly what
  `store_pose_frame_info` already produces. Verified to 5 decimals on
  `def_c_root_joint` of `deinosuchus$run`.
* **Scale is NOT swizzled for ACL.** `store_transform_data` writes `(z, y, x)`
  because it targets the dtype-0 writer; `import_manis` takes ACL scale straight.
  This is unobservable in vanilla data - all 70,374 shipped scale keys are
  uniform - so non-uniform scale authored here is the one untested case.

The sample array is (samples, tracks, 10): rot xyzw, pos xyz, scale xyz.
"""
from __future__ import annotations

import logging
import struct
from collections import defaultdict

import bpy
import numpy as np

from generated.formats.manis import POS, ORI, SCL, EUL
from plugin.modules_export.animation import store_pose_frame_info
from plugin.import_manis import anim_sys
from plugin.modules_export.armature import get_armature
from plugin.utils.anim import get_b_local_matrix
from plugin.utils.blender_util import bone_name_for_ovl
from plugin.utils.transforms import ManisCorrector

COMPONENTS = 10
QVVF_TRACK_TYPE = 12
JACL_VERSION = 1


def clip_track_bones(mani_info):
	"""Track index -> bone name for one clip; None where the clip names no bone.

	The docstring above says "track index == the armature's bone order, no
	remapping", and that holds only while the armature IS the clip's skeleton. It
	breaks as soon as the ms2 import adds bones the clip does not carry: rig-edit
	`NODE_` bones are inserted BETWEEN a bone and its parent, so they shift every
	bone after them, and the LOD nodes add more at the end. UltimasaurusCE imports
	as 184 bones against a 172-bone ms2, while its clips carry 178 and 180 tracks.
	Sampling in armature order then writes each bone's animation onto whatever
	track happens to sit at that index - which passes every offline check and
	wrecks the animal in game.

	The importer reads a clip as

	    k.ori_bones[:] = transforms[:, k.ori_channel_to_bone, 0:4]

	so channel `c` of a group belongs to bone `<group>_bones_names[c]` and lives
	at track `<group>_channel_to_bone[c]`. Inverting that gives the clip's own
	track order. Verified conflict-free on this bundle: ori/pos/scl always agree
	on which track a bone occupies.
	"""
	count = int(mani_info.target_bone_count)
	keys = mani_info.keys
	slots = [None] * count
	for group in ("ori", "pos", "scl"):
		names = getattr(keys, f"{group}_bones_names", None)
		channel_to_bone = getattr(keys, f"{group}_channel_to_bone", None)
		if names is None or channel_to_bone is None:
			continue
		for channel, bone in enumerate(names):
			if channel >= len(channel_to_bone):
				break
			track = int(channel_to_bone[channel])
			if not 0 <= track < count:
				logging.warning(f"{mani_info.name}: {group} channel {channel} maps to "
								f"track {track}, outside 0..{count - 1} - ignored")
				continue
			bone = str(bone)
			if slots[track] is not None and slots[track] != bone:
				raise ValueError(
					f"{mani_info.name}: track {track} is claimed by both "
					f"{slots[track]!r} and {bone!r}; the channel maps disagree")
			slots[track] = bone
	return slots


GROUP_SPANS = {"ori": slice(0, 4), "pos": slice(4, 7), "scl": slice(7, 10)}


def bundle_track_index(manis):
	"""bone name -> track index, unioned over every clip in the bundle.

	A clip that does not name a bone still HAS its track - `target_bone_count` is
	178 against a 172-bone ms2 - so the track number has to come from somewhere
	else. Any clip in the bundle that does name the bone gives it, and the widest
	clip usually names the whole skeleton. Raises if two clips disagree, which
	would mean track index is not a bundle-wide bone id after all.
	"""
	out = {}
	for mani_info in manis.mani_infos:
		for track, name in enumerate(clip_track_bones(mani_info)):
			if not name:
				continue
			if out.setdefault(name, track) != track:
				raise ValueError(
					f"{name!r} is track {out[name]} in one clip and {track} in "
					f"{mani_info.name}; track index is not a bundle-wide bone id")
	return out


def missing_channels(values, bone_names, mani_info, tracks):
	"""(group, bone, track) for bones the animator authored that this clip cannot pose.

	`sample_action` already NaN-fills every channel group it considers unauthored,
	so "has non-NaN data" is exactly "the animator touched this". A bone in that
	state whose group has no channel in this clip would otherwise be dropped
	SILENTLY - the export succeeds and the bone simply never moves in game.
	"""
	out, unknown = [], []
	for group, span in GROUP_SPANS.items():
		named = {str(x) for x in (getattr(mani_info.keys, f"{group}_bones_names", None) or [])}
		lo = int(getattr(mani_info, f"{group}_bone_min"))
		hi = int(getattr(mani_info, f"{group}_bone_max"))
		for index, bone in enumerate(bone_names):
			if bone in named or np.isnan(values[:, index, span]).all():
				continue
			track = tracks.get(bone)
			if track is None or not lo <= track <= hi:
				unknown.append((group, bone))
				continue
			out.append((group, bone, track))
	return out, unknown


def to_clip_tracks(values, bone_names, mani_info):
	"""Reorder armature-ordered samples into one clip's own track order.

	`values` is (samples, armature bones, 10) from `sample_action`. The result is
	(samples, target_bone_count, 10). Tracks the clip names no bone for stay NaN,
	which is the same "stripped, equals the bind pose" sentinel the importer reads.
	"""
	order = clip_track_bones(mani_info)
	index = {name: i for i, name in enumerate(bone_names)}
	columns = np.array([index.get(name, -1) if name else -1 for name in order])
	missing = [name for name in order if name and name not in index]
	if missing:
		logging.warning(f"{mani_info.name}: {len(missing)} track bone(s) are not in the "
						f"armature and stay at the bind pose, e.g. {missing[:4]}")
	out = np.full((values.shape[0], len(order), values.shape[2]), np.nan, dtype=values.dtype)
	present = columns >= 0
	out[:, present, :] = values[:, columns[present], :]
	return out


def sample_action(b_armature_ob, b_action, sample_rate=30.0003):
	"""Return (values, bone_names) for `b_action` on `b_armature_ob`.

	`values` is (samples, tracks, 10) float32 in ACL's parent-local qvv layout,
	with one sample per integer frame of the action's range.
	"""
	corrector = ManisCorrector(False)
	bone_names = [bone_name_for_ovl(b.name) for b in b_armature_ob.data.bones]
	b_local_mats = {name: get_b_local_matrix(bone)
					for name, bone in zip(bone_names, b_armature_ob.data.bones)}

	# ASSIGN the action before sampling. Without this the frame range and the
	# authored-channel test come from `b_action` while the POSE read back is
	# whatever action happens to be active - so sampling every action in a scene
	# returned the active one's animation for all of them, and an "export all
	# clips" wrote the same motion into every clip. Restored in the finally below.
	_anim = b_armature_ob.animation_data or b_armature_ob.animation_data_create()
	_prev_action = _anim.action
	_prev_slot = getattr(_anim, "action_slot", None)
	_anim.action = b_action
	if hasattr(_anim, "action_slot") and len(getattr(b_action, "slots", ())):
		_anim.action_slot = b_action.slots[0]

	start, end = (int(round(v)) for v in b_action.frame_range)
	frames = list(range(start, end + 1))
	storage = defaultdict(lambda: defaultdict(dict))
	for name in bone_names:
		storage[name][POS] = np.zeros((len(frames), 3))
		storage[name][ORI] = np.zeros((len(frames), 4))
		storage[name][SCL] = np.zeros((len(frames), 3))
		storage[name][EUL] = np.zeros((len(frames), 3))

	# Mute every constraint first. A JWE3 clip stores per-bone FK, but these rigs
	# carry IK, COPY_ROTATION, FLOOR, TRACK_TO and LOCKED_TRACK constraints, and
	# sampling the POSE reads the re-solved result instead of the animation. On
	# viralsarcosuchus$walktodrink that put all six rear-leg bones out by up to
	# 0.52 in quaternion terms while the other 121 tracks were within 0.02.
	# Record every constraint's ORIGINAL state and restore exactly that. Collecting
	# only the unmuted ones leaves them stuck muted if a run is interrupted, and the
	# next run then sees them as already-muted and never restores them - the rig ends
	# up silently broken in the viewport.
	states = [(c, c.mute) for pb in b_armature_ob.pose.bones for c in pb.constraints]
	previous = bpy.context.scene.frame_current
	try:
		for constraint, _was in states:
			constraint.mute = True
		for slot, frame in enumerate(frames):
			store_pose_frame_info(b_armature_ob, frame, slot, b_local_mats,
								  storage, corrector, None)
	finally:
		for constraint, was in states:
			constraint.mute = was
		bpy.context.scene.frame_set(previous)

	values = np.zeros((len(frames), len(bone_names), COMPONENTS), dtype="<f4")
	for track, name in enumerate(bone_names):
		values[:, track, 0:4] = storage[name][ORI]
		values[:, track, 4:7] = storage[name][POS]
		# store_transform_data swizzles scale to (z, y, x) for the dtype-0 writer.
		# ACL takes it in the same order as translation, so undo that here.
		values[:, track, 7:10] = storage[name][SCL][:, ::-1]

	# A bone whose fcurves hold a SINGLE key carries no authored animation. cobra's
	# importer calls key_unanimated_channels(), which drops one identity key on every
	# bone the ManiBlock's channel list does not name - the Target*, *Squash*,
	# *Jiggle* and *AllTwist* helpers, and def_rearLegUpr_joint.L, whose motion the
	# rig reproduces through an IK constraint rather than through fcurves.
	#
	# Sampling those with constraints muted therefore yields the identity key, not the
	# clip's real motion, and def_rearLegUpr_joint.L proves that silently storing it is
	# not safe: vanilla DOES store that track, so the stripped-set mask does not catch
	# the mistake and the bone comes out frozen 50 degrees off.
	#
	# NaN means "not authored here" - the same convention a stripped sub-track decodes
	# to - so the re-encoder keeps the template's own values for these tracks.
	# Per CHANNEL, not per bone: def_c_spine1Squash_joint has 185 scale keys and a
	# single rotation key, so treating the bone as a unit would still export a fake
	# identity rotation for it.
	groups = (("rotation_quaternion", slice(0, 4)),
			  ("location", slice(4, 7)),
			  ("scale", slice(7, 10)))
	unauthored = []
	for track in range(len(bone_names)):
		bone = b_armature_ob.data.bones[track].name
		blank = 0
		for prop, span in groups:
			if not authored(b_action, bone, prop):
				values[:, track, span] = np.float32("nan")
				blank += 1
		if blank == len(groups):
			unauthored.append(track)

	# put the armature back the way we found it - sampling must not leave the
	# animator looking at a different action than the one they had open
	_anim.action = _prev_action
	if _prev_action is not None and _prev_slot is not None and hasattr(_anim, "action_slot"):
		try:
			_anim.action_slot = _prev_slot
		except Exception:
			pass
	return values, bone_names, unauthored


def authored(b_action, bone_name, prop):
	"""True if `bone_name`.`prop` carries more than one keyframe in `b_action`.

	One keyframe means the channel was never authored: cobra's importer calls
	key_unanimated_channels(), which drops a single identity key on every channel
	the ManiBlock's list does not name. Sampling those with constraints muted
	yields that identity pose rather than the clip's real motion.

	Matched exactly, not by prefix - a bone's IK constraint influence lives at
	`pose.bones["x"].constraints["IK"].influence` and is often the only densely
	keyed channel on an IK-driven bone, which would otherwise make it look authored.
	"""
	path = f'pose.bones["{bone_name}"].{prop}'
	# Blender 5.0 keeps `action.fcurves` for compatibility but leaves it EMPTY -
	# the real curves live in a channelbag on an action slot. Going through
	# anim_sys.get_data() is what export_manis already does; reading the raw
	# attribute here raised "'Action' object has no attribute 'fcurves'".
	for fcurve in anim_sys.get_data(b_action).fcurves:
		if fcurve.data_path == path and len(fcurve.keyframe_points) > 1:
			return True
	return False


def write_jacl(path, values, sample_rate=30.0003, track_type=QVVF_TRACK_TYPE):
	"""Write the sample array in the format jwe3_acl_decode.exe emits."""
	samples, tracks, comps = values.shape
	with open(path, "wb") as stream:
		stream.write(b"JACL")
		stream.write(struct.pack("<IIIII", JACL_VERSION, track_type, tracks,
								 samples, comps))
		stream.write(struct.pack("<f", sample_rate))
		stream.write(np.ascontiguousarray(values, dtype="<f4").tobytes())


def export_action(filepath, b_action=None, b_armature_ob=None,
				  sample_rate=30.0003):
	"""Sample the active (or given) action and write it to `filepath`."""
	scene = bpy.context.scene
	if b_armature_ob is None:
		b_armature_ob = get_armature(scene.objects)
	if b_armature_ob is None:
		raise ValueError("no armature in the scene to sample")
	if b_action is None:
		anim = b_armature_ob.animation_data
		b_action = anim.action if anim else None
	if b_action is None:
		raise ValueError(f"{b_armature_ob.name} has no active action")

	values, bone_names, unauthored = sample_action(b_armature_ob, b_action, sample_rate)
	write_jacl(filepath, values, sample_rate)
	authored_count = values.shape[1] - len(unauthored)
	logging.info(f"wrote {filepath}: {values.shape[0]} samples x "
				 f"{values.shape[1]} tracks, {authored_count} authored, "
				 f"{len(unauthored)} left to the template")
	return {
		"action": b_action.name,
		"armature": b_armature_ob.name,
		"samples": values.shape[0],
		"tracks": values.shape[1],
		"authored": authored_count,
		"unauthored": unauthored,
		"sample_rate": sample_rate,
		"path": filepath,
	}
