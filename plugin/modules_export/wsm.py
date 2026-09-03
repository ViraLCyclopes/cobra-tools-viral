"""Write the srb bone's world path back out as a .wsm, the mirror of import_wsm.

`import_manis.import_wsm` reads `<clip>_srb.wsm` from beside the .manis and keys the
srb bone's location and rotation into the SAME action as the clip, so an imported
social already animates along its real world route. Nothing wrote it back, which
meant a trajectory could be posed in Blender and then silently lost.

The transform is the exact inverse of import's:

    import :  M_basis = b_local_inv_mat @ corrector.to_blender(M_file)
    export :  M_file  = corrector.from_blender(b_local_mat @ M_basis)

`b_local_inv_mats[name]` is `get_b_local_matrix(bone).inverted()`, and the values
land in the bone's `location` / `rotation_quaternion` fcurves - the POSE BASIS, not
`pose.matrix` - so that is what gets read back.

Timing note: .wsm runs at exactly 30.0 fps, NOT the MANIS rate of 30.0003. Measured
across the shipped files, `duration == (frame_count - 1) / 30.0` to eight decimals
every time. Using the MANIS rate here puts the trajectory a fraction out of step
with the clip it accompanies.

The XML is written directly rather than through `to_xml_file` so the layout matches
cobra's extract byte for byte - verified on all 146 shipped .wsm files.
"""
from __future__ import annotations

import logging
import os

import bpy
import mathutils

from plugin.utils.anim import get_b_local_matrix
from plugin.utils.transforms import ManisCorrector
from plugin.import_manis import anim_sys

# .wsm timing is exactly 30 fps; see the module docstring
WSM_FPS = 30.0
MOTION_BONE = "srb"


def _fmt(value) -> str:
	return repr(float(value))


def has_motion_keys(b_action, bone_name: str = MOTION_BONE) -> bool:
	"""True if the action actually animates this bone, rather than holding one key.

	A single key is what `key_unanimated_channels` leaves on a bone the clip does not
	drive; writing a .wsm from that would pin the animal to one spot.
	"""
	prefix = f'pose.bones["{bone_name}"].'
	# Blender 5.0 keeps `action.fcurves` for compatibility but leaves it EMPTY -
	# the real curves live in a channelbag on an action slot. Going through
	# anim_sys.get_data() is what export_manis already does; reading the raw
	# attribute here raised "'Action' object has no attribute 'fcurves'".
	for fcurve in anim_sys.get_data(b_action).fcurves:
		if fcurve.data_path.startswith(prefix) and len(fcurve.keyframe_points) > 1:
			return True
	return False


def sample_motion(b_armature_ob, b_action, bone_name: str = MOTION_BONE,
				  is_old_orientation: bool = False):
	"""Return (locs, quats) in FILE space for every integer frame of the action."""
	bone = b_armature_ob.data.bones.get(bone_name)
	pose_bone = b_armature_ob.pose.bones.get(bone_name)
	if bone is None or pose_bone is None:
		raise ValueError(f"the armature has no '{bone_name}' bone")

	corrector = ManisCorrector(is_old_orientation)
	b_local_mat = get_b_local_matrix(bone)
	start, end = (int(round(v)) for v in b_action.frame_range)

	# Constraints are muted exactly as the .jacl sampler does: reading the SOLVED
	# pose gives whatever IK decided, not the authored trajectory. Original states
	# are restored even if this raises, or the rig is left silently broken.
	states = [(c, c.mute) for pb in b_armature_ob.pose.bones for c in pb.constraints]
	previous = bpy.context.scene.frame_current
	locs, quats = [], []
	try:
		for constraint, _was in states:
			constraint.mute = True
		for frame in range(start, end + 1):
			bpy.context.scene.frame_set(frame)
			basis_loc = mathutils.Matrix.Translation(pose_bone.location)
			file_loc = corrector.from_blender(b_local_mat @ basis_loc).to_translation()
			locs.append([file_loc.x, file_loc.y, file_loc.z])
			basis_rot = pose_bone.rotation_quaternion.to_matrix().to_4x4()
			file_rot = corrector.from_blender(b_local_mat @ basis_rot).to_quaternion()
			# file order is xyzw; mathutils gives wxyz
			quats.append([file_rot.x, file_rot.y, file_rot.z, file_rot.w])
	finally:
		for constraint, was in states:
			constraint.mute = was
		bpy.context.scene.frame_set(previous)
	return locs, quats


def write_wsm(path, locs, quats, unknowns: str = "0.0 0.0 0.0 -0.0 0.0 -0.0 1.0 0.0",
			  game: str = "Jurassic World Evolution 3"):
	"""Write the .wsm XML in cobra's exact extract layout."""
	if len(locs) != len(quats):
		raise ValueError(f"{len(locs)} locs but {len(quats)} quats")
	count = len(locs)
	duration = (count - 1) / WSM_FPS if count > 1 else 0.0
	lines = [f'<WsmHeader duration="{_fmt(duration)}" frame_count="{count}" '
			 f'game="{game}">',
			 f'\t<unknowns>{unknowns}</unknowns>',
			 '\t<locs>']
	for x, y, z in locs:
		lines.append(f'\t\t<vector3 x="{_fmt(x)}" y="{_fmt(y)}" z="{_fmt(z)}" />')
	lines.append('\t</locs>')
	lines.append('\t<quats>')
	for x, y, z, w in quats:
		lines.append(f'\t\t<vector4 x="{_fmt(x)}" y="{_fmt(y)}" '
					 f'z="{_fmt(z)}" w="{_fmt(w)}" />')
	lines.append('\t</quats>')
	lines.append('</WsmHeader>')
	# newline="" - the shipped files are LF; the Windows default would write \r\n
	with open(path, "w", encoding="utf-8", newline="") as stream:
		stream.write("\n".join(lines) + "\n")
	return path


def export_wsm(folder, clip_name: str, b_armature_ob, b_action,
			   bone_name: str = MOTION_BONE, unknowns: str = None,
			   is_old_orientation: bool = False):
	"""Write `<clip_name>_<bone>.wsm` into `folder`, or return None if there is none.

	`unknowns` is carried through from the donor when re-exporting an imported clip -
	those 8 floats are UNDECODED (they look like a position plus a quaternion), so
	inventing them is worse than copying them.
	"""
	if not has_motion_keys(b_action, bone_name):
		logging.info(f"{b_action.name}: '{bone_name}' has no animation, no .wsm written")
		return None
	locs, quats = sample_motion(b_armature_ob, b_action, bone_name, is_old_orientation)
	safe = clip_name.replace("/", "_")
	path = os.path.join(folder, f"{safe}_{bone_name}.wsm")
	kwargs = {} if unknowns is None else {"unknowns": unknowns}
	write_wsm(path, locs, quats, **kwargs)
	logging.info(f"wrote {os.path.basename(path)}: {len(locs)} frames")
	return path
