"""Export one action to .jacl, the interchange the JWE3 ACL encoder reads.

Deliberately NOT a .manis export. `export_manis` writes uncompressed dtype 0,
which for JWE3 costs ~10x the size, loses animated scale and brings the charge
bug. This writes samples only; the vanilla bundle stays the template and one
clip's animation is swapped into it, so the result stays compressed and
database-backed:

    export .jacl here
    -> manis_database_cmd.py IN --out OUT --ms2 models.ms2
           --replace-clip 'species$clip' --jacl exported.jacl
    -> ovl_tool_cmd.py inject STAGED.ovl -f OUT --in-place --update --update-aux

Two things the round trip depends on, both measured rather than assumed:

* Constraints are muted while sampling. These rigs carry IK, COPY_ROTATION,
  FLOOR, TRACK_TO and LOCKED_TRACK, and reading the posed result instead of the
  animation put all six rear-leg bones out by up to 0.52 in quaternion terms.
* Scale is written UNSWIZZLED. `store_transform_data` swizzles to (z, y, x) for
  the dtype-0 writer; ACL takes it in translation order.

With both, a shipped clip round-trips exactly: rotation and scale to the bit,
translation at the ACL precision floor, 127/127 tracks.
"""
import logging
import os

import bpy

from plugin.modules_export.armature import get_armature
from plugin.modules_export.jacl import export_action


def save(reporter, filepath="", action_source="ACTIVE", sample_rate=30.0003):
	"""Write the chosen action to `filepath` (or one file per action)."""
	scene = bpy.context.scene
	b_armature_ob = get_armature(scene.objects)
	if b_armature_ob is None:
		raise ValueError("No armature in the scene to sample")

	if action_source == "ACTIVE":
		anim = b_armature_ob.animation_data
		action = anim.action if anim else None
		if action is None:
			raise ValueError(f"{b_armature_ob.name} has no active action to export")
		actions = [action]
	else:
		actions = [a for a in bpy.data.actions if a.users]
		if not actions:
			raise ValueError("No actions in the scene")

	folder = os.path.dirname(filepath)
	written = []
	previous = b_armature_ob.animation_data.action if b_armature_ob.animation_data else None
	try:
		for action in actions:
			if len(actions) == 1:
				out = filepath
			else:
				safe = action.name.replace("$", "_").replace("/", "_")
				out = os.path.join(folder, f"{safe}.jacl")
			if b_armature_ob.animation_data:
				b_armature_ob.animation_data.action = action
			info = export_action(out, b_action=action, b_armature_ob=b_armature_ob,
								 sample_rate=sample_rate)
			written.append(info)
			reporter.show_info(
				f"{action.name}: {info['samples']} samples x {info['tracks']} tracks "
				f"-> {os.path.basename(out)}")
	finally:
		if b_armature_ob.animation_data:
			b_armature_ob.animation_data.action = previous

	reporter.show_info(f"Exported {len(written)} action(s). Re-encode with "
					   f"manis_database_cmd.py --replace-clip <name> --jacl <file>")
	return written
