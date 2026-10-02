"""Read a skeleton's bind pose out of an .ms2 for use as ACL default sub-track values.

ACL strips any sub-track whose samples all equal that track's `default_value`, and
the runtime supplies the stripped value back through
`track_writer::get_variable_default_rotation()` and friends. Frontier compresses
JWE3 clips with the bind pose as that default, which is why a decoded clip has
~63% of its components missing and why `import_manis.py` substitutes the bone's
bind transform wherever the decoder wrote a NaN marker.

Re-encoding therefore needs the same bind pose, otherwise the defaults fall back to
identity/zero/one and the blob reports `has_trivial_default_values() == true`, which
vanilla never does.

The ACL track index is the target skeleton's bone index: every animated channel's
`ori_channel_to_bone` entry resolves to the .ms2 bone of the same name (verified
157/157 bones, 135/135 channels on Indoraptor), so bone order can be used directly.
"""
from __future__ import annotations

import struct

import numpy as np

JBIND_MAGIC = b"JBND"
JBIND_VERSION = 1
NO_PARENT = 0xFFFFFFFF
# .ms2 stores "no parent" as a ushort sentinel
MS2_NO_PARENT = 0xFFFF


def read_ms2_bind(ms2_path: str, model_index: int = 0):
    """Return (parents, values) for the skeleton in an .ms2.

    `parents` is uint32 per bone, NO_PARENT for roots. `values` is (num_bones, 10)
    float32 holding rotation xyzw, translation xyz and scale xyz, in the .ms2's own
    parent-local space - the same space the ACL qvv samples use.
    """
    from generated.formats.ms2 import Ms2File

    ms2 = Ms2File()
    ms2.load(ms2_path)
    bone_info = ms2.model_infos[model_index].bone_info
    if bone_info is None:
        raise ValueError(f"{ms2_path} model {model_index} has no bone_info")

    bones = bone_info.bones
    count = len(bones)
    values = np.zeros((count, 10), dtype="<f4")
    for i, bone in enumerate(bones):
        rot, loc = bone.rot, bone.loc
        # cobra stores the bind rotation as a quaternion with named components;
        # ACL wants xyzw in that order
        values[i, 0:4] = (rot.x, rot.y, rot.z, rot.w)
        values[i, 4:7] = (loc.x, loc.y, loc.z)
        # .ms2 bind scale is a single uniform factor, ACL's qvv scale is a vector3
        scale = float(bone.scale)
        values[i, 7:10] = (scale, scale, scale)

    raw_parents = [int(p) for p in bone_info.parents]
    if len(raw_parents) != count:
        raise ValueError(
            f"{ms2_path}: {len(raw_parents)} parents for {count} bones")
    parents = np.array(
        [NO_PARENT if p == MS2_NO_PARENT else p for p in raw_parents], dtype="<u4")
    return parents, values


def read_ms2_bone_names(ms2_path: str, model_index: int = 0):
    """Return the skeleton's bone names, in track order.

    `Ms2File.load()` resolves `bone.name` from the file's name pool, so this is the
    same order the ACL track indices use. Returns an empty list rather than raising
    if the names did not resolve - a caller that only wants to look one up can fall
    back to numeric indices.
    """
    from generated.formats.ms2 import Ms2File

    ms2 = Ms2File()
    ms2.load(ms2_path)
    bone_info = ms2.model_infos[model_index].bone_info
    if bone_info is None:
        return []
    try:
        return [str(bone.name) for bone in bone_info.bones]
    except AttributeError:
        return []


def bind_bytes(parents: np.ndarray, values: np.ndarray) -> bytes:
    """Serialise a bind pose into the .jbind blob jwe3_acl_encode.exe reads."""
    count = len(parents)
    if values.shape != (count, 10):
        raise ValueError(f"expected ({count}, 10) values, got {values.shape}")
    return (JBIND_MAGIC
            + struct.pack("<II", JBIND_VERSION, count)
            + np.ascontiguousarray(parents, dtype="<u4").tobytes()
            + np.ascontiguousarray(values, dtype="<f4").tobytes())


def write_jbind(path: str, parents: np.ndarray, values: np.ndarray) -> None:
    with open(path, "wb") as fh:
        fh.write(bind_bytes(parents, values))


def extend_bind_pose(parents: np.ndarray, values: np.ndarray, num_tracks: int):
    """Pad a bind pose out to `num_tracks`, for clips authored on a LARGER rig.

    `target_bone_count` names the skeleton a clip was authored on, and Frontier
    routinely ships clips built on a different one: 4 of Acrocanthosaurus' 22
    idle-bundle clips declare 172 bones against its own 170, and Deinosuchus
    carries 24 built on Dimetrodon's 212. A single bundle therefore needs a bind
    long enough for its longest clip, or the encoder rejects those clips with
    "bind pose track count does not match".

    The padded entries address bones this species does not have. The runtime
    retargets by bone name and drops them, so identity - parentless, no rotation,
    no translation, unit scale - is the honest filler. It also keeps ACL from
    stripping a real sub-track against a fabricated default.
    """
    count = len(parents)
    if num_tracks <= count:
        return parents, values
    extra = num_tracks - count
    parents = np.concatenate(
        (parents, np.full(extra, NO_PARENT, dtype="<u4")))
    identity = np.zeros((extra, 10), dtype="<f4")
    identity[:, 3] = 1.0      # rotation w
    identity[:, 7:10] = 1.0   # unit scale
    return parents, np.concatenate((values, identity))


def bind_for_manis(ms2_path: str, num_tracks: int, model_index: int = 0):
    """Bind pose for a clip with `num_tracks` ACL tracks, or None if it cannot apply.

    A bundle whose track count does not match the skeleton is not something we can
    supply defaults for; the caller should fall back to trivial defaults rather than
    silently pair a clip with the wrong skeleton.
    """
    parents, values = read_ms2_bind(ms2_path, model_index)
    if len(parents) != num_tracks:
        return None
    return parents, values


# component groups inside a qvv sample: rotation xyzw, translation xyz, scale xyz
SUB_TRACKS = ((0, 4), (4, 7), (7, 10))


def clip_defaults(values, bind, keep_all=False):
	"""Per-clip ACL defaults that reproduce a vanilla clip's stripped set exactly.

	ACL strips a sub-track when every sample equals `track_desc::default_value`, and
	it does NOT store that value - at playback the host supplies it back through
	`track_writer::get_variable_default_*()`. So the stripped set is a contract with
	the game, and getting it wrong is invisible in a sample-value comparison: the
	blob decodes correctly in our tools and renders as a crushed or stretched animal
	in game, because the game substituted its own number for a component we chose
	not to store.

	Vanilla is the only description of that contract we have. `values` is the vanilla
	decode, where a stripped sub-track reads as all-NaN, so:

	- **all-NaN sub-track**: vanilla stripped it, so we must too. The encoder fills
	  NaN with the default, which makes the sub-track constant-equal-to-default and
	  ACL strips it. The default's actual value is irrelevant - it never reaches the
	  file - which is what makes a clip authored on a foreign rig safe to encode
	  against this species' bind.
	- **sub-track vanilla stored**: move the default away from the samples so ACL
	  cannot strip it. This has to cover varying sub-tracks too, not just exactly
	  constant ones: ACL collapses a NEAR-constant track to a constant first
	  (constant_rotation_threshold_angle and friends), and would then strip it for
	  matching the default. A default the samples cannot reach is ignored by a
	  genuinely varying track, so applying it everywhere costs nothing.

	Returns a (num_tracks, 10) float32 array.
	"""
	import numpy as np

	count = values.shape[1]
	out = np.array(bind[:count], dtype="<f4", copy=True)
	for lo, hi in SUB_TRACKS:
		block = values[:, :, lo:hi]
		missing = np.isnan(block)
		stripped = missing.all(axis=0).all(axis=-1)
		# keep_all forces EVERY sub-track to be stored, so every bone becomes
		# editable from Blender. Storing more than vanilla is a superset: the
		# runtime reads an explicit value instead of substituting the bind it
		# would otherwise supply, so an unedited clip is unchanged. (Stripping
		# MORE than vanilla is the dangerous direction - that is what made
		# animals crush and stretch.) Costs blob size; verify in game before
		# relying on it.
		kept = np.ones_like(stripped) if keep_all else ~stripped
		if lo == 0:
			# A stripped rotation is filled with its default, so it only strips if ACL
			# reads the fill back as that default. ACL drops w and rebuilds it from
			# xyz; a bind with w ~ 0 (Acro def_c_lipLwr_joint) rebuilds ~0.03 deg off
			# and is STORED at precision 0.001. The value never reaches the file, so
			# use identity, which rebuilds exactly.
			out[~kept, 0:4] = (0.0, 0.0, 0.0, 1.0)
		if not kept.any():
			continue
		held = np.nan_to_num(np.nanmax(block, axis=0), nan=0.0)
		if lo == 0:
			# quaternions are unit length, so "far away" has to stay on the sphere:
			# compose a 90 degree turn about X, which no sample can coincide with
			x, y, z, w = (held[:, i] for i in range(4))
			root = np.float32(0.70710678)
			out[kept, 0] = (w * root + x * root)[kept]
			out[kept, 1] = (y * root + z * root)[kept]
			out[kept, 2] = (z * root - y * root)[kept]
			out[kept, 3] = (w * root - x * root)[kept]
		else:
			out[kept, lo:hi] = held[kept] + np.float32(1.0)
	return out
