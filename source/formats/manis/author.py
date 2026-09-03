"""Splice authored clips into a JWE3 .manis without rebuilding the bundle.

This is the path a Blender-authored animation takes. Two things make it different
from `manis_database_cmd`:

**DO NOT USE FOR A SHIPPING BUNDLE.** `encode_tracks` produces a SELF-CONTAINED
blob (`has_database=0`) while every other clip in the bundle stays database-backed
(`has_database=1`). The bundle's ACL database still holds the clip's ORIGINAL bulk,
so the engine has two copies and alternates between them as it streams LOD tiers -
in game the animal **blinks between two poses** (observed 2026-09-02). That is what
`manis_database_cmd`'s Gate F guards against. Use the rebuild path there instead;
this function is kept for research where one clip must be swapped without touching
the database.

* **Only the edited clips change.** Each one is re-encoded SELF-CONTAINED and its
  blob spliced in place, so the bundle's ACL database and every other clip come
  through byte-identical. `manis_database_cmd --replace-clip` rebuilds the whole
  database and re-encodes all 30 clips, which is fine for research but needless
  churn - and it would silently re-encode clips like rest03 that were carefully
  built and verified.

* **Authored bones survive.** The older splice re-applied vanilla's NaN mask to
  the incoming samples, on the belief that storing more than vanilla makes animals
  crush and stretch. That was wrong: storing more is inert, because the engine
  poses only bones set in the clip's bone mask. So a sub-track the animator
  actually moved is kept, and its mask bit is set (see bonemask.py).
  Game-verified 2026-09-02.

"Authored" means the incoming track VARIES over time. Blender hands back a full
pose for every bone, so presence is not authorship - a bone nobody touched arrives
as a constant rest pose, and un-stripping those would store 170 constant tracks
for nothing.
"""
import numpy as np

from generated.formats.manis import ManisFile
from generated.formats.manis.acl import decode_file, decode_blob, encode_tracks
from source.formats.manis.bindpose import (
	read_ms2_bind, bind_bytes, extend_bind_pose, clip_defaults)
from source.formats.manis.bonemask import find_mask, set_bits
from source.formats.manis.limbs import replace_clip_blob
from source.formats.manis.splice import list_clip_blobs, read_blob_header

QVVF = 12
GATE_A_THRESHOLD = 0.01


ORI = slice(0, 4)


def _authored_bones(template, incoming):
	"""Bones whose ROTATION vanilla stripped and the incoming samples animate.

	Rotation specifically: the bone mask's population equals the clip's
	`anim_rot + const_rot`, so it gates rotation sub-tracks. Testing all ten
	components instead misses a bone whose ori is stripped while its pos or scl is
	stored - which is the common case, and made this return nothing at all.
	"""
	stripped = np.isnan(template[:, :, ORI]).all(axis=0).all(axis=-1)
	with np.errstate(invalid="ignore"):
		block = incoming[:, :, ORI]
		spread = np.nanmax(block, axis=0) - np.nanmin(block, axis=0)
	varies = np.nan_to_num(spread, nan=0.0).max(axis=-1) > 1e-5
	return stripped & varies


def splice_clips(bundle_path, out_path, edits, ms2_path,
				 unstrip=True, logger=print):
	"""Splice authored samples into `bundle_path`, writing `out_path`.

	`edits` maps clip name -> sample array shaped (samples, tracks, components),
	as Blender produces. Returns {clip name: [bone indices un-stripped]}.
	"""
	data = bytearray(open(bundle_path, "rb").read())
	manis = ManisFile()
	manis.game = "Jurassic World Evolution 3"
	manis.load(bundle_path)
	names = [str(i.name) for i in manis.mani_infos]
	streams = decode_file(bundle_path)
	blobs = list_clip_blobs(bytes(data))
	transform_idx = [i for i, (o, _s) in enumerate(blobs)
					 if read_blob_header(bytes(data), o)["track_type"] == QVVF]

	parents, bind_values = read_ms2_bind(ms2_path)
	widest = max(streams[i].values.shape[1] for i in transform_idx)
	if widest > len(bind_values):
		parents, bind_values = extend_bind_pose(parents, bind_values, widest)

	unstripped = {}
	for clip_name, incoming in edits.items():
		if clip_name not in names:
			raise ValueError(f"no clip named '{clip_name}' in {bundle_path}")
		blob_index = transform_idx[names.index(clip_name)]
		template = streams[blob_index].values
		if incoming.shape[1] != template.shape[1]:
			raise ValueError(
				f"'{clip_name}': {incoming.shape[1]} tracks authored but the clip has "
				f"{template.shape[1]}. The channel list is fixed by the resident "
				f"ManiBlock - export against this clip's own skeleton.")
		header = read_blob_header(bytes(data), blobs[blob_index][0])
		wrap = header["wrap_optimized"]
		# a wrap-optimised clip's last frame is implicit; Blender returns it
		if wrap and incoming.shape[0] == template.shape[0] + 1:
			incoming = incoming[:-1]
		if incoming.shape[0] != template.shape[0]:
			raise ValueError(
				f"'{clip_name}': {incoming.shape[0]} samples authored, clip has "
				f"{template.shape[0]}. Retiming through this path is not supported.")

		keep = _authored_bones(template, incoming) if unstrip else \
			np.zeros(template.shape[1], dtype=bool)
		mask = np.isnan(template).copy()
		if keep.any():
			# lift the mask only on the rotation components we are enabling; the
			# bone's pos/scl stay exactly as vanilla left them
			mask[:, keep, ORI] = False
		values = np.where(mask, np.float32("nan"), incoming)
		# where the author declined to speak but the template has something, keep it
		silent = np.isnan(values) & ~mask
		if silent.any():
			values = np.where(silent, template, values)

		bind = bind_bytes(parents, clip_defaults(values, bind_values))
		blob = encode_tracks(values, QVVF, streams[blob_index].sample_rate,
							 bind=bind, wrap=wrap)
		check = decode_blob(blob)
		finite = np.isfinite(values) & np.isfinite(check.values)
		err = float(np.abs(values[finite] - check.values[finite]).max()) if finite.any() else 0.0
		if err > GATE_A_THRESHOLD:
			raise ValueError(f"'{clip_name}': Gate A failed, encoder error {err:.6f}")
		logger(f"  {clip_name}: encoded {len(blob)} bytes "
			   f"(was {blobs[blob_index][1]}), Gate A {err:.6f}")
		data = bytearray(replace_clip_blob(bytes(data), blob_index, blob))
		blobs = list_clip_blobs(bytes(data))          # offsets moved
		if keep.any():
			bones = [int(b) for b in np.nonzero(keep)[0]]
			unstripped[clip_name] = bones
			logger(f"  {clip_name}: un-stripped {len(bones)} authored bone(s) {bones[:8]}")

	# mask bits last, once every blob has settled at its final offset
	for clip_name, bones in unstripped.items():
		blob_index = transform_idx[names.index(clip_name)]
		off, size = blobs[blob_index]
		tracks = read_blob_header(bytes(data), off)["num_tracks"]
		mask_off = find_mask(bytes(data), off, tracks)
		if mask_off is None:
			logger(f"  {clip_name}: WARNING mask record not located - these bones "
				   f"will NOT animate in game")
			continue
		before, after = set_bits(data, mask_off, tracks, bones)
		logger(f"  {clip_name}: bone mask {before} -> {after}")
	with open(out_path, "wb") as fh:
		fh.write(bytes(data))
	return unstripped
