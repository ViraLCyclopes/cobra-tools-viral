"""Per-clip bone mask - the gate that decides which bones a clip may pose.

Each clip is preceded by a record the engine loads VERBATIM as a runtime struct
(confirmed field-by-field against a live dump). The part that matters:

    mask     ceil(num_tracks / 64) qwords, bit n = bone n
    +0x40    bone count (== num_tracks), then the u32 flag 1

**The count+flag pair sits exactly 0x40 after the mask start**, whatever the mask
width. That invariant is the locator: the record's distance from its blob varies
(0x88 for most clips, 0x80 for eat/sleep/social ones, other values for the
212-bone foreign rigs), so anchoring on the blob does not work, but anchoring on
the count pair does.

A sub-track can be stored in the ACL blob and still never animate, because the
engine poses only bones whose bit is set here. Storing without setting the bit is
half a change that passes every offline check and does nothing in game - that
mistake cost eight sessions, so nothing here guesses: a candidate is accepted only
when the bone count, the flag AND the popcount implied by the ACL blob all agree.
"""
import struct

COUNT_AFTER_MASK = 0x40      # count+flag pair, relative to the mask start
SEARCH_BACK = 0x4000
RECORD_SIZE = 0x48           # mask .. 0x40 count .. 0x44 flag .. end

# The whole struct the record is the tail of. See build_container.
CONTAINER_SIZE = 0x80
MASK_FIELD = 0x20            # each mask field is 0x20 bytes, so 256 bones max
TRA_MASK_AT = 0x18
ROT_MASK_AT = 0x38
SCL_MASK_AT = 0x58
COUNT_AT = 0x78
MASK_KINDS = (("translation", TRA_MASK_AT),
			  ("rotation", ROT_MASK_AT),
			  ("scale", SCL_MASK_AT))


def build_container(num_tracks, rotation=(), translation=(), scale=(), flag=1):
	"""Construct the whole 0x80 struct that precedes a compressed ManiBlock's blob.

	`build_record` covers only its last 0x48. That is enough to EDIT a vanilla
	container, because the part it leaves out is already in the file, and it is
	not enough to WRITE one: the translation mask sits 0x20 earlier, and a
	container emitted without it stores every bone's translation in the ACL blob
	with its mask bit clear. That is the same silent half-change as an unmasked
	rotation - valid file, every offline check green, nothing moves in game.

	Layout, measured over 605 shipped clips from 32 bundles on rigs of 142, 170,
	172, 175, 206 and 212 bones:

		0x00  0x18 zero bytes (three qwords the runtime fills in at load)
		0x18  translation mask, ceil(num_tracks/64) qwords, zero to 0x20
		0x38  rotation mask,    same shape   <- the region build_record builds
		0x58  scale mask,       same shape   (zero on every rig measured, all of
		      which have has_scale false)
		0x78  uint32 num_tracks
		0x7C  uint32 flag

	**605/605 rebuild BYTE-IDENTICALLY from the three sub-track sets, the bone
	count and the flag.** Nothing else in the struct varies.

	**The scale mask is NOT keyed the same way as the other two.** Rotation and
	translation carry what the ACL blob stores; scale carries what the ManiInfo's
	`scl_channel_to_bone` declares, even where ACL then strips every one of them.
	Measured 1137/1137 with no exceptions - see `acl_writer.container_masks`,
	which is where a writer should get all three from.

	Nothing on a scale-free rig can tell those apart, so the first 1042 clips
	measured here said nothing about it: Acrocanthosaurus and friends have
	`has_scale == False` everywhere. Deinosuchus separates them.

	Note that this is the AUTHORING convention, verified. Whether the runtime
	honours the scale mask is untested: Deinosuchus does not scale bones in game,
	because its prefab does not enable `BoneScaling`, so the engine never reads
	the field on the one rig that populates it.

	The container sits at align8 after the block's channel tables and the ACL
	blob at align16 after the container, which is why the gap between the two is
	always 0x80 or 0x88 - see `find_container`.

	`flag` is the one field no measurement explains: 592 of the 605 carry 1, and
	`baryonyx$partial_mouth01` carries 0 where `dimetrodon$partial_mouth01` -
	same role, same shape - carries 1, so it is not a function of the clip. 1 is
	the default because it is the overwhelming majority and because every clip
	that poses no bone at all is among the zeros, which a from-scratch clip never
	is.

	Each of `rotation` / `translation` / `scale` is the set of bones whose
	sub-track of that kind the ACL blob actually stores - constant or animated,
	i.e. `sub_track_types(...) != 0`. A sub-track stored without its bit set here
	is inert in game.
	"""
	nw = mask_words(num_tracks)
	if nw * 8 > MASK_FIELD:
		raise ValueError(f"{num_tracks} bones needs {nw} mask qwords, which would "
						 f"overrun the {MASK_FIELD:#x}-byte mask field")
	if flag not in (0, 1):
		raise ValueError(f"container flag is {flag}; vanilla only ever has 0 or 1")
	container = bytearray(CONTAINER_SIZE)
	for bones, offset in ((translation, TRA_MASK_AT), (rotation, ROT_MASK_AT),
						  (scale, SCL_MASK_AT)):
		words = [0] * nw
		for b in bones:
			if not 0 <= b < num_tracks:
				raise ValueError(f"bone {b} outside this clip's {num_tracks} tracks")
			words[b >> 6] |= 1 << (b & 63)
		struct.pack_into(f"<{nw}Q", container, offset, *words)
	struct.pack_into("<II", container, COUNT_AT, num_tracks, flag)
	return bytes(container)


def find_container(data, blob_offset, num_tracks):
	"""File offset of the 0x80 container preceding the ACL blob at `blob_offset`.

	The container is at align8 after the channel tables and the blob at align16
	after the container, so the gap is 0x80 when the container starts 16-aligned
	in buffer coordinates and 0x88 when it starts 8 past. Those are the only two
	possibilities, and the bone count at +0x78 tells them apart - no search, and
	no dependence on knowing the ManiBlock's own start.

	Returns None when neither candidate carries the count, which is what a
	non-transform blob or an uncompressed block looks like.
	"""
	for gap in (CONTAINER_SIZE, CONTAINER_SIZE + 8):
		start = blob_offset - gap
		if start < 0 or start + CONTAINER_SIZE > len(data):
			continue
		count, flag = struct.unpack_from("<II", data, start + COUNT_AT)
		if count == num_tracks and flag in (0, 1):
			return start
	return None


def read_container(data, start, num_tracks):
	"""Unpack a container into its three bone sets, the count and the flag."""
	nw = mask_words(num_tracks)
	out = {}
	for kind, offset in MASK_KINDS:
		words = list(struct.unpack_from(f"<{nw}Q", data, start + offset))
		out[kind] = [b for b in range(num_tracks) if (words[b >> 6] >> (b & 63)) & 1]
	out["num_tracks"], out["flag"] = struct.unpack_from("<II", data, start + COUNT_AT)
	return out


def build_record(num_tracks, bones):
    """Construct a per-clip bone-mask record from scratch.

    This is what makes writing a JWE3 ACL bundle from scratch possible: the
    module could previously only FIND and EDIT a record, so every from-scratch
    path had to adopt and patch a vanilla one.

    Layout, measured (see below):

        0x00  mask, ceil(num_tracks/64) qwords, bit n = bone n
              zero-filled from the end of the mask up to 0x40
        0x40  uint32 num_tracks
        0x44  uint32 1
        total 0x48, fixed whatever the mask width - the mask cannot exceed
        0x40 bytes because that would need more than 512 bones

    **Vanilla-verified: 208/208 Acrocanthosaurus records (num_tracks 170 and
    172, both 0x48 and 0x50 blob gaps) rebuild BYTE-IDENTICALLY from nothing but
    `(num_tracks, posed bones)`.** Records were located by popcount-verified
    match, so none of those 208 is a false positive. The 0x48 vs 0x50 distance
    to the ACL blob is alignment padding after the record, not part of it.

    `bones` is the set the clip may pose - normally the bones whose rotation the
    ACL blob stores, i.e. `num_animated_rotation + num_constant_rotation`. A
    sub-track stored without its bit set here is inert in game.
    """
    nw = mask_words(num_tracks)
    if nw * 8 > COUNT_AFTER_MASK:
        raise ValueError(f"{num_tracks} bones needs {nw} mask qwords, which would "
                         f"overrun the count+flag pair at 0x{COUNT_AFTER_MASK:x}")
    words = [0] * nw
    for b in bones:
        if not 0 <= b < num_tracks:
            raise ValueError(f"bone {b} outside this clip's {num_tracks} tracks")
        words[b >> 6] |= 1 << (b & 63)
    record = bytearray(RECORD_SIZE)
    struct.pack_into(f"<{nw}Q", record, 0, *words)
    struct.pack_into("<II", record, COUNT_AFTER_MASK, num_tracks, 1)
    return bytes(record)


def mask_words(num_tracks):
	"""Mask width in qwords. 212-bone rigs need 4, so this is not fixed at 3."""
	return (num_tracks + 63) // 64


def read_mask(data, mask_off, num_tracks):
	return list(struct.unpack_from(f"<{mask_words(num_tracks)}Q", data, mask_off))


def popcount(words):
	return sum(bin(w).count("1") for w in words)


def bones_in(words):
	return [b for b in range(len(words) * 64) if (words[b >> 6] >> (b & 63)) & 1]


def find_mask(data, blob_offset, num_tracks, expect_pop=None):
	"""Locate a clip's bone mask. Returns its file offset, or None.

	Scans back from the blob for the (num_tracks, 1) count+flag pair and takes the
	mask 0x40 before it. `expect_pop` - the clip's num_animated_rotation +
	num_constant_rotation - is what makes the match trustworthy; without it an
	unrelated count pair can be picked up, which happened during development.
	"""
	nw = mask_words(num_tracks)
	start = max(0, blob_offset - SEARCH_BACK)
	for p in range(blob_offset - 8, start - 1, -4):
		if struct.unpack_from("<II", data, p) != (num_tracks, 1):
			continue
		mask_off = p - COUNT_AFTER_MASK
		if mask_off < 0 or mask_off + nw * 8 > len(data):
			continue
		if expect_pop is not None and \
		   popcount(read_mask(data, mask_off, num_tracks)) != expect_pop:
			continue
		return mask_off
	return None


def set_bits(data, mask_off, num_tracks, bones):
	"""Set bones' bits. `data` must be a bytearray. Returns (before, after) popcounts."""
	nw = mask_words(num_tracks)
	words = read_mask(data, mask_off, num_tracks)
	before = popcount(words)
	for b in bones:
		if not 0 <= b < num_tracks:
			raise ValueError(f"bone {b} outside this clip's {num_tracks} tracks")
		words[b >> 6] |= 1 << (b & 63)
	struct.pack_into(f"<{nw}Q", data, mask_off, *words)
	return before, popcount(words)


def _has_acl_blobs(data):
	"""True if the bundle carries compressed ACL clips."""
	from source.formats.manis.splice import list_clip_blobs, read_blob_header
	try:
		return any(read_blob_header(data, o)["track_type"] == 12
				   for o, _s in list_clip_blobs(data))
	except Exception:
		return False


def sync_file(path, logger=None):
	"""Set every clip's mask bits from the bones that actually have stored data.

	This is what makes Blender authoring work. Blender knows which bones have
	keyframes; it knows nothing about the mask, so without this an exported clip
	animates a bone only if Frontier happened to leave its bit set - and the
	failure is silent: the file is valid, every gate passes, the bone does not
	move. Returns (clips_changed, clips_not_located).
	"""
	from generated.formats.manis import ManisFile
	from source.formats.manis.splice import list_clip_blobs, read_blob_header
	from source.formats.manis.acl_patch import parse_transform_layout
	from source.formats.manis.acl import decode_file
	import numpy as np

	QVVF = 12
	data = bytearray(open(path, "rb").read())
	# A Blender-exported bundle is UNCOMPRESSED (dtype 0) - it has no ACL blobs and
	# therefore no mask records. That is not an error, it just means the mask step
	# belongs after the ACL re-encode, not here. Detect it and say so plainly
	# instead of raising "ACL decoding failed", which reads like a broken export.
	if not _has_acl_blobs(bytes(data)):
		if logger:
			logger("bone mask: this bundle is uncompressed (no ACL blobs) - the mask "
				   "is set during the ACL re-encode, so nothing to do here")
		return 0, 0
	mf = ManisFile()
	mf.game = "Jurassic World Evolution 3"
	mf.load(path)
	names = [str(i.name) for i in mf.mani_infos]
	blobs = [(o, s) for o, s in list_clip_blobs(bytes(data))
			 if read_blob_header(bytes(data), o)["track_type"] == QVVF]
	decoded = [st for st in decode_file(path) if st.track_type == QVVF]
	if not (len(names) == len(blobs) == len(decoded)):
		if logger:
			logger(f"bone mask: skipped - {len(names)} names, {len(blobs)} blobs, "
				   f"{len(decoded)} decoded streams do not line up")
		return 0, 0
	changed = missing = 0
	for name, (off, size), st in zip(names, blobs, decoded):
		vals = st.values
		want = [b for b in range(vals.shape[1]) if not np.isnan(vals[:, b, 0:4]).all()]
		if not want:
			continue
		layout = parse_transform_layout(bytes(data[off:off + size]))
		mask_off = find_mask(bytes(data), off, layout.num_tracks)
		if mask_off is None:
			missing += 1
			if logger:
				logger(f"bone mask: '{name}' record not located - its bones will "
					   f"NOT animate in game")
			continue
		before, after = set_bits(data, mask_off, layout.num_tracks, want)
		if after != before:
			changed += 1
			if logger:
				logger(f"bone mask: '{name}' {before} -> {after} bones enabled")
	if changed:
		with open(path, "wb") as fh:
			fh.write(bytes(data))
	return changed, missing
