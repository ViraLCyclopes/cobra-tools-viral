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
