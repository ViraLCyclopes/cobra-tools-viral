"""The limb structure that follows a compressed ManiBlock, and where it must sit.

cobra does not model this at all on JWE3 - `ManiBlock`'s `LimbTrackData` field is
`vercond="!#PC2#"`, and `KeysReader` keeps the bytes as opaque `inter_block_data`
while it *searches* for the next block. That is fine for a byte-for-byte round
trip and wrong the moment a clip changes size, because the runtime does not
search: it computes the address arithmetically.

`JWE3.exe+0x1697ED3..0x1697FFA` walks a compressed clip as

    align8 -> 0x90 container -> align16 -> transform blob -> align16 -> scalar
    blob -> align8 -> LIMB STRUCTURE

taking each blob's length from its own ACL header. **The limb structure is at
align 8, not 16.** Vanilla agrees: on Acrocanthosaurus' idle bundle a plausible
limb header is present at align8 for 22/22 clips and at align16 for only 4/22 -
the four where the two happen to coincide.

Re-encoding changes blob sizes, so a tool that pads to 16 leaves the limb
structure a few bytes from where the runtime looks. The runtime then reads
counts out of the middle of the previous clip's padding, walks a cursor
megabytes past the buffer, and dies writing a pointer into it:

    01697FBD  mov qword ptr [rdx + r9*8 + 0x10], rax   <- ACCESS_VIOLATION

Layout, read off that routine:

    limb_base = align8(end of the clip's last ACL blob)
      +0x00  qword   pointer, filled in at load, zero on disk
      +0x08  ushort  outer_count
      +0x10  outer_count x 16-byte entries
                 +0x00  qword  pointer, filled in at load
                 +0x0C  ushort inner_count
    cursor = limb_base + 0x10 + outer_count * 16
    for each outer entry:
        sub_base = align8(cursor)
        inner_count x 0x30-byte entries
                 +0x10  qword  pointer, filled in at load
                 +0x18  qword  pointer, filled in at load
                 +0x2C  ushort n
        cursor = sub_base + inner_count * 0x30
        for each inner entry, in order:
            cursor += n * 16      (block A)
            cursor += n * 12      (block B)

Offsets here are relative to the keys buffer, whose residue class mod 16 in file
coordinates is what `splice.ref_alignment` returns.
"""
from __future__ import annotations

import struct

OUTER_ENTRY = 16
INNER_ENTRY = 0x30
# an outer count past this is not a limb count, it is garbage read from padding
MAX_PLAUSIBLE_OUTER = 64
MAX_PLAUSIBLE_INNER = 4096


def align(offset: int, alignment: int) -> int:
	return offset + (-offset % alignment)


def limb_extent(data: bytes, base: int, start: int) -> int:
	"""Length in bytes of the limb structure whose header is at file offset `start`.

	`base` is the keys buffer's residue class mod 16, so that the internal align8
	steps are measured in buffer coordinates rather than file coordinates.

	Raises ValueError if the structure does not parse, which is the signal that
	`start` is not actually where the limb data begins.
	"""
	def rel(offset):
		return offset - base

	def align8(offset):
		return offset + (-rel(offset) % 8)

	if start + 0x10 > len(data):
		raise ValueError(f"limb header at {start} runs past the end")
	pointer, outer_count = struct.unpack_from("<QH", data, start)
	if pointer:
		raise ValueError(f"limb header at {start} has a non-zero load pointer {pointer:#x}")
	if outer_count > MAX_PLAUSIBLE_OUTER:
		raise ValueError(f"limb header at {start} claims {outer_count} limbs")
	cursor = start + 0x10
	if not outer_count:
		return cursor - start
	entries = cursor
	cursor = entries + outer_count * OUTER_ENTRY
	if cursor > len(data):
		raise ValueError(f"limb entries at {entries} run past the end")
	for index in range(outer_count):
		inner_count, = struct.unpack_from("<H", data, entries + index * OUTER_ENTRY + 0xC)
		if inner_count > MAX_PLAUSIBLE_INNER:
			raise ValueError(f"limb {index} at {start} claims {inner_count} entries")
		sub_base = align8(cursor)
		cursor = sub_base + inner_count * INNER_ENTRY
		if cursor > len(data):
			raise ValueError(f"limb {index} sub-entries run past the end")
		for inner in range(inner_count):
			count, = struct.unpack_from("<H", data, sub_base + inner * INNER_ENTRY + 0x2C)
			cursor += count * 16 + count * 12
			if cursor > len(data):
				raise ValueError(f"limb {index} entry {inner} data runs past the end")
	return cursor - start


def _plausible(data: bytes, base: int, start: int, limit: int, tidy: bool):
	"""(start, extent) if a real limb structure begins at `start`, else None.

	`tidy` additionally demands that the structure end on a 16-byte boundary with
	nothing but zeros in between, which is true of every file Frontier shipped but
	NOT of one a 16-padding tool has already shifted - there the trailing padding
	is whatever the original layout left behind.

	The non-zero outer count is what stops the common false positive: the align8
	candidate can land 8 bytes *before* a structure that actually sits at align16.
	Those 8 bytes are the tail of the previous padding, so the load pointer reads
	as 0 and the count reads as the low half of the real structure's zeroed
	pointer - i.e. 0. Without this test that parses as an empty limb structure and
	the repair quietly does nothing.
	"""
	try:
		extent = limb_extent(data, base, start)
	except ValueError:
		return None
	outer_count, = struct.unpack_from("<H", data, start + 8)
	if not outer_count:
		return None
	limb_end = start + extent
	if limb_end > limit:
		return None
	if tidy:
		slack = -(limb_end - base) % 16
		if data[limb_end:limb_end + slack] != b"\x00" * slack:
			return None
	return start, extent


def find_limb_start(data: bytes, base: int, blob_end: int, limit: int = None):
	"""Where the limb structure for a clip ending at `blob_end` actually is.

	Prefers the runtime's align8, and falls back to align16 so a bundle a
	16-padding tool already wrote can be read and repaired. Both alignments are
	tried with the strict trailing-padding test first, then without it: a file
	that still has vanilla's layout should never need the loose pass, while a
	misaligned one has trailing padding that no longer matches its own end.

	Returns None when nothing validates, which covers a clip whose `has_list` is
	below 2 and so has no limb structure at all - the runtime's gate is
	`test [dtype], 0x40`, and 17 of Acrocanthosaurus' clips are exactly that. It
	also covers the (unobserved) case of a genuinely empty limb structure; the
	repair leaves those alone rather than guess.
	"""
	if limit is None:
		limit = len(data)
	for tidy in (True, False):
		for alignment in (8, 16):
			found = _plausible(data, base, blob_end + (-(blob_end - base) % alignment),
							   limit, tidy)
			if found:
				return found
	return None


def buffer_residue(blobs) -> int:
	"""The keys buffer's residue class mod 16, from any ACL blob's file offset.

	Every blob starts 16-aligned in buffer coordinates - the transform blob at
	`align16(container + 0x8F)`, the scalar blob at `align16(transform end)` - so
	they all share the buffer's own class. `splice.ref_alignment` derives the same
	number but only looks at scalar blobs, and some bundles have none.
	"""
	classes = {offset % 16 for offset, _size in blobs}
	if len(classes) != 1:
		raise ValueError(f"ACL blobs disagree on alignment: {sorted(classes)}")
	return classes.pop()


def block_tail_blobs(blobs, headers, transform_track_type: int = 12):
	"""Indices of the blobs a limb structure follows: the last one in each block.

	A compressed ManiBlock holds a transform blob and, when the clip has scalar
	channels, a second blob after it. The limb structure follows whichever came
	last.
	"""
	out = []
	for index, head in enumerate(headers):
		if head["track_type"] != transform_track_type:
			out.append(index)
		elif index + 1 >= len(headers) or headers[index + 1]["track_type"] == transform_track_type:
			out.append(index)
	return out


def repair_limb_alignment(data: bytes, blobs, headers):
	"""Move every limb structure back to align8 after its clip's last ACL blob.

	Re-encoding changes blob sizes, and `splice.replace_blob` pads to the 16-byte
	class that keeps cobra's own parse working. The runtime wants align8, so on a
	re-encoded bundle some clips end up with their limb structure a few bytes off
	and the game dies at `JWE3.exe+0x1697FBD`. This rewrites the padding on both
	sides of the structure: align8 before it, and enough after it to keep the next
	ManiBlock 16-aligned, which it must be for cobra to find it again.

	Returns (data, moved, skipped). On a vanilla bundle it is a byte-for-byte
	no-op, which is the regression test worth keeping.
	"""
	base = buffer_residue(blobs)
	moved = 0
	skipped = 0
	# back to front, so the offsets of the clips not yet visited stay valid
	for index in reversed(block_tail_blobs(blobs, headers)):
		offset, size = blobs[index]
		end = offset + size
		found = find_limb_start(data, base, end)
		if found is None:
			skipped += 1
			continue
		start, extent = found
		limb_end = start + extent
		old_post = -(limb_end - base) % 16
		if data[limb_end:limb_end + old_post] != b"\x00" * old_post:
			# not padding, so this is not the end of the structure after all
			skipped += 1
			continue
		want = end + (-(end - base) % 8)
		if want == start:
			continue
		body = data[start:start + extent]
		new_post = -(want + extent - base) % 16
		data = (data[:end] + b"\x00" * (want - end) + body + b"\x00" * new_post
				+ data[limb_end + old_post:])
		moved += 1
	return data, moved, skipped

def block_layout(data: bytes, blobs, headers, transform_track_type: int = 12):
	"""Describe each compressed ManiBlock's tail, as it currently sits in the file.

	Returns one dict per block:

	    blobs        [(offset, size), ...] - the transform blob and, if the clip has
	                 scalar channels, the scalar blob that follows it
	    limb         (offset, extent) or None
	    end          where the next block begins: the first 16-byte boundary, in
	                 buffer coordinates, after whatever came last

	This is the unit an edit has to be applied to. Replacing a blob on its own
	cannot work: `splice.replace_blob` assumes the padding after every blob rounds
	to 16, but the padding after a block's LAST blob rounds to 8, so it consumes up
	to 8 bytes too many and eats the head of the limb structure - which is a run of
	zeros, so nothing complains and the structure silently shifts.
	"""
	base = buffer_residue(blobs)
	out = []
	index = 0
	while index < len(blobs):
		group = [blobs[index]]
		if (headers[index]["track_type"] == transform_track_type
				and index + 1 < len(headers)
				and headers[index + 1]["track_type"] != transform_track_type):
			index += 1
			group.append(blobs[index])
		index += 1
		last_end = group[-1][0] + group[-1][1]
		limit = blobs[index][0] if index < len(blobs) else len(data)
		limb = find_limb_start(data, base, last_end, limit)
		tail = limb[0] + limb[1] if limb else last_end
		out.append({
			"blobs": group,
			"limb": limb,
			"end": tail + (-(tail - base) % 16),
		})
	return out


def rebuild_block(data: bytes, block: dict, base: int, new_blobs) -> bytes:
	"""Splice one ManiBlock's tail back in with `new_blobs` in place of its blobs.

	Lays the tail out the way the runtime reads it: align16 between a block's two
	blobs, **align8 before the limb structure**, then align16 so the next block
	still starts where cobra's KeysReader expects to find it.
	"""
	if len(new_blobs) != len(block["blobs"]):
		raise ValueError(f"block has {len(block['blobs'])} blobs, got {len(new_blobs)}")
	start = block["blobs"][0][0]
	pieces = []
	cursor = start
	for position, blob in enumerate(new_blobs):
		if position:
			pad = -(cursor - base) % 16
			pieces.append(b"\x00" * pad)
			cursor += pad
		pieces.append(blob)
		cursor += len(blob)
	if block["limb"]:
		limb_start, extent = block["limb"]
		pad = -(cursor - base) % 8
		pieces.append(b"\x00" * pad)
		cursor += pad
		pieces.append(data[limb_start:limb_start + extent])
		cursor += extent
	pad = -(cursor - base) % 16
	pieces.append(b"\x00" * pad)
	return data[:start] + b"".join(pieces) + data[block["end"]:]


def replace_clip_blob(data: bytes, index: int, new_blob: bytes) -> bytes:
	"""Swap one ACL blob, keeping its ManiBlock's limb structure where it belongs.

	Use this instead of `splice.replace_blob` whenever the replacement can be a
	different size. `replace_blob` pads to 16 after every blob; the padding after a
	block's LAST blob rounds to 8, so it eats the head of the limb structure and
	the game dies on spawn at `JWE3.exe+0x1697FBD`. This re-lays the whole block
	instead, which is the only unit the layout is well defined over.

	`splice.replace_blob` remains correct for a same-size swap, where nothing moves.
	"""
	from source.formats.manis.splice import list_clip_blobs, read_blob_header

	blobs = list_clip_blobs(data)
	if not 0 <= index < len(blobs):
		raise IndexError(f"blob {index} out of range (found {len(blobs)})")
	headers = [read_blob_header(data, offset) for offset, _size in blobs]
	base = buffer_residue(blobs)
	first = 0
	for block in block_layout(data, blobs, headers):
		count = len(block["blobs"])
		if first <= index < first + count:
			replacement = [data[offset:offset + size] for offset, size in block["blobs"]]
			replacement[index - first] = new_blob
			return rebuild_block(data, block, base, replacement)
		first += count
	raise IndexError(f"blob {index} belongs to no ManiBlock")

