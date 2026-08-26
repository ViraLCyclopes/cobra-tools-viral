"""Locate and verify the ACL compressed_database bulk blocks in a .manis.

JWE3 strips per-frame detail into database bulk blocks which cobra appends to the
.manis tail as [low tier][medium tier]EOF, each padded to an alignment boundary.
The database header stores each tier's size and an FNV1a-32 hash, so a candidate
offset can be verified rather than guessed - important, because ACL's
database_context::initialize does not validate bulk contents and a wrong offset
crashes during decompression.
"""
from __future__ import annotations

import struct

TRACKS_TAG = 0xAC11AC11
DB_TAG = 0xAC11DB01
MAX_PADDING = 256


def _fnv1a_32(buf) -> int:
	h = 2166136261
	for b in buf:
		h = ((h ^ b) * 16777619) & 0xFFFFFFFF
	return h


def find_database(data: bytes):
	"""Return (offset, size) of the compressed_database blob, or None."""
	tag = struct.pack("<I", DB_TAG)
	off = data.find(tag)
	while off != -1:
		start = off - 8
		if start >= 0:
			size = struct.unpack_from("<I", data, start)[0]
			if 32 <= size and start + size <= len(data):
				return start, size
		off = data.find(tag, off + 1)
	return None


def read_bulk_info(data: bytes):
	"""Return the database header's bulk sizes and hashes, or None if no database.

	The header lays out bulk_data_size[2], bulk_data_offset[2], bulk_data_hash[2]
	consecutively. Stripped bulk data is marked by offsets of 0xFFFFFFFF, which is
	what we anchor on to find the right position in the header.
	"""
	found = find_database(data)
	if found is None:
		return None
	db_off, db_size = found
	hdr = data[db_off:db_off + db_size]
	for i in range(0, max(0, db_size - 24), 4):
		med, low = struct.unpack_from("<II", hdr, i)
		if med == 0 or low == 0:
			continue
		if med > len(data) or low > len(data):
			continue
		o0, o1 = struct.unpack_from("<II", hdr, i + 8)
		if o0 != 0xFFFFFFFF or o1 != 0xFFFFFFFF:
			continue
		h_med, h_low = struct.unpack_from("<II", hdr, i + 16)
		return {
			"db_offset": db_off,
			"db_size": db_size,
			"medium_size": med,
			"low_size": low,
			"medium_hash": h_med,
			"low_hash": h_low,
		}
	return None


def _find_before(data: bytes, size: int, want_hash: int, search_end: int):
	"""Search backwards from search_end for a block of `size` matching want_hash."""
	if size > search_end or search_end > len(data):
		return None
	for pad in range(MAX_PADDING + 1):
		if size + pad > search_end:
			break
		off = search_end - pad - size
		if _fnv1a_32(data[off:off + size]) == want_hash:
			return off
	return None


def locate_bulk(data: bytes):
	"""Return {'medium_offset', 'low_offset'} verified by hash, or None."""
	info = read_bulk_info(data)
	if info is None:
		return None
	med = _find_before(data, info["medium_size"], info["medium_hash"], len(data))
	if med is None:
		return None
	low = _find_before(data, info["low_size"], info["low_hash"], med)
	if low is None:
		return None
	return {"medium_offset": med, "low_offset": low}


def verify_bulk(path: str):
	"""Return (ok, message) describing whether the database bulk is intact."""
	data = open(path, "rb").read()
	info = read_bulk_info(data)
	if info is None:
		return True, "no database in this file (nothing to verify)"
	med = _find_before(data, info["medium_size"], info["medium_hash"], len(data))
	if med is None:
		return False, "medium tier bulk not found or hash mismatch"
	low = _find_before(data, info["low_size"], info["low_hash"], med)
	if low is None:
		return False, "low tier bulk not found or hash mismatch"
	return True, (f"bulk intact: low @{low} ({info['low_size']}), "
				  f"medium @{med} ({info['medium_size']})")


def _patch_bulk_clip_hash(out: bytearray, bulk_offset: int, bulk_size: int,
						  descriptions: tuple[tuple[int, int], ...],
						  clip_header_offset: int, old_hash: int,
						  new_hash: int) -> int:
	"""Patch chunk segment headers belonging to one runtime clip record."""
	bulk_end = bulk_offset + bulk_size
	patched = 0
	for described_size, relative_offset in descriptions:
		chunk = bulk_offset + relative_offset
		if chunk < bulk_offset or chunk + 12 > bulk_end:
			raise ValueError("database chunk description points outside bulk data")
		index, chunk_size, num_segments = struct.unpack_from("<III", out, chunk)
		if chunk_size != described_size or chunk + chunk_size > bulk_end:
			raise ValueError(
				f"database chunk {index} size {chunk_size} does not match "
				f"description {described_size}")
		headers_end = chunk + 12 + num_segments * 20
		if headers_end > chunk + chunk_size:
			raise ValueError(f"database chunk {index} has truncated segment headers")
		for segment in range(num_segments):
			header = chunk + 12 + segment * 20
			segment_hash, = struct.unpack_from("<I", out, header)
			segment_clip_offset, = struct.unpack_from("<I", out, header + 12)
			if segment_clip_offset != clip_header_offset:
				continue
			if segment_hash != old_hash:
				raise ValueError(
					f"database chunk {index} clip hash 0x{segment_hash:08X} "
					f"does not match old hash 0x{old_hash:08X}")
			struct.pack_into("<I", out, header, new_hash)
			patched += 1
	return patched


def rekey_database_clip(data: bytes, clip_blob: bytes, old_hash: int,
						new_hash: int):
	"""Return a same-size MANIS with one database-backed ACL clip re-keyed.

	ACL keys database clips by the hash in the compressed_tracks raw header.  A
	size-preserving payload edit therefore has to update the matching clip metadata
	entry and its streamed chunk segment headers.  The clip's runtime-header offset
	is used as the identity, avoiding ambiguous hash-only replacement.
	"""
	if len(clip_blob) < 84:
		raise ValueError("ACL transform blob is too small for database metadata")
	transform = 32
	database_header_offset, = struct.unpack_from("<I", clip_blob, transform + 32)
	if database_header_offset == 0xFFFFFFFF:
		raise ValueError("ACL clip is not database-backed")
	tracks_database_header = transform + database_header_offset
	if tracks_database_header + 4 > len(clip_blob):
		raise ValueError("ACL tracks database header points outside clip")
	clip_header_offset, = struct.unpack_from("<I", clip_blob, tracks_database_header)
	if clip_header_offset == 0xFFFFFFFF:
		raise ValueError("ACL clip has no runtime database record")

	found = find_database(data)
	info = read_bulk_info(data)
	bulk = locate_bulk(data)
	if found is None or info is None or bulk is None:
		raise ValueError("MANIS compressed database or verified bulk data not found")
	db_offset, db_size = found
	db_header = db_offset + 8
	if db_header + 56 > db_offset + db_size:
		raise ValueError("compressed database header is truncated")
	num_medium, num_low = struct.unpack_from("<II", data, db_header + 8)
	num_clips, = struct.unpack_from("<I", data, db_header + 20)
	clip_metadata_offset, = struct.unpack_from("<I", data, db_header + 28)
	metadata = db_header + clip_metadata_offset
	if metadata < db_header or metadata + num_clips * 8 > db_offset + db_size:
		raise ValueError("database clip metadata points outside database")

	out = bytearray(data)
	matches = []
	for clip_index in range(num_clips):
		entry = metadata + clip_index * 8
		entry_hash, entry_offset = struct.unpack_from("<II", out, entry)
		if entry_offset == clip_header_offset:
			matches.append((clip_index, entry, entry_hash))
	if len(matches) != 1:
		raise ValueError(
			f"runtime clip offset {clip_header_offset} matched {len(matches)} metadata entries")
	clip_index, entry, entry_hash = matches[0]
	if entry_hash != old_hash:
		raise ValueError(
			f"database clip {clip_index} hash 0x{entry_hash:08X} does not match "
			f"old hash 0x{old_hash:08X}")
	struct.pack_into("<I", out, entry, new_hash)

	descriptions_base = db_header + 56
	def descriptions(start: int, count: int):
		end = start + count * 8
		if end > db_offset + db_size:
			raise ValueError("database chunk descriptions are truncated")
		return tuple(struct.unpack_from("<II", out, start + index * 8)
					 for index in range(count))
	medium_descriptions = descriptions(descriptions_base, num_medium)
	low_descriptions = descriptions(descriptions_base + num_medium * 8, num_low)
	medium_segments = _patch_bulk_clip_hash(
		out, bulk["medium_offset"], info["medium_size"], medium_descriptions,
		clip_header_offset, old_hash, new_hash)
	low_segments = _patch_bulk_clip_hash(
		out, bulk["low_offset"], info["low_size"], low_descriptions,
		clip_header_offset, old_hash, new_hash)

	medium_hash = _fnv1a_32(
		out[bulk["medium_offset"]:bulk["medium_offset"] + info["medium_size"]])
	low_hash = _fnv1a_32(
		out[bulk["low_offset"]:bulk["low_offset"] + info["low_size"]])
	struct.pack_into("<II", out, db_header + 48, medium_hash, low_hash)
	struct.pack_into("<I", out, db_offset + 4,
				 _fnv1a_32(out[db_offset + 8:db_offset + db_size]))
	return bytes(out), {
		"clip_index": clip_index,
		"clip_header_offset": clip_header_offset,
		"medium_segments": medium_segments,
		"low_segments": low_segments,
		"medium_hash": medium_hash,
		"low_hash": low_hash,
	}
