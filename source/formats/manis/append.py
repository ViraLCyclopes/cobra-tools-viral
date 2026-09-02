"""Append a clip to a COMPRESSED JWE3 bundle, at byte level.

`ManisFile.duplicate()` + `save()` can add a clip, and that is how
`acrocanthosaurus$preen2` was first made - but `save()` cannot write JWE3's
compressed form. It drops the ACL database, the compressed blobs and the limb
structure, so the bundle it produces is entirely dtype 0: 23 clips, 0 ACL blobs,
13.8 MB against the compressed original's 1.36 MB. That path also brings the
charge bug and loses animated scale, which is why it was abandoned.

This adds a clip while everything stays compressed, by copying bytes rather than
re-serialising. A bundle is laid out as:

    [0  ..8)  version u16, mani_version u16, mani_count u32
    [8  ..)   stream ZString            e.g. "Anim_L0"
              mani_count clip-name ZStrings
              ManisRoot                 mani_files_size = 16 * count
                                        hash_block_size = 4 * string count
              ManiInfo x mani_count     304 bytes each
              compressed_database       ~8 bytes per clip, regenerated later
              name buffer
              keys buffer               one ManiBlock per clip
    [low_offset..]  ACL bulk, low then medium tier

A clip therefore needs: one name, one ManiInfo, one ManiBlock, and two counts
updated. The ManiBlock is copied whole from a donor - `[io_start, next io_start)`
- which carries its channel tables, its ACL container and blobs, AND the limb
structure that lives in the gap before the next block. Nothing inside it has to
be understood to move it.

Offsets are never stored absolutely (the database's bulk offsets are the stripped
0xFFFFFFFF sentinel and ManiInfos are positional), so a pure splice is safe.

The caller must then rebuild through `manis_database_cmd`, because the ACL
database is sized per clip and the copied blobs still carry the donor's database
bindings.
"""
from __future__ import annotations

import struct

MANI_INFO_SIZE = 304
COUNT_OFFSET = 4
MANI_FILES_SIZE_STRIDE = 16


def _zstring(text: str) -> bytes:
	return text.encode("utf-8") + b"\x00"


def preamble_layout(manis):
	"""Byte offsets of the regions before the compressed database.

	Mirrors how the writer lays them out; the caller checks the result against
	what the parser decoded before writing anything.
	"""
	from modules.helpers import as_bytes

	head = 8 + len(as_bytes(str(manis.stream or "")))
	names = sum(len(as_bytes(str(n))) for n in manis.names)
	root = len(as_bytes(manis.header))
	info_start = head + names + root
	return {
		"stream_end": head,
		"names_start": head,
		"names_end": head + names,
		"root_start": head + names,
		"root_end": info_start,
		"info_start": info_start,
		"info_end": info_start + len(manis.mani_infos) * MANI_INFO_SIZE,
	}


def block_extents(manis, keys_end):
	"""(start, end) of every ManiBlock including its trailing limb data.

	KeysReader records each block's `io_start`; the bytes between one block's
	start and the next are the block itself plus the limb structure that follows
	it, which is exactly the unit a clip owns.
	"""
	starts = []
	for info in manis.mani_infos:
		keys = getattr(info, "keys", None)
		start = getattr(keys, "io_start", None)
		if start is None:
			raise ValueError(f"clip {info.name} has no parsed ManiBlock; the bundle "
							 f"did not fully read and cannot be appended to")
		starts.append(int(start))
	if starts != sorted(starts):
		raise ValueError("ManiBlocks are not in file order; cannot take extents")
	return [(s, starts[i + 1] if i + 1 < len(starts) else keys_end)
			for i, s in enumerate(starts)]


def append_clip(data: bytes, manis, donor: str, new_name: str, keys_end: int) -> bytes:
	"""Return `data` with `donor` duplicated under `new_name`, still compressed.

	`keys_end` is where the keys buffer stops - the ACL bulk's low_offset.
	"""
	names = [str(i.name) for i in manis.mani_infos]
	if donor not in names:
		raise ValueError(f"donor {donor!r} is not in this bundle")
	if new_name in names:
		raise ValueError(f"{new_name!r} already exists")
	index = names.index(donor)
	count = len(names)

	layout = preamble_layout(manis)
	stored_count = struct.unpack_from("<I", data, COUNT_OFFSET)[0]
	if stored_count != count:
		raise ValueError(f"mani_count says {stored_count}, parser found {count}")

	extents = block_extents(manis, keys_end)
	donor_start, donor_end = extents[index]
	block = data[donor_start:donor_end]
	if len(block) % 16:
		raise ValueError(f"donor block is {len(block)} bytes, not a multiple of 16; "
						 f"appending it would break ManiBlock alignment")

	info = data[layout["info_start"] + index * MANI_INFO_SIZE:
				layout["info_start"] + (index + 1) * MANI_INFO_SIZE]

	# ManisRoot.mani_files_size is 16 * mani_count
	root = bytearray(data[layout["root_start"]:layout["root_end"]])
	files_size = struct.unpack_from("<H", root, 0)[0]
	if files_size != count * MANI_FILES_SIZE_STRIDE:
		raise ValueError(f"mani_files_size is {files_size}, expected "
						 f"{count * MANI_FILES_SIZE_STRIDE}")
	struct.pack_into("<H", root, 0, (count + 1) * MANI_FILES_SIZE_STRIDE)

	header = bytearray(data[:layout["stream_end"]])
	struct.pack_into("<I", header, COUNT_OFFSET, count + 1)

	return b"".join((
		bytes(header),
		data[layout["names_start"]:layout["names_end"]], _zstring(new_name),
		bytes(root),
		data[layout["info_start"]:layout["info_end"]], info,
		data[layout["info_end"]:keys_end], block,
		data[keys_end:],
	))


def retoken_clips(data: bytes, manis, old: str, new: str) -> bytes:
	"""Rewrite the species token in a bundle's clip names, in place.

	Lets a pristine bundle from a donor species be dropped into a renamed mod
	without going through the OVL: the token lives ONLY in the clip-name ZStrings
	(measured on Deinosuchus' idle bundle - 29 occurrences, all in the names
	region, none elsewhere in the file), so this is the same preamble splice
	`append_clip` performs.

	Pass the token WITH its `$` separator - `deinosuchus$` not `deinosuchus` - or
	an opponent reference like `fightfinishadeinosuchusleft` gets rewritten too and
	the fight pairing silently stops matching.

	`mani_files_size` and `hash_block_size` are untouched: the clip COUNT and the
	bone-name string count are both unchanged, and those are what they measure.
	"""
	if "$" not in old:
		raise ValueError(f"refusing to retoken on {old!r}: pass the token with its '$' "
						 f"separator, or opponent names in clip titles are hit too")
	names = [str(i.name) for i in manis.mani_infos]
	hits = [n for n in names if old in n]
	if not hits:
		raise ValueError(f"no clip name contains {old!r}")

	layout = preamble_layout(manis)
	rebuilt = b"".join(_zstring(n.replace(old, new)) for n in names)
	return (data[:layout["names_start"]] + rebuilt + data[layout["names_end"]:])
