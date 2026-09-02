"""Repair graph references left on a species' PRE-RENAME clip names.

Renaming a species rewrites the clip names inside the .manis and creates new
`<NewToken>$Clip` strings in the OVL, but references in the motiongraph can be
left pointing at the old `<OldToken>$Clip` strings. Such a reference names a clip
that no longer exists in any bundle, so whatever plays it has no animation - and
nothing reports an error, at load or at runtime.

Measured on SarcosuchusViral (renamed from the `sarcmimsaee` donor token):
80 fragments still pointed at `Sarcmimsaee$` strings, including 20 for
`StandIdle01` and 12 across the FightFinish set.

The repair is a pure FRAGMENT repoint: every affected pointer is aimed at the
already-present correctly-named string. No pool grows, no bytes are written into
any pool, the fragment count does not change, and the archive does not need
recompressing beyond a normal rewrite. That makes this the safest class of edit
available - "prefer operations that preserve bytes you do not understand".

The old strings are deliberately LEFT in place. They become unreferenced, which
shows up as dead tail bytes; overwriting them is a separate decision, and this
archive proves why caution is warranted - cobra's dead-space audit reported 441
"dead" bytes at pool 2 offset 8461 that turned out to hold live strings.

Cobra reload is not proof of engine acceptance. Verify in game.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .clone import COMPRESSED_SIZE_OFFSET, UNCOMPRESSED_SIZE_OFFSET
from .edit import DEFAULT_GAME
from .surgical_growth import _load_quiet

STRING_POOL_TYPE = 2


@dataclass(frozen=True)
class RenameRepairReport:
	output: Path
	old_token: str
	new_token: str
	repointed: int
	missing: tuple[str, ...]
	distinct_names: tuple[str, ...]


def _index_strings(ovl, ovs):
	"""Return (name -> (local_pool, offset), (local_pool, offset) -> name)."""
	local = {id(pool): i for i, pool in enumerate(ovs.pools)}
	by_name, by_site = {}, {}
	for pool in ovl.pools:
		if int(pool.type) != STRING_POOL_TYPE or id(pool) not in local:
			continue
		local_pool = local[id(pool)]
		data = pool.data.getvalue()
		offset = 0
		while offset < len(data):
			end = data.find(b"\0", offset)
			if end < 0:
				break
			raw = data[offset:end]
			if raw:
				try:
					name = raw.decode("ascii")
				except UnicodeDecodeError:
					name = None
				if name:
					by_name.setdefault(name, (local_pool, offset))
					by_site[(local_pool, offset)] = name
			offset = end + 1
	return by_name, by_site


def survey_stale_references(source: Path, old_token: str, new_token: str,
							game: str = DEFAULT_GAME):
	"""Return (repointable, missing) without modifying anything.

	`repointable` is a list of (old_name, new_name, count); `missing` lists old
	names whose renamed counterpart does not exist and so cannot be repaired here.
	"""
	ovl, static = _load_quiet(Path(source).resolve(), game)
	ovs = static.content
	by_name, by_site = _index_strings(ovl, ovs)
	prefix = old_token if old_token.endswith("$") else old_token + "$"
	replacement = new_token if new_token.endswith("$") else new_token + "$"

	counts, missing = {}, set()
	for row in ovs.fragments:
		name = by_site.get((int(row["struct_pool"]), int(row["struct_offset"])))
		if not name or not name.startswith(prefix):
			continue
		wanted = replacement + name[len(prefix):]
		if wanted in by_name:
			counts[(name, wanted)] = counts.get((name, wanted), 0) + 1
		else:
			missing.add(name)
	repointable = [(old, new, n) for (old, new), n in sorted(counts.items())]
	return repointable, tuple(sorted(missing))


def repoint_stale_species_strings(source: Path, output: Path,
								  old_token: str, new_token: str,
								  game: str = DEFAULT_GAME) -> RenameRepairReport:
	"""Aim every stale `old_token$` reference at its `new_token$` counterpart."""
	source, output = Path(source).resolve(), Path(output).resolve()
	if source == output:
		raise ValueError("Refusing to overwrite the source OVL")
	if source.name.lower() != output.name.lower():
		raise ValueError("Source and staged output OVL basenames must match")
	if not output.is_file():
		raise ValueError("Copy the complete archive family to the stage directory first")

	ovl, static = _load_quiet(source, game)
	ovs = static.content
	by_name, by_site = _index_strings(ovl, ovs)
	prefix = old_token if old_token.endswith("$") else old_token + "$"
	replacement = new_token if new_token.endswith("$") else new_token + "$"

	fragments = ovs.fragments
	old_pool_sizes = tuple(int(pool.size) for pool in ovs.pools)
	old_fragments = int(static.num_fragments)
	old_uncompressed = int(static.uncompressed_size)
	old_compressed = int(static.compressed_size)

	repointed, missing, touched = 0, set(), set()
	for row in fragments:
		site = (int(row["struct_pool"]), int(row["struct_offset"]))
		name = by_site.get(site)
		if not name or not name.startswith(prefix):
			continue
		wanted = replacement + name[len(prefix):]
		target = by_name.get(wanted)
		if target is None:
			missing.add(name)
			continue
		row["struct_pool"], row["struct_offset"] = target
		repointed += 1
		touched.add(name)
	if not repointed:
		raise ValueError(f"No repointable {prefix} references found")

	ovs.fragments.sort(order=("link_pool", "struct_pool", "link_offset", "struct_offset"))
	ovs.write_pools()
	uncompressed = ovs.write_archive()
	if tuple(int(pool.size) for pool in ovs.pools) != old_pool_sizes:
		raise ValueError("A pool changed size while repointing references")
	if len(uncompressed) != old_uncompressed:
		raise ValueError(
			f"STATIC size changed while repointing: {old_uncompressed} -> "
			f"{len(uncompressed)}; this edit must not resize anything")
	if int(static.num_fragments) != old_fragments:
		raise ValueError("Fragment count changed while repointing references")

	_, new_compressed, compressed = ovs.compress(uncompressed, True)
	source_bytes = source.read_bytes()
	result = bytearray(source_bytes[:len(source_bytes) - old_compressed])
	result.extend(compressed)
	head = int(static.io_start)
	struct.pack_into("<I", result, head + COMPRESSED_SIZE_OFFSET, new_compressed)
	struct.pack_into("<Q", result, head + UNCOMPRESSED_SIZE_OFFSET, len(uncompressed))
	output.write_bytes(result)

	check_ovl, check_static = _load_quiet(output, game)
	_, check_sites = _index_strings(check_ovl, check_static.content)
	left = 0
	for row in check_static.content.fragments:
		name = check_sites.get((int(row["struct_pool"]), int(row["struct_offset"])))
		if name and name.startswith(prefix) and name not in missing:
			left += 1
	if left:
		raise ValueError(f"{left} repairable {prefix} references survived the rewrite")

	return RenameRepairReport(
		output=output, old_token=prefix, new_token=replacement,
		repointed=repointed, missing=tuple(sorted(missing)),
		distinct_names=tuple(sorted(touched)))
