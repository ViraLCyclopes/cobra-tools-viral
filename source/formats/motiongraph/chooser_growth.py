"""Grow a RandomAnimationActivity chooser by one clip.

A `RandomAnimationActivityData` picks among clips by NAME - its entries are
`ActivityAnimationInfo` records holding a ZString pointer, not references to
`Activity` objects. That is why `Rest01` has an entry but no `AnimationActivity`
anywhere in the graph, and it is what makes adding a clip to a chooser far
cheaper than cloning an activity: no wrapper, no payload, no state, no edge.

    RandomAnimationActivityData          ActivityAnimationInfo (16 bytes)
      +0  NumAnimations  uint64            +0   activity_name  Pointer -> ZString
      +8  Animations     ArrayPointer      +8   Offset         float
                                           +12  Weight         uint

`Animations` is an ArrayPointer, so the entry array is separately addressed and
can be RELOCATED rather than grown in place. This matters: every chooser array in
the shipped Sarcosuchus graph is exactly `count * 16` bytes with zero slack, and
both neighbours of the rest chooser's array are full, so in-place growth is
impossible. Relocation sidesteps that entirely.

Composed from primitives that are individually game-verified:

* append into the final PARTIAL page tail of a pool (`append_tail_pool_bytes`),
  never past a page - Tests F/F2 crash every time when a full page is exceeded
* relocate the end-of-pool sentinels before occupying that address
  (`repoint_pool_end_fragments`) - Test G, ~1,463 type-3 sentinels share it
* strings may live outside the string pool: append to a type-2 tail and repoint
  (Tests X, Y - any length, any count)
* a grown counted array is accepted (Test D, State.activities 15 -> 16) but a
  NULL in the new slot is not: that build reached the menu, spawned, and ran over
  a minute before an access violation at JWE3.exe+0x1B0A117 reading 0x10 from a
  null base. The delay makes it look like a pass, so the new entry here always
  carries a real string fragment.

The donor array is left in place and simply stops being referenced, which shows
up as dead tail bytes rather than as a decode failure. ALWAYS follow this with
`motiongraph-census NEW --against OLD`: raw and semantic reload both pass on a
build that silently dropped an object.

Cobra reload is NOT proof of engine acceptance. Every one of these needs a game
launch.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .clone import (
	COMPRESSED_SIZE_OFFSET,
	NUM_FRAGMENTS_OFFSET,
	POOLS_END_OFFSET,
	TYPE_3_PAGE_LIMIT,
	UNCOMPRESSED_SIZE_OFFSET,
	clone_fragment_sources,
	internal_fragment_mask,
)
from .edit import DEFAULT_GAME, load_motiongraph
from .report import build_deref
from .surgical_growth import _load_quiet, append_tail_pool_bytes, repoint_pool_end_fragments

ENTRY_SIZE = 16
COUNT_FIELD_OFFSET = 0
ARRAY_POINTER_OFFSET = 8


def list_choosers(source: Path, name: str | None = None, game: str = DEFAULT_GAME):
	"""Return every RandomAnimationActivity chooser with its clips and odds.

	`weight` is relative, not a percentage - the engine draws from the sum - so the
	share is reported alongside it, which is what anyone actually wants to tune.
	"""
	ovl, loader = load_motiongraph(Path(source).resolve(), name or None, game)
	deref = build_deref(loader)
	out = []
	for (pool, offset), obj in loader.context.recursion.items():
		if type(obj).__name__ != "RandomAnimationActivityData":
			continue
		entries = list(deref(obj.animations) or [])
		if not entries:
			continue
		weights = [int(e.weight) for e in entries]
		total = sum(weights) or 1
		out.append({
			"pool": int(pool.i), "offset": int(offset),
			"array_pool": int(obj.animations.target_pool.i),
			"array_offset": int(obj.animations.target_offset),
			"count": int(obj.num_animations),
			"blend_time": float(obj.blend_time),
			"flags": int(obj.random_animation_flags),
			"clips": [{
				"name": str(deref(e.activity_name)),
				"short": str(deref(e.activity_name)).split("$")[-1],
				"weight": w,
				"share": w / total,
				"weight_at": (int(obj.animations.target_pool.i),
							  int(obj.animations.target_offset) + i * ENTRY_SIZE + 12),
			} for i, (e, w) in enumerate(zip(entries, weights))],
		})
	return sorted(out, key=lambda r: (r["pool"], r["offset"]))


def set_chooser_weights(source: Path, output: Path, chooser_pool: int,
						chooser_offset: int, weights, name: str | None = None,
						game: str = DEFAULT_GAME):
	"""Rewrite one chooser's entry weights. Sizes never change, so this is a
	fixed-topology byte edit - the safest operation available."""
	source, output = Path(source).resolve(), Path(output).resolve()
	if source == output:
		raise ValueError("Refusing to overwrite the source OVL")
	if source.name.lower() != output.name.lower() or not output.is_file():
		raise ValueError("Output must be a staged same-basename OVL family")

	rows = [r for r in list_choosers(source, name, game)
			if r["pool"] == chooser_pool and r["offset"] == chooser_offset]
	if not rows:
		raise ValueError(f"No chooser at {chooser_pool}:{chooser_offset}")
	row = rows[0]
	weights = [int(w) for w in weights]
	if len(weights) != len(row["clips"]):
		raise ValueError(f"chooser has {len(row['clips'])} clips, got {len(weights)} weights")
	if any(w < 0 or w > 0xFFFFFFFF for w in weights):
		raise ValueError("weights must fit in a uint32")
	if not any(weights):
		raise ValueError("at least one weight must be non-zero or nothing can be drawn")

	ovl, static = _load_quiet(source, game)
	ovs = static.content
	pool = ovl.pools[row["array_pool"]]
	old_sizes = tuple(int(p.size) for p in ovs.pools)
	old_uncompressed = int(static.uncompressed_size)
	old_compressed = int(static.compressed_size)
	for clip, weight in zip(row["clips"], weights):
		pool.data.seek(clip["weight_at"][1])
		pool.data.write(struct.pack("<I", weight))
	pool.data.seek(0)

	ovs.write_pools()
	uncompressed = ovs.write_archive()
	if len(uncompressed) != old_uncompressed or \
			tuple(int(p.size) for p in ovs.pools) != old_sizes:
		raise ValueError("a weight edit must not resize anything")
	_, new_compressed, compressed = ovs.compress(uncompressed, True)
	raw = source.read_bytes()
	result = bytearray(raw[:len(raw) - old_compressed])
	result.extend(compressed)
	head = int(static.io_start)
	struct.pack_into("<I", result, head + COMPRESSED_SIZE_OFFSET, new_compressed)
	struct.pack_into("<Q", result, head + UNCOMPRESSED_SIZE_OFFSET, len(uncompressed))
	output.write_bytes(result)

	check = [r for r in list_choosers(output, name, game)
			 if r["pool"] == chooser_pool and r["offset"] == chooser_offset][0]
	got = [c["weight"] for c in check["clips"]]
	if got != weights:
		raise ValueError(f"reloaded weights are {got}, expected {weights}")
	return check


@dataclass(frozen=True)
class ChooserGrowthReport:
	output: Path
	chooser: tuple[int, int]
	old_array: tuple[int, int]
	new_array: tuple[int, int]
	string: tuple[int, int]
	clip_name: str
	entries_before: int
	entries_after: int
	names: tuple[str, ...]
	fragments_before: int
	fragments_after: int
	pool_growth: int


def grow_random_animation_chooser(
	source: Path,
	output: Path,
	chooser_pool: int,
	chooser_offset: int,
	clip_name: str,
	weight: int | None = None,
	offset_value: float | None = None,
	name: str | None = None,
	game: str = DEFAULT_GAME,
) -> ChooserGrowthReport:
	"""Append `clip_name` to the chooser at `chooser_pool:chooser_offset`.

	Pool indices are GLOBAL at the API boundary, matching `activity_probe.py` and
	the capacity audit. `weight` and `offset_value` default to whatever the
	chooser's LAST existing entry uses, so the new clip is drawn on the same terms
	as its siblings unless the caller says otherwise.
	"""
	source, output = Path(source).resolve(), Path(output).resolve()
	if source == output:
		raise ValueError("Refusing to overwrite the source OVL")
	if source.name.lower() != output.name.lower():
		raise ValueError("Source and staged output OVL basenames must match")
	if not output.is_file():
		raise ValueError("Copy the complete archive family to the stage directory first")
	if not clip_name or any(ord(c) > 126 or ord(c) < 32 for c in clip_name):
		raise ValueError(f"Clip name must be printable ASCII: {clip_name!r}")

	semantic_ovl, semantic_loader = load_motiongraph(source, name or None, game)
	deref = build_deref(semantic_loader)
	chooser = semantic_loader.context.recursion.get(
		(semantic_ovl.pools[chooser_pool], chooser_offset))
	if chooser is None:
		for (pool, offset), obj in semantic_loader.context.recursion.items():
			if int(pool.i) == chooser_pool and int(offset) == chooser_offset:
				chooser = obj
				break
	if chooser is None or type(chooser).__name__ != "RandomAnimationActivityData":
		raise ValueError(
			f"No RandomAnimationActivityData at {chooser_pool}:{chooser_offset} "
			f"(found {type(chooser).__name__ if chooser else 'nothing'})")

	entries = list(deref(chooser.animations) or [])
	count = int(chooser.num_animations)
	if len(entries) != count:
		raise ValueError(f"num_animations is {count} but {len(entries)} entries decoded")
	existing = tuple(str(deref(entry.activity_name)) for entry in entries)
	if clip_name in existing:
		raise ValueError(f"{clip_name!r} is already in this chooser: {existing}")
	if weight is None:
		weight = int(entries[-1].weight)
	if offset_value is None:
		offset_value = float(entries[-1].offset)

	array_global = int(chooser.animations.target_pool.i)
	array_offset = int(chooser.animations.target_offset)

	ovl, static = _load_quiet(source, game)
	ovs = static.content

	def local(global_index: int) -> int:
		if global_index < 0 or global_index >= len(ovl.pools):
			raise ValueError(f"Global pool is out of range: {global_index}")
		pool = ovl.pools[global_index]
		if pool not in ovs.pools:
			raise ValueError(f"Global pool {global_index} is not in STATIC")
		return ovs.pools.index(pool)

	chooser_local, array_local = local(chooser_pool), local(array_global)
	old_bytes = ovl.pools[array_global].data.getvalue()[
		array_offset:array_offset + count * ENTRY_SIZE]
	if len(old_bytes) != count * ENTRY_SIZE:
		raise ValueError("Could not read the complete chooser entry array")

	fragments = ovs.fragments
	array_rows = fragments[
		internal_fragment_mask(fragments, array_local, array_offset, len(old_bytes))].copy()
	expected = [index * ENTRY_SIZE for index in range(count)]
	if sorted(int(x) - array_offset for x in array_rows["link_offset"]) != expected:
		raise ValueError(
			f"Chooser entry array does not have exactly one name pointer per entry "
			f"at {expected}")
	pointer_mask = (
		(fragments["link_pool"] == chooser_local)
		& (fragments["link_offset"] == chooser_offset + ARRAY_POINTER_OFFSET))
	if int(pointer_mask.sum()) != 1:
		raise ValueError(
			f"Expected exactly one Animations ArrayPointer fragment at "
			f"{chooser_local}:{chooser_offset + ARRAY_POINTER_OFFSET}, "
			f"found {int(pointer_mask.sum())}")

	# the new entry's own bytes: the name pointer is carried by a fragment, so the
	# in-pool 8 bytes stay zero exactly as they are for every existing entry
	new_entry = bytearray(ENTRY_SIZE)
	struct.pack_into("<f", new_entry, 8, float(offset_value))
	struct.pack_into("<I", new_entry, 12, int(weight))
	new_array = bytes(old_bytes) + bytes(new_entry)

	old_fragments = int(static.num_fragments)
	old_uncompressed = int(static.uncompressed_size)
	old_compressed = int(static.compressed_size)
	old_pools_end = int(static.pools_end)
	old_pool_sizes = tuple(int(pool.size) for pool in ovs.pools)
	static_index = ovl.archives.index(static)
	old_reservation = int(ovl.archives_meta[static_index].unk_0)

	# Prefer an EXISTING string. Appending one needs room in the final type-2 pool,
	# and this archive's is exactly full (31,544 of 31,544) where Acro had ~700 names
	# spare - so the Test X/Y append is not always available. A name already present
	# costs nothing and cannot dangle. The usual way to get one is to overwrite an
	# orphaned pre-rename string in place via `motiongraph-string-slot`, which is
	# capped at the donor's length but needs no pool growth at all.
	#
	# Type 2 is not negotiable for strings: all 1,269 shipped string allocations live
	# in type-2 pools and none in type 3, so putting one in the type-3 tail - which
	# does have room - would be unprecedented.
	reused_site = None
	for pool_index, pool in enumerate(ovl.pools):
		if int(pool.type) != 2 or pool not in ovs.pools:
			continue
		blob = pool.data.getvalue()
		needle = clip_name.encode("ascii") + b"\0"
		at = blob.find(needle)
		while at != -1:
			if at == 0 or blob[at - 1] == 0:
				reused_site = (ovs.pools.index(pool), at)
				break
			at = blob.find(needle, at + 1)
		if reused_site:
			break

	if reused_site is not None:
		class _Reused:
			local_pool, offset = reused_site
			old_size = new_size = int(ovs.pools[reused_site[0]].size)
			global_pool = next(i for i, candidate in enumerate(ovl.pools)
							   if candidate is ovs.pools[reused_site[0]])
		string_allocation = _Reused()
	else:
		# strings do not have to live in the string pool - Tests X and Y
		string_allocation = append_tail_pool_bytes(
			ovl.pools, ovs.pools, 2, clip_name.encode("ascii") + b"\0", alignment=16)
		repoint_pool_end_fragments(
			fragments, string_allocation.local_pool,
			string_allocation.old_size, string_allocation.new_size)
	array_allocation = append_tail_pool_bytes(
		ovl.pools, ovs.pools, 3, new_array, alignment=16, page_size=TYPE_3_PAGE_LIMIT)
	repoint_pool_end_fragments(
		fragments, array_allocation.local_pool,
		array_allocation.old_size, array_allocation.new_size)

	# the owner now points at the relocated, longer array
	fragments["struct_pool"][pointer_mask] = array_allocation.local_pool
	fragments["struct_offset"][pointer_mask] = array_allocation.offset

	cloned = clone_fragment_sources(
		array_rows, array_allocation.local_pool, array_offset, array_allocation.offset)
	added = np.array(
		[(array_allocation.local_pool, array_allocation.offset + count * ENTRY_SIZE,
		  string_allocation.local_pool, string_allocation.offset)],
		dtype=fragments.dtype)
	new_rows = np.concatenate((cloned, added))
	ovs.fragments = np.concatenate((fragments, new_rows))
	ovs.fragments.sort(order=("link_pool", "struct_pool", "link_offset", "struct_offset"))
	static.num_fragments = old_fragments + len(new_rows)

	# NumAnimations is a uint64 at the head of the payload
	chooser_data = ovl.pools[chooser_pool].data
	chooser_data.seek(chooser_offset + COUNT_FIELD_OFFSET)
	if struct.unpack("<Q", chooser_data.read(8))[0] != count:
		raise ValueError("NumAnimations is not where the schema says it is")
	chooser_data.seek(chooser_offset + COUNT_FIELD_OFFSET)
	chooser_data.write(struct.pack("<Q", count + 1))
	chooser_data.seek(0)

	ovs.write_pools()
	uncompressed = ovs.write_archive()
	pool_growth = (
		string_allocation.new_size - string_allocation.old_size
		+ array_allocation.new_size - array_allocation.old_size)
	expected_uncompressed = old_uncompressed + pool_growth + len(new_rows) * 16
	if len(uncompressed) != expected_uncompressed:
		raise ValueError(
			f"Unexpected STATIC growth: {len(uncompressed)} vs {expected_uncompressed}")
	expected_sizes = list(old_pool_sizes)
	for allocation in (string_allocation, array_allocation):
		expected_sizes[allocation.local_pool] = allocation.new_size
	if tuple(int(pool.size) for pool in ovs.pools) != tuple(expected_sizes):
		raise ValueError("An unrelated pool changed size while growing the chooser")

	_, new_compressed, compressed = ovs.compress(uncompressed, True)
	source_bytes = source.read_bytes()
	header_size = len(source_bytes) - old_compressed
	meta_offset = header_size - len(ovl.archives_meta) * 8 + static_index * 8
	result = bytearray(source_bytes[:header_size])
	result.extend(compressed)
	head = int(static.io_start)
	struct.pack_into("<I", result, head + NUM_FRAGMENTS_OFFSET, static.num_fragments)
	struct.pack_into("<I", result, head + COMPRESSED_SIZE_OFFSET, new_compressed)
	struct.pack_into("<Q", result, head + UNCOMPRESSED_SIZE_OFFSET, expected_uncompressed)
	struct.pack_into("<I", result, head + POOLS_END_OFFSET, old_pools_end + pool_growth)
	struct.pack_into("<I", result, meta_offset, old_reservation + pool_growth)
	output.write_bytes(result)

	check_ovl, check_loader = load_motiongraph(output, name or None, game)
	check_deref = build_deref(check_loader)
	grown = None
	for (pool, offset), obj in check_loader.context.recursion.items():
		if int(pool.i) == chooser_pool and int(offset) == chooser_offset:
			grown = obj
			break
	if grown is None:
		raise ValueError("The chooser did not decode after growth")
	names = tuple(str(check_deref(entry.activity_name))
				  for entry in (check_deref(grown.animations) or []))
	wanted = existing + (clip_name,)
	if int(grown.num_animations) != count + 1 or names != wanted:
		raise ValueError(
			f"Reloaded chooser is wrong.\n"
			f"  num_animations {int(grown.num_animations)} (wanted {count + 1})\n"
			f"  got    {names!r}\n"
			f"  wanted {wanted!r}")

	return ChooserGrowthReport(
		output=output, chooser=(chooser_pool, chooser_offset),
		old_array=(array_global, array_offset),
		new_array=(array_allocation.global_pool, array_allocation.offset),
		string=(string_allocation.global_pool, string_allocation.offset),
		clip_name=clip_name, entries_before=count, entries_after=count + 1,
		names=names, fragments_before=old_fragments,
		fragments_after=int(static.num_fragments), pool_growth=pool_growth)
