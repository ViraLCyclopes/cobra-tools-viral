"""Decision-layer routes and new states.

Two operations, both game-verified on UltimasaurusCE 2026-09-06. See
`Motiongraph Research/Ultima Decision Chooser Growth Build 1/NOTES.md` and
`.../Ultima New State Build 2/NOTES.md`.

**Adding a decision result.** `StandPreen` / `StandPreen02` look like they are
not chooser-driven - each is a lone `AnimationActivity` the decision layer routes
to - but they ARE chooser-driven at the DECISION layer. Both are results of a
`MotionGraphEventHandling.RandomChoiceEndDecisionScope`, whose `children` is an
`ArrayPointer` of 32-byte `MRFChild`:

    MRFMember1 (72 bytes)             MRFChild (32 bytes)
      +0  LuaMethod   -> ZString        +0  count_0  uint64   1-based ordinal
      +8  count_0     uint64            +8  MRFMember -> MRFMember1
      +16 ptr_0       -> State          +16 ptr_1     -> Something
      +24 MotiongraphVars               +24 count_1  uint64   7 on every shipped child
      +32 dtype       uint64  8=output
      +40 num_children uint64         Something (16 bytes)      MotiongraphResultParam (24)
      +48 children    -> MRFChild[]     +0  ptr -> ResultParam    +0  Field -> "Weight"
      +56 count_4     uint64            +8  unk uint64  MUST be 1 +8  Value uint64
      +64 id          -> ZString                                  +16 Type  uint64  2

That array is an `ArrayPointer`, so it relocates like any other - it does not
need slack where it sits.

**Adding a state.** `StateArray {uint64 count; ArrayPointer states}`; the array
is `count * 8` `StateReference`s with zero slack, and it can end exactly at its
pool's end. Relocate it, bump the count, and register a new `State`.

Two rules learned the hard way:

* **Allocate a NEW `StateOutput` node; never retarget a shipped one.** They are
  shared - in UltimasaurusCE both preen `StateOutput` nodes carry a third inbound
  reference from a *transition-layer* decision block on top of the chooser array.
  Retargeting one in place silently redirects that route too. `inbound_count` is
  reported for exactly this reason.
* **Share the twin state's `array_2` by pointer.** Exits become identical to the
  twin's and there is no `TransStruct` tree to deep-clone - that tree holds a
  nested `StateArray` and a `TransitionConditionRecord` array. The *activities*
  array is copied fresh, because vanilla gives every state its own even when the
  contents are identical (STATE[169] and [170] share all 13 activity nodes
  through two separate arrays).

Cobra reload is NOT proof of engine acceptance. Both operations are topology
growth and every operation of this class has needed a game launch.
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
from .report import analyze_states, build_deref
from .surgical_growth import _load_quiet, append_tail_pool_bytes, repoint_pool_end_fragments
from .staging import CandidatePublication, validate_staged_pair

# Type-2 pools in shipped archives reach ~19.6 KB, but stay conservative and use
# the same explicit bound the type-3 appends use.
TYPE_2_PAGE_LIMIT = 16384

CHILD_SIZE = 32
SOMETHING_SIZE = 16
RESULT_PARAM_SIZE = 24
STATE_SIZE = 40
MRF_NODE_SIZE = 72
SLOT = 8
SPEED_OFFSET = 32   # AnimationActivityData.Speed.float, same constant clone.py uses

# MRFMember1 field offsets
LUA_METHOD_OFFSET = 0
PTR_0_OFFSET = 16
MG_VARS_OFFSET = 24
DTYPE_OFFSET = 32
NUM_CHILDREN_OFFSET = 40
CHILDREN_PTR_OFFSET = 48

# State field offsets
STATE_NUM_ACTIVITIES_OFFSET = 4
STATE_ACTIVITIES_OFFSET = 8
STATE_ARRAY_2_COUNT_OFFSET = 16
STATE_ARRAY_2_OFFSET = 24

STATE_OUTPUT_METHOD = "MotionGraph.StateOutput"
RANDOM_CHOICE_METHOD = "RandomChoiceEndDecisionScope"


@dataclass(frozen=True)
class DecisionGrowthReport:
	output: Path
	node: tuple
	old_results: int
	new_results: int
	target_state: int
	weight: int
	pool_growth: int
	added_fragments: int
	moved_end_sentinels: int


@dataclass(frozen=True)
class NewStateReport:
	output: Path
	twin_state: int
	new_state_index: int
	new_state_at: tuple
	state_node_at: tuple
	old_count: int
	new_count: int
	pool_growth: int
	added_fragments: int
	moved_end_sentinels: int
	route: dict | None = None      # set when the state was routed to in the same pass


def _addr(pointer):
	try:
		return int(pointer.target_pool.i), int(pointer.target_offset)
	except Exception:
		return None


def _walk_decision(deref, root):
	"""Yield every reachable decision node once, following shared children."""
	seen = set()
	stack = [root]
	while stack:
		node = stack.pop()
		if node is None or id(node) in seen:
			continue
		seen.add(id(node))
		yield node
		for child in (deref(node.children) or []):
			stack.append(deref(child.m_r_f_member))


def list_decision_choosers(source: Path, name: str | None = None, game: str = DEFAULT_GAME):
	"""Every `RandomChoiceEndDecisionScope` with its results, targets and odds.

	Pool indices are GLOBAL at the API boundary, matching `activity_probe.py`,
	the capacity audit and `chooser_growth`.
	"""
	ovl, loader = load_motiongraph(Path(source).resolve(), name or None, game)
	deref = build_deref(loader)
	_, states, rows = analyze_states(loader)
	state_index = {id(state): index for index, state in enumerate(states) if state is not None}

	out = []
	for node in _walk_decision(deref, deref(loader.header.m_r_f_member_1)):
		method = str(deref(node.lua_method) or "")
		if RANDOM_CHOICE_METHOD not in method:
			continue
		results = []
		for child in (deref(node.children) or []):
			sub = deref(child.m_r_f_member)
			target = deref(sub.ptr_0) if sub is not None else None
			index = state_index.get(id(target))
			something = deref(child.ptr_1)
			param = deref(something.ptr) if something is not None else None
			results.append({
				"ordinal": int(child.count_0),
				"state": index,
				"label": rows[index]["label"] if index is not None else "?",
				"weight": int(param.value) if param is not None else 0,
				"weight_at": _addr(something.ptr) if something is not None else None,
				"node_at": _addr(child.m_r_f_member),
			})
		total = sum(r["weight"] for r in results) or 1
		for result in results:
			result["share"] = 100.0 * result["weight"] / total
		out.append({
			"pool": _node_pool(loader, node),
			"offset": int(node.io_start),
			"children_at": _addr(node.children),
			"num_children": int(node.num_children),
			"results": results,
		})
	return out


def _node_pool(loader, node):
	"""Global pool index a decoded MemStruct was read from."""
	for (pool, offset), obj in loader.context.recursion.items():
		if obj is node:
			return int(pool.i)
	raise ValueError("Could not resolve the pool of a decision node")


def list_states(source: Path, name: str | None = None, game: str = DEFAULT_GAME):
	"""Index, label, activity count and address for every registered state."""
	ovl, loader = load_motiongraph(Path(source).resolve(), name or None, game)
	deref = build_deref(loader)
	_, states, rows = analyze_states(loader)
	array = deref(deref(loader.header.state_output_entries).states)
	out = []
	for index, state in enumerate(states):
		if state is None:
			continue
		out.append({
			"index": index,
			"label": rows[index]["label"],
			"activities": int(state.num_activities),
			"transitions": int(state.array_2_count),
			"at": _addr(array[index].state),
		})
	return out


class _Staged:
	"""Shared setup + teardown for a topology-growth edit on a staged family."""

	def __init__(self, source: Path, output: Path, game: str):
		self.source, self.output = validate_staged_pair(source, output)
		self.game = game
		self.ovl, self.static = _load_quiet(self.source, game)
		self.ovs = self.static.content
		self.old_fragments = int(self.static.num_fragments)
		self.old_uncompressed = int(self.static.uncompressed_size)
		self.old_compressed = int(self.static.compressed_size)
		self.old_pools_end = int(self.static.pools_end)
		self.old_pool_sizes = tuple(int(pool.size) for pool in self.ovs.pools)
		self.static_index = self.ovl.archives.index(self.static)
		self.old_reservation = int(self.ovl.archives_meta[self.static_index].unk_0)
		self.allocations = []
		self.added = 0

	def local(self, global_index: int) -> int:
		pool = self.ovl.pools[global_index]
		if pool not in self.ovs.pools:
			raise ValueError(f"Global pool {global_index} is not in STATIC")
		return self.ovs.pools.index(pool)

	def append(self, pool_type: int, payload: bytes):
		limit = TYPE_3_PAGE_LIMIT if pool_type == 3 else TYPE_2_PAGE_LIMIT
		allocation = append_tail_pool_bytes(
			self.ovl.pools, self.ovs.pools, pool_type, payload, alignment=16, page_size=limit)
		moved = repoint_pool_end_fragments(
			self.ovs.fragments, allocation.local_pool, allocation.old_size, allocation.new_size)
		self.allocations.append((allocation, moved))
		return allocation, moved

	def add_rows(self, rows):
		"""Fold new fragments in immediately.

		Deliberately not deferred to `save`: a later append's end-sentinel
		relocation must be able to see fragments an earlier step added, or a
		second operation in the same session could strand one at the old end.
		"""
		self.ovs.fragments = np.concatenate((self.ovs.fragments, rows))
		self.added += len(rows)

	def save(self):
		ovs = self.ovs
		new_rows_count = self.added
		ovs.fragments.sort(order=("link_pool", "struct_pool", "link_offset", "struct_offset"))
		self.static.num_fragments = self.old_fragments + new_rows_count

		ovs.write_pools()
		uncompressed = ovs.write_archive()
		pool_growth = sum(a.new_size - a.old_size for a, _ in self.allocations)
		expected = self.old_uncompressed + pool_growth + new_rows_count * 16
		if len(uncompressed) != expected:
			raise ValueError(f"Unexpected STATIC growth: {len(uncompressed)} vs {expected}")
		expected_sizes = list(self.old_pool_sizes)
		for allocation, _ in self.allocations:
			expected_sizes[allocation.local_pool] = allocation.new_size
		if tuple(int(p.size) for p in ovs.pools) != tuple(expected_sizes):
			raise ValueError("An unrelated pool changed size during the edit")

		_, new_compressed, compressed = ovs.compress(uncompressed, True)
		source_bytes = self.source.read_bytes()
		header_size = len(source_bytes) - self.old_compressed
		meta_offset = header_size - len(self.ovl.archives_meta) * 8 + self.static_index * 8
		result = bytearray(source_bytes[:header_size])
		result.extend(compressed)
		head = int(self.static.io_start)
		struct.pack_into("<I", result, head + NUM_FRAGMENTS_OFFSET, self.static.num_fragments)
		struct.pack_into("<I", result, head + COMPRESSED_SIZE_OFFSET, new_compressed)
		struct.pack_into("<Q", result, head + UNCOMPRESSED_SIZE_OFFSET, expected)
		struct.pack_into("<I", result, head + POOLS_END_OFFSET, self.old_pools_end + pool_growth)
		struct.pack_into("<I", result, meta_offset, self.old_reservation + pool_growth)
		publication = CandidatePublication(self.source, self.output)
		publication.path.write_bytes(result)
		_load_quiet(publication.path, self.game)
		load_motiongraph(publication.path, None, self.game)
		self.publication = publication
		return pool_growth

	def commit(self):
		if not hasattr(self, "publication"):
			raise ValueError("The staged decision edit has not been saved and verified")
		return self.publication.commit()


def _find_state_output_donor(staged, deref, loader):
	"""Any shipped `StateOutput` node, used purely as a byte + fragment template."""
	for node in _walk_decision(deref, deref(loader.header.m_r_f_member_1)):
		if str(deref(node.lua_method) or "") != STATE_OUTPUT_METHOD:
			continue
		if int(node.dtype) != 8 or int(node.num_children) != 0:
			continue
		global_pool = _node_pool(loader, node)
		offset = int(node.io_start)
		local = staged.local(global_pool)
		mask = internal_fragment_mask(staged.ovs.fragments, local, offset, MRF_NODE_SIZE)
		rows = staged.ovs.fragments[mask].copy()
		got = sorted(int(x) - offset for x in rows["link_offset"])
		if got == [LUA_METHOD_OFFSET, PTR_0_OFFSET, MG_VARS_OFFSET, CHILDREN_PTR_OFFSET]:
			data = staged.ovl.pools[global_pool].data.getvalue()
			return (global_pool, offset), data[offset:offset + MRF_NODE_SIZE], rows
	raise ValueError("No usable StateOutput node found to use as a template")


def inbound_count(source: Path, pool: int, offset: int, game: str = DEFAULT_GAME) -> int:
	"""How many fragments target one exact address.

	Call this before retargeting anything. Shipped `StateOutput` nodes are shared
	between the chooser array and transition-layer decision blocks, and an
	in-place retarget silently redirects every one of them.
	"""
	ovl, static = _load_quiet(Path(source).resolve(), game)
	ovs = static.content
	local = ovs.pools.index(ovl.pools[pool])
	fragments = ovs.fragments
	return int(((fragments["struct_pool"] == local)
				& (fragments["struct_offset"] == offset)).sum())


def set_result_state(source: Path, output: Path, node_pool: int, node_offset: int,
					 index: int, target_state: int, name: str | None = None,
					 game: str = DEFAULT_GAME):
	"""Point one existing chooser result at a different state.

	Allocates a NEW `StateOutput` node rather than retargeting the one the result
	currently uses: shipped nodes are shared, and in UltimasaurusCE both preen
	`StateOutput` nodes also carry a transition-layer reference, so an in-place
	retarget silently redirects that route too.
	"""
	loader, deref, states, rows = _semantics(source, name, game)
	if target_state < 0 or target_state >= len(states) or states[target_state] is None:
		raise ValueError(f"No state with index {target_state}")
	chooser = None
	for node in _walk_decision(deref, deref(loader.header.m_r_f_member_1)):
		if int(node.io_start) == node_offset and _node_pool(loader, node) == node_pool:
			chooser = node
			break
	if chooser is None:
		raise ValueError(f"No decision node at {node_pool}:{node_offset}")
	kids = list(deref(chooser.children) or [])
	if not 0 <= index < len(kids):
		raise ValueError(f"result {index} is outside 0..{len(kids) - 1}")
	target_at = _addr(deref(deref(loader.header.state_output_entries).states)[target_state].state)
	children_global, children_offset = _addr(chooser.children)

	staged = _Staged(source, output, game)
	frags = staged.ovs.fragments
	children_local = staged.local(children_global)
	link = (children_local, children_offset + index * CHILD_SIZE + 8)
	mask = (frags["link_pool"] == link[0]) & (frags["link_offset"] == link[1])
	if int(mask.sum()) != 1:
		raise ValueError(f"Expected 1 MRFMember fragment at {link}, found {int(mask.sum())}")

	donor_at, donor_bytes, donor_rows = _find_state_output_donor(staged, deref, loader)
	alloc, _moved = staged.append(2, bytes(donor_bytes))
	cloned = clone_fragment_sources(donor_rows, alloc.local_pool, donor_at[1], alloc.offset)
	for field in (PTR_0_OFFSET, MG_VARS_OFFSET):
		sel = cloned["link_offset"] == alloc.offset + field
		cloned["struct_pool"][sel] = staged.local(target_at[0])
		cloned["struct_offset"][sel] = target_at[1]
	# Repoint BEFORE add_rows: it rebinds `ovs.fragments` to a new array, so a
	# reference taken earlier goes stale and writes land on the discarded copy.
	frags["struct_pool"][mask] = alloc.local_pool
	frags["struct_offset"][mask] = alloc.offset
	staged.add_rows(cloned)
	pool_growth = staged.save()
	staged.commit()
	return {"result": index, "state": target_state, "label": rows[target_state]["label"],
			"node_at": (alloc.local_pool, alloc.offset), "pool_growth": pool_growth}


def share_state_transitions(source: Path, output: Path, state_index: int, from_state: int,
							name: str | None = None, game: str = DEFAULT_GAME):
	"""Give one state another state's transition list, by pointer.

	Exits are a property of the state, so this is how you change where a state can
	go without deep-cloning a `TransStruct` tree. Both states must declare the same
	`array_2_count`, or the count would have to move too.
	"""
	loader, deref, states, rows = _semantics(source, name, game)
	for index in (state_index, from_state):
		if index < 0 or index >= len(states) or states[index] is None:
			raise ValueError(f"No state with index {index}")
	target, donor = states[state_index], states[from_state]
	if int(target.array_2_count) != int(donor.array_2_count):
		raise ValueError(f"array_2_count differs: STATE[{state_index}] has "
						 f"{int(target.array_2_count)}, STATE[{from_state}] has "
						 f"{int(donor.array_2_count)}")
	array = deref(deref(loader.header.state_output_entries).states)
	state_at = _addr(array[state_index].state)
	donor_array_2 = _addr(donor.array_2)

	staged = _Staged(source, output, game)
	frags = staged.ovs.fragments
	state_local = staged.local(state_at[0])
	link = (state_local, state_at[1] + STATE_ARRAY_2_OFFSET)
	mask = (frags["link_pool"] == link[0]) & (frags["link_offset"] == link[1])
	if int(mask.sum()) != 1:
		raise ValueError(f"Expected 1 array_2 fragment at {link}, found {int(mask.sum())}")
	frags["struct_pool"][mask] = staged.local(donor_array_2[0])
	frags["struct_offset"][mask] = donor_array_2[1]
	pool_growth = staged.save()
	if pool_growth:
		raise ValueError("sharing a transition list must not grow anything")
	staged.commit()
	return {"state": state_index, "took_transitions_from": from_state,
			"label": rows[state_index]["label"]}


def set_decision_weights(source: Path, output: Path, node_pool: int, node_offset: int,
						 weights, name: str | None = None, game: str = DEFAULT_GAME):
	"""Rewrite one decision chooser's result weights.

	Each result's odds live in its own `MotiongraphResultParam.Value`, a uint64,
	so this is a fixed-width edit inside allocations that already exist - no
	relocation, no fragments, no size change anywhere. Weights are relative and
	the engine draws from their sum, so the share is what matters.
	"""
	loader, deref, _states, rows = _semantics(source, name, game)
	chooser = None
	for node in _walk_decision(deref, deref(loader.header.m_r_f_member_1)):
		if int(node.io_start) == node_offset and _node_pool(loader, node) == node_pool:
			chooser = node
			break
	if chooser is None or RANDOM_CHOICE_METHOD not in str(deref(chooser.lua_method) or ""):
		raise ValueError(f"No RandomChoiceEndDecisionScope at {node_pool}:{node_offset}")

	kids = list(deref(chooser.children) or [])
	weights = [int(w) for w in weights]
	if len(weights) != len(kids):
		raise ValueError(f"chooser has {len(kids)} results, got {len(weights)} weights")
	if any(w < 0 or w > 0xFFFFFFFFFFFFFFFF for w in weights):
		raise ValueError("weights must fit in a uint64")
	if not any(weights):
		raise ValueError("at least one weight must be non-zero or nothing can be drawn")

	targets = []
	for child in kids:
		something = deref(child.ptr_1)
		param = deref(something.ptr) if something is not None else None
		if param is None or str(deref(param.field)) != "Weight":
			raise ValueError("a result does not carry a 'Weight' param")
		targets.append(_addr(something.ptr))

	staged = _Staged(source, output, game)
	old_sizes = tuple(int(p.size) for p in staged.ovs.pools)
	for (global_pool, offset), value in zip(targets, weights):
		data = staged.ovl.pools[global_pool].data
		data.seek(offset + 8)             # MotiongraphResultParam.Value
		data.write(struct.pack("<Q", value))
		data.seek(0)
	pool_growth = staged.save()
	if pool_growth or tuple(int(p.size) for p in staged.ovs.pools) != old_sizes:
		raise ValueError("a weight edit must not resize anything")
	staged.commit()
	total = sum(weights) or 1
	return [{"weight": w, "share": 100.0 * w / total} for w in weights]


def set_activity_speed(source: Path, output: Path, activity_pool: int, activity_offset: int,
					   speed: float, name: str | None = None, game: str = DEFAULT_GAME):
	"""Set one `AnimationActivity`'s playback speed - a fixed-width float edit.

	`clone_complete_animation_activity` can set this while cloning; this changes it
	afterwards, which is what you want once a clip has been retimed or replaced.
	"""
	if not (speed >= 0.0) or speed != speed or speed in (float("inf"),):
		raise ValueError("Playback speed must be a finite non-negative number")
	loader, deref, _states, _rows = _semantics(source, name, game)
	activity = None
	for (pool, offset), obj in loader.context.recursion.items():
		if int(pool.i) == activity_pool and int(offset) == activity_offset:
			activity = obj
			break
	if activity is None or type(activity).__name__ != "Activity":
		raise ValueError(f"No Activity at {activity_pool}:{activity_offset}")
	payload = deref(activity.data)
	if payload is None or not hasattr(payload, "speed"):
		raise ValueError(f"Activity at {activity_pool}:{activity_offset} has no Speed field")
	old = float(payload.speed.float)
	payload_at = _addr(activity.data)

	staged = _Staged(source, output, game)
	old_sizes = tuple(int(p.size) for p in staged.ovs.pools)
	data = staged.ovl.pools[payload_at[0]].data
	data.seek(payload_at[1] + SPEED_OFFSET)
	data.write(struct.pack("<f", float(speed)))
	data.seek(0)
	pool_growth = staged.save()
	if pool_growth or tuple(int(p.size) for p in staged.ovs.pools) != old_sizes:
		raise ValueError("a speed edit must not resize anything")
	staged.commit()
	return {"old": old, "new": float(speed)}


def _plan_new_result(staged, loader, deref, node_pool: int, node_offset: int,
					 target_local: int, target_offset: int, weight: int):
	"""Append one weighted result to a chooser. Target is given as an ADDRESS so
	this works for a state that was created earlier in the same session and does
	not exist in `loader`'s view yet."""
	chooser = None
	for node in _walk_decision(deref, deref(loader.header.m_r_f_member_1)):
		if int(node.io_start) == node_offset and _node_pool(loader, node) == node_pool:
			chooser = node
			break
	if chooser is None or RANDOM_CHOICE_METHOD not in str(deref(chooser.lua_method) or ""):
		raise ValueError(f"No RandomChoiceEndDecisionScope at {node_pool}:{node_offset}")

	count = int(chooser.num_children)
	kids = list(deref(chooser.children) or [])
	if len(kids) != count:
		raise ValueError(f"num_children is {count} but {len(kids)} children decoded")
	array_global, array_offset = _addr(chooser.children)
	weight_string = _addr(deref(deref(kids[0].ptr_1).ptr).field)

	frags = staged.ovs.fragments
	array_local = staged.local(array_global)
	node_local = staged.local(node_pool)

	owner = ((frags["struct_pool"] == array_local) & (frags["struct_offset"] == array_offset))
	if int(owner.sum()) != 1:
		raise ValueError(f"Expected 1 owner fragment for the children array, got {int(owner.sum())}")
	owner_link = (int(frags["link_pool"][owner][0]), int(frags["link_offset"][owner][0]))
	if owner_link != (node_local, node_offset + CHILDREN_PTR_OFFSET):
		raise ValueError(f"Children owner fragment at {owner_link}, expected "
						 f"{(node_local, node_offset + CHILDREN_PTR_OFFSET)}")

	old_size = count * CHILD_SIZE
	inside = internal_fragment_mask(frags, array_local, array_offset, old_size)
	child_rows = frags[inside].copy()
	expected = sorted([i * CHILD_SIZE + 8 for i in range(count)]
					  + [i * CHILD_SIZE + 16 for i in range(count)])
	if sorted(int(x) - array_offset for x in child_rows["link_offset"]) != expected:
		raise ValueError("Children array does not have exactly two pointers per element")

	donor_at, donor_bytes, donor_rows = _find_state_output_donor(staged, deref, loader)

	# type-2: the new StateOutput node
	node_alloc, _ = staged.append(2, bytes(donor_bytes))
	# type-3: grown children array + Something + ResultParam
	old_bytes = staged.ovl.pools[array_global].data.getvalue()[
		array_offset:array_offset + old_size]
	new_child = bytearray(CHILD_SIZE)
	struct.pack_into("<Q", new_child, 0, count + 1)
	struct.pack_into("<Q", new_child, 24, 7)
	something = bytearray(SOMETHING_SIZE)
	struct.pack_into("<Q", something, 8, 1)
	param = bytearray(RESULT_PARAM_SIZE)
	struct.pack_into("<Q", param, 8, int(weight))
	struct.pack_into("<Q", param, 16, 2)
	payload = bytes(old_bytes) + bytes(new_child) + bytes(something) + bytes(param)
	array_alloc, moved = staged.append(3, payload)
	base = array_alloc.offset
	something_at = base + (count + 1) * CHILD_SIZE
	param_at = something_at + SOMETHING_SIZE

	frags["struct_pool"][owner] = array_alloc.local_pool
	frags["struct_offset"][owner] = base

	cloned_children = clone_fragment_sources(child_rows, array_alloc.local_pool,
											 array_offset, base)
	cloned_node = clone_fragment_sources(donor_rows, node_alloc.local_pool,
										 donor_at[1], node_alloc.offset)
	for field in (PTR_0_OFFSET, MG_VARS_OFFSET):
		sel = cloned_node["link_offset"] == node_alloc.offset + field
		if int(sel.sum()) != 1:
			raise ValueError(f"Cloned StateOutput node missing field +{field}")
		cloned_node["struct_pool"][sel] = target_local
		cloned_node["struct_offset"][sel] = target_offset

	added = np.array(
		[
			(array_alloc.local_pool, base + count * CHILD_SIZE + 8,
			 node_alloc.local_pool, node_alloc.offset),
			(array_alloc.local_pool, base + count * CHILD_SIZE + 16,
			 array_alloc.local_pool, something_at),
			(array_alloc.local_pool, something_at, array_alloc.local_pool, param_at),
			(array_alloc.local_pool, param_at,
			 staged.local(weight_string[0]), weight_string[1]),
		],
		dtype=frags.dtype,
	)
	new_rows = np.concatenate((cloned_children, cloned_node, added))

	data = staged.ovl.pools[node_pool].data
	data.seek(node_offset + NUM_CHILDREN_OFFSET)
	if struct.unpack("<Q", data.read(8))[0] != count:
		raise ValueError("num_children is not where the schema says it is")
	data.seek(node_offset + NUM_CHILDREN_OFFSET)
	data.write(struct.pack("<Q", count + 1))
	data.seek(0)

	staged.add_rows(new_rows)
	return {"old_results": count, "new_results": count + 1, "weight": int(weight)}


def grow_decision_chooser(source: Path, output: Path, node_pool: int, node_offset: int,
						  target_state: int, weight: int = 1,
						  name: str | None = None,
						  game: str = DEFAULT_GAME) -> DecisionGrowthReport:
	"""Add one weighted result to a decision chooser, routed to an existing state.

	A NEW `StateOutput` node is always allocated rather than reusing one that
	already targets `target_state` - see the module docstring.
	"""
	if weight < 1 or weight > 0xFFFFFFFF:
		raise ValueError("Weight must be a positive uint32")
	sem_ovl, loader = load_motiongraph(Path(source).resolve(), name or None, game)
	deref = build_deref(loader)
	_, states, _rows = analyze_states(loader)
	if target_state < 0 or target_state >= len(states) or states[target_state] is None:
		raise ValueError(f"No state with index {target_state}")
	target_at = _addr(deref(deref(loader.header.state_output_entries).states)[target_state].state)

	staged = _Staged(source, output, game)
	info = _plan_new_result(staged, loader, deref, node_pool, node_offset,
							staged.local(target_at[0]), target_at[1], weight)
	pool_growth = staged.save()
	staged.commit()
	return DecisionGrowthReport(
		output=staged.output, node=(node_pool, node_offset),
		old_results=info["old_results"], new_results=info["new_results"],
		target_state=target_state, weight=info["weight"],
		pool_growth=pool_growth, added_fragments=staged.added,
		moved_end_sentinels=sum(m for _, m in staged.allocations),
	)


def _plan_new_state(staged, loader, deref, states, twin_state: int):
	"""Register a new state twinned on `twin_state`. Returns its address."""
	header_array = deref(loader.header.state_output_entries)
	old_count = int(header_array.states_count)
	count_at = _addr(loader.header.state_output_entries)
	array_at = _addr(header_array.states)
	twin = states[twin_state]
	twin_activities = _addr(twin.activities)
	twin_array_2 = _addr(twin.array_2)
	num_activities = int(twin.num_activities)
	if num_activities < 1:
		raise ValueError("Twin state has no activities to copy")
	twin_at = _addr(deref(header_array.states)[twin_state].state)

	frags = staged.ovs.fragments
	array_local = staged.local(array_at[0])
	count_local = staged.local(count_at[0])
	act_local = staged.local(twin_activities[0])

	owner = ((frags["struct_pool"] == array_local) & (frags["struct_offset"] == array_at[1]))
	if int(owner.sum()) != 1:
		raise ValueError(f"Expected 1 owner fragment for the state array, got {int(owner.sum())}")
	owner_link = (int(frags["link_pool"][owner][0]), int(frags["link_offset"][owner][0]))
	if owner_link != (count_local, count_at[1] + 8):
		raise ValueError(f"State array owner fragment at {owner_link}, expected "
						 f"{(count_local, count_at[1] + 8)}")

	slots = internal_fragment_mask(frags, array_local, array_at[1], old_count * SLOT)
	slot_rows = frags[slots].copy()
	if sorted(int(x) - array_at[1] for x in slot_rows["link_offset"]) != \
			[i * SLOT for i in range(old_count)]:
		raise ValueError("State array is not exactly one pointer per 8 bytes")

	acts = internal_fragment_mask(frags, act_local, twin_activities[1], num_activities * SLOT)
	act_rows = frags[acts].copy()
	if sorted(int(x) - twin_activities[1] for x in act_rows["link_offset"]) != \
			[i * SLOT for i in range(num_activities)]:
		raise ValueError("Twin activities array is not one pointer per element")

	twin_bytes = staged.ovl.pools[twin_at[0]].data.getvalue()[
		twin_at[1]:twin_at[1] + STATE_SIZE]
	if struct.unpack_from("<I", twin_bytes, STATE_NUM_ACTIVITIES_OFFSET)[0] != num_activities:
		raise ValueError("Twin State bytes disagree with the decoded activity count")

	# type-3: grown state array + a fresh activities array
	old_array = staged.ovl.pools[array_at[0]].data.getvalue()[
		array_at[1]:array_at[1] + old_count * SLOT]
	new_array = bytes(old_array) + bytes(SLOT)
	pad = (-len(new_array)) % 16
	payload3 = new_array + bytes(pad) + bytes(num_activities * SLOT)
	alloc3, moved3 = staged.append(3, payload3)
	array_base = alloc3.offset
	act_base = array_base + len(new_array) + pad

	# type-2: the new State
	alloc2, moved2 = staged.append(2, bytes(twin_bytes))
	state_base = alloc2.offset

	frags["struct_pool"][owner] = alloc3.local_pool
	frags["struct_offset"][owner] = array_base

	cloned_slots = clone_fragment_sources(slot_rows, alloc3.local_pool, array_at[1], array_base)
	cloned_acts = clone_fragment_sources(act_rows, alloc3.local_pool,
										 twin_activities[1], act_base)
	added = np.array(
		[
			(alloc3.local_pool, array_base + old_count * SLOT, alloc2.local_pool, state_base),
			(alloc2.local_pool, state_base + STATE_ACTIVITIES_OFFSET, alloc3.local_pool, act_base),
			(alloc2.local_pool, state_base + STATE_ARRAY_2_OFFSET,
			 staged.local(twin_array_2[0]), twin_array_2[1]),
		],
		dtype=frags.dtype,
	)
	new_rows = np.concatenate((cloned_slots, cloned_acts, added))

	data = staged.ovl.pools[count_at[0]].data
	data.seek(count_at[1])
	if struct.unpack("<Q", data.read(8))[0] != old_count:
		raise ValueError("states_count is not where the schema says it is")
	data.seek(count_at[1])
	data.write(struct.pack("<Q", old_count + 1))
	data.seek(0)

	staged.add_rows(new_rows)
	return {
		"new_index": old_count, "old_count": old_count, "new_count": old_count + 1,
		"state_local": alloc2.local_pool, "state_offset": state_base,
		"array_local": alloc3.local_pool, "array_offset": array_base,
	}


@dataclass(frozen=True)
class ClipRetargetReport:
	output: Path
	activity: tuple
	old_clip: str
	new_clip: str
	allocated: bool
	string_at: tuple
	pool_growth: int
	added_fragments: int
	moved_end_sentinels: int


def set_activity_clip(source: Path, output: Path, activity_pool: int, activity_offset: int,
					  new_clip: str, name: str | None = None,
					  game: str = DEFAULT_GAME) -> ClipRetargetReport:
	"""Point ONE `AnimationActivity` at a different clip name, allocating it if new.

	The gap between `clone_complete_animation_activity` (which duplicates the
	wrapper and payload but keeps the donor's clip string) and
	`static_patch.repoint_existing_string` (which moves EVERY reference to a
	string, and needs both to exist already). This moves exactly one activity's
	`mani` pointer, and will allocate the name if the archive does not have it.

	Strings must live in type-2 pools - all 1,269 shipped string allocations do
	and none is in type 3 - so a new name goes in the final type-2 tail. That is
	the game-verified Test X/Y route: append and repoint, any length, any count.

	NOTE: the name must correspond to a real clip in the `.manis` bundles. This
	writes the reference, not the animation.
	"""
	if not new_clip or any(ord(c) > 126 or ord(c) < 32 for c in new_clip):
		raise ValueError(f"Clip name must be printable ASCII: {new_clip!r}")

	loader, deref, _states, _rows = _semantics(source, name, game)
	activity = loader.context.recursion.get(
		(_pool_by_index(loader, activity_pool), activity_offset))
	if activity is None:
		for (pool, offset), obj in loader.context.recursion.items():
			if int(pool.i) == activity_pool and int(offset) == activity_offset:
				activity = obj
				break
	if activity is None or type(activity).__name__ != "Activity":
		raise ValueError(f"No Activity at {activity_pool}:{activity_offset} "
						 f"(found {type(activity).__name__ if activity else 'nothing'})")
	payload = deref(activity.data)
	if payload is None or not hasattr(payload, "mani"):
		raise ValueError(f"Activity at {activity_pool}:{activity_offset} is a "
						 f"{type(payload).__name__ if payload else 'null'}, which has no clip")
	old_clip = str(deref(payload.mani) or "")
	if old_clip == new_clip:
		raise ValueError(f"That activity already plays {new_clip!r}")
	payload_at = _addr(activity.data)

	staged = _Staged(source, output, game)
	frags = staged.ovs.fragments
	payload_local = staged.local(payload_at[0])

	mask = ((frags["link_pool"] == payload_local)
			& (frags["link_offset"] == payload_at[1]))     # mani is at payload +0
	if int(mask.sum()) != 1:
		raise ValueError(f"Expected exactly 1 mani fragment at {payload_local}:"
						 f"{payload_at[1]}, found {int(mask.sum())}")

	# reuse an identical existing string before allocating another
	needle = new_clip.encode("ascii") + b"\0"
	reused = None
	for index, pool in enumerate(staged.ovl.pools):
		if int(pool.type) != 2 or pool not in staged.ovs.pools:
			continue
		blob = pool.data.getvalue()
		at = blob.find(needle)
		while at != -1:
			if at == 0 or blob[at - 1] == 0:
				reused = (staged.ovs.pools.index(pool), at)
				break
			at = blob.find(needle, at + 1)
		if reused:
			break

	if reused is not None:
		string_local, string_offset = reused
		moved = 0
	else:
		allocation, moved = staged.append(2, needle)
		string_local, string_offset = allocation.local_pool, allocation.offset

	frags["struct_pool"][mask] = string_local
	frags["struct_offset"][mask] = string_offset

	pool_growth = staged.save()
	staged.commit()
	return ClipRetargetReport(
		output=staged.output, activity=(activity_pool, activity_offset),
		old_clip=old_clip, new_clip=new_clip, allocated=reused is None,
		string_at=(string_local, string_offset), pool_growth=pool_growth,
		added_fragments=staged.added, moved_end_sentinels=moved,
	)


def _pool_by_index(loader, global_index):
	for (pool, _offset) in loader.context.recursion:
		if int(pool.i) == global_index:
			return pool
	return None


def _semantics(source: Path, name, game):
	sem_ovl, loader = load_motiongraph(Path(source).resolve(), name or None, game)
	deref = build_deref(loader)
	_, states, rows = analyze_states(loader)
	return loader, deref, states, rows


def add_state(source: Path, output: Path, twin_state: int,
			  name: str | None = None, game: str = DEFAULT_GAME) -> NewStateReport:
	"""Register a new state, structurally twinned on an existing one.

	The new state copies the twin's `State` bytes, gets a FRESH activities array
	holding the twin's activity pointers, and SHARES the twin's `array_2`
	transition list by pointer. It is appended at the END of `StateArray`, which
	leaves `first_non_transition_state` and every existing index untouched.

	It has NO inbound edge, so it is dormant. Prefer `add_state_and_route`, which
	does both in one pass - two separate calls would each read `source` and the
	second would discard the first.
	"""
	loader, deref, states, _rows = _semantics(source, name, game)
	if twin_state < 0 or twin_state >= len(states) or states[twin_state] is None:
		raise ValueError(f"No state with index {twin_state}")
	staged = _Staged(source, output, game)
	info = _plan_new_state(staged, loader, deref, states, twin_state)
	pool_growth = staged.save()
	staged.commit()
	return NewStateReport(
		output=staged.output, twin_state=twin_state, new_state_index=info["new_index"],
		new_state_at=(info["state_local"], info["state_offset"]),
		state_node_at=(info["array_local"], info["array_offset"]),
		old_count=info["old_count"], new_count=info["new_count"],
		pool_growth=pool_growth, added_fragments=staged.added,
		moved_end_sentinels=sum(m for _, m in staged.allocations),
	)


def add_state_and_route(source: Path, output: Path, twin_state: int,
						node_pool: int, node_offset: int, weight: int = 1,
						name: str | None = None,
						game: str = DEFAULT_GAME) -> NewStateReport:
	"""Add a state AND the chooser result that reaches it, in ONE staged pass.

	This is the operation you almost always want: a state with no inbound edge is
	dormant by definition, and running `add_state` then `grow_decision_chooser`
	as two calls does NOT work - each reads `source`, so the second silently
	discards the first. Doing both against one `_Staged` also means one save,
	one recompress, and one consistent set of end-sentinel relocations.
	"""
	if weight < 1 or weight > 0xFFFFFFFF:
		raise ValueError("Weight must be a positive uint32")
	loader, deref, states, _rows = _semantics(source, name, game)
	if twin_state < 0 or twin_state >= len(states) or states[twin_state] is None:
		raise ValueError(f"No state with index {twin_state}")

	staged = _Staged(source, output, game)
	state_info = _plan_new_state(staged, loader, deref, states, twin_state)
	route_info = _plan_new_result(
		staged, loader, deref, node_pool, node_offset,
		state_info["state_local"], state_info["state_offset"], weight)
	pool_growth = staged.save()
	staged.commit()
	return NewStateReport(
		output=staged.output, twin_state=twin_state, new_state_index=state_info["new_index"],
		new_state_at=(state_info["state_local"], state_info["state_offset"]),
		state_node_at=(state_info["array_local"], state_info["array_offset"]),
		old_count=state_info["old_count"], new_count=state_info["new_count"],
		pool_growth=pool_growth, added_fragments=staged.added,
		moved_end_sentinels=sum(m for _, m in staged.allocations),
		route=route_info,
	)
