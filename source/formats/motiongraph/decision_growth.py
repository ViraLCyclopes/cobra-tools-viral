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

# Type-2 pools in shipped archives reach ~19.6 KB, but stay conservative and use
# the same explicit bound the type-3 appends use.
TYPE_2_PAGE_LIMIT = 16384

CHILD_SIZE = 32
SOMETHING_SIZE = 16
RESULT_PARAM_SIZE = 24
STATE_SIZE = 40
MRF_NODE_SIZE = 72
SLOT = 8

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
		self.source = Path(source).resolve()
		self.output = Path(output).resolve()
		if self.source == self.output:
			raise ValueError("Refusing to overwrite the source OVL")
		if self.source.name.lower() != self.output.name.lower():
			raise ValueError("Source and staged output OVL basenames must match")
		if not self.output.is_file():
			raise ValueError("Copy the complete archive family to the stage directory first")
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

	def save(self, new_rows):
		ovs = self.ovs
		ovs.fragments = np.concatenate((ovs.fragments, new_rows))
		ovs.fragments.sort(order=("link_pool", "struct_pool", "link_offset", "struct_offset"))
		self.static.num_fragments = self.old_fragments + len(new_rows)

		ovs.write_pools()
		uncompressed = ovs.write_archive()
		pool_growth = sum(a.new_size - a.old_size for a, _ in self.allocations)
		expected = self.old_uncompressed + pool_growth + len(new_rows) * 16
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
		self.output.write_bytes(result)
		return pool_growth


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
	_, states, rows = analyze_states(loader)
	if target_state < 0 or target_state >= len(states) or states[target_state] is None:
		raise ValueError(f"No state with index {target_state}")

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
	state_array = deref(deref(loader.header.state_output_entries).states)
	target_at = _addr(state_array[target_state].state)

	staged = _Staged(source, output, game)
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
		cloned_node["struct_pool"][sel] = staged.local(target_at[0])
		cloned_node["struct_offset"][sel] = target_at[1]

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

	pool_growth = staged.save(new_rows)
	return DecisionGrowthReport(
		output=staged.output, node=(node_pool, node_offset),
		old_results=count, new_results=count + 1,
		target_state=target_state, weight=int(weight),
		pool_growth=pool_growth, added_fragments=len(new_rows),
		moved_end_sentinels=sum(m for _, m in staged.allocations),
	)


def add_state(source: Path, output: Path, twin_state: int,
			  name: str | None = None, game: str = DEFAULT_GAME) -> NewStateReport:
	"""Register a new state, structurally twinned on an existing one.

	The new state copies the twin's `State` bytes, gets a FRESH activities array
	holding the twin's activity pointers, and SHARES the twin's `array_2`
	transition list by pointer. It is appended at the end of `StateArray`, which
	leaves `first_non_transition_state` and every existing index untouched.

	It has no inbound edge yet - route to it with `grow_decision_chooser`.
	"""
	sem_ovl, loader = load_motiongraph(Path(source).resolve(), name or None, game)
	deref = build_deref(loader)
	_, states, rows = analyze_states(loader)
	if twin_state < 0 or twin_state >= len(states) or states[twin_state] is None:
		raise ValueError(f"No state with index {twin_state}")

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

	staged = _Staged(source, output, game)
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

	pool_growth = staged.save(new_rows)
	return NewStateReport(
		output=staged.output, twin_state=twin_state, new_state_index=old_count,
		new_state_at=(alloc2.local_pool, state_base), state_node_at=(alloc3.local_pool, array_base),
		old_count=old_count, new_count=old_count + 1,
		pool_growth=pool_growth, added_fragments=len(new_rows),
		moved_end_sentinels=moved3 + moved2,
	)
