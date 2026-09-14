"""Experimental fixed-capacity RandomSelectActivityActivity cloning for JWE3."""

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
    WRAPPER_SIZE,
    clone_fragment_sources,
    exact_inbound_mask,
    internal_fragment_mask,
)
from .edit import DEFAULT_GAME, load_motiongraph
from .report import build_deref
from .surgical_growth import _load_quiet, append_tail_pool_bytes, repoint_pool_end_fragments
from .staging import CandidatePublication, validate_staged_pair


SELECTOR_DATA_SIZE = 56
SELECTOR_ENTRY_SIZE = 16
SELECTOR_COUNT = 4


@dataclass(frozen=True)
class RandomSelectorCloneReport:
    output: Path
    donor_wrapper: tuple[int, int]
    clone_wrapper: tuple[int, int]
    clone_data: tuple[int, int]
    clone_array: tuple[int, int]
    redirected_references: int
    cloned_fragments: int
    children: tuple[tuple[int, int, int], ...]


def _activity(loader, ovl, pool: int, offset: int):
    if pool < 0 or pool >= len(ovl.pools):
        raise ValueError(f"Activity pool is out of range: {pool}")
    value = loader.context.recursion.get((ovl.pools[pool], offset))
    if type(value).__name__ != "Activity":
        raise ValueError(f"No Activity decodes at global pool {pool}, offset {offset}")
    return value


def clone_fixed_random_selector(
    source: Path,
    output: Path,
    donor_pool: int,
    donor_offset: int,
    route_targets: tuple[tuple[int, int], ...],
    children: tuple[tuple[int, int, int], ...],
    name: str | None = None,
    game: str = DEFAULT_GAME,
) -> RandomSelectorCloneReport:
    """Clone one four-entry selector and route existing edges through the clone.

    ``children`` contains ``(global pool, offset, weighting)`` tuples. No array,
    string, state, or MANI is added; the donor selector's fixed four-slot shape
    is preserved exactly.
    """
    source, output = validate_staged_pair(source, output)
    if len(children) != SELECTOR_COUNT:
        raise ValueError(f"Exactly {SELECTOR_COUNT} selector children are required")
    if not route_targets:
        raise ValueError("At least one existing route target is required")
    if len(set(route_targets)) != len(route_targets):
        raise ValueError("Route targets must be unique")
    if any(weight < 0 or weight > 0xFFFFFFFF for _pool, _offset, weight in children):
        raise ValueError("Selector weights must fit uint32")

    semantic_ovl, semantic_loader = load_motiongraph(source, name or None, game)
    deref = build_deref(semantic_loader)
    donor = _activity(semantic_loader, semantic_ovl, donor_pool, donor_offset)
    if donor.data_type.data != "RandomSelectActivityActivity":
        raise ValueError("Donor is not RandomSelectActivityActivity")
    payload = deref(donor.data)
    entries = deref(payload.activities)
    if type(payload).__name__ != "RandomSelectActivityActivityData":
        raise ValueError("Selector donor has the wrong payload type")
    if int(payload.io_size) != SELECTOR_DATA_SIZE or int(payload.num_activities) != SELECTOR_COUNT:
        raise ValueError("Selector donor is not the verified 56-byte/four-entry shape")
    if int(entries.io_size) != SELECTOR_ENTRY_SIZE * SELECTOR_COUNT:
        raise ValueError("Selector donor does not have a contiguous 64-byte entry array")
    for pool, offset in route_targets:
        _activity(semantic_loader, semantic_ovl, pool, offset)
    for pool, offset, _weight in children:
        _activity(semantic_loader, semantic_ovl, pool, offset)

    data_global = int(donor.data.target_pool.i)
    data_offset = int(donor.data.target_offset)
    array_global = int(payload.activities.target_pool.i)
    array_offset = int(payload.activities.target_offset)

    ovl, static = _load_quiet(source, game)
    ovs = static.content

    def local(global_index: int) -> int:
        if global_index < 0 or global_index >= len(ovl.pools):
            raise ValueError(f"Global pool is out of range: {global_index}")
        pool = ovl.pools[global_index]
        if pool not in ovs.pools:
            raise ValueError(f"Global pool {global_index} is not in STATIC")
        return ovs.pools.index(pool)

    wrapper_local, data_local, array_local = map(local, (donor_pool, data_global, array_global))
    wrapper = ovl.pools[donor_pool].data.getvalue()[donor_offset:donor_offset + WRAPPER_SIZE]
    data = ovl.pools[data_global].data.getvalue()[data_offset:data_offset + SELECTOR_DATA_SIZE]
    array = bytearray(
        ovl.pools[array_global].data.getvalue()[
            array_offset:array_offset + SELECTOR_ENTRY_SIZE * SELECTOR_COUNT
        ]
    )
    if len(wrapper) != WRAPPER_SIZE or len(data) != SELECTOR_DATA_SIZE or len(array) != 64:
        raise ValueError("Could not read the complete selector allocations")
    for index, (_pool, _offset, weight) in enumerate(children):
        struct.pack_into("<I", array, index * SELECTOR_ENTRY_SIZE + 8, weight)

    fragments = ovs.fragments
    wrapper_rows = fragments[
        internal_fragment_mask(fragments, wrapper_local, donor_offset, WRAPPER_SIZE)
    ].copy()
    data_rows = fragments[
        internal_fragment_mask(fragments, data_local, data_offset, SELECTOR_DATA_SIZE)
    ].copy()
    array_rows = fragments[
        internal_fragment_mask(fragments, array_local, array_offset, len(array))
    ].copy()
    if sorted(int(x) - donor_offset for x in wrapper_rows["link_offset"]) != [0, 8]:
        raise ValueError("Unexpected selector Activity pointer layout")
    if sorted(int(x) - data_offset for x in data_rows["link_offset"]) != [0, 48]:
        raise ValueError("Unexpected selector payload pointer layout")
    if sorted(int(x) - array_offset for x in array_rows["link_offset"]) != [0, 16, 32, 48]:
        raise ValueError("Unexpected selector entry pointer layout")

    inbound_masks = [exact_inbound_mask(fragments, local(pool), offset) for pool, offset in route_targets]
    inbound_counts = [int(mask.sum()) for mask in inbound_masks]
    if any(count < 1 for count in inbound_counts):
        raise ValueError(f"Every route target must have an inbound reference: {inbound_counts}")

    old_fragments = int(static.num_fragments)
    old_uncompressed = int(static.uncompressed_size)
    old_compressed = int(static.compressed_size)
    old_pools_end = int(static.pools_end)
    old_pool_sizes = tuple(int(pool.size) for pool in ovs.pools)
    static_index = ovl.archives.index(static)
    old_reservation = int(ovl.archives_meta[static_index].unk_0)

    wrapper_allocation = append_tail_pool_bytes(ovl.pools, ovs.pools, 2, wrapper, alignment=16)
    repoint_pool_end_fragments(
        fragments, wrapper_allocation.local_pool,
        wrapper_allocation.old_size, wrapper_allocation.new_size,
    )
    data_allocation = append_tail_pool_bytes(
        ovl.pools, ovs.pools, 3, data, alignment=16, page_size=TYPE_3_PAGE_LIMIT
    )
    repoint_pool_end_fragments(
        fragments, data_allocation.local_pool,
        data_allocation.old_size, data_allocation.new_size,
    )
    array_allocation = append_tail_pool_bytes(
        ovl.pools, ovs.pools, 3, bytes(array), alignment=16, page_size=TYPE_3_PAGE_LIMIT
    )
    repoint_pool_end_fragments(
        fragments, array_allocation.local_pool,
        array_allocation.old_size, array_allocation.new_size,
    )

    for mask in inbound_masks:
        fragments["struct_pool"][mask] = wrapper_allocation.local_pool
        fragments["struct_offset"][mask] = wrapper_allocation.offset

    cloned_wrapper = clone_fragment_sources(
        wrapper_rows, wrapper_allocation.local_pool, donor_offset, wrapper_allocation.offset
    )
    wrapper_data = cloned_wrapper["link_offset"] == wrapper_allocation.offset + 8
    cloned_wrapper["struct_pool"][wrapper_data] = data_allocation.local_pool
    cloned_wrapper["struct_offset"][wrapper_data] = data_allocation.offset

    cloned_data = clone_fragment_sources(
        data_rows, data_allocation.local_pool, data_offset, data_allocation.offset
    )
    data_array = cloned_data["link_offset"] == data_allocation.offset
    cloned_data["struct_pool"][data_array] = array_allocation.local_pool
    cloned_data["struct_offset"][data_array] = array_allocation.offset

    cloned_array = clone_fragment_sources(
        array_rows, array_allocation.local_pool, array_offset, array_allocation.offset
    )
    for index, (pool, offset, _weight) in enumerate(children):
        slot = cloned_array["link_offset"] == array_allocation.offset + index * SELECTOR_ENTRY_SIZE
        if int(slot.sum()) != 1:
            raise ValueError(f"Selector slot {index} does not have one pointer fragment")
        cloned_array["struct_pool"][slot] = local(pool)
        cloned_array["struct_offset"][slot] = offset

    new_rows = np.concatenate((cloned_wrapper, cloned_data, cloned_array))
    ovs.fragments = np.concatenate((fragments, new_rows))
    ovs.fragments.sort(order=("link_pool", "struct_pool", "link_offset", "struct_offset"))
    static.num_fragments = old_fragments + len(new_rows)

    ovs.write_pools()
    uncompressed = ovs.write_archive()
    pool_growth = (
        wrapper_allocation.new_size - wrapper_allocation.old_size
        + data_allocation.new_size - data_allocation.old_size
        + array_allocation.new_size - array_allocation.old_size
    )
    expected_uncompressed = old_uncompressed + pool_growth + len(new_rows) * 16
    if len(uncompressed) != expected_uncompressed:
        raise ValueError("Unexpected STATIC growth while cloning selector")
    expected_sizes = list(old_pool_sizes)
    for allocation in (wrapper_allocation, data_allocation, array_allocation):
        expected_sizes[allocation.local_pool] = allocation.new_size
    if tuple(int(pool.size) for pool in ovs.pools) != tuple(expected_sizes):
        raise ValueError("An unrelated pool changed size while cloning selector")

    _, new_compressed, compressed = ovs.compress(uncompressed, True)
    source_bytes = source.read_bytes()
    header_size = len(source_bytes) - old_compressed
    meta_offset = header_size - len(ovl.archives_meta) * 8 + static_index * 8
    result = bytearray(source_bytes[:header_size])
    result.extend(compressed)
    struct.pack_into("<I", result, int(static.io_start) + NUM_FRAGMENTS_OFFSET, static.num_fragments)
    struct.pack_into("<I", result, int(static.io_start) + COMPRESSED_SIZE_OFFSET, new_compressed)
    struct.pack_into("<Q", result, int(static.io_start) + UNCOMPRESSED_SIZE_OFFSET, expected_uncompressed)
    struct.pack_into("<I", result, int(static.io_start) + POOLS_END_OFFSET, old_pools_end + pool_growth)
    struct.pack_into("<I", result, meta_offset, old_reservation + pool_growth)
    publication = CandidatePublication(source, output)
    publication.path.write_bytes(result)

    check_ovl, check_loader = load_motiongraph(publication.path, name or None, game)
    check_deref = build_deref(check_loader)
    clone = _activity(
        check_loader, check_ovl, wrapper_allocation.global_pool, wrapper_allocation.offset
    )
    check_payload = check_deref(clone.data)
    check_entries = check_deref(check_payload.activities)
    actual = tuple(
        (int(entry.activity.target_pool.i), int(entry.activity.target_offset), int(entry.weighting))
        for entry in check_entries
    )
    if clone.data_type.data != "RandomSelectActivityActivity" or actual != children:
        raise ValueError(f"Reloaded selector children are wrong: {actual!r}")

    publication.commit()

    return RandomSelectorCloneReport(
        output=output,
        donor_wrapper=(donor_pool, donor_offset),
        clone_wrapper=(wrapper_allocation.global_pool, wrapper_allocation.offset),
        clone_data=(data_allocation.global_pool, data_allocation.offset),
        clone_array=(array_allocation.global_pool, array_allocation.offset),
        redirected_references=sum(inbound_counts),
        cloned_fragments=len(new_rows),
        children=children,
    )
