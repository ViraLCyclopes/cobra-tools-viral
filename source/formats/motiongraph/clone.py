"""Game-verified complete ``AnimationActivity`` cloning for JWE3 STATIC archives.

This deliberately implements only the topology proven by motiongraph Test L:
one 48-byte Activity wrapper in the final partial type-2 page, one 96-byte
AnimationActivityData payload in the final partial type-3 page, cloned internal
fragments, and redirection of all or an explicit subset of the exact donor
wrapper's existing inbound edges.
It does not add a state entry, selector entry, MANI, or string.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .capacity import census, diff_census
from .edit import DEFAULT_GAME, load_motiongraph
from .report import build_deref
from .surgical_growth import (
    _load_quiet,
    append_tail_pool_bytes,
    repoint_pool_end_fragments,
)


NUM_FRAGMENTS_OFFSET = 28
COMPRESSED_SIZE_OFFSET = 44
UNCOMPRESSED_SIZE_OFFSET = 48
POOLS_END_OFFSET = 60
WRAPPER_SIZE = 48
DATA_SIZE = 96
TYPE_3_PAGE_LIMIT = 16384
SPEED_OFFSET = 32


@dataclass(frozen=True)
class CompleteActivityCloneReport:
    output: Path
    donor_wrapper: tuple[int, int]
    clone_wrapper: tuple[int, int]
    donor_data: tuple[int, int]
    clone_data: tuple[int, int]
    inbound_references: int
    cloned_fragments: int
    moved_end_sentinels: int
    speed: float | None
    orphaned_donor: bool
    decoded_added: int
    decoded_removed: int
    fragments_before: int
    fragments_after: int
    uncompressed_before: int
    uncompressed_after: int
    compressed_before: int
    compressed_after: int


def internal_fragment_mask(fragments, local_pool: int, offset: int, size: int):
    """Select fragments whose pointer source lies inside one fixed allocation."""
    if size <= 0:
        raise ValueError("Allocation size must be positive")
    return (
        (fragments["link_pool"] == local_pool)
        & (fragments["link_offset"] >= offset)
        & (fragments["link_offset"] < offset + size)
    )


def clone_fragment_sources(rows, new_local_pool: int, old_offset: int, new_offset: int):
    """Copy fragment rows while translating only their pointer-source addresses."""
    result = rows.copy()
    result["link_pool"] = new_local_pool
    result["link_offset"] = (
        result["link_offset"].astype(np.int64) + new_offset - old_offset
    ).astype(result.dtype["link_offset"])
    return result


def exact_inbound_mask(fragments, target_local_pool: int, target_offset: int):
    """Select all fragments targeting one exact allocation address."""
    return (
        (fragments["struct_pool"] == target_local_pool)
        & (fragments["struct_offset"] == target_offset)
    )


def inbound_source_mask(fragments, inbound, sources):
    """Restrict an exact-target inbound mask to explicit local pointer sources."""
    selected = np.zeros(len(fragments), dtype=bool)
    for pool, offset in sources:
        match = (
            inbound
            & (fragments["link_pool"] == pool)
            & (fragments["link_offset"] == offset)
        )
        count = int(match.sum())
        if count != 1:
            raise ValueError(
                f"Inbound source {pool}/{offset} matched {count} exact references, expected 1"
            )
        selected |= match
    return selected


def _activity_at(source: Path, name: str | None, pool: int, offset: int, game: str):
    ovl, loader = load_motiongraph(source, name or None, game)
    if pool < 0 or pool >= len(ovl.pools):
        raise ValueError(f"Activity pool is out of range: {pool}")
    activity = loader.context.recursion.get((ovl.pools[pool], offset))
    if type(activity).__name__ != "Activity":
        raise ValueError(f"No Activity decodes at global pool {pool}, offset {offset}")
    activity_type = getattr(getattr(activity, "data_type", None), "data", None)
    if activity_type != "AnimationActivity":
        raise ValueError(
            f"Complete cloning currently supports AnimationActivity only, not {activity_type!r}"
        )
    payload = build_deref(loader)(activity.data)
    if type(payload).__name__ != "AnimationActivityData" or int(payload.io_size) != DATA_SIZE:
        raise ValueError("Selected activity does not have one 96-byte AnimationActivityData payload")
    data_pointer = activity.data
    return (
        ovl,
        loader,
        activity,
        payload,
        int(data_pointer.target_pool.i),
        int(data_pointer.target_offset),
    )


def clone_complete_animation_activity(
    source: Path,
    output: Path,
    activity_pool: int,
    activity_offset: int,
    name: str | None = None,
    speed: float | None = None,
    redirect_sources: tuple[tuple[int, int], ...] | None = None,
    game: str = DEFAULT_GAME,
) -> CompleteActivityCloneReport:
    """Clone one exact activity and redirect selected inbound edges.

    ``source`` remains untouched. ``output`` must already be the same-basename OVL
    inside a separately copied complete archive family. ``redirect_sources`` uses
    global pool/offset pointer-source addresses; omit it to preserve the original
    all-inbound behavior.
    """
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Refusing to overwrite the source OVL")
    if source.name.lower() != output.name.lower():
        raise ValueError("Source and staged output OVL basenames must match")
    if not output.is_file():
        raise ValueError("Copy the complete archive family to the stage directory first")
    if speed is not None and (not np.isfinite(speed) or speed < 0.0):
        raise ValueError("Playback speed must be a finite non-negative number")

    semantic_ovl, source_loader, _activity, source_payload, data_global, data_offset = _activity_at(
        source, name, activity_pool, activity_offset, game
    )
    # The semantic and quiet loads must expose the same global pool numbering.
    if activity_pool >= len(semantic_ovl.pools) or data_global >= len(semantic_ovl.pools):
        raise ValueError("Selected activity uses an invalid global pool")

    ovl, static = _load_quiet(source, game)
    ovs = static.content

    def local(global_index: int) -> int:
        if global_index < 0 or global_index >= len(ovl.pools):
            raise ValueError(f"Global pool is out of range: {global_index}")
        pool = ovl.pools[global_index]
        if pool not in ovs.pools:
            raise ValueError(f"Global pool {global_index} is not in STATIC")
        return ovs.pools.index(pool)

    wrapper_local, data_local = local(activity_pool), local(data_global)
    wrapper_pool, data_pool = ovl.pools[activity_pool], ovl.pools[data_global]
    wrapper = wrapper_pool.data.getvalue()[activity_offset:activity_offset + WRAPPER_SIZE]
    payload = data_pool.data.getvalue()[data_offset:data_offset + DATA_SIZE]
    if len(wrapper) != WRAPPER_SIZE:
        raise ValueError("Selected Activity wrapper is not readable as 48 bytes")
    if len(payload) != DATA_SIZE:
        raise ValueError("Selected AnimationActivityData is not readable as 96 bytes")

    fragments = ovs.fragments
    wrapper_rows = fragments[
        internal_fragment_mask(fragments, wrapper_local, activity_offset, WRAPPER_SIZE)
    ].copy()
    wrapper_layout = sorted(int(value) - activity_offset for value in wrapper_rows["link_offset"])
    if wrapper_layout != [0, 8]:
        raise ValueError(f"Unexpected Activity pointer layout: {wrapper_layout}")
    data_pointer_rows = wrapper_rows[wrapper_rows["link_offset"] == activity_offset + 8]
    if len(data_pointer_rows) != 1 or (
        int(data_pointer_rows[0]["struct_pool"]), int(data_pointer_rows[0]["struct_offset"])
    ) != (data_local, data_offset):
        raise ValueError("Activity +8 does not point to the decoded AnimationActivityData")

    data_rows = fragments[
        internal_fragment_mask(fragments, data_local, data_offset, DATA_SIZE)
    ].copy()
    if not len(data_rows):
        raise ValueError("AnimationActivityData has no serialized internal pointers")
    all_inbound = exact_inbound_mask(fragments, wrapper_local, activity_offset)
    all_inbound_count = int(all_inbound.sum())
    if all_inbound_count < 1:
        raise ValueError("Selected activity has no inbound references to redirect")
    if redirect_sources is not None:
        if not redirect_sources:
            raise ValueError("Choose at least one inbound source to redirect")
        local_sources = tuple((local(pool), int(offset)) for pool, offset in redirect_sources)
        inbound = inbound_source_mask(fragments, all_inbound, local_sources)
    else:
        inbound = all_inbound
    inbound_count = int(inbound.sum())

    old_fragments = int(static.num_fragments)
    old_uncompressed = int(static.uncompressed_size)
    old_compressed = int(static.compressed_size)
    old_pools_end = int(static.pools_end)
    old_pool_sizes = tuple(int(pool.size) for pool in ovs.pools)
    static_index = ovl.archives.index(static)
    old_reservation = int(ovl.archives_meta[static_index].unk_0)

    wrapper_allocation = append_tail_pool_bytes(
        ovl.pools, ovs.pools, 2, wrapper, alignment=16
    )
    wrapper_sentinels = repoint_pool_end_fragments(
        fragments,
        wrapper_allocation.local_pool,
        wrapper_allocation.old_size,
        wrapper_allocation.new_size,
    )
    # The bound now derives from the largest type-3 pool this archive ships;
    # Test W game-verified 17,600, well past the old hardcoded 16,384.
    data_allocation = append_tail_pool_bytes(
        ovl.pools, ovs.pools, 3, payload, alignment=16
    )
    data_sentinels = repoint_pool_end_fragments(
        fragments,
        data_allocation.local_pool,
        data_allocation.old_size,
        data_allocation.new_size,
    )

    # Redirect only references to this exact wrapper instance, never every clip-name match.
    fragments["struct_pool"][inbound] = wrapper_allocation.local_pool
    fragments["struct_offset"][inbound] = wrapper_allocation.offset

    cloned_wrapper = clone_fragment_sources(
        wrapper_rows, wrapper_allocation.local_pool, activity_offset, wrapper_allocation.offset
    )
    cloned_data_pointer = cloned_wrapper["link_offset"] == wrapper_allocation.offset + 8
    if int(cloned_data_pointer.sum()) != 1:
        raise ValueError("Cloned Activity does not have one data pointer")
    cloned_wrapper["struct_pool"][cloned_data_pointer] = data_allocation.local_pool
    cloned_wrapper["struct_offset"][cloned_data_pointer] = data_allocation.offset
    cloned_data = clone_fragment_sources(
        data_rows, data_allocation.local_pool, data_offset, data_allocation.offset
    )
    new_fragments = np.concatenate((cloned_wrapper, cloned_data))
    ovs.fragments = np.concatenate((fragments, new_fragments))
    ovs.fragments.sort(order=("link_pool", "struct_pool", "link_offset", "struct_offset"))
    static.num_fragments = old_fragments + len(new_fragments)

    if speed is not None:
        target_pool = ovl.pools[data_allocation.global_pool]
        target_pool.data.seek(data_allocation.offset + SPEED_OFFSET)
        target_pool.data.write(struct.pack("<f", float(speed)))
        target_pool.data.seek(0)

    ovs.write_pools()
    uncompressed = ovs.write_archive()
    pool_growth = (
        wrapper_allocation.new_size - wrapper_allocation.old_size
        + data_allocation.new_size - data_allocation.old_size
    )
    expected_uncompressed = old_uncompressed + pool_growth + len(new_fragments) * 16
    if len(uncompressed) != expected_uncompressed:
        raise ValueError(
            f"Unexpected STATIC growth: {old_uncompressed} -> {len(uncompressed)}, "
            f"expected {expected_uncompressed}"
        )
    expected_pool_sizes = list(old_pool_sizes)
    expected_pool_sizes[wrapper_allocation.local_pool] = wrapper_allocation.new_size
    expected_pool_sizes[data_allocation.local_pool] = data_allocation.new_size
    if tuple(int(pool.size) for pool in ovs.pools) != tuple(expected_pool_sizes):
        raise ValueError("A pool other than the selected type-2/type-3 tails changed size")

    _, new_compressed, compressed = ovs.compress(uncompressed, True)
    source_bytes = source.read_bytes()
    archive_header = int(static.io_start)
    header_size = len(source_bytes) - old_compressed
    meta_offset = header_size - len(ovl.archives_meta) * 8 + static_index * 8
    result = bytearray(source_bytes[:header_size])
    result.extend(compressed)
    struct.pack_into("<I", result, archive_header + NUM_FRAGMENTS_OFFSET, static.num_fragments)
    struct.pack_into("<I", result, archive_header + COMPRESSED_SIZE_OFFSET, new_compressed)
    struct.pack_into("<Q", result, archive_header + UNCOMPRESSED_SIZE_OFFSET, expected_uncompressed)
    struct.pack_into("<I", result, archive_header + POOLS_END_OFFSET, old_pools_end + pool_growth)
    struct.pack_into("<I", result, meta_offset, old_reservation + pool_growth)
    output.write_bytes(result)

    check_ovl, check_static = _load_quiet(output, game)
    if int(check_static.num_fragments) != old_fragments + len(new_fragments):
        raise ValueError("Reloaded fragment count is wrong")
    if int(check_static.uncompressed_size) != expected_uncompressed:
        raise ValueError("Reloaded uncompressed size is wrong")
    if int(check_ovl.archives_meta[static_index].unk_0) != old_reservation + pool_growth:
        raise ValueError("Reloaded STATIC reservation is wrong")
    check_fragments = check_static.content.fragments
    donor_inbound_after = int(exact_inbound_mask(
        check_fragments, wrapper_local, activity_offset
    ).sum())
    clone_inbound_after = int(exact_inbound_mask(
        check_fragments, wrapper_allocation.local_pool, wrapper_allocation.offset
    ).sum())
    # The clone's own pointers never target its wrapper, so this count is external only.
    expected_donor_inbound = all_inbound_count - inbound_count
    if donor_inbound_after != expected_donor_inbound or clone_inbound_after != inbound_count:
        raise ValueError(
            f"Reloaded inbound references are wrong: donor={donor_inbound_after}, "
            f"expected donor={expected_donor_inbound}, clone={clone_inbound_after}, "
            f"expected clone={inbound_count}"
        )
    for allocation, moved, new_allocation_targets in (
        (wrapper_allocation, wrapper_sentinels, inbound_count),
        (data_allocation, data_sentinels, 1),
    ):
        old_end = int(((check_fragments["struct_pool"] == allocation.local_pool)
                       & (check_fragments["struct_offset"] == allocation.old_size)).sum())
        new_end = int(((check_fragments["struct_pool"] == allocation.local_pool)
                       & (check_fragments["struct_offset"] == allocation.new_size)).sum())
        expected_old_end = (
            new_allocation_targets if allocation.offset == allocation.old_size else 0
        )
        if old_end != expected_old_end or new_end != moved:
            raise ValueError(
                f"Reloaded type-{allocation.pool_type} end sentinels are wrong: "
                f"old={old_end} (expected {expected_old_end}), "
                f"new={new_end} (expected {moved})"
            )

    _decoded_ovl, decoded_loader, clone_activity, clone_payload, decoded_data_pool, decoded_data_offset = (
        _activity_at(
            output, name, wrapper_allocation.global_pool, wrapper_allocation.offset, game
        )
    )
    if (decoded_data_pool, decoded_data_offset) != (
        data_allocation.global_pool, data_allocation.offset
    ):
        raise ValueError("Reloaded clone points to the wrong data allocation")
    if clone_activity.data_type.data != "AnimationActivity":
        raise ValueError("Reloaded clone changed activity type")
    if speed is not None and not np.isclose(float(clone_payload.speed.float), speed):
        raise ValueError("Reloaded clone does not contain the requested playback speed")
    # Preserve the exact donor MANI and all other payload semantics.
    donor_mani = build_deref(source_loader)(source_payload.mani)
    clone_mani = build_deref(decoded_loader)(clone_payload.mani)
    if clone_mani != donor_mani:
        raise ValueError("Reloaded clone changed the donor MANI reference")

    # Redirecting a donor's LAST inbound reference orphans it: nothing points at
    # the wrapper any more, so neither it nor its payload is decoded again and
    # the object count stays flat. Prove the census moved exactly as intended
    # rather than trusting that a successful reload means nothing was lost.
    orphaned_donor = expected_donor_inbound == 0
    delta = 0 if orphaned_donor else 1
    expected_added = {
        (wrapper_allocation.global_pool, wrapper_allocation.offset),
        (data_allocation.global_pool, data_allocation.offset),
    }
    expected_removed = (
        {(activity_pool, activity_offset), (data_global, data_offset)}
        if orphaned_donor else set()
    )
    before_census, after_census = census(source_loader), census(decoded_loader)
    difference = diff_census(before_census, after_census)
    added = {address for row in difference["changed_types"].values() for address in row["added"]}
    removed = {
        address for row in difference["changed_types"].values() for address in row["removed"]
    }
    if added != expected_added or removed != expected_removed:
        raise ValueError(
            f"Decoded object census changed unexpectedly: added {sorted(added)}, "
            f"expected {sorted(expected_added)}; removed {sorted(removed)}, "
            f"expected {sorted(expected_removed)}"
        )
    for type_name in ("Activity", "AnimationActivityData"):
        moved = after_census["counts"].get(type_name, 0) - before_census["counts"].get(type_name, 0)
        if moved != delta:
            raise ValueError(
                f"{type_name} count moved by {moved}, expected a delta of {delta}"
            )

    return CompleteActivityCloneReport(
        output=output,
        donor_wrapper=(activity_pool, activity_offset),
        clone_wrapper=(wrapper_allocation.global_pool, wrapper_allocation.offset),
        donor_data=(data_global, data_offset),
        clone_data=(data_allocation.global_pool, data_allocation.offset),
        inbound_references=inbound_count,
        cloned_fragments=len(new_fragments),
        moved_end_sentinels=wrapper_sentinels + data_sentinels,
        speed=speed,
        orphaned_donor=orphaned_donor,
        decoded_added=difference["added"],
        decoded_removed=difference["removed"],
        fragments_before=old_fragments,
        fragments_after=old_fragments + len(new_fragments),
        uncompressed_before=old_uncompressed,
        uncompressed_after=expected_uncompressed,
        compressed_before=old_compressed,
        compressed_after=new_compressed,
    )
