"""Experimental fixed-pool motiongraph topology growth.

This does not append bytes or rebuild pools. It consumes verified zero padding
immediately before a counted array, moves that array's target boundary backward,
and prepends one null element. The experiment isolates logical count growth from
pool-size growth and fragment-count growth.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from generated.formats.ovl import OvlFile
from generated.formats.ovl_base.structs.MemStruct import MemStruct
from generated.formats.ovl_base.structs.Pointer import Pointer

from .capacity import _iter_pointers, audit_allocations
from .edit import COMPRESSED_SIZE_OFFSET, DEFAULT_GAME, load_motiongraph
from .staging import CandidatePublication, validate_staged_pair


NUM_FRAGMENTS_OFFSET = 28
UNCOMPRESSED_SIZE_OFFSET = 48
TYPE_2_PAGE_SIZE = 16416


@dataclass(frozen=True)
class TailPoolAllocation:
    global_pool: int
    local_pool: int
    pool_type: int
    old_size: int
    offset: int
    padding: int
    payload_size: int
    new_size: int
    page_size: int


def observed_type_capacity(archive_pools, pool_type: int) -> int:
    """Largest size the archive itself uses for this pool type.

    Test W (2026-08-27) game-verified growing the final type-3 pool to 17,600,
    well past the 16,384 this module used to enforce, and a stock Acrocanthosaurus
    archive contains 49 type-2 pools above 16,416 (up to 22,408) and fourteen
    type-3 pools at 24,000. A size the shipped file uses freely is not a cap, so
    the bound is derived from the data rather than hardcoded.
    """
    sizes = [int(pool.size) for pool in archive_pools if int(pool.type) == pool_type]
    if not sizes:
        raise ValueError(f"Archive has no pool of type {pool_type}")
    return max(sizes)


def plan_tail_pool_allocation(global_pools, archive_pools, pool_type: int,
                              payload_size: int, alignment: int = 16,
                              page_size: int | None = None) -> TailPoolAllocation:
    """Plan an allocation in the final partial pool of one pool type.

    The bound defaults to :func:`observed_type_capacity` - the largest size this
    archive already uses for the type. Tests F/F2 did crash after growing global
    pool 67 from 16,416, but that pool was a fully packed table of 342 * 48-byte
    Activity wrappers, which is a per-pool property and not a global page size.
    Creating an additional pool remains a separate, unsupported operation.
    """
    if payload_size <= 0:
        raise ValueError("Payload size must be positive")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("Alignment must be a positive power of two")
    if page_size is None:
        page_size = observed_type_capacity(archive_pools, pool_type)
    candidates = [pool for pool in archive_pools if int(pool.type) == pool_type]
    if not candidates:
        raise ValueError(f"Archive has no pool of type {pool_type}")
    target = candidates[-1]
    old_size = int(target.size)
    offset = (old_size + alignment - 1) & ~(alignment - 1)
    new_size = offset + payload_size
    if new_size > page_size:
        raise ValueError(
            f"Final type-{pool_type} pool has no room: {old_size} -> {new_size}, "
            f"page limit {page_size}; creating a new page is not supported"
        )
    try:
        local_pool = next(index for index, pool in enumerate(archive_pools) if pool is target)
        global_pool = next(index for index, pool in enumerate(global_pools) if pool is target)
    except StopIteration as exc:
        raise ValueError("Tail pool is not present in both pool tables") from exc
    return TailPoolAllocation(
        global_pool=global_pool,
        local_pool=local_pool,
        pool_type=pool_type,
        old_size=old_size,
        offset=offset,
        padding=offset - old_size,
        payload_size=payload_size,
        new_size=new_size,
        page_size=page_size,
    )


def append_tail_pool_bytes(global_pools, archive_pools, pool_type: int,
                           payload: bytes, alignment: int = 16,
                           page_size: int | None = None) -> TailPoolAllocation:
    """Append bytes according to :func:`plan_tail_pool_allocation`.

    This only mutates the selected pool's bytes and size. The caller must add
    pointer fragments and update archive compression/size/reservation metadata.
    Unknown `MemPool` counters are deliberately preserved.
    """
    allocation = plan_tail_pool_allocation(
        global_pools, archive_pools, pool_type, len(payload), alignment, page_size
    )
    pool = global_pools[allocation.global_pool]
    pool.data.seek(0, 2)
    if pool.data.tell() != allocation.old_size:
        raise ValueError("Tail pool stream length does not match MemPool.size")
    pool.data.write(b"\0" * allocation.padding)
    pool.data.write(payload)
    pool.data.seek(0)
    pool.size = allocation.new_size
    return allocation


def repoint_pool_end_fragments(fragments, local_pool: int,
                               old_end: int, new_end: int) -> int:
    """Keep serialized end-sentinel pointers at the end after pool growth.

    A pointer target equal to the old pool size cannot address an existing byte.
    Test G found 1,463 such fragments sharing the type-3 tail's empty-array
    sentinel. Leaving them at the old end aliases newly appended storage.
    """
    if new_end <= old_end:
        raise ValueError("New pool end must be greater than the old end")
    new_mask = (
        (fragments["struct_pool"] == local_pool)
        & (fragments["struct_offset"] == new_end)
    )
    if int(new_mask.sum()):
        raise ValueError("New pool end already has inbound fragments")
    old_mask = (
        (fragments["struct_pool"] == local_pool)
        & (fragments["struct_offset"] == old_end)
    )
    moved = int(old_mask.sum())
    fragments["struct_offset"][old_mask] = new_end
    return moved


@dataclass(frozen=True)
class NullPrefixGrowthReport:
    output: Path
    pool: int
    old_offset: int
    new_offset: int
    old_count: int
    new_count: int
    element_size: int
    count_pool: int
    count_offset: int
    repointed_fragments: int
    pools: int
    fragments: int
    uncompressed_size: int
    compressed_before: int
    compressed_after: int


@dataclass(frozen=True)
class FragmentSourceMoveReport:
    output: Path
    source: tuple[int, int]
    destination: tuple[int, int]
    target: tuple[int, int]
    moved_fragments: int
    pools: int
    fragments: int
    uncompressed_size: int
    compressed_before: int
    compressed_after: int
    count_change: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class FragmentGrowthReport:
    output: Path
    source: tuple[int, int]
    target: tuple[int, int]
    pools: int
    fragments_before: int
    fragments_after: int
    uncompressed_before: int
    uncompressed_after: int
    compressed_before: int
    compressed_after: int


def _field_size(field_type: type, value: Any, context: Any, arguments: tuple) -> int:
    arg = arguments[0] if arguments else 0
    template = arguments[1] if len(arguments) > 1 else None
    try:
        return int(field_type.get_size(value, context, arg, template))
    except TypeError:
        return int(field_type.get_size(value, context))


def _count_field(owner: MemStruct, array_field: str, source_pool: Any):
    attributes = list(type(owner)._get_filtered_attribute_list(owner, include_abstract=False))
    field_names = {attribute[0] for attribute in attributes}
    offset = int(owner.io_start)
    for name, field_type, arguments, _default in attributes:
        value = getattr(owner, name)
        size = _field_size(field_type, value, owner.context, arguments)
        if MemStruct.is_array_count(name, field_names) == array_field:
            return source_pool, offset, size, int(value), name
        offset += size
    raise ValueError(
        f"Could not find the serialized count field for {type(owner).__name__}.{array_field}"
    )


def _load_quiet(path: Path, game: str):
    ovl = OvlFile()
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        ovl.load(str(path), {"game": game})
    finally:
        logging.disable(previous_disable)
    static = next((archive for archive in ovl.archives if archive.name == "STATIC"), None)
    if static is None:
        raise ValueError("OVL has no STATIC archive")
    return ovl, static


def grow_null_prefix(source: Path, output: Path, array_pool: int, array_offset: int,
                     name: str | None = None, game: str = DEFAULT_GAME,
                     allow_eight_alignment: bool = False) -> NullPrefixGrowthReport:
    """Prepend one null element without changing any pool or fragment count.

    ``allow_eight_alignment`` is deliberately explicit: cobra's writer normally
    aligns non-string pointer targets to 16 bytes. The known land candidates use
    8-byte ActivityReference elements and would move to an 8-mod-16 address.
    """
    source, output = validate_staged_pair(source, output)

    ovl, loader = load_motiongraph(source, name, game)
    static = next((archive for archive in ovl.archives if archive.name == "STATIC"), None)
    if static is None:
        raise ValueError("OVL has no STATIC archive")
    matches = []
    for owner, field, pointer in _iter_pointers(loader):
        pool = getattr(pointer, "target_pool", None)
        offset = getattr(pointer, "target_offset", None)
        if pool is not None and offset is not None and int(pool.i) == array_pool \
                and int(offset) == array_offset:
            matches.append((owner, field, pointer))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one decoded pointer to array {array_pool}:{array_offset}, found {len(matches)}"
        )
    owner, array_field, pointer = matches[0]
    data = pointer.data or loader.context.recursion.get((pointer.target_pool, pointer.target_offset))
    try:
        old_count = len(data)
    except TypeError as exc:
        raise ValueError("Selected pointer target is not an array") from exc
    used = int(getattr(data, "io_size", 0))
    if old_count <= 0 or used <= 0 or used % old_count:
        raise ValueError("Could not derive a fixed array element size")
    element_size = used // old_count
    new_offset = array_offset - element_size
    if new_offset % 16 and not (allow_eight_alignment and new_offset % 8 == 0):
        raise ValueError(
            f"New target {new_offset} is not 16-byte aligned; use the explicit experimental "
            f"8-byte-alignment opt-in only for an 8-aligned target"
        )

    allocations = [row for row in audit_allocations(loader) if row["pool"] == array_pool]
    allocations.sort(key=lambda row: row["offset"])
    index = next(
        (index for index, row in enumerate(allocations) if row["offset"] == array_offset), None
    )
    if index is None or index == 0:
        raise ValueError("Could not identify the allocation immediately before the array")
    previous = allocations[index - 1]
    if not previous["zero_tail"] or previous["slack_bytes"] < element_size:
        raise ValueError("The preceding allocation has insufficient verified zero padding")
    if previous["offset"] + previous["allocation_bytes"] != array_offset:
        raise ValueError("The preceding allocation is not contiguous with the array")
    target_pool = pointer.target_pool
    target_local_pool = static.content.pools.index(target_pool)
    prefix = target_pool.data.getvalue()[new_offset:array_offset]
    if prefix != b"\0" * element_size:
        raise ValueError("Candidate prefix bytes are not all zero")

    count_pool, count_offset, count_width, count_value, count_name = _count_field(
        owner, str(array_field), pointer.src_pool
    )
    if count_value != old_count:
        raise ValueError(
            f"{type(owner).__name__}.{count_name}={count_value}, decoded array has {old_count} entries"
        )
    count_blob = count_pool.data.getvalue()[count_offset:count_offset + count_width]
    if int.from_bytes(count_blob, "little") != old_count:
        raise ValueError("Serialized array count does not match the decoded value")
    new_count = old_count + 1
    try:
        replacement_count = new_count.to_bytes(count_width, "little")
    except OverflowError as exc:
        raise ValueError("Incremented count does not fit its serialized field") from exc

    fragments = static.content.fragments
    mask = ((fragments["struct_pool"] == target_local_pool)
            & (fragments["struct_offset"] == array_offset))
    repointed = int(mask.sum())
    if repointed != 1:
        raise ValueError(
            f"Expected one inbound fragment for the selected array, found {repointed}"
        )
    before_topology = (
        int(static.num_pools), int(static.num_fragments), int(static.uncompressed_size),
        tuple(int(pool.size) for pool in static.content.pools),
    )
    compressed_before = int(static.compressed_size)

    count_pool.data.seek(count_offset)
    count_pool.data.write(replacement_count)
    count_pool.data.seek(0)
    # Prefix remains zero, making the new ActivityReference/StateReference null.
    fragments["struct_offset"][mask] = new_offset
    static.content.write_pools()
    uncompressed = static.content.write_archive()
    if len(uncompressed) != before_topology[2]:
        raise ValueError("STATIC uncompressed size changed during fixed-pool growth")
    after_topology = (
        int(static.num_pools), int(static.num_fragments), len(uncompressed),
        tuple(int(pool.size) for pool in static.content.pools),
    )
    if after_topology != before_topology:
        raise ValueError(f"Archive topology changed unexpectedly: {before_topology} -> {after_topology}")
    _, compressed_size, compressed = static.content.compress(uncompressed, True)
    source_bytes = source.read_bytes()
    header_size = len(source_bytes) - compressed_before
    result = bytearray(source_bytes[:header_size])
    result.extend(compressed)
    struct.pack_into("<I", result, static.io_start + COMPRESSED_SIZE_OFFSET, compressed_size)
    publication = CandidatePublication(source, output)
    publication.path.write_bytes(result)

    check, check_static = _load_quiet(publication.path, game)
    check_topology = (
        int(check_static.num_pools), int(check_static.num_fragments),
        int(check_static.uncompressed_size),
        tuple(int(pool.size) for pool in check_static.content.pools),
    )
    if check_topology != before_topology:
        raise ValueError(f"Reloaded topology changed: {before_topology} -> {check_topology}")
    check_fragments = check_static.content.fragments
    old_targets = int(((check_fragments["struct_pool"] == target_local_pool)
                       & (check_fragments["struct_offset"] == array_offset)).sum())
    new_targets = int(((check_fragments["struct_pool"] == target_local_pool)
                       & (check_fragments["struct_offset"] == new_offset)).sum())
    if old_targets or new_targets != repointed:
        raise ValueError(
            f"Reloaded fragment targets are wrong: old={old_targets}, new={new_targets}"
        )
    reloaded_count = int.from_bytes(
        check.pools[int(count_pool.i)].data.getvalue()[count_offset:count_offset + count_width],
        "little",
    )

    if reloaded_count != new_count:
        raise ValueError(f"Reloaded count is {reloaded_count}, expected {new_count}")
    check_loader = next(
        (item for item in check.loaders.values() if item.name.lower() == loader.name.lower()), None
    )
    if check_loader is None:
        raise ValueError("Reloaded output lost the motiongraph loader")
    decoded_matches = []
    for check_owner, check_field, check_pointer in _iter_pointers(check_loader):
        pool = getattr(check_pointer, "target_pool", None)
        offset = getattr(check_pointer, "target_offset", None)
        if (pool is not None and offset is not None and int(pool.i) == array_pool
                and int(offset) == new_offset and str(check_field) == str(array_field)
                and type(check_owner).__name__ == type(owner).__name__):
            decoded_matches.append(check_pointer)
    if len(decoded_matches) != 1:
        raise ValueError(
            f"Reloaded output resolved {len(decoded_matches)} matching grown arrays, expected one"
        )
    decoded_pointer = decoded_matches[0]
    decoded = decoded_pointer.data or check_loader.context.recursion.get(
        (decoded_pointer.target_pool, decoded_pointer.target_offset)
    )
    if len(decoded) != new_count:
        raise ValueError(f"Reloaded array has {len(decoded)} entries, expected {new_count}")
    first_pointers = list(MemStruct.get_instances_recursive(decoded[0], Pointer))
    if any(item.target_pool is not None for item, _field, _args in first_pointers):
        raise ValueError("Prepended array element is not null after reload")
    publication.commit()
    return NullPrefixGrowthReport(
        output=output, pool=array_pool, old_offset=array_offset, new_offset=new_offset,
        old_count=old_count, new_count=new_count, element_size=element_size,
        count_pool=int(count_pool.i), count_offset=count_offset,
        repointed_fragments=repointed, pools=before_topology[0],
        fragments=before_topology[1], uncompressed_size=before_topology[2],
        compressed_before=compressed_before, compressed_after=compressed_size,
    )


def move_fragment_source(source: Path, output: Path,
                         from_pool: int, from_offset: int,
                         to_pool: int, to_offset: int,
                         target_pool: int, target_offset: int,
                         count_pool: int | None = None,
                         count_offset: int | None = None,
                         old_count: int | None = None,
                         new_count: int | None = None,
                         game: str = DEFAULT_GAME) -> FragmentSourceMoveReport:
    """Move one existing pointer fragment to an unused pointer slot.

    Global pool indices are used at the API boundary; fragment records use
    archive-local indices internally. No fragment or pool is added or removed.

    The optional count arguments make fragment harvesting safe for a donor
    counted array: after moving its final reference, shorten that donor array so
    the now-null source slot is outside its logical extent. All four arguments
    must be supplied together and currently describe a little-endian uint32.
    """
    source, output = validate_staged_pair(source, output)
    ovl, static = _load_quiet(source, game)

    def local_pool(global_index: int):
        if global_index < 0 or global_index >= len(ovl.pools):
            raise ValueError(f"Global pool index is out of range: {global_index}")
        pool = ovl.pools[global_index]
        if pool not in static.content.pools:
            raise ValueError(f"Global pool {global_index} is not in STATIC")
        return static.content.pools.index(pool)

    from_local, to_local, target_local = (
        local_pool(from_pool), local_pool(to_pool), local_pool(target_pool)
    )
    fragments = static.content.fragments
    source_mask = (
        (fragments["link_pool"] == from_local)
        & (fragments["link_offset"] == from_offset)
        & (fragments["struct_pool"] == target_local)
        & (fragments["struct_offset"] == target_offset)
    )
    moved = int(source_mask.sum())
    if moved != 1:
        raise ValueError(f"Expected one exact donor fragment, found {moved}")
    destination_mask = (
        (fragments["link_pool"] == to_local)
        & (fragments["link_offset"] == to_offset)
    )
    if int(destination_mask.sum()):
        raise ValueError("Destination pointer slot already has a fragment")
    count_args = (count_pool, count_offset, old_count, new_count)
    has_count_change = any(value is not None for value in count_args)
    if has_count_change and not all(value is not None for value in count_args):
        raise ValueError("Supply all count adjustment arguments or none")
    if has_count_change:
        if new_count < 0 or new_count >= old_count:
            raise ValueError("Donor count adjustment must reduce a non-negative count")
        donor_count_pool = ovl.pools[count_pool]
        if donor_count_pool not in static.content.pools:
            raise ValueError(f"Global count pool {count_pool} is not in STATIC")
        count_blob = donor_count_pool.data.getvalue()[count_offset:count_offset + 4]
        if len(count_blob) != 4 or int.from_bytes(count_blob, "little") != old_count:
            raise ValueError("Serialized donor count does not match the expected old value")
    before_topology = (
        int(static.num_pools), int(static.num_fragments), int(static.uncompressed_size),
        tuple(int(pool.size) for pool in static.content.pools),
    )
    compressed_before = int(static.compressed_size)
    fragments["link_pool"][source_mask] = to_local
    fragments["link_offset"][source_mask] = to_offset
    if has_count_change:
        donor_count_pool.data.seek(count_offset)
        donor_count_pool.data.write(int(new_count).to_bytes(4, "little"))
        donor_count_pool.data.seek(0)
    static.content.write_pools()
    uncompressed = static.content.write_archive()
    after_topology = (
        int(static.num_pools), int(static.num_fragments), len(uncompressed),
        tuple(int(pool.size) for pool in static.content.pools),
    )
    if after_topology != before_topology:
        raise ValueError(f"Archive topology changed unexpectedly: {before_topology} -> {after_topology}")
    _, compressed_size, compressed = static.content.compress(uncompressed, True)
    source_bytes = source.read_bytes()
    result = bytearray(source_bytes[:len(source_bytes) - compressed_before])
    result.extend(compressed)
    struct.pack_into("<I", result, static.io_start + COMPRESSED_SIZE_OFFSET, compressed_size)
    publication = CandidatePublication(source, output)
    publication.path.write_bytes(result)

    check, check_static = _load_quiet(publication.path, game)
    check_topology = (
        int(check_static.num_pools), int(check_static.num_fragments),
        int(check_static.uncompressed_size),
        tuple(int(pool.size) for pool in check_static.content.pools),
    )
    if check_topology != before_topology:
        raise ValueError(f"Reloaded topology changed: {before_topology} -> {check_topology}")
    check_fragments = check_static.content.fragments
    donor_left = int((
        (check_fragments["link_pool"] == from_local)
        & (check_fragments["link_offset"] == from_offset)
        & (check_fragments["struct_pool"] == target_local)
        & (check_fragments["struct_offset"] == target_offset)
    ).sum())
    destination_after = int((
        (check_fragments["link_pool"] == to_local)
        & (check_fragments["link_offset"] == to_offset)
        & (check_fragments["struct_pool"] == target_local)
        & (check_fragments["struct_offset"] == target_offset)
    ).sum())

    if donor_left or destination_after != 1:
        raise ValueError(
            f"Reloaded fragment move failed: donor={donor_left}, destination={destination_after}"
        )
    if has_count_change:
        reloaded_count = int.from_bytes(
            check.pools[count_pool].data.getvalue()[count_offset:count_offset + 4], "little"
        )
        if reloaded_count != new_count:
            raise ValueError(
                f"Reloaded donor count is {reloaded_count}, expected {new_count}"
            )
    publication.commit()
    return FragmentSourceMoveReport(
        output=output, source=(from_pool, from_offset), destination=(to_pool, to_offset),
        target=(target_pool, target_offset), moved_fragments=moved,
        pools=before_topology[0], fragments=before_topology[1],
        uncompressed_size=before_topology[2], compressed_before=compressed_before,
        compressed_after=compressed_size,
        count_change=(count_pool, count_offset, old_count, new_count)
        if has_count_change else None,
    )


def add_fragment_source(source: Path, output: Path,
                        link_pool: int, link_offset: int,
                        target_pool: int, target_offset: int,
                        game: str = DEFAULT_GAME) -> FragmentGrowthReport:
    """Append one fragment record while leaving every pool byte unchanged.

    This is deliberately a narrow experimental writer for the next game gate.
    The source and target use global pool indices at the API boundary. The STATIC
    fragment table and uncompressed archive grow by exactly one 16-byte record.
    """
    source, output = validate_staged_pair(source, output)
    ovl, static = _load_quiet(source, game)

    def local_pool(global_index: int) -> int:
        if global_index < 0 or global_index >= len(ovl.pools):
            raise ValueError(f"Global pool index is out of range: {global_index}")
        pool = ovl.pools[global_index]
        if pool not in static.content.pools:
            raise ValueError(f"Global pool {global_index} is not in STATIC")
        return static.content.pools.index(pool)

    link_local, target_local = local_pool(link_pool), local_pool(target_pool)
    fragments = static.content.fragments
    source_mask = (
        (fragments["link_pool"] == link_local)
        & (fragments["link_offset"] == link_offset)
    )
    if int(source_mask.sum()):
        raise ValueError("New fragment source slot already has a fragment")
    before_pools = tuple(int(pool.size) for pool in static.content.pools)
    fragments_before = int(static.num_fragments)
    uncompressed_before = int(static.uncompressed_size)
    compressed_before = int(static.compressed_size)
    if len(fragments) != fragments_before:
        raise ValueError("Decoded fragment array length does not match archive header")

    added = np.array(
        [(link_local, link_offset, target_local, target_offset)], dtype=fragments.dtype
    )
    static.content.fragments = np.concatenate((fragments, added))
    static.content.fragments.sort(
        order=("link_pool", "struct_pool", "link_offset", "struct_offset")
    )
    static.num_fragments = fragments_before + 1
    static.content.write_pools()
    uncompressed = static.content.write_archive()
    uncompressed_after = len(uncompressed)
    if tuple(int(pool.size) for pool in static.content.pools) != before_pools:
        raise ValueError("A pool size changed while adding a fragment")
    if uncompressed_after != uncompressed_before + 16:
        raise ValueError(
            f"Expected one 16-byte fragment of growth, got "
            f"{uncompressed_before} -> {uncompressed_after}"
        )
    _, compressed_after, compressed = static.content.compress(uncompressed, True)

    source_bytes = source.read_bytes()
    archive_header = int(static.io_start)
    if struct.unpack_from("<I", source_bytes, archive_header + NUM_FRAGMENTS_OFFSET)[0] \
            != fragments_before:
        raise ValueError("Unexpected num_fragments location in ArchiveEntry")
    if struct.unpack_from("<I", source_bytes, archive_header + COMPRESSED_SIZE_OFFSET)[0] \
            != compressed_before:
        raise ValueError("Unexpected compressed_size location in ArchiveEntry")
    if struct.unpack_from("<Q", source_bytes, archive_header + UNCOMPRESSED_SIZE_OFFSET)[0] \
            != uncompressed_before:
        raise ValueError("Unexpected uncompressed_size location in ArchiveEntry")
    result = bytearray(source_bytes[:len(source_bytes) - compressed_before])
    result.extend(compressed)
    struct.pack_into(
        "<I", result, archive_header + NUM_FRAGMENTS_OFFSET, fragments_before + 1
    )
    struct.pack_into(
        "<I", result, archive_header + COMPRESSED_SIZE_OFFSET, compressed_after
    )
    struct.pack_into(
        "<Q", result, archive_header + UNCOMPRESSED_SIZE_OFFSET, uncompressed_after
    )
    publication = CandidatePublication(source, output)
    publication.path.write_bytes(result)

    check, check_static = _load_quiet(publication.path, game)
    if int(check_static.num_fragments) != fragments_before + 1:
        raise ValueError("Reloaded fragment count did not grow by one")
    if int(check_static.uncompressed_size) != uncompressed_after:
        raise ValueError("Reloaded uncompressed size does not match the grown archive")
    if tuple(int(pool.size) for pool in check_static.content.pools) != before_pools:
        raise ValueError("Reloaded pool sizes changed")
    check_fragments = check_static.content.fragments
    exact = int((
        (check_fragments["link_pool"] == link_local)
        & (check_fragments["link_offset"] == link_offset)
        & (check_fragments["struct_pool"] == target_local)
        & (check_fragments["struct_offset"] == target_offset)
    ).sum())
    if exact != 1:
        raise ValueError(f"Reloaded archive has {exact} copies of the new fragment")
    publication.commit()
    return FragmentGrowthReport(
        output=output, source=(link_pool, link_offset),
        target=(target_pool, target_offset), pools=len(before_pools),
        fragments_before=fragments_before, fragments_after=fragments_before + 1,
        uncompressed_before=uncompressed_before,
        uncompressed_after=uncompressed_after,
        compressed_before=compressed_before, compressed_after=compressed_after,
    )
