"""Direct STATIC patches for game-proven, fixed-topology motiongraph edits."""

from __future__ import annotations

import logging
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

from generated.formats.ovl import OvlFile
from .clone import (NUM_FRAGMENTS_OFFSET, POOLS_END_OFFSET,
                    UNCOMPRESSED_SIZE_OFFSET)
from .surgical_growth import (_load_quiet, append_tail_pool_bytes,
                              observed_type_capacity, repoint_pool_end_fragments)
from .staging import CandidatePublication, copy_motiongraph_family, validate_staged_pair


DEFAULT_GAME = "Jurassic World Evolution 3"
COMPRESSED_SIZE_OFFSET = 44
# Appending STRINGS to the last type-2 pool past the archive's own largest pool is
# game-verified (MegaraptorAnims 28,312 -> 28,723, 2026-10-02), and installed mods
# load type-2 pools of 164,796 (UltimasaurusCE.ovl) up to 1,180,896. Object/table
# growth keeps the archive bound: appending to a packed 48-byte Activity table
# crashed (Tests F/F2).
STRING_TAIL_CAPACITY = 200_000

if not hasattr(logging, "success"):
    logging.success = logging.info  # type: ignore[attr-defined]


@dataclass(frozen=True)
class StaticPatchReport:
    output: Path
    changed_bytes: int
    compressed_before: int
    compressed_after: int
    topology: tuple[int, int, int, int]
    source_pool: int
    source_offset: int
    target_pool: int | None = None
    target_offset: int | None = None
    references: int | None = None


def _load(path: Path, game: str):
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        ovl = OvlFile()
        ovl.load(str(path), {"game": game})
    finally:
        logging.disable(previous_disable)
    static = next((archive for archive in ovl.archives if archive.name == "STATIC"), None)
    if static is None:
        raise ValueError("OVL has no STATIC archive")
    return ovl, static


def _topology(ovl) -> tuple[int, int, int, int]:
    return (
        len(ovl.loaders), len(ovl.archives),
        sum(len(archive.content.pools) for archive in ovl.archives),
        sum(len(archive.content.fragments) for archive in ovl.archives),
    )


def _decompressed_static(source: Path, ovl, static) -> bytes:
    source_bytes = source.read_bytes()
    compressed = source_bytes[len(source_bytes) - int(static.compressed_size):]
    with static.content.decompress(ovl.reporter, compressed, static.uncompressed_size) as stream:
        result = stream.read()
    if len(result) != int(static.uncompressed_size):
        raise ValueError("Decompressed STATIC size disagrees with its header")
    return result


def _require_staged_output(source: Path, output: Path):
    return validate_staged_pair(source, output)


def _publish(source: Path, output: Path, ovl, static, original: bytes,
             modified: bytes) -> tuple[int, int]:
    if len(modified) != len(original):
        raise ValueError("STATIC uncompressed size changed")
    old_size = int(static.compressed_size)
    _, new_size, compressed = static.content.compress(modified, True)
    source_bytes = source.read_bytes()
    compressed_start = len(source_bytes) - old_size
    result = bytearray(source_bytes[:compressed_start])
    result.extend(compressed)
    struct.pack_into(
        "<I", result, int(static.io_start) + COMPRESSED_SIZE_OFFSET, new_size
    )
    publication = CandidatePublication(source, output)
    publication.path.write_bytes(result)
    return old_size, new_size, publication


def _find_unique_pool_string(static, text: str) -> tuple[int, int, bytes]:
    try:
        encoded = text.encode("ascii") + b"\0"
    except UnicodeEncodeError as exc:
        raise ValueError("Motiongraph strings must be ASCII") from exc
    matches = []
    for pool_index, pool in enumerate(static.content.pools):
        data = pool.data.getvalue()
        offset = data.find(encoded)
        while offset >= 0:
            matches.append((pool_index, offset))
            offset = data.find(encoded, offset + 1)
    if len(matches) != 1:
        raise ValueError(f"Expected one STATIC occurrence of {text!r}, found {len(matches)}")
    return matches[0][0], matches[0][1], encoded


def patch_string_slot(source: Path, output: Path, source_text: str, target_text: str,
                      *, expected_offset: int | None = None,
                      game: str = DEFAULT_GAME) -> StaticPatchReport:
    """Replace one string within its existing allocation; shorter text is null padded."""
    source, output = _require_staged_output(source, output)
    if source_text == target_text:
        raise ValueError("Source and target strings are identical")
    ovl, static = _load(source, game)
    pool_index, source_offset, encoded_source = _find_unique_pool_string(static, source_text)
    if expected_offset is not None and source_offset != expected_offset:
        raise ValueError(f"Source offset {source_offset} != expected {expected_offset}")
    try:
        target_bytes = target_text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Motiongraph strings must be ASCII") from exc
    if len(target_bytes) + 1 > len(encoded_source):
        raise ValueError(
            f"Replacement needs {len(target_bytes) + 1} bytes; slot has {len(encoded_source)}"
        )
    replacement = target_bytes + b"\0" * (len(encoded_source) - len(target_bytes))
    pool_bytes = static.content.pools[pool_index].data.getvalue()
    original = _decompressed_static(source, ovl, static)
    pool_at = original.find(pool_bytes)
    if pool_at < 0 or original.find(pool_bytes, pool_at + 1) >= 0:
        raise ValueError("Selected pool does not have a unique raw STATIC location")
    absolute = pool_at + source_offset
    if original[absolute:absolute + len(encoded_source)] != encoded_source:
        raise ValueError("Decoded string and raw STATIC bytes disagree")
    modified = bytearray(original)
    modified[absolute:absolute + len(encoded_source)] = replacement
    changed = {index for index, (left, right) in enumerate(zip(original, modified))
               if left != right}
    if not changed.issubset(range(absolute, absolute + len(encoded_source))):
        raise ValueError("Bytes outside the string allocation changed")
    old_size, new_size, publication = _publish(source, output, ovl, static, original, bytes(modified))

    check, check_static = _load(publication.path, game)
    if _topology(check) != _topology(ovl):
        raise ValueError(f"Topology changed: {_topology(ovl)} -> {_topology(check)}")
    checked_pool = check_static.content.pools[pool_index].data.getvalue()
    if checked_pool[source_offset:source_offset + len(encoded_source)] != replacement:
        raise ValueError("String replacement did not survive staged-family reload")
    publication.commit()
    return StaticPatchReport(
        output, len(changed), old_size, new_size, _topology(check),
        pool_index, source_offset,
    )


def repoint_existing_string(source: Path, output: Path, source_text: str,
                            target_text: str, *, expected_count: int,
                            game: str = DEFAULT_GAME) -> StaticPatchReport:
    """Repoint existing fragment references to another existing same-pool string."""
    source, output = _require_staged_output(source, output)
    if source_text == target_text:
        raise ValueError("Source and target strings are identical")
    ovl, static = _load(source, game)
    source_pool, source_offset, _ = _find_unique_pool_string(static, source_text)
    target_pool, target_offset, _ = _find_unique_pool_string(static, target_text)
    if source_pool != target_pool:
        raise ValueError(
            f"Cross-pool repoint is not game-verified ({source_pool} -> {target_pool}); refusing"
        )
    fragments = static.content.fragments
    mask = ((fragments["struct_pool"] == source_pool)
            & (fragments["struct_offset"] == source_offset))
    count = int(mask.sum())
    if count != expected_count:
        raise ValueError(f"Found {count} source fragments, expected {expected_count}")
    target_before = int(((fragments["struct_pool"] == target_pool)
                         & (fragments["struct_offset"] == target_offset)).sum())
    original = _decompressed_static(source, ovl, static)
    fragment_blob = fragments.tobytes()
    fragments_at = original.find(fragment_blob)
    if fragments_at < 0 or original.find(fragment_blob, fragments_at + 1) >= 0:
        raise ValueError("Fragment array does not have a unique raw STATIC location")
    modified = bytearray(original)
    record_size = fragments.dtype.itemsize
    allowed = set()
    changed_records = 0
    for index, record in enumerate(fragments):
        if (int(record["struct_pool"]) == source_pool
                and int(record["struct_offset"]) == source_offset):
            field_at = fragments_at + index * record_size + 12
            if struct.unpack_from("<I", modified, field_at)[0] != source_offset:
                raise ValueError(f"Fragment {index} raw offset disagrees with decoded value")
            struct.pack_into("<I", modified, field_at, target_offset)
            allowed.update(range(field_at, field_at + 4))
            changed_records += 1
    if changed_records != expected_count:
        raise ValueError(f"Patched {changed_records} fragments, expected {expected_count}")
    changed = {index for index, (left, right) in enumerate(zip(original, modified))
               if left != right}
    if not changed.issubset(allowed):
        raise ValueError("Bytes outside the selected fragment offsets changed")
    old_size, new_size, publication = _publish(source, output, ovl, static, original, bytes(modified))

    check, check_static = _load(publication.path, game)
    if _topology(check) != _topology(ovl):
        raise ValueError(f"Topology changed: {_topology(ovl)} -> {_topology(check)}")
    check_fragments = check_static.content.fragments
    source_after = int(((check_fragments["struct_pool"] == source_pool)
                        & (check_fragments["struct_offset"] == source_offset)).sum())
    target_after = int(((check_fragments["struct_pool"] == target_pool)
                        & (check_fragments["struct_offset"] == target_offset)).sum())
    if source_after or target_after != target_before + expected_count:
        raise ValueError(
            f"Fragment reload mismatch: source {count}->{source_after}, "
            f"target {target_before}->{target_after}"
        )
    publication.commit()
    return StaticPatchReport(
        output, len(changed), old_size, new_size, _topology(check),
        source_pool, source_offset, target_pool, target_offset, expected_count,
    )


@dataclass(frozen=True)
class BatchPatchReport:
    output: Path
    renamed: int
    changed_bytes: int
    compressed_before: int
    compressed_after: int
    topology: tuple[int, int, int, int]
    skipped: list[tuple[str, str]]          # (name, why)


def patch_string_slots(source: Path, output: Path, pairs, *,
                       game: str = DEFAULT_GAME) -> BatchPatchReport:
    """Replace MANY strings in one pass, each within its existing allocation.

    patch_string_slot() reloads, republishes and re-verifies the whole family per
    string, so renaming a species' 54 audio events meant 54 full rewrites of a
    14 MB OVL - about 25 minutes. The work itself is a handful of byte writes, so
    this collects every replacement against one decompressed STATIC buffer and
    publishes once.

    A pair whose source string is missing, ambiguous, or too long for its slot is
    SKIPPED and reported rather than aborting the batch - one unusable name
    should not cost the other fifty-three.
    """
    source, output = _require_staged_output(source, output)
    ovl, static = _load(source, game)
    original = _decompressed_static(source, ovl, static)

    # Resolve every replacement against the UNMODIFIED buffer before writing any
    # of them, so one edit cannot move another's offset out from under it.
    plan: list[tuple[int, bytes, bytes]] = []
    skipped: list[tuple[str, str]] = []
    claimed: list[range] = []
    for source_text, target_text in pairs:
        if source_text == target_text:
            skipped.append((source_text, "source and target identical"))
            continue
        try:
            pool_index, pool_offset, encoded_source = _find_unique_pool_string(static, source_text)
        except ValueError as exc:
            skipped.append((source_text, str(exc)))
            continue
        try:
            target_bytes = target_text.encode("ascii")
        except UnicodeEncodeError:
            skipped.append((source_text, "replacement is not ASCII"))
            continue
        if len(target_bytes) + 1 > len(encoded_source):
            skipped.append((source_text,
                            f"needs {len(target_bytes) + 1} bytes, slot has {len(encoded_source)}"))
            continue

        pool_bytes = static.content.pools[pool_index].data.getvalue()
        pool_at = original.find(pool_bytes)
        if pool_at < 0 or original.find(pool_bytes, pool_at + 1) >= 0:
            skipped.append((source_text, "pool has no unique raw STATIC location"))
            continue
        absolute = pool_at + pool_offset
        if original[absolute:absolute + len(encoded_source)] != encoded_source:
            skipped.append((source_text, "decoded string and raw STATIC bytes disagree"))
            continue
        span = range(absolute, absolute + len(encoded_source))
        if any(span.start < c.stop and c.start < span.stop for c in claimed):
            skipped.append((source_text, "allocation overlaps another rename in this batch"))
            continue
        claimed.append(span)
        replacement = target_bytes + b"\0" * (len(encoded_source) - len(target_bytes))
        plan.append((absolute, encoded_source, replacement))

    if not plan:
        raise ValueError("No usable renames in this batch")

    modified = bytearray(original)
    for absolute, encoded_source, replacement in plan:
        modified[absolute:absolute + len(encoded_source)] = replacement

    allowed = set()
    for absolute, encoded_source, _replacement in plan:
        allowed.update(range(absolute, absolute + len(encoded_source)))
    changed = {index for index, (left, right) in enumerate(zip(original, modified))
               if left != right}
    if not changed.issubset(allowed):
        raise ValueError("Bytes outside the string allocations changed")

    old_size, new_size, publication = _publish(source, output, ovl, static, original, bytes(modified))

    # A successful write is not verification: reload the staged family and read
    # every replacement back out of the pools.
    check, check_static = _load(publication.path, game)
    if _topology(check) != _topology(ovl):
        raise ValueError(f"Topology changed: {_topology(ovl)} -> {_topology(check)}")
    check_bytes = _decompressed_static(publication.path, check, check_static)
    for absolute, encoded_source, replacement in plan:
        if check_bytes[absolute:absolute + len(replacement)] != replacement:
            raise ValueError("A replacement did not survive the staged-family reload")

    publication.commit()

    return BatchPatchReport(output, len(plan), len(changed), old_size, new_size,
                            _topology(check), skipped)


@dataclass(frozen=True)
class RelocateReport:
    output: Path
    relocated: int
    fragments_repointed: int
    pool_growth: int
    skipped: list


@dataclass(frozen=True)
class RenameReport:
    output: Path
    in_place: int
    relocated: int
    fragments_repointed: int


@dataclass(frozen=True)
class ReusedStringSlot:
    local_pool: int
    offset: int
    old_size: int
    new_size: int


def _reuse_unreferenced_clip_slot(ovl, static, payload: bytes):
    """Reuse one orphaned clip-name slot without growing or shifting a pool.

    Content renaming can leave old donor clip strings behind after their
    fragments move elsewhere. Only accept complete clip-shaped strings inside
    an entirely ASCII/NUL type-2 pool. Any fragment target within the slot
    (including a suffix or its terminator) protects it. A root, dependency or
    fragment SOURCE in the pool disqualifies the entire pool: it may hold a
    structure or resource rather than just standalone strings.

    This is deliberately narrower than treating unreferenced bytes or zero
    runs as free memory. No general pool limit is relaxed.
    """
    ovs = static.content
    choices = []
    for index, pool in enumerate(ovs.pools):
        if int(pool.type) != 2:
            continue
        blob = pool.data.getvalue()
        if not blob or any(byte != 0 and not 32 <= byte <= 126 for byte in blob):
            continue
        global_index = next(i for i, p in enumerate(ovl.pools) if p is pool)
        # Protect both numbering interpretations across archive tables. This
        # may reject spare space, but cannot make a live allocation reusable.
        indices = {index, global_index}
        targets, structured = set(), False
        for archive in ovl.archives:
            for fragment in archive.content.fragments:
                if int(fragment['link_pool']) in indices:
                    structured = True
                if int(fragment['struct_pool']) in indices:
                    targets.add(int(fragment['struct_offset']))
            for root in archive.content.root_entries:
                if int(root['struct_ptr']['pool_index']) in indices:
                    structured = True
        for dependency in ovl.dependencies:
            if int(dependency['link_ptr']['pool_index']) in indices:
                structured = True
        if structured or not targets:
            continue
        start = 0
        while start < len(blob):
            end = blob.find(b'\0', start)
            if end < 0:
                break  # Never reclaim a string cut by a pool boundary.
            text = blob[start:end]
            if (end + 1 - start >= len(payload)
                    and re.fullmatch(rb'[A-Za-z][A-Za-z0-9_]*[$@][A-Za-z0-9_]+', text)
                    and not any(start <= offset <= end for offset in targets)):
                choices.append((end + 1 - start, index, start))
            start = end + 1
    if not choices:
        return None
    size, index, offset = min(choices)
    pool = ovs.pools[index]
    pool.data.seek(offset)
    pool.data.write(payload + b'\0' * (size - len(payload)))
    pool.data.seek(0)
    return ReusedStringSlot(index, offset, int(pool.size), int(pool.size))


def relocate_strings(source: Path, output: Path, pairs, *,
                     game: str = DEFAULT_GAME) -> RelocateReport:
    """Rename strings that do NOT fit their existing slot, by ALLOCATING new ones.

    `patch_string_slot` writes into the bytes the old string already occupies, so
    a longer replacement has nowhere to go - which is why an audio prefix could
    never be longer than the donor's (`Indoraptor` -> `Indocapi` fine,
    `IndominusRex` -> `UltimasaurusCE` refused).

    This lifts that limit the same way datastream growth does: append the new
    string to the tail of a type-2 pool and repoint every fragment that referenced
    the old one. The old string is left in place, abandoned - nothing is deleted
    and no offsets shift.

    Use it for the pairs `patch_string_slots` skipped as too long; short-enough
    renames are cheaper as in-place patches.

    Allocation first tries the LAST type-2 pool, bounded by STRING_TAIL_CAPACITY
    (or the archive's own largest type-2 pool if that is bigger). If that is
    exhausted, an unreferenced old clip-name slot in a string-only pool may be
    reused without growth. If neither route fits, the edit fails; creating a new
    pool remains unsupported.
    """
    source, output = _require_staged_output(source, output)
    ovl, static = _load_quiet(source, game)
    ovs = static.content
    fragments = ovs.fragments

    old_uncompressed = int(static.uncompressed_size)
    old_compressed = int(static.compressed_size)
    old_pools_end = int(static.pools_end)
    old_pool_sizes = tuple(int(pool.size) for pool in ovs.pools)
    old_fragment_count = int(static.num_fragments)
    static_index = ovl.archives.index(static)
    old_reservation = int(ovl.archives_meta[static_index].unk_0)

    # Keep unmapped bytes intact. A reader/writer mismatch is a separate format
    # problem, not permission to publish collateral changes during a rename.
    original_static = _decompressed_static(source, ovl, static)
    ovs.write_pools()
    if ovs.write_archive() != original_static:
        raise ValueError("STATIC does not round-trip unchanged; refusing string relocation")

    allocations, skipped, repointed = [], [], 0
    for source_text, target_text in pairs:
        if source_text == target_text:
            skipped.append((source_text, "source and target are identical"))
            continue
        needle = source_text.encode("ascii") + b"\0"
        found = None
        for index, pool in enumerate(ovs.pools):
            if int(pool.type) != 2:
                continue
            blob = pool.data.getvalue()
            at = blob.find(needle)
            while at != -1:
                if at == 0 or blob[at - 1] == 0:
                    found = (index, at)
                    break
                at = blob.find(needle, at + 1)
            if found:
                break
        if found is None:
            skipped.append((source_text, "not found in any string pool"))
            continue
        old_pool, old_offset = found
        mask = ((fragments["struct_pool"] == old_pool)
                & (fragments["struct_offset"] == old_offset))
        count = int(mask.sum())
        if not count:
            skipped.append((source_text, "no fragment references it"))
            continue
        try:
            allocation = append_tail_pool_bytes(
                ovl.pools, ovs.pools, 2, target_text.encode("ascii") + b"\0", alignment=16,
                page_size=max(observed_type_capacity(ovs.pools, 2), STRING_TAIL_CAPACITY))
        except ValueError as error:
            allocation = _reuse_unreferenced_clip_slot(
                ovl, static, target_text.encode("ascii") + b"\0")
            if allocation is None:
                skipped.append((source_text, str(error) + "; no reusable orphaned clip-name slot fits"))
                continue
        if allocation.new_size > allocation.old_size:
            repoint_pool_end_fragments(fragments, allocation.local_pool,
                                       allocation.old_size, allocation.new_size)
        allocations.append(allocation)
        # recompute the mask: repoint_pool_end_fragments may have moved records
        mask = ((fragments["struct_pool"] == old_pool)
                & (fragments["struct_offset"] == old_offset))
        fragments["struct_pool"][mask] = allocation.local_pool
        fragments["struct_offset"][mask] = allocation.offset
        repointed += int(mask.sum())

    if not allocations:
        raise ValueError("Nothing could be relocated: " + "; ".join(
            f"{name}: {why}" for name, why in skipped) or "no pairs given")

    ovs.write_pools()
    uncompressed = ovs.write_archive()
    pool_growth = sum(a.new_size - a.old_size for a in allocations)
    expected = old_uncompressed + pool_growth
    if len(uncompressed) != expected:
        raise ValueError(f"Unexpected STATIC growth: {len(uncompressed)} vs {expected}")
    if int(static.num_fragments) != old_fragment_count:
        raise ValueError("Fragment count changed; a string relocation must not add any")
    expected_sizes = list(old_pool_sizes)
    for a in allocations:
        expected_sizes[a.local_pool] = a.new_size
    if tuple(int(pool.size) for pool in ovs.pools) != tuple(expected_sizes):
        raise ValueError("An unrelated pool changed size while relocating strings")

    _, new_compressed, compressed = ovs.compress(uncompressed, True)
    source_bytes = source.read_bytes()
    header_size = len(source_bytes) - old_compressed
    meta_offset = header_size - len(ovl.archives_meta) * 8 + static_index * 8
    result = bytearray(source_bytes[:header_size])
    result.extend(compressed)
    head = int(static.io_start)
    struct.pack_into("<I", result, head + NUM_FRAGMENTS_OFFSET, static.num_fragments)
    struct.pack_into("<I", result, head + COMPRESSED_SIZE_OFFSET, new_compressed)
    struct.pack_into("<Q", result, head + UNCOMPRESSED_SIZE_OFFSET, expected)
    struct.pack_into("<I", result, head + POOLS_END_OFFSET, old_pools_end + pool_growth)
    struct.pack_into("<I", result, meta_offset, old_reservation + pool_growth)
    publication = CandidatePublication(source, output)
    publication.path.write_bytes(result)

    # a written file is not a verified one: every new string must be readable back
    check_ovl, check_static = _load_quiet(publication.path, game)
    pools = check_static.content.pools
    for _source_text, target_text in pairs:
        if any(name == _source_text for name, _ in skipped):
            continue
        needle = target_text.encode("ascii") + b"\0"
        if not any(int(p.type) == 2 and needle in p.data.getvalue() for p in pools):
            raise ValueError(f"Reloaded archive has no string {target_text!r}")
    publication.commit()
    logging.info("Relocated %d strings, repointed %d fragments, +%d bytes",
                 len(allocations), repointed, pool_growth)
    return RelocateReport(output, len(allocations), repointed, pool_growth, skipped)


def apply_string_renames(source: Path, output: Path, pairs, *,
                         game: str = DEFAULT_GAME) -> RenameReport:
    """Apply an all-or-nothing mixture of in-place and relocated string renames."""
    source, output = validate_staged_pair(source, output)
    publication = CandidatePublication(source, output)
    pairs = list(pairs)
    if not pairs:
        raise ValueError("Choose at least one string to rename")
    in_place, relocated = [], []
    seen = set()
    for old, new in pairs:
        if old in seen:
            raise ValueError(f"Duplicate source string in rename request: {old!r}")
        seen.add(old)
        try:
            old_bytes, new_bytes = old.encode("ascii"), new.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("Motiongraph strings must be ASCII") from exc
        if old == new:
            raise ValueError(f"Source and target are identical: {old!r}")
        (in_place if len(new_bytes) <= len(old_bytes) else relocated).append((old, new))

    def string_references(path):
        _ovl, archive = _load(path, game)
        pools, fragments = archive.content.pools, archive.content.fragments
        result = {}
        for row in fragments:
            pool_index, offset = int(row["struct_pool"]), int(row["struct_offset"])
            if pool_index >= len(pools) or int(pools[pool_index].type) != 2:
                continue
            blob = pools[pool_index].data.getvalue()
            if offset >= len(blob) or (offset and blob[offset - 1] != 0):
                continue
            end = blob.find(b"\0", offset)
            if end < 0:
                continue
            try:
                value = blob[offset:end].decode("ascii")
            except UnicodeDecodeError:
                continue
            result[(int(row["link_pool"]), int(row["link_offset"]))] = value
        return result

    references_before = string_references(source)
    replacements = dict(pairs)
    with tempfile.TemporaryDirectory(prefix="motiongraph-rename-", dir=output.parent) as raw:
        root = Path(raw)
        source_files = copy_motiongraph_family(source, root / "source")
        current = root / "source" / source.name
        in_place_report = None
        if in_place:
            copy_motiongraph_family(current, root / "short")
            destination = root / "short" / source.name
            in_place_report = patch_string_slots(current, destination, in_place, game=game)
            if in_place_report.skipped:
                raise ValueError("Unsupported audio renames: " + "; ".join(
                    f"{name}: {why}" for name, why in in_place_report.skipped))
            current = destination

        relocate_report = None
        if relocated:
            copy_motiongraph_family(current, root / "relocate-source")
            relocation_source = root / "relocate-source" / source.name
            copy_motiongraph_family(relocation_source, root / "relocate-output")
            destination = root / "relocate-output" / source.name
            relocate_report = relocate_strings(
                relocation_source, destination, relocated, game=game)
            if relocate_report.skipped:
                raise ValueError("Unsupported audio relocations: " + "; ".join(
                    f"{name}: {why}" for name, why in relocate_report.skipped))
            current = destination

        # Final semantic gate: each requested target has at least one fragment.
        _check, static = _load(current, game)
        fragments = static.content.fragments
        for _old, target in pairs:
            encoded = target.encode("ascii") + b"\0"
            refs = 0
            for pool_index, pool in enumerate(static.content.pools):
                blob = pool.data.getvalue()
                at = blob.find(encoded)
                while at >= 0:
                    if at == 0 or blob[at - 1] == 0:
                        refs += int(((fragments["struct_pool"] == pool_index)
                                    & (fragments["struct_offset"] == at)).sum())
                    at = blob.find(encoded, at + 1)
            if refs == 0:
                raise ValueError(f"Renamed target has no fragment references after reload: {target!r}")
        references_after = string_references(current)
        for site, old_value in references_before.items():
            wanted = replacements.get(old_value, old_value)
            if references_after.get(site) != wanted:
                raise ValueError(
                    f"String reference {site[0]}:{site[1]} changed unexpectedly: "
                    f"{old_value!r} -> {references_after.get(site)!r}; wanted {wanted!r}"
                )
        publication.path.write_bytes(current.read_bytes())
        publication.commit()
        return RenameReport(
            output, len(in_place), len(relocated),
            relocate_report.fragments_repointed if relocate_report else 0)
