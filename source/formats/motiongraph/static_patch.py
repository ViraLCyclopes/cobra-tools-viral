"""Direct STATIC patches for game-proven, fixed-topology motiongraph edits."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

from generated.formats.ovl import OvlFile


DEFAULT_GAME = "Jurassic World Evolution 3"
COMPRESSED_SIZE_OFFSET = 44

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
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Refusing to overwrite the source OVL; use a complete staged family")
    if source.name.lower() != output.name.lower():
        raise ValueError(
            "Source and output OVL basenames must match so the staged OVS/AUX family can reload"
        )
    return source, output


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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result)
    return old_size, new_size


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
    old_size, new_size = _publish(source, output, ovl, static, original, bytes(modified))

    check, check_static = _load(output, game)
    if _topology(check) != _topology(ovl):
        raise ValueError(f"Topology changed: {_topology(ovl)} -> {_topology(check)}")
    checked_pool = check_static.content.pools[pool_index].data.getvalue()
    if checked_pool[source_offset:source_offset + len(encoded_source)] != replacement:
        raise ValueError("String replacement did not survive staged-family reload")
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
    old_size, new_size = _publish(source, output, ovl, static, original, bytes(modified))

    check, check_static = _load(output, game)
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
    return StaticPatchReport(
        output, len(changed), old_size, new_size, _topology(check),
        source_pool, source_offset, target_pool, target_offset, expected_count,
    )
