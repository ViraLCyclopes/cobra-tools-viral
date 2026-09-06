"""Add a datastream to an animation activity.

An activity fires sounds and effects through its ``additional_data_streams`` -
a ``DataStreamResourceDataList``, which is a ``Uint64`` count plus an
``ArrayPointer``. That is the same shape as a random chooser's ``Animations``,
so this is the relocate-and-repoint growth that ``chooser_growth`` already does,
applied to a richer entry.

Why growth rather than repurposing an existing entry: ``ds_name`` strings are
POOLED, so renaming one rewrites every activity that shares it. On the Indominus
roar, four of the five datastreams are shared 3-102 ways; only ``Interruptible``
is unique, and spending it costs a gameplay flag and still yields one effect.

Entry layout, measured from a shipped graph rather than inferred from the schema
(the schema gives field order, not stride, and not which fields carry fragments):

    stride 56 bytes, FIVE fragments per entry
      +0   curve_type    Uint64      65537 on every VFX entry observed
      +8   ds_name       -> ZString
      +16  type          -> ZString  e.g. "VFXEnable"
      +24  bone_i_d      -> ZString  empty, but the pointer still exists
      +32  location      -> ZString  empty or e.g. "Head_Default"
      +40  curve count   Uint64
      +48  curve points  -> array

and on the owning AnimationActivityData:

      +80  count         Uint64
      +88  ArrayPointer  -> the entry array

Only ``ds_name`` needs a new allocation. ``type`` points at a string the archive
already has, ``bone_i_d``/``location`` reuse the empty string every entry
already points at, and the curve can share an existing on/off curve - a curve is
read-only key data, so sharing one costs nothing.
"""
from __future__ import annotations

import logging
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .clone import (COMPRESSED_SIZE_OFFSET, NUM_FRAGMENTS_OFFSET, POOLS_END_OFFSET,
                    UNCOMPRESSED_SIZE_OFFSET, clone_fragment_sources,
                    internal_fragment_mask)
from .edit import DEFAULT_GAME, load_motiongraph
from .report import build_deref
from .surgical_growth import (_load_quiet, append_tail_pool_bytes,
                              repoint_pool_end_fragments)

ENTRY_SIZE = 56
CURVE_TYPE_OFFSET = 0
CURVE_COUNT_OFFSET = 40
POINTER_SUBOFFSETS = (8, 16, 24, 32, 48)      # ds_name, type, bone_id, location, curve
COUNT_FIELD_OFFSET = 80
ARRAY_POINTER_OFFSET = 88
STRING_POOL_TYPE = 2                           # every shipped string allocation is type 2
CURVE_POINT_SIZE = 12                          # x float, y short, subtype short, 2 params
SUBCURVE_CONSTANT = 0                          # SubCurveType.Constant - what every VFX key uses

# The two VFX families behave differently and their curve_type is part of the
# contract, so pair them here rather than leaving it to the caller:
#   VFXEnable  LATCHES  - stays on until the activity ends, off keys ignored
#   VFXToggle  honours off keys mid-clip, whatever the activity does
# Anything that must stop needs VFXToggle. See jwe3-dinosaur-vfx notes.
CURVE_TYPE_FOR = {"VFXEnable": 65537, "VFXToggle": 0}


def curve_type_for(ds_type: str, default: int = 65537) -> int:
    """The curve_type that belongs with a datastream type."""
    return CURVE_TYPE_FOR.get(ds_type, default)


def is_held(animation_flags: int) -> bool:
    """True if the activity loops/holds rather than ending.

    Bit 0 of AnimationFlags. Measured: Sleep, Rest02, StandIdle_Aggressive,
    RestPreen and TerritoryDefend are all 17; Partial_Roar is 2 and Eat01 is 0.
    A VFXEnable on a held activity NEVER drops, which is the classic cause of an
    effect that will not switch off.
    """
    return bool(int(animation_flags) & 1)


def encode_curve_y(value: float) -> int:
    """Encode a curve y/param as the archive's biased bfloat16.

    Stored as ``top16(float32(value + 2.0))``. The +2 bias pins the exponent so
    the 7-bit mantissa behaves as fixed point - step 1/64 for value >= 0. Boolean
    signals (0.0 / 1.0) therefore round-trip exactly.
    """
    return struct.unpack("<HH", struct.pack("<f", float(value) + 2.0))[1]


def decode_curve_y(raw: int) -> float:
    """Inverse of :func:`encode_curve_y`."""
    return struct.unpack("<f", struct.pack("<HH", 0, int(raw) & 0xFFFF))[0] - 2.0


def _pack_curve_points(points: "Sequence[tuple[float, float]]") -> bytes:
    if len(points) < 2:
        raise ValueError("A curve needs at least two points")
    blob = bytearray()
    last_x = None
    for x, y in points:
        x = float(x)
        if not 0.0 <= x <= 4.0:
            raise ValueError(f"Curve x must be normalised time 0..4, got {x}")
        if last_x is not None and x < last_x:
            raise ValueError("Curve points must be sorted by ascending x")
        last_x = x
        # The two curve params use the SAME +2-biased encoding as y, so a plain 0
        # here is not 0.0 - it decodes to -2.0. Every shipped point stores 16384,
        # i.e. encode_curve_y(0.0). Writing a raw 0 produces a malformed curve
        # that the engine holds high forever, which looks exactly like "the effect
        # never turns off" and survives both recurving and swapping the particle.
        blob += struct.pack("<fhHhh", x, encode_curve_y(y), SUBCURVE_CONSTANT,
                            encode_curve_y(0.0), encode_curve_y(0.0))
    return bytes(blob)


@dataclass(frozen=True)
class DataStreamGrowthReport:
    output: Path
    activity: str
    ds_name: str
    ds_type: str
    old_count: int
    new_count: int
    fragments_added: int
    array_pool: int
    array_offset: int
    string_reused: bool


def find_activity(loader, deref, clip: str):
    """The AnimationActivityData whose clip name contains `clip`."""
    matches = []
    for (pool, offset), obj in list(loader.context.recursion.items()):
        if type(obj).__name__ != "AnimationActivityData":
            continue
        try:
            mani = str(deref(obj.mani))
        except Exception:
            continue
        if clip in mani:
            matches.append((int(pool.i), int(offset), mani, obj))
    if not matches:
        raise ValueError(f"No animation activity whose clip name contains {clip!r}")
    if len(matches) > 1:
        names = ", ".join(sorted(m[2] for m in matches))
        raise ValueError(f"{clip!r} is ambiguous, it matches: {names}")
    return matches[0]


def list_activities(source: Path, game: str = DEFAULT_GAME) -> list[str]:
    """Every animation activity name in the graph, sorted. Read-only."""
    _ovl, loader = load_motiongraph(Path(source), None, game)
    deref = build_deref(loader)
    names = set()
    for _key, obj in list(loader.context.recursion.items()):
        if type(obj).__name__ != "AnimationActivityData":
            continue
        try:
            names.add(str(deref(obj.mani)))
        except Exception:
            continue
    return sorted(names)


def describe_activity(source: Path, clip: str, game: str = DEFAULT_GAME) -> dict:
    """Read-only: what an activity currently fires. Use before growing."""
    _ovl, loader = load_motiongraph(Path(source), None, game)
    deref = build_deref(loader)
    pool, offset, mani, act = find_activity(loader, deref, clip)
    lst = act.additional_data_streams
    ptr = lst.data_stream_resource_data
    streams = []
    for entry in (deref(ptr) or []):
        streams.append({"ds_name": str(deref(entry.ds_name)),
                        "type": str(deref(entry.type)),
                        "curve_type": int(entry.curve_type)})
    try:
        flags = int(act.animation_flags)
    except Exception:
        flags = 0
    return {
        "activity": mani,
        "animation_flags": flags,
        "held": is_held(flags),
        "activity_pool": pool,
        "activity_offset": offset,
        "count": int(lst.data_stream_resource_data_count),
        "array_pool": int(ptr.target_pool.i),
        "array_offset": int(ptr.target_offset),
        "streams": streams,
    }


def _find_string(ovl, ovs, text: str):
    """(local_pool, offset) of an existing NUL-terminated string, or None.

    Must start on a string boundary: a suffix match would produce a pointer into
    the middle of another name.
    """
    needle = text.encode("ascii") + b"\0"
    for pool in ovs.pools:
        if int(pool.type) != STRING_POOL_TYPE:
            continue
        blob = pool.data.getvalue()
        at = blob.find(needle)
        while at != -1:
            if at == 0 or blob[at - 1] == 0:
                return ovs.pools.index(pool), at
            at = blob.find(needle, at + 1)
    return None


def _fragment_target(fragments, local_pool: int, offset: int):
    mask = ((fragments["link_pool"] == local_pool) & (fragments["link_offset"] == offset))
    rows = fragments[mask]
    if len(rows) != 1:
        return None
    return int(rows[0]["struct_pool"]), int(rows[0]["struct_offset"])


def grow_data_stream(
    source: Path,
    output: Path,
    clip: str,
    ds_name: str,
    ds_type: str = "VFXEnable",
    *,
    curve_from: str | None = None,
    curve_points: "Sequence[tuple[float, float]] | None" = None,
    curve_type: int | None = None,
    game: str = DEFAULT_GAME,
) -> DataStreamGrowthReport:
    """Append a datastream named `ds_name` of type `ds_type` to `clip`'s activity.

    `curve_from` names an existing datastream whose curve should be shared; it
    defaults to any entry already using `ds_type`, so a VFXEnable addition picks
    up a real on/off curve rather than an invented one.

    `curve_points` authors a curve instead of sharing one: a sequence of
    ``(x, y)`` pairs, x normalised clip time 0..1 and y the signal value. Pass it
    whenever the borrowed curve is the wrong shape.

    Borrowing is the safe default for a BURST effect, because every shipped
    VFXEnable curve measured has the form ``(0,0) (t,1) (1,1)`` - it switches on
    and never switches off. That is harmless for a one-shot dust puff, and wrong
    for a CONTINUOUS emitter such as a fire, which then burns for good. For those,
    author an explicit off key, e.g.::

        curve_points=[(0.0, 0.0), (0.35, 1.0), (0.75, 0.0), (1.0, 0.0)]
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Refusing to overwrite the source OVL; stage a full family")
    if source.name.lower() != output.name.lower():
        raise ValueError("Source and staged output basenames must match")
    if not output.is_file():
        raise ValueError("Copy the complete archive family to the stage directory first")
    if not ds_name or any(ord(c) < 32 or ord(c) > 126 for c in ds_name):
        raise ValueError(f"Datastream name must be printable ASCII: {ds_name!r}")

    _sem_ovl, loader = load_motiongraph(source, None, game)
    deref = build_deref(loader)
    act_global, act_offset, mani, act = find_activity(loader, deref, clip)
    lst = act.additional_data_streams
    count = int(lst.data_stream_resource_data_count)
    ptr = lst.data_stream_resource_data
    arr_global, arr_offset = int(ptr.target_pool.i), int(ptr.target_offset)
    existing = [str(deref(e.ds_name)) for e in (deref(ptr) or [])]
    if ds_name in existing:
        raise ValueError(f"{mani} already fires {ds_name!r}")

    ovl, static = _load_quiet(source, game)
    ovs = static.content
    fragments = ovs.fragments

    def local(global_index: int) -> int:
        pool = ovl.pools[global_index]
        if pool not in ovs.pools:
            raise ValueError(f"Global pool {global_index} is not in STATIC")
        return ovs.pools.index(pool)

    act_local, arr_local = local(act_global), local(arr_global)

    # verify the measured layout holds for THIS activity before writing anything
    act_blob = ovl.pools[act_global].data.getvalue()
    if struct.unpack_from("<Q", act_blob, act_offset + COUNT_FIELD_OFFSET)[0] != count:
        raise ValueError(
            f"Count field at activity+{COUNT_FIELD_OFFSET} does not read {count}; "
            f"the layout assumption does not hold for this graph")
    pointer_mask = ((fragments["link_pool"] == act_local)
                    & (fragments["link_offset"] == act_offset + ARRAY_POINTER_OFFSET))
    if int(pointer_mask.sum()) != 1:
        raise ValueError(
            f"Expected exactly one ArrayPointer fragment at "
            f"{act_local}:{act_offset + ARRAY_POINTER_OFFSET}")

    span = count * ENTRY_SIZE
    old_bytes = ovl.pools[arr_global].data.getvalue()[arr_offset:arr_offset + span]
    if len(old_bytes) != span:
        raise ValueError("Could not read the complete datastream entry array")

    array_rows = fragments[
        internal_fragment_mask(fragments, arr_local, arr_offset, span)].copy()
    expected = sorted(index * ENTRY_SIZE + sub
                      for index in range(count) for sub in POINTER_SUBOFFSETS)
    actual = sorted(int(x) - arr_offset for x in array_rows["link_offset"])
    if actual != expected:
        raise ValueError(
            f"Entry array does not carry {len(POINTER_SUBOFFSETS)} pointers per entry; "
            f"expected sub-offsets {POINTER_SUBOFFSETS} on every entry")

    # ---- what the new entry's five pointers will aim at ---------------------
    type_site = _find_string(ovl, ovs, ds_type)
    if type_site is None:
        raise ValueError(
            f"The string {ds_type!r} is not in this archive; a datastream type must "
            f"already exist (VFXEnable, AudioEvent, ...)")

    # bone_i_d and location: reuse the empty string the existing entries point at
    empty_site = _fragment_target(fragments, arr_local, arr_offset + 24)
    if empty_site is None:
        raise ValueError("Could not read the existing bone_i_d pointer to reuse")

    # curve: share one from an entry that already uses this type
    curve_site = None
    curve_count = 0
    curve_type_borrowed = 65537
    probe = curve_from or ds_type
    for pool_index, pool in enumerate(ovs.pools):
        blob = pool.data.getvalue()
        mask = (fragments["link_pool"] == pool_index)
        for row in fragments[mask]:
            off = int(row["link_offset"])
            tgt = (int(row["struct_pool"]), int(row["struct_offset"]))
            if tgt[0] >= len(ovs.pools):
                continue
            tblob = ovs.pools[tgt[0]].data.getvalue()
            end = tblob.find(b"\0", tgt[1])
            if end < 0:
                continue
            if tblob[tgt[1]:end].decode("ascii", "replace") != probe:
                continue
            base = off - 16                      # 'type' sits at entry+16
            if base < 0 or base + ENTRY_SIZE > len(blob):
                continue
            site = _fragment_target(fragments, pool_index, base + 48)
            if site is None:
                continue
            curve_site = site
            curve_count = struct.unpack_from("<Q", blob, base + CURVE_COUNT_OFFSET)[0]
            curve_type_borrowed = struct.unpack_from("<Q", blob, base + CURVE_TYPE_OFFSET)[0]
            break
        if curve_site:
            break
    if curve_site is None and curve_points is None:
        raise ValueError(
            f"No existing {probe!r} datastream to borrow a curve from; pass curve_from= "
            f"or author one with curve_points=")

    # An authored curve replaces the borrowed one entirely; the borrow loop still
    # ran because it is where curve_type comes from, and that field is shared.
    if curve_type is not None:
        # the caller knows the type/curve_type pairing; a VFXToggle entry that
        # keeps a borrowed VFXEnable curve_type of 65537 will still latch
        curve_type_value = int(curve_type)
    else:
        curve_type_value = curve_type_borrowed
    curve_blob = None
    if curve_points is not None:
        curve_blob = _pack_curve_points(curve_points)
        curve_count = len(curve_points)

    # ---- build the new entry ------------------------------------------------
    new_entry = bytearray(ENTRY_SIZE)
    struct.pack_into("<Q", new_entry, CURVE_TYPE_OFFSET, curve_type_value)
    struct.pack_into("<Q", new_entry, CURVE_COUNT_OFFSET, curve_count)
    new_array = bytes(old_bytes) + bytes(new_entry)

    old_fragment_count = int(static.num_fragments)
    old_uncompressed = int(static.uncompressed_size)
    old_compressed = int(static.compressed_size)
    old_pools_end = int(static.pools_end)
    old_pool_sizes = tuple(int(pool.size) for pool in ovs.pools)
    static_index = ovl.archives.index(static)
    old_reservation = int(ovl.archives_meta[static_index].unk_0)
    allocations = []

    # ---- allocate --------------------------------------------------------
    reused = _find_string(ovl, ovs, ds_name)
    if reused is not None:
        string_local, string_offset = reused
    else:
        allocation = append_tail_pool_bytes(
            ovl.pools, ovs.pools, STRING_POOL_TYPE,
            ds_name.encode("ascii") + b"\0", alignment=16)
        repoint_pool_end_fragments(fragments, allocation.local_pool,
                                   allocation.old_size, allocation.new_size)
        string_local, string_offset = allocation.local_pool, allocation.offset
        allocations.append(allocation)

    # an authored curve gets its own point array, in the same pool kind that
    # already holds the shipped curves
    if curve_blob is not None:
        curve_allocation = append_tail_pool_bytes(
            ovl.pools, ovs.pools, int(ovl.pools[arr_global].type),
            curve_blob, alignment=16)
        repoint_pool_end_fragments(fragments, curve_allocation.local_pool,
                                   curve_allocation.old_size, curve_allocation.new_size)
        allocations.append(curve_allocation)
        curve_site = (curve_allocation.local_pool, curve_allocation.offset)

    array_type = int(ovl.pools[arr_global].type)
    array_allocation = append_tail_pool_bytes(
        ovl.pools, ovs.pools, array_type, new_array, alignment=16)
    repoint_pool_end_fragments(fragments, array_allocation.local_pool,
                               array_allocation.old_size, array_allocation.new_size)
    allocations.append(array_allocation)

    # ---- repoint and add fragments -----------------------------------------
    fragments["struct_pool"][pointer_mask] = array_allocation.local_pool
    fragments["struct_offset"][pointer_mask] = array_allocation.offset

    cloned = clone_fragment_sources(array_rows, array_allocation.local_pool,
                                    arr_offset, array_allocation.offset)
    new_base = array_allocation.offset + count * ENTRY_SIZE
    added = np.array(
        [
            (array_allocation.local_pool, new_base + 8,  string_local,  string_offset),
            (array_allocation.local_pool, new_base + 16, type_site[0],  type_site[1]),
            (array_allocation.local_pool, new_base + 24, empty_site[0], empty_site[1]),
            (array_allocation.local_pool, new_base + 32, empty_site[0], empty_site[1]),
            (array_allocation.local_pool, new_base + 48, curve_site[0], curve_site[1]),
        ],
        dtype=fragments.dtype,
    )
    new_rows = np.concatenate((cloned, added))
    ovs.fragments = np.concatenate((fragments, new_rows))
    ovs.fragments.sort(order=("link_pool", "struct_pool", "link_offset", "struct_offset"))
    static.num_fragments = old_fragment_count + len(new_rows)

    # ---- bump the count ----------------------------------------------------
    data = ovl.pools[act_global].data
    data.seek(act_offset + COUNT_FIELD_OFFSET)
    data.write(struct.pack("<Q", count + 1))

    logging.info("Growing %s: %d -> %d datastreams, +%d fragments",
                 mani, count, count + 1, len(new_rows))

    # ovl.save() rebuilds mime tables and needs config this path never loads.
    # Write the pools, compress, and splice over the source's STATIC exactly as
    # chooser growth does - then every header field it touched must be patched.
    ovs.write_pools()
    uncompressed = ovs.write_archive()
    pool_growth = sum(a.new_size - a.old_size for a in allocations)
    expected_uncompressed = old_uncompressed + pool_growth + len(new_rows) * 16
    if len(uncompressed) != expected_uncompressed:
        raise ValueError(
            f"Unexpected STATIC growth: {len(uncompressed)} vs {expected_uncompressed}")
    expected_sizes = list(old_pool_sizes)
    for a in allocations:
        expected_sizes[a.local_pool] = a.new_size
    if tuple(int(pool.size) for pool in ovs.pools) != tuple(expected_sizes):
        raise ValueError("An unrelated pool changed size while growing the datastream list")

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

    # a written file is not a verified one: reload and read the list back
    _check_ovl, check_loader = load_motiongraph(output, None, game)
    check_deref = build_deref(check_loader)
    _p, _o, _m, grown = find_activity(check_loader, check_deref, clip)
    got = [str(check_deref(e.ds_name))
           for e in (check_deref(grown.additional_data_streams.data_stream_resource_data) or [])]
    wanted = existing + [ds_name]
    reloaded = int(grown.additional_data_streams.data_stream_resource_data_count)
    if reloaded != count + 1 or got != wanted:
        raise ValueError(
            "Reloaded activity is wrong: count %d (wanted %d)\n  got    %r\n  wanted %r"
            % (reloaded, count + 1, got, wanted))

    return DataStreamGrowthReport(
        output=output, activity=mani, ds_name=ds_name, ds_type=ds_type,
        old_count=count, new_count=count + 1, fragments_added=len(new_rows),
        array_pool=array_allocation.local_pool, array_offset=array_allocation.offset,
        string_reused=reused is not None,
    )


def read_data_stream_curve(source: Path, clip: str, ds_name: str,
                           game: str = DEFAULT_GAME) -> list[tuple[float, float]]:
    """Read one datastream's curve back as ``(x, y)`` pairs.

    This is the read half a curve editor needs: load, show, edit, write back
    with :func:`set_data_stream_curve`.
    """
    _sem_ovl, loader = load_motiongraph(Path(source).resolve(), None, game)
    deref = build_deref(loader)
    _p, _o, mani, act = find_activity(loader, deref, clip)
    for entry in (deref(act.additional_data_streams.data_stream_resource_data) or []):
        if str(deref(entry.ds_name)) != ds_name:
            continue
        return [(float(point.x), decode_curve_y(int(point.y)))
                for point in (deref(entry.curve.points) or [])]
    raise ValueError(f"{mani} has no datastream named {ds_name!r}")


def set_data_stream_curve(source: Path, output: Path, clip: str, ds_name: str,
                          points: "Sequence[tuple[float, float]]",
                          game: str = DEFAULT_GAME,
                          *, ds_type: str | None = None,
                          curve_type: int | None = None) -> dict:
    """Replace the curve on an EXISTING datastream, in place.

    Growth adds an entry; this only changes when an entry is on. It allocates a
    fresh point array and repoints the entry's curve fragment at it, so the
    borrowed curve is left untouched for whatever else still shares it - curves
    ARE shared, so editing the borrowed array in place would change every
    datastream pointing at it.

    Adds no fragments and needs no string, which is why it works on archives
    whose final string pool is already full.

    `ds_type` and `curve_type` retype the entry. This matters more than the curve
    shape, because the two shipped VFX families behave differently:

        VFXEnable  curve_type 65537   (0,0) (t,1) (1,1)      latches ON
        VFXToggle  curve_type 0       (0,0) (t,1) (u,0) ...  honours OFF keys

    Every shipped VFXEnable curve stays high to the end of the clip, and the
    continuous emitters that must actually stop - the preen dust puffs - are all
    VFXToggle with curve_type 0. So an off key on a VFXEnable entry does nothing;
    a continuous effect such as fire has to be retyped, not just recurved.

    `ds_type` must already exist as a string in the archive; retyping never
    allocates one.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Refusing to overwrite the source OVL; stage a full family")
    if source.name.lower() != output.name.lower():
        raise ValueError("Source and staged output basenames must match")
    if not output.is_file():
        raise ValueError("Copy the complete archive family to the stage directory first")

    curve_blob = _pack_curve_points(points)

    _sem_ovl, loader = load_motiongraph(source, None, game)
    deref = build_deref(loader)
    act_global, act_offset, mani, act = find_activity(loader, deref, clip)
    ptr = act.additional_data_streams.data_stream_resource_data
    arr_global, arr_offset = int(ptr.target_pool.i), int(ptr.target_offset)
    names = [str(deref(e.ds_name)) for e in (deref(ptr) or [])]
    if ds_name not in names:
        raise ValueError(f"{mani} has no datastream named {ds_name!r}; it fires {names}")
    index = names.index(ds_name)

    ovl, static = _load_quiet(source, game)
    ovs = static.content
    fragments = ovs.fragments
    arr_pool = ovl.pools[arr_global]
    if arr_pool not in ovs.pools:
        raise ValueError("Entry array pool is not in STATIC")
    arr_local = ovs.pools.index(arr_pool)

    entry_base = arr_offset + index * ENTRY_SIZE
    curve_mask = ((fragments["link_pool"] == arr_local)
                  & (fragments["link_offset"] == entry_base + 48))
    if int(curve_mask.sum()) != 1:
        raise ValueError(
            f"Expected exactly one curve fragment at {arr_local}:{entry_base + 48}; "
            f"the measured entry layout does not hold for this graph")

    old_fragment_count = int(static.num_fragments)
    old_uncompressed = int(static.uncompressed_size)
    old_compressed = int(static.compressed_size)
    old_pools_end = int(static.pools_end)
    old_pool_sizes = tuple(int(pool.size) for pool in ovs.pools)
    static_index = ovl.archives.index(static)
    old_reservation = int(ovl.archives_meta[static_index].unk_0)

    allocation = append_tail_pool_bytes(
        ovl.pools, ovs.pools, int(arr_pool.type), curve_blob, alignment=16)
    repoint_pool_end_fragments(fragments, allocation.local_pool,
                               allocation.old_size, allocation.new_size)

    fragments["struct_pool"][curve_mask] = allocation.local_pool
    fragments["struct_offset"][curve_mask] = allocation.offset

    data = ovl.pools[arr_global].data
    data.seek(entry_base + CURVE_COUNT_OFFSET)
    data.write(struct.pack("<Q", len(points)))

    if ds_type is not None:
        site = _find_string(ovl, ovs, ds_type)
        if site is None:
            raise ValueError(
                f"{ds_type!r} is not already a string in this archive; retyping "
                f"never allocates one")
        type_mask = ((fragments["link_pool"] == arr_local)
                     & (fragments["link_offset"] == entry_base + 16))
        if int(type_mask.sum()) != 1:
            raise ValueError(f"Expected one type fragment at {arr_local}:{entry_base + 16}")
        fragments["struct_pool"][type_mask] = site[0]
        fragments["struct_offset"][type_mask] = site[1]

    if curve_type is not None:
        data.seek(entry_base + CURVE_TYPE_OFFSET)
        data.write(struct.pack("<Q", int(curve_type)))

    logging.info("Recurving %s / %s: %d points", mani, ds_name, len(points))

    ovs.write_pools()
    uncompressed = ovs.write_archive()
    pool_growth = allocation.new_size - allocation.old_size
    expected_uncompressed = old_uncompressed + pool_growth
    if len(uncompressed) != expected_uncompressed:
        raise ValueError(
            f"Unexpected STATIC growth: {len(uncompressed)} vs {expected_uncompressed}")
    expected_sizes = list(old_pool_sizes)
    expected_sizes[allocation.local_pool] = allocation.new_size
    if tuple(int(pool.size) for pool in ovs.pools) != tuple(expected_sizes):
        raise ValueError("An unrelated pool changed size while setting the curve")
    if int(static.num_fragments) != old_fragment_count:
        raise ValueError("Fragment count changed; a curve edit must not add fragments")

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

    # a written file is not a verified one: read the curve back off disk
    got = read_data_stream_curve(output, clip, ds_name, game)
    wanted = [(float(x), float(y)) for x, y in points]
    if len(got) != len(wanted) or any(
            abs(a[0] - b[0]) > 1e-4 or abs(a[1] - b[1]) > 1.0 / 64
            for a, b in zip(got, wanted)):
        raise ValueError(f"Reloaded curve is wrong:\n  got    {got}\n  wanted {wanted}")

    reloaded_type = next(
        (s["type"] for s in describe_activity(output, clip, game)["streams"]
         if s["ds_name"] == ds_name), None)
    if ds_type is not None and reloaded_type != ds_type:
        raise ValueError(f"Reloaded type is {reloaded_type!r}, wanted {ds_type!r}")

    return {"output": output, "activity": mani, "ds_name": ds_name,
            "points": got, "type": reloaded_type,
            "curve_pool": allocation.local_pool,
            "curve_offset": allocation.offset}


def activities_with_stream(source: Path, ds_name: str,
                           game: str = DEFAULT_GAME) -> list[dict]:
    """Every activity that already fires `ds_name`, with its type and held flag.

    Held activities are the ones where a ``VFXEnable`` can never drop, so this is
    what tells you which entries must be ``VFXToggle``.
    """
    _ovl, loader = load_motiongraph(Path(source), None, game)
    deref = build_deref(loader)
    found = []
    for _key, obj in list(loader.context.recursion.items()):
        if type(obj).__name__ != "AnimationActivityData":
            continue
        try:
            entries = list(deref(obj.additional_data_streams.data_stream_resource_data) or [])
        except Exception:
            continue
        for entry in entries:
            try:
                if str(deref(entry.ds_name)) != ds_name:
                    continue
                mani = str(deref(obj.mani))
                flags = int(obj.animation_flags)
            except Exception:
                continue
            found.append({"activity": mani, "type": str(deref(entry.type)),
                          "curve_type": int(entry.curve_type),
                          "animation_flags": flags, "held": is_held(flags)})
    found.sort(key=lambda d: d["activity"])
    return found


def set_curve_everywhere(source: Path, output: Path, ds_name: str,
                         points: "Sequence[tuple[float, float]]",
                         game: str = DEFAULT_GAME, *,
                         ds_type: str | None = None,
                         curve_type: int | None = None,
                         only: "Sequence[str] | None" = None) -> list[dict]:
    """Apply one curve/type to EVERY activity that fires `ds_name`.

    Each :func:`set_data_stream_curve` call rewrites the whole archive, so they
    have to be chained source -> output; doing that by hand is six staging
    directories and easy to get wrong. `only` restricts it to named activities.
    """
    import shutil
    import tempfile

    source, output = Path(source).resolve(), Path(output).resolve()
    targets = [d["activity"] for d in activities_with_stream(source, ds_name, game)]
    if only is not None:
        wanted = set(only)
        targets = [t for t in targets if t in wanted or t.split("$")[-1] in wanted]
    if not targets:
        raise ValueError(f"No activity fires {ds_name!r}")

    applied = []
    with tempfile.TemporaryDirectory(prefix="ds_chain_") as tmp:
        tmp = Path(tmp)
        current = source
        for index, clip in enumerate(targets):
            last = index == len(targets) - 1
            if last:
                step_dir = output.parent
            else:
                step_dir = tmp / ("step%d" % index)
                step_dir.mkdir(parents=True, exist_ok=True)
                for f in source.parent.iterdir():
                    if f.is_file():
                        shutil.copy2(f, step_dir / f.name)
            result = set_data_stream_curve(current, step_dir / source.name, clip,
                                           ds_name, points, game,
                                           ds_type=ds_type, curve_type=curve_type)
            applied.append({"activity": result["activity"], "type": result["type"]})
            current = step_dir / source.name
    return applied
