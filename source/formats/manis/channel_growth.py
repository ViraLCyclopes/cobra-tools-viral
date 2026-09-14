"""Add a channel to a compressed clip's channel list, at byte level.

GAME-VERIFIED 2026-09-07 on UltimasaurusCE: `standpreen03` was given an ori
channel for `def_l_armLwrHalfTwist_joint` (track 120), a bone its channel list
never named, and the twist is visible in game. This retires limit #1 - "a clip
can only animate the bones its channel list names".

Limit #1 in the handoff - "a clip can only animate the bones its channel list
names" - framed as needing a new name+hash in Buffer1, moving hash_block_size and
everything after it. Measured against real data that is only true when the bone's
name is NOT already in the bundle's shared name buffer. For a bone another clip in
the same bundle already animates, the name is there, and the edit collapses to:

    OriBonesNames         += 4 bytes   (uint index into the shared name buffer)
    ori_channel_to_bone   += 1 byte    (the track this channel reads)
    ori_bone_to_channel[bone] = new channel index, IN PLACE (255 meant "none")
    ManiInfo.ori_bone_count += 1
    target_bone_count       unchanged  - the track already exists in the ACL blob
    bone_to_channel size    unchanged  - the bone is inside BoneMin..BoneMax
    name buffer/hashes      unchanged

Layout, verified exact on all 22 clips of this bundle (delta 0):

    channel_region = (pos + ori + scl + float) * 4        names, uint each
                   + (pos + ori + scl)                    channel_to_bone, ubyte
                   + sum(max - min + 1) per group         bone_to_channel, ubyte
                   -> padded to 4

Two invariants the splice must not break:

* every ManiBlock starts at file offset == 4 (mod 16) and every block SIZE is a
  multiple of 16 - all 22, no exceptions.
* the align-4 pad after the channel region and the align-16 pad after the ACL
  blob are DERIVED, not stored. Filler cannot be inserted anywhere: the engine
  recomputes those lengths, so any size change must come from a real count
  change, and the trailing pad has to absorb the difference.

`rebuild` with no edit must return the input byte-for-byte; that is the gate that
separates a broken splice from a wrong edit.
"""
from __future__ import annotations

import struct
import sys

from generated.formats.manis import ManisFile
from .append import block_extents, preamble_layout
from .database import locate_bulk

MANI_INFO_SIZE = 304
OFF_POS_COUNT = 24      # ushort, verified on all 22 clips
OFF_ORI_COUNT = 26
OFF_FLOAT_COUNT = 36
OFF_TARGET_COUNT = 42
NAME_WIDTH = 4          # ChannelName is Uint at bundle version > 257
PREFIX_STRUCT = 128     # fixed structure between the channel region and the ACL blob
ACL_TAG = bytes.fromhex("11ac11ac")   # ACL blob magic
GROUPS = ("pos", "ori", "scl")


def channel_layout(mani_info):
    """Byte offsets of every array inside one clip's channel region."""
    counts = {g: int(getattr(mani_info, f"{g}_bone_count")) for g in GROUPS}
    floats = int(mani_info.float_count)
    spans = {}
    for g in GROUPS:
        lo = int(getattr(mani_info, f"{g}_bone_min"))
        hi = int(getattr(mani_info, f"{g}_bone_max"))
        spans[g] = (hi - lo + 1) if lo <= hi else 0

    out, at = {}, 0
    for g in GROUPS + ("float",):
        n = floats if g == "float" else counts[g]
        out[f"{g}_names"] = (at, n * NAME_WIDTH)
        at += n * NAME_WIDTH
    for g in GROUPS:
        out[f"{g}_c2b"] = (at, counts[g])
        at += counts[g]
    for g in GROUPS:
        out[f"{g}_b2c"] = (at, spans[g])
        at += spans[g]
    out["_used"] = at
    out["_padded"] = at + (-at) % 4
    out["_counts"] = counts
    out["_spans"] = spans
    return out


def rebuild(data: bytes, manis, clip: str, add=None) -> bytes:
    """Return `data` with one clip's channel region rebuilt.

    `add` is `(group, bone_name, track)` or None. None is the no-op gate and must
    reproduce `data` exactly.
    """
    names = [str(i.name) for i in manis.mani_infos]
    if clip not in names:
        raise ValueError(f"{clip!r} is not in this bundle")
    index = names.index(clip)
    mani_info = manis.mani_infos[index]
    layout = preamble_layout(manis)
    keys_end = locate_bulk(data)["low_offset"]
    start, end = block_extents(manis, keys_end)[index]
    block = data[start:end]
    if (end - start) % 16:
        raise ValueError(f"{clip}: block is {end - start} bytes, not a multiple of 16")

    plan = channel_layout(mani_info)
    region = block[:plan["_padded"]]
    rest = block[plan["_padded"]:]
    if len(region) != plan["_padded"]:
        raise ValueError(f"{clip}: block is shorter than its own channel region")

    # filter before unpacking: a for-target unpack runs before the if clause, and
    # the metadata entries are plain ints
    pieces = {k: region[v[0]:v[0] + v[1]] for k, v in plan.items() if not k.startswith("_")}

    new_counts = dict(plan["_counts"])
    if add is not None:
        group, bone_name, track = add
        if group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}")
        buffer_names = [str(x) for x in manis.name_buffer.target_names]
        if bone_name not in buffer_names:
            raise ValueError(
                f"{bone_name!r} is not in the bundle's name buffer; adding a NAME is the "
                f"expensive case this routine does not cover")
        if not 0 <= track < int(mani_info.target_bone_count):
            raise ValueError(f"track {track} outside 0..{int(mani_info.target_bone_count) - 1}")
        if track in set(pieces[f"{group}_c2b"]):
            raise ValueError(f"{group} already has a channel reading track {track}")
        lo = int(getattr(mani_info, f"{group}_bone_min"))
        hi = int(getattr(mani_info, f"{group}_bone_max"))
        if not lo <= track <= hi:
            raise ValueError(
                f"track {track} is outside {group} bone range {lo}..{hi}; the "
                f"bone_to_channel array would have to resize, which this does not do")
        new_channel = new_counts[group]
        if new_channel > 254:
            raise ValueError("channel index would not fit the ubyte bone_to_channel")
        pieces[f"{group}_names"] += struct.pack("<I", buffer_names.index(bone_name))
        pieces[f"{group}_c2b"] += bytes([track])
        b2c = bytearray(pieces[f"{group}_b2c"])
        slot = track - lo
        if b2c[slot] != 255:
            raise ValueError(f"bone {track} already maps to channel {b2c[slot]}")
        b2c[slot] = new_channel
        pieces[f"{group}_b2c"] = bytes(b2c)
        new_counts[group] += 1

    ordered = [f"{g}_names" for g in GROUPS] + ["float_names"]
    ordered += [f"{g}_c2b" for g in GROUPS] + [f"{g}_b2c" for g in GROUPS]
    new_region = b"".join(pieces[k] for k in ordered)
    new_region += bytes((-len(new_region)) % 4)

    # What follows the channel region is NOT free space. Measured on all 22 clips
    # of this bundle, the first ACL blob sits at exactly
    #     align16(padded_region + PREFIX_STRUCT)
    # i.e. a fixed 128-byte structure, then zero padding to the next 16. Each
    # CompressedManiDataPC2 then ends on a 16-boundary RELATIVE TO THE BLOCK
    # START, so moving a blob off that boundary breaks the next one - the first
    # attempt shifted them by 8 and the parse died in `compressed_floats`.
    #
    # So rebuild the prefix rather than shifting it: structure verbatim, then
    # re-pad. When the growth fits the existing alignment slack the blob does not
    # move at all and the block keeps its exact size - no other block shifts.
    old_blob = plan["_padded"] + PREFIX_STRUCT
    old_blob += (-old_blob) % 16
    if block.find(ACL_TAG, plan["_padded"]) - 8 != old_blob:
        raise ValueError(f"{clip}: ACL blob is not at align16(region + {PREFIX_STRUCT})")
    # The structure is NOT a fixed 128 bytes - measured 125 or 129 across this
    # bundle - so keep the prefix verbatim and treat only its TRAILING zeros as
    # adjustable. Anything else would silently truncate real data.
    prefix = block[plan["_padded"]:old_blob]
    structure = prefix.rstrip(b"\0")

    new_blob = len(new_region) + PREFIX_STRUCT
    new_blob += (-new_blob) % 16
    available = new_blob - len(new_region)
    if available < len(structure):
        raise ValueError(
            f"{clip}: the pre-blob structure is {len(structure)} bytes but only "
            f"{available} fit after growing the channel region; the ACL blob would "
            f"have to move")
    new_block = new_region + structure + bytes(available - len(structure))
    new_block += block[old_blob:]
    moved = new_blob - old_blob
    if len(new_block) != len(block) + moved:
        raise ValueError(f"{clip}: rebuilt block is {len(new_block)}, expected {len(block) + moved}")
    if len(new_block) % 16:
        raise ValueError(f"{clip}: rebuilt block is {len(new_block)} bytes, not a multiple of 16")

    out = bytearray(data)
    out[start:end] = new_block
    if add is not None:
        info_at = layout["info_start"] + index * MANI_INFO_SIZE
        for g, off in (("pos", OFF_POS_COUNT), ("ori", OFF_ORI_COUNT)):
            struct.pack_into("<H", out, info_at + off, new_counts[g])
    return bytes(out)


if __name__ == "__main__":
    path = sys.argv[1]
    data = open(path, "rb").read()
    manis = ManisFile()
    manis.load(path)
    clip = sys.argv[2]
    same = rebuild(data, manis, clip, None)
    print("NO-OP GATE: %s (%d -> %d bytes)"
          % ("byte-identical" if same == data else "*** DIFFERS ***", len(data), len(same)))
