"""Write a JWE3 ACL bundle. Knows the format; knows nothing about the source.

`acl.py` is the bridge to the ACL encoder/decoder and `manis_database_cmd.py`
REBUILDS a bundle that already has one - it gets its samples by decoding the ACL
blobs in the file (line ~641), so it cannot be pointed at a dtype-0 bundle, a
Planet Zoo one, or a Blender export. This module is the other half: given
samples from anywhere, it writes a database-backed bundle from scratch.

That matters for more than one caller:

* Osteoblast's PZ -> JWE3 port, which is where this was first written and which
  now only supplies the samples;
* `plugin/export_manis.py`, which can only write dtype 0 today. `bonemask.sync_file`
  says so in as many words - "a Blender-exported bundle is UNCOMPRESSED (dtype 0)
  ... the mask step belongs after the ACL re-encode" - and there was no ACL
  re-encode for it to belong after;
* any future Cobra game whose clips have to be re-emitted as JWE3 ACL.

WHAT THIS DOES *NOT* HAND-ASSEMBLE
-----------------------------------
`splice.py` says re-serialising a JWE3 manis through `ManisFile.save()` drops the
compressed_database, the inter-block auxiliary data and the trailing bulk, and
every other ACL tool here splices bytes because of it. Two thirds of that is out
of date: `CompressedHeaderReader` keeps the database blob verbatim and
`KeysReader` keeps `inter_block_data`.

Re-measured over 46 shipped bundles, `save()` reproduces the whole keys region
BYTE-IDENTICALLY unless the last clip carries a limb structure, where the output
is a strict prefix ending at `eoh` - those bytes sit past where the reader stops
and are never read. See `tests/test_manis_compressed_write.py`.

We write `has_list = 1`, which has no limb structure at all (vanilla-attested by
17 shipped Acrocanthosaurus clips), so that limitation cannot bite - and it also
keeps us away from the structure behind the `JWE3.exe+0x1697FBD` spawn crash.
Only the bulk tiers are appended by hand.

The resulting dtype is **48** (unk 0, use_ushort 0, compression 1, has_list 1),
which `manis.xml` lists among the values PC2/JWE3 ship and which
`CompressedHeaderReader` calls out as "the dtype 48/49 bundle, which has
has_list == 1 yet still carries a database".

STRIPPED SUB-TRACKS ARE THE CONTRACT, NOT AN OPTIMISATION
----------------------------------------------------------
ACL omits a sub-track whose samples all equal its default and the runtime
supplies the default back. Frontier's default is the BIND POSE, so a bone a clip
does not animate is stripped and the animal holds its bind pose there.

`samples_from_keys` therefore writes **NaN** for a sub-track with no channel, not
identity. `bindpose.clip_defaults` reads all-NaN as "vanilla stripped this".
Writing identity instead would store every bone's rotation as identity and pose
the whole skeleton flat.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from source.formats.manis import bonemask

QVVF = 12                     # ACL track_type for a qvv transform stream
SCALAR = 0                    # ACL track_type for a float1 stream
BULK_ALIGNMENT = 16

JWE3_VERSION = (262, 282)     # (version, mani_version)

# unk 0, use_ushort 0, compression 1, has_list 1 -> the uint 48.
DTYPE_ACL_BITS = {"unk": 0, "use_ushort": 0, "compression": 1, "has_list": 1}

# qvv sample component groups, as both cobra and ACL order them
SUB_TRACKS = {"rotation": (0, 4), "translation": (4, 7), "scale": (7, 10)}


def _build_database():
    """`manis_database_cmd.build_database`, which lives in a top-level script.

    Imported lazily and by path if need be: that script imports from this
    package, so importing it at module scope would be circular.
    """
    try:
        from manis_database_cmd import build_database
        return build_database
    except ImportError:
        import importlib.util
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        path = os.path.join(root, "manis_database_cmd.py")
        spec = importlib.util.spec_from_file_location("manis_database_cmd", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_database


def pad(data, alignment=BULK_ALIGNMENT):
    return data + b"\x00" * (-len(data) % alignment)


def channel_table_extent(mani_info):
    """Bytes from a ManiBlock's start to the end of its channel tables, padded to 4.

    Every ManiBlock opens with this prologue whatever its dtype - the schema gates
    only the KEY data on `compression`, not the names or the channel maps. So it
    sizes the uncompressed key arrays and it is also where a COMPRESSED block's
    ACL container begins, at the next align8.

    The subtle part is `bone_to_channel`: one byte per bone in `min..max`, not one
    per channel. Two clips animating 8 bones each differ by 152 bytes when one
    spans 105 bones and the other 29.
    """
    pos = int(mani_info.pos_bone_count)
    ori = int(mani_info.ori_bone_count)
    scl = int(mani_info.scl_bone_count)
    flo = int(mani_info.float_count)

    size = 4 * (pos + ori + scl + flo)          # channel names, uint index each
    size += pos + ori + scl                     # channel_to_bone, one byte each
    for low, high in ((mani_info.pos_bone_min, mani_info.pos_bone_max),
                      (mani_info.ori_bone_min, mani_info.ori_bone_max),
                      (mani_info.scl_bone_min, mani_info.scl_bone_max)):
        if int(low) <= int(high):
            size += int(high) - int(low) + 1
    return size + (-size % 4)


def block_geometry(mani_info):
    """(prefix_start, blob_start) for one block, relative to the block's start.

    The runtime walks a compressed clip as `align8 -> 0x80 container -> align16 ->
    transform blob`, which is why the container-to-blob gap is always 0x80 or 0x88
    and never anything else - measured over 1042 shipped clips.
    """
    prefix_start = channel_table_extent(mani_info)
    container_start = prefix_start + (-prefix_start % 8)
    blob_start = container_start + bonemask.CONTAINER_SIZE
    blob_start += -blob_start % 16
    return prefix_start, blob_start


def stored_subtracks(blob, layout=None):
    """Which bones the ACL blob actually stores a rotation/translation/scale for.

    This is what the container's masks have to agree with. A stored sub-track
    whose mask bit is clear is inert in game, and a mask bit with nothing stored
    behind it is a lie about the clip - so the masks are derived from the encoded
    blob rather than from what we asked for.
    """
    from source.formats.manis.acl_patch import parse_transform_layout, sub_track_types

    if layout is None:
        layout = parse_transform_layout(blob)
    out = {}
    for kind in SUB_TRACKS:
        if kind == "scale" and not layout.has_scale:
            out[kind] = []
            continue
        out[kind] = [i for i, t in enumerate(sub_track_types(blob, layout, kind)) if t]
    return out, layout


def container_masks(blob, mani_info, layout=None):
    """The three bone sets a clip's container must carry. Returns (masks, layout).

    **Scale is keyed differently from the other two, and this is not a guess.**
    Measured over 1137 clips - vanilla Acrocanthosaurus, Baryonyx, Dimetrodon,
    Spinosaurus, Deinosuchus and the SarcoViral mod - with zero exceptions:

        rotation     mask == what the ACL blob STORES   1137/1137
        translation  mask == what the ACL blob STORES   1137/1137
        scale        mask == what the CHANNEL TABLE declares  1137/1137

    Nothing distinguished the two rules until a rig with scale turned up: on a
    rig without it all three sets are empty together. Deinosuchus separates them
    - 227 of its clips declare scale channels that ACL then strips, so `has_scale`
    is false while `scl_channel_to_bone` is not empty, and vanilla still masks the
    declared bones. Rotation is the control: it disagrees with the declared set on
    1133 clips and follows the blob every time.

    **What that does NOT establish**: whether the runtime honours the scale mask.
    Deinosuchus is the only rig here that carries the data, and it does not scale
    bones in game because its prefab does not enable `BoneScaling` - so the engine
    never reads the field. This is the authoring convention, verified; the runtime
    behaviour behind it is untested.
    """
    from source.formats.manis.acl_patch import parse_transform_layout

    if layout is None:
        layout = parse_transform_layout(blob)
    stored, layout = stored_subtracks(blob, layout)
    masks = dict(stored)
    masks["scale"] = sorted({int(t) for t in mani_info.keys.scl_channel_to_bone})
    return masks, layout


def build_prefix(blob, mani_info, prefix_start, blob_start, flag=1):
    """The `acl_prefix` bytes for one block: pad8, the 0x80 container, pad16.

    cobra models everything between the channel tables and the ACL blob as one
    opaque run (`AclPaddingReader`), so the padding on both sides of the container
    is the writer's to emit.

    Returns (prefix bytes, masks, layout).
    """
    posed, layout = container_masks(blob, mani_info)
    container = bonemask.build_container(
        layout.num_tracks, rotation=posed["rotation"],
        translation=posed["translation"], scale=posed["scale"], flag=flag)

    lead = -prefix_start % 8
    body = prefix_start + lead + len(container)
    if body > blob_start:
        raise AssertionError(
            f"container runs to {body} but the blob starts at {blob_start}")
    return b"\x00" * lead + container + b"\x00" * (blob_start - body), posed, layout


def sample_rate(mani_info):
    """Frames per second for the ACL stream.

    A clip of `n` frames spans `n - 1` intervals, so the rate is
    `(n - 1) / duration`. Vanilla stores 30.0003 rather than a clean 30 - the
    float32 nearest whatever Frontier's exporter computed - so this does not
    reproduce a magic constant, it reproduces the RELATIONSHIP, which is what
    keeps the ManiInfo's `duration` and the blob's rate from disagreeing.
    """
    frames = int(mani_info.frame_count)
    duration = float(getattr(mani_info, "duration", 0.0) or 0.0)
    if frames < 2 or duration <= 0.0:
        return 30.0
    return (frames - 1) / duration


def samples_from_keys(manis, mani_info, unroll=True):
    """One clip's uncompressed keys as ACL track-space samples.

    Shape is `(frames, target_bone_count, 10)`: rotation xyzw, translation xyz,
    scale xyz - the component order `jwe3_acl_decode.exe` reports for a QVVF
    stream.

    **A sub-track with no channel is NaN, not identity.** `clip_defaults` reads
    all-NaN as "strip this", so the runtime substitutes the bone's bind pose,
    exactly as it does for a vanilla clip. Filling identity instead - which a
    dtype-0 writer must do, because it has to write a key for every track - would
    store the whole rig as identity rotation and flatten the skeleton.

    `unroll` makes each rotation track hemisphere-continuous first. `q` and `-q`
    are the same rotation, so a sign flip between frames is invisible to every
    offline check, but it is what the encoder's error metric sees.

    The source must be uncompressed keys, or keys in a codec `ManisFile.decompress`
    handles - dtype 0, or Planet Zoo / JWE2's segmented quantiser. A clip that is
    ALREADY JWE3 ACL has no key arrays to read; decode it with
    `generated.formats.manis.acl.decode_file`, which returns samples in this same
    shape.
    """
    if mani_info.dtype.compression:
        if manis.context.version == 262 and manis.context.mani_version == 282:
            raise ValueError(
                f"{mani_info.name} is already a JWE3 ACL clip, which has no key "
                f"arrays to read - decode it with acl.decode_file() instead, which "
                f"returns samples in this same (frames, tracks, 10) shape")
        manis.decompress(mani_info)
    k = mani_info.keys
    frames = int(mani_info.frame_count)
    tracks = int(mani_info.target_bone_count)

    samples = np.full((frames, tracks, 10), np.nan, dtype=np.float32)

    ori = np.array(k.ori_bones, dtype=np.float32, copy=True)
    if unroll:
        unroll_quaternions(ori)
    pos = np.asarray(k.pos_bones, dtype=np.float32)
    scl = np.asarray(k.scl_bones, dtype=np.float32) if int(mani_info.scl_bone_count) else None

    for channel, track in enumerate(int(x) for x in k.ori_channel_to_bone):
        samples[:, track, 0:4] = ori[:, channel]
    for channel, track in enumerate(int(x) for x in k.pos_channel_to_bone):
        samples[:, track, 4:7] = pos[:, channel]
    if scl is not None:
        for channel, track in enumerate(int(x) for x in k.scl_channel_to_bone):
            samples[:, track, 7:10] = scl[:, channel, :3]

    scalars = None
    if int(mani_info.float_count):
        flo = np.asarray(k.floats, dtype=np.float32)
        scalars = flo.reshape(flo.shape[0], flo.shape[1], 1)
    return samples, scalars


def unroll_quaternions(ori):
    """Make a rotation track hemisphere-continuous in time. Modifies in place.

    `ori` is (frames, channels, 4).
    """
    if ori.shape[0] < 2:
        return ori
    for frame in range(1, ori.shape[0]):
        flip = np.einsum("cj,cj->c", ori[frame], ori[frame - 1]) < 0.0
        ori[frame][flip] *= -1.0
    return ori


def count_hemisphere_flips(ori):
    """Sign flips between consecutive frames; (frames, channels, 4)."""
    if ori.shape[0] < 2 or ori.size == 0:
        return 0
    return int((np.einsum("fcj,fcj->fc", ori[1:], ori[:-1]) < 0.0).sum())


def set_acl_dtype(manis, preserve_unk=True):
    """Retag every clip as compressed JWE3 ACL in a fresh PC2 dtype object.

    The two dtype bitfields order their bits differently, so an object read
    through `ManisDtype` cannot be written through `ManisDtypePC2` - it has to be
    rebuilt, not edited.

    **Only `compression` and `has_list` are ours to set.** `unk` is three bits
    nothing here understands, and forcing it to 0 rewrote 15 of 17 clips on an
    Acrocanthosaurus bundle whose source had `unk = 1` - a silent, unexplained
    deviation from both the source and vanilla, found only by diffing against
    `manis_database_cmd.py`, which does not touch the dtype at all. `manis.xml`
    lists 48 AND 49 among the values PC2/JWE3 ship, so both are legal and the
    source's choice is the one to keep.
    """
    from generated.formats.manis.imports import name_type_map

    manis.context.version, manis.context.mani_version = JWE3_VERSION
    manis.version, manis.mani_version = JWE3_VERSION
    for mani_info in manis.mani_infos:
        was = mani_info.dtype
        dtype = name_type_map["ManisDtypePC2"](manis.context, 0, None)
        for field, value in DTYPE_ACL_BITS.items():
            setattr(dtype, field, value)
        if preserve_unk and was is not None:
            for field in ("unk", "use_ushort"):
                carried = getattr(was, field, None)
                if carried is not None:
                    setattr(dtype, field, int(carried))
        mani_info.dtype = dtype


def _set_blob(manis, keys, field, blob):
    """Put a raw ACL blob into one of a ManiBlock's CompressedManiDataPC2 slots."""
    from generated.formats.manis.imports import name_type_map

    holder = name_type_map["CompressedManiDataPC2"](manis.context, keys, None)
    holder.size = len(blob)
    holder.acl_data = np.frombuffer(blob[4:], dtype=np.uint8).copy()
    setattr(keys, field, holder)


def write_bundle(manis, clips, ms2, dst, precision=None, medium=None, low=None,
                 flag=1, database=True, rates=None, flags=None, wraps=None,
                 stream_name="Anim_L0", reporter=None):
    """Write `manis` out as a JWE3 ACL bundle.

    `manis` is a `ManisFile` whose `mani_infos` are already shaped for the output
    - names, counts, channel tables - as a loaded bundle is. `clips` is a list of
    `(samples, scalars)` parallel to `manis.mani_infos`, from `samples_from_keys`
    or anywhere else. `ms2` supplies the bind pose ACL uses as its per-track
    defaults, so it must be the rig the clips were authored on.

    `database` picks how the bulk data travels, and the two are not
    interchangeable - which loader reads the file decides:

    * **True** - a shared ACL database whose bulk is split into the `_L0`/`_L1`
      tiers the caller appends to the OVL as `.ovs.anim_l0` / `_l1`. This is what
      every vanilla DINOSAUR ships, and `manis_database_cmd.py` records that the
      game rejects the alternative there: re-encoding a dinosaur's clips
      self-contained crashes on spawn whether the old database is kept or not.
    * **False** - self-contained clips carrying their bulk inline. No database,
      no bulk tiers, no extra streams. Bigger than the split form but with no
      dependency on anything outside the one file.

    **Which one you need is not a size choice, it is a hard requirement, and the
    two loaders want OPPOSITE things. Both directions are game-verified:**

        dinosaur       database-backed REQUIRED  (self-contained crashes on spawn)
        scenery deco   self-contained REQUIRED   (database-backed aborts 0xFDEAD)

    The scenery result is VLAardvark's PZ Aardvark `restloop01`, 2026-09-26: the
    database form aborted, the self-contained form plays. 0xFDEAD is a DELIBERATE
    abort - the game validated and refused - so it does not mean the file is
    malformed: every offline gate passed on the bundle that aborted and it
    round-tripped through the packer byte-identically. The likely mechanism is
    that a deco animated by a plain `Animation` component never asks the
    streaming system for the `Anim_L0`/`_L1` archives, so ACL initialises a
    database whose bulk was never mounted.

    Self-contained is also the better clip where it is allowed: on that Aardvark
    93,680 bytes against 603,667 for dtype 0, and worst rotation error 0.044 deg
    against the database form's 0.230, because nothing is demoted to a lossy tier.

    Vanilla JWE3 ships no animated scenery at all, so there was no precedent to
    copy here and the absence of one is not evidence either way - it took a game
    test to settle it.

    Returns a per-clip report for `verify_bundle`.
    """
    from generated.formats.manis.acl import AclSamples, encode_tracks
    from source.formats.manis.bindpose import (
        bind_bytes, extend_bind_pose, read_ms2_bind)
    from source.formats.manis.splice import read_blob_header

    say = reporter or (lambda message: None)
    set_acl_dtype(manis)
    # Carry the source's own values wherever there are any. Everything this
    # writer INVENTS is a place it can silently disagree with the game.
    rates = list(rates) if rates is not None else [None] * len(manis.mani_infos)
    flags = list(flags) if flags is not None else [None] * len(manis.mani_infos)
    wraps = list(wraps) if wraps is not None else [None] * len(manis.mani_infos)

    # One bind serves the bundle, so it has to cover the widest clip.
    parents, bind_values = read_ms2_bind(ms2)
    widest = max(int(mi.target_bone_count) for mi in manis.mani_infos)
    if widest > len(parents):
        parents, bind_values = extend_bind_pose(parents, bind_values, widest)
        say(f"  padded bind to {widest} tracks")
    elif widest < len(parents):
        # Not an error. Tracks are index-aligned with bones and build_database
        # slices the per-clip bind to the clip's own width. It is ALSO what a
        # mismatched .ms2 looks like, so say it out loud.
        say(f"  note: {os.path.basename(ms2)} has {len(parents)} bones but the "
            f"widest clip declares {widest} tracks - fine if this is the right "
            f"rig, wrong .ms2 if it is not")
    bind = bind_bytes(parents, bind_values)
    say(f"  bind pose: {len(parents)} bones from {os.path.basename(ms2)}")

    # build_database only ever reads .track_type / .values / .sample_rate per
    # stream plus headers[i]['wrap_optimized'], so nothing here has to come from
    # a decoded ACL blob.
    streams, headers, index_of = [], [], []
    for position, (mani_info, (samples, scalars)) in enumerate(
            zip(manis.mani_infos, clips)):
        rate = rates[position] if rates[position] else sample_rate(mani_info)
        index_of.append(len(streams))
        streams.append(AclSamples(QVVF, samples.shape[1], samples.shape[0],
                                  samples.shape[2], rate, samples))
        # wrap_optimized drops the repeated final sample and shifts the reported
        # duration by one frame. CARRY it when the source has one: hardcoding
        # False here changed 15 of 17 blobs on an Acrocanthosaurus rebuild, and
        # that was the fourth place this writer invented a value the source
        # already held (the others: dtype `unk`, the container flag, the sample
        # rate). Only a from-scratch clip with no source has to choose, and False
        # is the safe choice there because vanilla ships plenty of them.
        # An entry may be a bool for the transform stream, or a
        # (transform, scalar) pair. They are INDEPENDENT: vanilla ships clips
        # whose transform blob is wrap-optimized and whose scalar blob is not,
        # and making the scalar inherit the transform's flag left 50 bytes
        # differing from the game-verified rebuild - one hash and one
        # misc_packed bit 30 per scalar blob.
        entry = wraps[position]
        if isinstance(entry, (tuple, list)):
            wrap, scalar_wrap = bool(entry[0]), bool(entry[1])
        else:
            wrap = bool(entry) if entry is not None else False
            scalar_wrap = False
        headers.append({"wrap_optimized": wrap})
        if scalars is not None and int(mani_info.float_count):
            streams.append(AclSamples(SCALAR, scalars.shape[1], scalars.shape[0],
                                      1, rate, scalars))
            headers.append({"wrap_optimized": scalar_wrap})

    if database:
        with tempfile.TemporaryDirectory(prefix="cobra_acl_write_") as work:
            bound, database_blob, low_bulk, medium_bulk = _build_database()(
                streams, headers, bind, work, parents, bind_values,
                precision=precision, medium_proportion=medium, low_proportion=low)
        say(f"  database {len(database_blob)} bytes, bulk low={len(low_bulk)} "
            f"medium={len(medium_bulk)}")
    else:
        # Self-contained: each clip carries its own bulk. The per-clip defaults
        # still come from `clip_defaults`, not the raw bind, because they are what
        # decides the STRIPPED SET - hand the encoder the plain bind instead and
        # every sub-track that happens to equal it is silently dropped.
        from source.formats.manis.bindpose import clip_defaults

        bound, database_blob = {}, b""
        low_bulk = medium_bulk = b""
        for index, stream in enumerate(streams):
            if stream.track_type != QVVF:
                continue
            tracks = stream.values.shape[1]
            defaults = clip_defaults(stream.values, bind_values)
            bound[index] = encode_tracks(
                stream.values, QVVF, stream.sample_rate,
                bind=bind_bytes(parents[:tracks], defaults),
                wrap=headers[index]["wrap_optimized"])
        say(f"  self-contained: {len(bound)} clip(s), no database, no bulk tiers")

    report = []
    geometries = []
    for index, (mani_info, (samples, scalars)) in enumerate(zip(manis.mani_infos, clips)):
        transform_index = index_of[index]
        blob = bound[transform_index]
        bound_ok = bool(read_blob_header(blob).get("has_database"))
        if database and not bound_ok:
            raise AssertionError(f"{mani_info.name}: build_database did not bind this clip")
        if not database and bound_ok:
            raise AssertionError(f"{mani_info.name}: asked for a self-contained clip "
                                 f"but the encoder produced a database-backed one")

        prefix_start, blob_start = block_geometry(mani_info)
        geometries.append((prefix_start, blob_start))
        clip_flag = flags[index] if flags[index] is not None else flag
        prefix, posed, layout = build_prefix(blob, mani_info, prefix_start,
                                             blob_start, flag=clip_flag)

        keys = mani_info.keys
        keys.acl_prefix.data = prefix
        _set_blob(manis, keys, "compressed", blob)
        if scalars is not None and int(mani_info.float_count):
            scalar_blob = encode_tracks(streams[transform_index + 1].values, SCALAR,
                                        streams[transform_index].sample_rate,
                                        wrap=headers[transform_index + 1]["wrap_optimized"])
            _set_blob(manis, keys, "compressed_floats", scalar_blob)
        report.append(dict(name=str(mani_info.name), samples=samples, scalars=scalars,
                           num_tracks=layout.num_tracks, posed=posed,
                           blob_size=len(blob), frames=int(mani_info.frame_count)))

    # The preamble's stream-name field names the OVS archive the database bulk
    # ships in. **A database-backed bundle MUST carry one.**
    #
    # Measured: all 57 shipped bundles have a non-empty name - Anim_L0 (32),
    # Anim_Hatchery, Anim_JumpAttack, Anim_DynamicFight, Anim_Hunting - and 0 of
    # 977 vanilla ACL clips are self-contained. A bundle built from a Planet Zoo
    # source inherits PZ's empty name, and the database-backed VLAardvark that
    # aborted the game with 0xFDEAD had exactly that: a database to feed and no
    # archive named to feed it from. The self-contained build with the same empty
    # name plays, which fits - with the bulk inline there is nothing to locate.
    #
    # MANI.py already defaults the archive it CREATES to `Anim`, so this changes
    # nothing about packing; it changes what the bundle itself declares.
    if database_blob and not str(manis.stream or ""):
        manis.stream = stream_name
        say(f"  stream name set to {stream_name!r} - a database-backed bundle has "
            f"to name the archive its bulk ships in")
    manis.compressed_header.data = pad(database_blob) if database_blob else None
    manis.save(dst)

    # save() rebuilds bones_lut and re-derives every clip's bone min/max
    # (update_key_indices), and those size the channel tables the prefix above was
    # measured against. The values come back out of the same channel_to_bone
    # arrays they went in on, so this has never been observed to move - but the
    # prefix is already written by then, so if it ever did, every ACL blob would
    # shift off its 16-byte boundary. KeysReader swallows its own failures, so the
    # symptom would be a clean parse of a broken file. Check, do not assume.
    for mani_info, was in zip(manis.mani_infos, geometries):
        now = block_geometry(mani_info)
        if now != was:
            raise AssertionError(
                f"{mani_info.name}: save() changed the channel-table layout from "
                f"{was} to {now}, so the container and blob are at the wrong offsets")

    if low_bulk or medium_bulk:
        with open(dst, "ab") as stream:
            stream.write(pad(low_bulk))
            stream.write(pad(medium_bulk))
    say(f"  wrote {os.path.basename(dst)} ({os.path.getsize(dst)} bytes)")
    return report


# Accuracy bars, and why they differ by mode.
#
# A self-contained clip keeps every key inline, so the only loss is ACL's own
# quantisation - measured 0.053 deg worst across the Aardvark's 38 clips.
#
# A database-backed clip DELIBERATELY demotes keys into the low and medium bulk
# tiers; that is the whole point of the format. Measured on the same 38 clips at
# ACL's default proportions: median 0.347 deg, 11 clips over 0.5, worst 1.236.
# That is not us degrading the clip - rebuilding a vanilla bundle at those same
# defaults reproduces it BYTE FOR BYTE, so the defaults are what Frontier used,
# and this is vanilla's own quality level for this content.
#
# Holding the database path to the self-contained bar rejected a correct build.
# These numbers are a corruption detector, not a quality target: pass
# `max_rotation_deg` explicitly to tighten them, and lower `medium` / `low` in
# `write_bundle` if you want the accuracy rather than the streaming.
SELF_CONTAINED_ACCURACY = (0.5, 0.01)      # (degrees, units)
DATABASE_ACCURACY = (3.0, 0.05)


def verify_bundle(dst, report, reporter=None, max_rotation_deg=None,
                  max_translation=None, database=True):
    """Re-read a written bundle and prove it is what was meant. Returns problems.

    A clean parse is not among the gates. cobra parses a desynchronised bundle
    happily, and `KeysReader` logs and SWALLOWS its own failures, so
    `ManisFile.load()` comes back clean on a file whose block chain is broken.
    """
    from generated.formats.manis import ManisFile
    from generated.formats.manis.acl import decode_file
    from source.formats.manis.acl_patch import parse_transform_layout
    from source.formats.manis.database import (
        check_name_buffer, locate_bulk, read_bulk_info)
    from source.formats.manis.selfcontained import count_parsed_maniblocks
    from source.formats.manis.splice import list_clip_blobs, read_blob_header

    say = reporter or (lambda message: None)
    default_deg, default_pos = (DATABASE_ACCURACY if database
                                else SELF_CONTAINED_ACCURACY)
    if max_rotation_deg is None:
        max_rotation_deg = default_deg
    if max_translation is None:
        max_translation = default_pos
    problems = []
    with open(dst, "rb") as stream:
        data = stream.read()

    parsed, total = count_parsed_maniblocks(dst)
    if parsed != total:
        problems.append(f"only {parsed} of {total} ManiBlocks parse")

    out = ManisFile()
    out.game = "Jurassic World Evolution 3"
    out.load(dst)
    if (out.context.version, out.context.mani_version) != JWE3_VERSION:
        problems.append(f"wrote manis {out.context.version}/{out.context.mani_version}")
    if len(out.mani_infos) != len(report):
        problems.append(f"{len(out.mani_infos)} clips out, {len(report)} in")

    ok, message = check_name_buffer(dst)
    if not ok:
        problems.append(f"name buffer: {message}")

    info = read_bulk_info(data)
    if not database:
        # A self-contained bundle must carry NO database, or the runtime is told
        # to go looking for bulk that was never shipped.
        if info is not None:
            problems.append("asked for a self-contained bundle but it carries an "
                            "ACL database header")
        else:
            say("  self-contained: no database, no bulk tiers, no extra streams")
    elif info is None:
        problems.append("no ACL database header in the written file")
    elif locate_bulk(data) is None:
        problems.append("the bulk tiers do not hash-match the database header - "
                        "ACL's database_context::initialize does NOT validate this, "
                        "it crashes during decompression instead")
    else:
        say(f"  database {info['db_size']} B, bulk low={info['low_size']} "
            f"medium={info['medium_size']}, hashes match")

    blobs = [(o, s) for o, s in list_clip_blobs(data)
             if read_blob_header(data, o)["track_type"] == QVVF]
    if len(blobs) != len(report):
        problems.append(f"{len(blobs)} transform blobs for {len(report)} clips")

    for mani_info, expected, (offset, size) in zip(out.mani_infos, report, blobs):
        name = str(mani_info.name)
        blob = data[offset:offset + size]
        head = read_blob_header(data, offset)

        if database and not head.get("has_database"):
            problems.append(f"{name}: transform clip is not database-backed - a "
                            f"dinosaur's loader rejects self-contained clips")
        if not database and head.get("has_database"):
            problems.append(f"{name}: clip is database-backed but no database was "
                            f"written - the runtime will look for bulk that is "
                            f"not in the file")
        if head.get("trivial_defaults"):
            problems.append(f"{name}: trivial_defaults is set, so the runtime will "
                            f"not supply the bind pose for stripped sub-tracks")

        # extra_count is JWE3-only. cobra sizes the scalar section from
        # float_count, so a wrong one plays perfectly in Blender; the GAME reads
        # extra_count and every channel table after it shifts.
        if int(getattr(mani_info, "extra_count", 0)) != int(mani_info.float_count):
            problems.append(
                f"{name}: extra_count={int(getattr(mani_info, 'extra_count', 0))} but "
                f"float_count={int(mani_info.float_count)} - vanilla is always equal")

        if (int(mani_info.dtype.compression), int(mani_info.dtype.has_list)) != (1, 1):
            problems.append(f"{name}: dtype compression/has_list is "
                            f"{int(mani_info.dtype.compression)}/"
                            f"{int(mani_info.dtype.has_list)}, expected 1/1")

        start = bonemask.find_container(data, offset, expected["num_tracks"])
        if start is None:
            problems.append(f"{name}: no 0x80 container before the blob - the clip "
                            f"has no bone mask and can pose nothing")
            continue
        got = bonemask.read_container(data, start, expected["num_tracks"])
        # Rotation and translation are checked against what the blob STORES;
        # scale against what the channel table DECLARES. That asymmetry is
        # vanilla, 1137/1137 - see container_masks.
        want_masks, _layout = container_masks(blob, mani_info,
                                              parse_transform_layout(blob))
        for kind in SUB_TRACKS:
            if got[kind] != want_masks[kind]:
                unmasked = sorted(set(want_masks[kind]) - set(got[kind]))
                spurious = sorted(set(got[kind]) - set(want_masks[kind]))
                problems.append(
                    f"{name}: {kind} mask disagrees with the blob - {len(unmasked)} "
                    f"stored but unmasked (INERT IN GAME, first {unmasked[:4]}), "
                    f"{len(spurious)} masked but not stored (first {spurious[:4]})")
        if not got["rotation"] and expected["posed"]["rotation"]:
            problems.append(f"{name}: rotation mask is empty but the blob stores "
                            f"{len(expected['posed']['rotation'])} rotations")

    streams = [st for st in decode_file(dst) if st.track_type == QVVF]
    if len(streams) != len(report):
        problems.append(f"{len(streams)} decoded transform streams for "
                        f"{len(report)} clips")

    worst_deg = worst_pos = 0.0
    flips = 0
    for expected, stream in zip(report, streams):
        want = expected["samples"]
        got_values = stream.values
        if got_values.shape != want.shape:
            problems.append(f"{expected['name']}: decoded {got_values.shape}, "
                            f"wrote {want.shape}")
            continue

        # A sub-track we left NaN must come back stripped, and one we filled must
        # come back present. Backwards is the crush-and-stretch failure: the game
        # substitutes the bind pose for anything not stored.
        for kind, (lo, hi) in SUB_TRACKS.items():
            meant = np.isnan(want[:, :, lo:hi]).all(axis=0).all(axis=-1)
            came = np.isnan(got_values[:, :, lo:hi]).all(axis=0).all(axis=-1)
            lost = int((~meant & came).sum())
            gained = int((meant & ~came).sum())
            if lost:
                problems.append(f"{expected['name']}: ACL stripped {lost} {kind} "
                                f"sub-track(s) we meant to store - the game will "
                                f"substitute the bind pose there")
            if gained:
                problems.append(f"{expected['name']}: {gained} {kind} sub-track(s) "
                                f"stored that we meant to strip")

        live = np.isfinite(want) & np.isfinite(got_values)
        rot = live[:, :, 0:4].all(axis=-1)
        if rot.any():
            a = got_values[:, :, 0:4][rot].astype(np.float64)
            b = want[:, :, 0:4][rot].astype(np.float64)
            na = np.linalg.norm(a, axis=-1)
            nb = np.linalg.norm(b, axis=-1)
            usable = (na > 1e-6) & (nb > 1e-6)
            if usable.any():
                dot = np.abs(np.sum(a[usable] * b[usable], axis=-1)
                             / (na[usable] * nb[usable]))
                worst_deg = max(worst_deg, float(np.degrees(
                    2.0 * np.arccos(np.clip(dot, -1.0, 1.0))).max()))
        tra = live[:, :, 4:7]
        if tra.any():
            worst_pos = max(worst_pos, float(
                np.abs(got_values[:, :, 4:7][tra] - want[:, :, 4:7][tra]).max()))

        # Hemisphere flips are checked on the INPUT, not on what comes back.
        #
        # On a dtype-0 path the stored sign is the writer's to choose, 0 flips is
        # achievable, and a flip makes the runtime send the bone the 360 degree
        # way round. Carrying that gate to ACL unchanged is a FALSE POSITIVE: ACL's
        # rotation formats drop W and reconstruct it non-negative, so the decoded
        # stream is canonicalised by the codec. Measured on the shipped corpus, 962
        # such "flips" across 213 Acrocanthosaurus clips against 8 negative-w
        # samples in total. What is still ours is the continuity of the samples
        # handed TO the encoder, which is what its error metric sees. The rotations
        # that come back are covered by the angular error above, which compares
        # |dot| and so is sign-blind by construction.
        flips += count_hemisphere_flips(np.nan_to_num(want[:, :, 0:4], nan=0.0))

    if worst_deg > max_rotation_deg:
        problems.append(f"worst rotation error {worst_deg:.4g} deg > {max_rotation_deg}")
    if worst_pos > max_translation:
        problems.append(f"worst translation error {worst_pos:.4g} > {max_translation}")
    if flips:
        problems.append(f"{flips} quaternion hemisphere flips in the samples handed "
                        f"to the encoder - unroll them before encoding")

    say(f"  {len(out.mani_infos)} clips parse, {parsed}/{total} blocks, "
        f"masks agree with the blobs, rot <= {worst_deg:.4g} deg, "
        f"pos <= {worst_pos:.4g}, input hemisphere flips={flips}")
    return problems
