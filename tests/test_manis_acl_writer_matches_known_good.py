"""`acl_writer.write_bundle` must reproduce `manis_database_cmd.py` byte for byte.

This is the strongest gate available for the writer, and it is the one that was
missing. `manis_database_cmd.py` is game-verified but it SPLICES: preamble,
ManiInfo array and database position all come from the original file, so anything
it does not touch is preserved for free. `write_bundle` builds the whole file, so
every field it does not carry across is a field it INVENTS.

Run against the first version of the writer, this found four such fields, and
none of them were visible to any other check - every offline gate passed, the
bundle round-tripped through the OVL packer byte-identically, and the samples
decoded back correctly:

    dtype `unk`        forced to 0, source had 1        15 of 17 clips
    container flag     forced to 1, source had 0         5 of 17 clips
    sample_rate        recomputed (n-1)/duration        all clips
                       instead of the source's 30.00029945373535
    wrap_optimized     hardcoded False                  15 of 17 clips
                       then, once carried, wrongly inherited by the SCALAR
                       blob, which vanilla keeps independent

The lesson is one rule: carry what the source has, invent only what it does not.

The equality is over the KEYS REGION and the bulk both tools produce; ACL
encoding is deterministic for identical inputs, so a difference here is a real
difference in what we assemble.
"""
import glob
import os
import subprocess
import sys

import pytest

from generated.formats.manis import ManisFile
from generated.formats.manis.acl import decode_file
from source.formats.manis import acl_writer, bonemask
from source.formats.manis.acl_patch import parse_transform_layout
from source.formats.manis.database import locate_bulk
from source.formats.manis.splice import list_clip_blobs, read_blob_header

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = [
    (r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game"
     r"\Dinosaur Files\Acro Female", "notmotionextracted.manisetd0bec0f7.manis"),
]


def pairs():
    for folder, bundle in CORPUS:
        src = os.path.join(folder, bundle)
        ms2 = os.path.join(folder, "models.ms2")
        if os.path.isfile(src) and os.path.isfile(ms2):
            with open(src, "rb") as fh:
                data = fh.read()
            # only bundles that are already the shape we write: database-backed,
            # every clip has_list == 1 so nothing carries a limb structure
            if locate_bulk(data) is None:
                continue
            manis = ManisFile()
            manis.game = "Jurassic World Evolution 3"
            manis.load(src)
            if any(int(mi.dtype.has_list) > 1 for mi in manis.mani_infos):
                continue
            yield src, ms2


ALL = list(pairs())
pytestmark = pytest.mark.skipif(
    not ALL, reason="no database-backed limb-free reference bundle present")


def carried(src):
    """Everything the source already holds, which the writer must not reinvent."""
    with open(src, "rb") as fh:
        raw = fh.read()
    manis = ManisFile()
    manis.game = "Jurassic World Evolution 3"
    manis.load(src)
    decoded = decode_file(src)
    transforms = [s for s in decoded if s.track_type == acl_writer.QVVF]
    scalars = iter([s for s in decoded if s.track_type != acl_writer.QVVF])
    blob_offsets = [o for o, _s in list_clip_blobs(raw)]
    transform_blobs = [(o, s) for o, s in list_clip_blobs(raw)
                       if read_blob_header(raw, o)["track_type"] == acl_writer.QVVF]

    clips, rates, flags, wraps = [], [], [], []
    for mani_info, stream, (off, size) in zip(manis.mani_infos, transforms,
                                              transform_blobs):
        scalar = next(scalars).values if int(mani_info.float_count) else None
        clips.append((stream.values, scalar))
        rates.append(stream.sample_rate)

        layout = parse_transform_layout(raw[off:off + size])
        start = bonemask.find_container(raw, off, layout.num_tracks)
        flags.append(bonemask.read_container(raw, start, layout.num_tracks)["flag"]
                     if start is not None else None)

        transform_wrap = read_blob_header(raw, off)["wrap_optimized"]
        scalar_wrap = False
        if int(mani_info.float_count):
            later = [o for o in blob_offsets if o > off]
            if later and read_blob_header(raw, later[0])["track_type"] != acl_writer.QVVF:
                scalar_wrap = read_blob_header(raw, later[0])["wrap_optimized"]
        wraps.append((transform_wrap, scalar_wrap))
    return manis, clips, rates, flags, wraps


@pytest.mark.slow
@pytest.mark.parametrize("src,ms2", ALL, ids=[os.path.basename(s) for s, _ in ALL])
def test_write_bundle_reproduces_manis_database_cmd(src, ms2, tmp_path):
    known = str(tmp_path / "known_good.manis")
    result = subprocess.run(
        [sys.executable, "manis_database_cmd.py", src, "--ms2", ms2, "--out", known],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, (result.stderr or result.stdout)[-2000:]

    mine = str(tmp_path / "mine.manis")
    manis, clips, rates, flags, wraps = carried(src)
    acl_writer.write_bundle(manis, clips, ms2, mine,
                            rates=rates, flags=flags, wraps=wraps)

    with open(known, "rb") as fh:
        expected = fh.read()
    with open(mine, "rb") as fh:
        got = fh.read()
    if got != expected:
        differing = [i for i in range(min(len(got), len(expected)))
                     if got[i] != expected[i]]
        pytest.fail(
            f"{len(differing)} bytes differ from the game-verified rebuild "
            f"(sizes {len(got)} vs {len(expected)}), first at "
            f"{differing[0] if differing else 'n/a'} - the writer is inventing a "
            f"value the source already holds")


@pytest.mark.parametrize("src,ms2", ALL, ids=[os.path.basename(s) for s, _ in ALL])
def test_set_acl_dtype_keeps_unk_and_only_forces_what_it_owns(src, ms2):
    """Only `compression` and `has_list` are the writer's to set."""
    manis = ManisFile()
    manis.game = "Jurassic World Evolution 3"
    manis.load(src)
    before = [(int(mi.dtype.unk), int(mi.dtype.use_ushort)) for mi in manis.mani_infos]
    acl_writer.set_acl_dtype(manis)
    after = [(int(mi.dtype.unk), int(mi.dtype.use_ushort)) for mi in manis.mani_infos]
    assert after == before
    assert all((int(mi.dtype.compression), int(mi.dtype.has_list)) == (1, 1)
               for mi in manis.mani_infos)
