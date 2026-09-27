"""The format half of writing a JWE3 ACL bundle, with no Planet Zoo in sight.

The load-bearing claims are not "it produces a file" - cobra parses a broken
bundle happily - they are:

* the container and the blob land where the runtime computes them to be;
* a sub-track the clip does not animate is STRIPPED, so the runtime supplies the
  bone's bind pose, rather than stored as identity and flattening the skeleton;
* every sub-track that IS stored has its mask bit set, or it is inert in game.

The encoder tests need a rig and a clip to encode. They use whatever JWE3 bundle
the corpus offers and skip when there is none.
"""
import glob
import os

import numpy as np
import pytest

from source.formats.manis import acl_writer, bonemask

CORPUS_GLOBS = [
    r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Acro Female\*.manis",
    r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Base Game Dinos\*\*\*.manis",
    r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Animation Research\retest_20260823\deino_full_extract_runtime_meta\*.manis",
]
MS2_GLOBS = [
    r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Acro Female\models.ms2",
    r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Animation Research\retest_20260823\deino_full_extract_runtime_meta\models.ms2",
]


class FakeInfo:
    """The handful of ManiInfo fields the block layout depends on."""

    def __init__(self, pos=0, ori=0, scl=0, flo=0, frames=1, duration=0.0, spans=None):
        self.pos_bone_count, self.ori_bone_count = pos, ori
        self.scl_bone_count, self.float_count = scl, flo
        self.frame_count, self.duration = frames, duration
        spans = spans or {}
        for kind, count in (("pos", pos), ("ori", ori), ("scl", scl)):
            low, high = spans.get(kind, (0, count - 1) if count else (255, 0))
            setattr(self, f"{kind}_bone_min", low)
            setattr(self, f"{kind}_bone_max", high)


LAYOUTS = [
    FakeInfo(),
    FakeInfo(pos=1, ori=1),
    FakeInfo(pos=143, ori=143, flo=10),
    FakeInfo(pos=30, ori=30, spans={"pos": (5, 109), "ori": (5, 109)}),
    FakeInfo(pos=150, ori=146, scl=4, flo=9),
    FakeInfo(pos=212, ori=212, scl=212, flo=32),
]


@pytest.mark.parametrize("info", LAYOUTS)
def test_the_blob_sits_0x80_or_0x88_after_the_container(info):
    """The only two gaps vanilla ever shows, across 1042 measured clips.

    Container at align8 after the channel tables, blob at align16 after the
    container - so the gap can only be 0x80 (container already 16-aligned) or
    0x88 (container 8 past). A third value means the geometry is wrong.
    """
    prefix_start, blob_start = acl_writer.block_geometry(info)
    container_start = prefix_start + (-prefix_start % 8)
    assert blob_start % 16 == 0
    assert blob_start - container_start in (0x80, 0x88)
    assert container_start + bonemask.CONTAINER_SIZE <= blob_start


@pytest.mark.parametrize("info", LAYOUTS)
def test_channel_table_extent_is_four_aligned_and_covers_every_table(info):
    extent = acl_writer.channel_table_extent(info)
    assert extent % 4 == 0
    least = 4 * (info.pos_bone_count + info.ori_bone_count
                 + info.scl_bone_count + info.float_count)
    assert extent >= least


@pytest.mark.parametrize("frames,duration,want", [
    (61, 2.0, 30.0),
    (55, 1.8, 30.0),
    (209, 6.933333, 30.0),
    (1, 1.0, 30.0),      # a single frame spans no interval; fall back, do not /0
    (10, 0.0, 30.0),     # duration 0 would divide by zero
])
def test_sample_rate_is_the_frames_to_duration_relationship(frames, duration, want):
    info = FakeInfo(frames=frames, duration=duration)
    assert acl_writer.sample_rate(info) == pytest.approx(want, rel=1e-4)


def test_unrolling_removes_every_hemisphere_flip():
    rng = np.random.default_rng(7)
    track = rng.normal(size=(40, 3, 4)).astype(np.float32)
    track /= np.linalg.norm(track, axis=-1, keepdims=True)
    assert acl_writer.count_hemisphere_flips(track) > 0
    acl_writer.unroll_quaternions(track)
    assert acl_writer.count_hemisphere_flips(track) == 0


def test_unrolling_preserves_the_rotation_each_sample_represents():
    """q and -q are the same rotation; the unroll must only ever change the sign."""
    rng = np.random.default_rng(11)
    track = rng.normal(size=(30, 5, 4)).astype(np.float32)
    track /= np.linalg.norm(track, axis=-1, keepdims=True)
    before = track.copy()
    acl_writer.unroll_quaternions(track)
    same = np.isclose(track, before).all(axis=-1)
    negated = np.isclose(track, -before).all(axis=-1)
    assert (same | negated).all()


# -- the stripping contract, which needs the ACL encoder ---------------------

def _bundles_beside_an_ms2():
    """Every corpus bundle that sits next to a models.ms2, with that .ms2."""
    out = []
    for pattern in MS2_GLOBS:
        for ms2 in sorted(glob.glob(pattern)):
            folder = os.path.dirname(ms2)
            for corpus in CORPUS_GLOBS:
                for path in sorted(glob.glob(corpus)):
                    if os.path.dirname(path) == folder:
                        out.append((path, ms2))
    return out


PAIRS = _bundles_beside_an_ms2()
needs_corpus = pytest.mark.skipif(
    not PAIRS, reason="no JWE3 bundle with a sibling models.ms2 present")


class FakeKeys:
    def __init__(self, frames, ori_tracks, pos_tracks):
        self.ori_channel_to_bone = list(ori_tracks)
        self.pos_channel_to_bone = list(pos_tracks)
        self.scl_channel_to_bone = []
        self.ori_bones = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (frames, len(ori_tracks), 1))
        self.pos_bones = np.full((frames, len(pos_tracks), 3), 2.0, dtype=np.float32)
        self.scl_bones = np.zeros((frames, 0, 3), dtype=np.float32)
        self.floats = np.zeros((frames, 0), dtype=np.float32)


class FakeDtype:
    compression = 0


class FakeClip:
    def __init__(self, frames, tracks, ori_tracks, pos_tracks):
        self.dtype = FakeDtype()
        self.frame_count, self.target_bone_count = frames, tracks
        self.scl_bone_count = self.float_count = 0
        self.keys = FakeKeys(frames, ori_tracks, pos_tracks)


def test_samples_from_keys_marks_unkeyed_subtracks_nan():
    """NaN, not identity - and no assets needed to say so.

    An identity quaternion written explicitly is a STORED sub-track: ACL keeps
    it, the container has to mask it, and the runtime applies our number instead
    of the bone's bind rotation. Writing the whole rig that way poses it flat,
    which is the single most consequential line in this module.
    """
    clip = FakeClip(frames=6, tracks=8, ori_tracks=[0, 3, 5], pos_tracks=[3, 7])
    samples, scalars = acl_writer.samples_from_keys(None, clip)
    assert samples.shape == (6, 8, 10)
    assert scalars is None
    for track in range(8):
        assert bool(np.isnan(samples[:, track, 0:4]).all()) == (track not in (0, 3, 5))
        assert bool(np.isnan(samples[:, track, 4:7]).all()) == (track not in (3, 7))
    # scale is keyed nowhere, so the whole group is stripped
    assert np.isnan(samples[:, :, 7:10]).all()
    # and the keyed ones carry their real values, not the NaN fill
    assert np.allclose(samples[:, 3, 4:7], 2.0)


def test_samples_from_keys_refuses_a_clip_that_is_already_acl():
    """Otherwise it dies inside cobra with 'no attribute segments'.

    `ManisFile.decompress` handles dtype 0 and the PZ/JWE2 segmented quantiser,
    not a JWE3 ACL blob - which has no key arrays at all. Say which tool to reach
    for instead.
    """
    class AclDtype:
        compression = 1

    class Context:
        version, mani_version = 262, 282

    class Manis:
        context = Context()

    clip = FakeClip(frames=2, tracks=2, ori_tracks=[0], pos_tracks=[0])
    clip.dtype = AclDtype()
    clip.name = "species$clip"
    with pytest.raises(ValueError, match="decode_file"):
        acl_writer.samples_from_keys(Manis(), clip)


@pytest.fixture(scope="module")
def clip_samples():
    """A real clip on the same rig as the .ms2, with rotations actually keyed.

    Taken through `acl.decode_file`, not `samples_from_keys`: the corpus bundles
    are already JWE3 ACL, so they have no key arrays to read. The decoder returns
    the same (frames, tracks, 10) shape with NaN for the sub-tracks Frontier
    stripped, which is exactly the input this module is built to re-encode.
    """
    from generated.formats.manis import ManisFile
    from generated.formats.manis.acl import decode_file
    from source.formats.manis.bindpose import read_ms2_bind
    from source.formats.manis.splice import list_clip_blobs, read_blob_header

    tried = []
    for path, ms2 in PAIRS:
        with open(path, "rb") as stream:
            raw = stream.read()
        # decode_file raises "acl_blob_count=0" on a dtype-0 bundle, and
        # hatcheryexitcamera sits in the same folder as every rig's models.ms2.
        if not any(read_blob_header(raw, o)["track_type"] == acl_writer.QVVF
                   for o, _s in list_clip_blobs(raw)):
            continue
        parents, bind_values = read_ms2_bind(ms2)
        manis = ManisFile()
        manis.game = "Jurassic World Evolution 3"
        manis.load(path)
        transforms = [st for st in decode_file(path) if st.track_type == acl_writer.QVVF]
        for mani_info, stream in zip(manis.mani_infos, transforms):
            if int(mani_info.target_bone_count) != len(parents):
                continue
            keyed = ~np.isnan(stream.values[:, :, 0:4]).all(axis=0).all(axis=-1)
            if int(keyed.sum()) >= 12:
                return stream.values, parents, bind_values, mani_info
        tried.append(os.path.basename(path))
    pytest.skip(f"no clip on the .ms2's own rig in {len(tried)} bundle(s)")


@needs_corpus
def test_unanimated_subtracks_are_stripped_and_unmasked(clip_samples, tmp_path):
    """The contract that makes the whole thing work.

    Blank the rotation of every track but the first ten and check that ACL strips
    exactly those and the container masks exactly the rest. If the mask were
    built from the rig rather than from what the blob stores, this is where it
    shows - and a stored-but-unmasked sub-track is inert in game while passing
    every other check.
    """
    from source.formats.manis.bindpose import bind_bytes
    from generated.formats.manis.acl import AclSamples

    samples, parents, bind_values, mani_info = clip_samples
    keyed = [t for t in range(samples.shape[1])
             if not np.isnan(samples[:, t, 0:4]).all()]
    keep = keyed[:10]
    assert len(keep) >= 2

    edited = samples.copy()
    edited[:, :, 0:4] = np.nan
    for track in keep:
        edited[:, track, 0:4] = samples[:, track, 0:4]

    stream = AclSamples(acl_writer.QVVF, edited.shape[1], edited.shape[0],
                        edited.shape[2], 30.0, edited)
    bound, _db, _low, _med = acl_writer._build_database()(
        [stream], [{"wrap_optimized": False}], bind_bytes(parents, bind_values),
        str(tmp_path), parents, bind_values)
    blob = bound[0]

    stored, layout = acl_writer.stored_subtracks(blob)
    assert stored["rotation"] == keep, (
        f"expected rotation stored for {keep}, got {stored['rotation'][:16]}")

    prefix_start, blob_start = acl_writer.block_geometry(mani_info)
    _prefix, posed, _layout = acl_writer.build_prefix(blob, mani_info,
                                                      prefix_start, blob_start)
    assert posed["rotation"] == keep
    # Scale comes from the channel table, not the blob - a different rule from
    # the other two, vanilla 1137/1137. See acl_writer.container_masks.
    assert posed["scale"] == sorted({int(t) for t in mani_info.keys.scl_channel_to_bone})

    container = bonemask.build_container(
        layout.num_tracks, rotation=posed["rotation"],
        translation=posed["translation"], scale=posed["scale"])
    got = bonemask.read_container(container, 0, layout.num_tracks)
    assert got["rotation"] == keep
    assert got["translation"] == posed["translation"]
