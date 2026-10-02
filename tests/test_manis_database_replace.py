"""`manis_database_cmd.py --replace-clip` takes several clips in ONE rebuild.

Every rebuild re-encodes every clip in the bundle from the previous rebuild's
output, so the Blender splice export used to chain one rebuild per edited clip
and compound the loss: ~1 deg of visible stepping after 10 rebuilds at ACL's old
0.01 default. These tests pin the one-rebuild path and the stripped-rotation
default that the tighter 0.001 precision depends on.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from generated.formats.manis import ManisFile
from generated.formats.manis.acl import decode_file, _write_jacl
from source.formats.manis.bindpose import clip_defaults, read_ms2_bone_names
from source.formats.manis.inspect import _clip_blob_map, transform_blob_indices

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = (r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game"
          r"\Dinosaur Files\Acro Female")
BUNDLE = os.path.join(CORPUS, "notmotionextracted.manisetd0bec0f7.manis")
MS2 = os.path.join(CORPUS, "models.ms2")

needs_corpus = pytest.mark.skipif(
    not (os.path.isfile(BUNDLE) and os.path.isfile(MS2)),
    reason="Acro Female corpus bundle not present")


def clips_of(path):
    manis = ManisFile()
    manis.load(path)
    with open(path, "rb") as fh:
        data = fh.read()
    names = [str(info.name) for info in manis.mani_infos]
    order = transform_blob_indices(_clip_blob_map(data))
    streams = decode_file(path)
    return {name: streams[order[i]] for i, name in enumerate(names)}


def worst_angle(a, b):
    """Largest rotation difference in degrees over sub-tracks both sides store."""
    qa, qb = a[:, :, 0:4], b[:, :, 0:4]
    both = ~(np.isnan(qa).any(axis=-1) | np.isnan(qb).any(axis=-1))
    if not both.any():
        return 0.0
    dot = np.clip(np.abs(np.sum(qa[both] * qb[both], axis=-1)), -1.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dot)).max())


def yawed(values, track, degrees):
    """Compose a yaw about Y onto one bone, so a replacement is visible."""
    half = np.radians(degrees) / 2.0
    s, c = np.sin(half), np.cos(half)
    out = values.copy()
    x, y, z, w = (out[:, track, i].astype(np.float64) for i in range(4))
    out[:, track, 0] = c * x + s * z
    out[:, track, 1] = c * y + s * w
    out[:, track, 2] = c * z - s * x
    out[:, track, 3] = c * w - s * y
    return out


def editable_track(values, bone_names):
    for track in range(values.shape[1]):
        if track < len(bone_names) and bone_names[track].startswith("def_") \
                and not np.isnan(values[:, track, 0:4]).any():
            return track
    return None


@needs_corpus
def test_several_clips_replace_in_one_rebuild(tmp_path):
    before = clips_of(BUNDLE)
    bone_names = read_ms2_bone_names(MS2)
    # some clips store no rotation at all (partial_blank01)
    targets = [(clip, track) for clip, stream in before.items()
               if (track := editable_track(stream.values, bone_names)) is not None][:2]
    assert len(targets) == 2, "corpus needs two clips with a stored def_ rotation"
    edits = {}
    argv = [sys.executable, "manis_database_cmd.py", BUNDLE, "--ms2", MS2,
            "--out", str(tmp_path / "out.manis")]
    for n, (clip, track) in enumerate(targets):
        stream = before[clip]
        values = yawed(stream.values, track, 10.0 + 5.0 * n)
        jacl = str(tmp_path / f"clip{n}.jacl")
        _write_jacl(jacl, values, stream.track_type, stream.sample_rate)
        edits[clip] = (values, track)
        argv += ["--replace-clip", clip, "--jacl", jacl]

    result = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, (result.stderr or result.stdout)[-2000:]
    after = clips_of(str(tmp_path / "out.manis"))

    for clip, (values, track) in edits.items():
        got = after[clip].values
        assert worst_angle(got[:, track:track + 1], before[clip].values[:, track:track + 1]) > 5.0, \
            f"{clip}: the edit did not land"
        assert worst_angle(got, values) < 0.2, f"{clip}: replacement not faithful"
    for clip in before:
        if clip in edits:
            continue
        assert worst_angle(after[clip].values, before[clip].values) < 0.2, \
            f"{clip}: untouched clip drifted"
        for lo, hi in ((0, 4), (4, 7), (7, 10)):
            was = np.isnan(before[clip].values[:, :, lo:hi]).all(axis=(0, 2))
            now = np.isnan(after[clip].values[:, :, lo:hi]).all(axis=(0, 2))
            assert (was == now).all(), f"{clip}: stripped set changed in components {lo}:{hi}"


@needs_corpus
@pytest.mark.parametrize("extra, message", [
    (["--replace-clip", "a", "--replace-clip", "b", "--jacl", "x"], "must pair up"),
    (["--replace-clip", "a", "--jacl", "x", "--replace-clip", "a", "--jacl", "y"],
     "more than once"),
])
def test_replace_clip_arguments_are_validated(tmp_path, extra, message):
    result = subprocess.run(
        [sys.executable, "manis_database_cmd.py", BUNDLE, "--ms2", MS2,
         "--out", str(tmp_path / "out.manis")] + extra,
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode != 0
    assert message in (result.stderr + result.stdout)


def test_stripped_rotation_default_is_identity():
    """ACL rebuilds w from xyz, so a bind with w ~ 0 does not read back as itself.

    Acro's def_c_lipLwr_joint binds at (0, 0.707, 0.707, -0): filled with that
    default it came back 0.03 deg off and was STORED at precision 0.001 - 17
    stripped-set differences across a vanilla bundle. Identity rebuilds exactly.
    """
    frames, tracks = 4, 3
    values = np.full((frames, tracks, 10), np.nan, dtype="<f4")
    values[:, 1, 0:4] = (0.0, 0.0, 0.0, 1.0)
    bind = np.zeros((tracks, 10), dtype="<f4")
    bind[:, 0:4] = (0.0, 0.70710677, 0.70710677, -0.0)
    out = clip_defaults(values, bind)
    assert out[0, 0:4].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert out[2, 0:4].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert out[1, 0:4].tolist() != [0.0, 0.0, 0.0, 1.0]
