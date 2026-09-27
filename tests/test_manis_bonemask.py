"""The 0x80 container that precedes every compressed ManiBlock's ACL blob.

`bonemask.build_record` builds 0x48 of it - the rotation mask plus the bone
count and flag - and that much is vanilla-verified. It is NOT the whole thing.
Measured over 605 shipped clips on four different rigs (142, 170, 172, 175, 206
and 212 bones), the region between a ManiBlock's channel tables and its ACL
transform blob is a fixed 0x80-byte struct:

    0x00  0x18 zero bytes
    0x18  translation mask, ceil(num_tracks/64) qwords, zero-filled to 0x20
    0x38  rotation mask,    same shape          <- what build_record covers
    0x58  scale mask,       same shape
    0x78  uint32 num_tracks
    0x7C  uint32 flag

placed at align8 after the channel tables, with the ACL blob at align16 after
it - which is why the gap from container to blob is 0x80 or 0x88 and never
anything else.

A writer that emits only `build_record` produces a container with **no
translation mask**. Every bone's translation would then be stored in the ACL
blob and masked off, which is the same silent half-change that the rotation
mask cost eight sessions to find - it passes every offline check and does
nothing in game.

These tests are the 605/605 proof, kept as a regression so the model cannot
quietly rot the way `uncompressed_pad_PC2` did.
"""
import glob
import os
import struct

import pytest

from source.formats.manis import bonemask
from source.formats.manis.acl_patch import parse_transform_layout, sub_track_types
from source.formats.manis.splice import list_clip_blobs, read_blob_header

QVVF = 12

# Any directories of shipped JWE3 bundles; the tests skip when none is present.
CORPUS_GLOBS = [
	r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Acro Female\*.manis",
	r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Spino Female\*.manis",
	r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Base Game Dinos\*\*\*.manis",
	r"D:\JWE2 Stuff\Personal Mods\JWE3\Images and Models\Dinosaurs\Land (Base)\Indoraptor\*.manis",
	r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Animation Research\retest_20260823\deino_full_extract_runtime_meta\*.manis",  # the only rig here with SCALE
]


def bundles():
	seen = set()
	for pattern in CORPUS_GLOBS:
		for path in sorted(glob.glob(pattern)):
			key = os.path.realpath(path)
			if key in seen:
				continue
			seen.add(key)
			with open(path, "rb") as fh:
				data = fh.read()
			if any(read_blob_header(data, o)["track_type"] == QVVF
				   for o, _s in list_clip_blobs(data)):
				yield path, data


ALL = list(bundles())
pytestmark = pytest.mark.skipif(not ALL, reason="no reference manis bundles present")


def clips(data, path=None):
	"""(blob, layout, container_offset, mani_info) per compressed transform clip.

	The ManiInfo is needed because the SCALE mask is keyed on the channel table,
	not on the blob - see `acl_writer.container_masks`. It is None when no path is
	given, which is fine for a scale-free rig and wrong for Deinosuchus.
	"""
	infos = []
	if path is not None:
		from generated.formats.manis import ManisFile
		manis = ManisFile()
		manis.game = "Jurassic World Evolution 3"
		manis.load(path)
		infos = list(manis.mani_infos)
	index = 0
	for offset, size in list_clip_blobs(data):
		if read_blob_header(data, offset)["track_type"] != QVVF:
			continue
		blob = data[offset:offset + size]
		layout = parse_transform_layout(blob)
		start = bonemask.find_container(data, offset, layout.num_tracks)
		yield blob, layout, start, (infos[index] if index < len(infos) else None)
		index += 1


def posed(blob, layout, kind, mani_info=None):
	"""What the container's `kind` mask carries.

	Rotation and translation are what the ACL blob STORES. Scale is what the
	channel table DECLARES - a different rule, vanilla 1137/1137, and invisible
	on any rig without scale. See `acl_writer.container_masks`.
	"""
	if kind == "scale":
		if mani_info is not None:
			return sorted({int(t) for t in mani_info.keys.scl_channel_to_bone})
		if not layout.has_scale:
			return []
	return [i for i, t in enumerate(sub_track_types(blob, layout, kind)) if t]


@pytest.mark.parametrize("path,data", ALL, ids=[os.path.basename(p) for p, _ in ALL])
def test_every_compressed_clip_has_a_locatable_container(path, data):
	found = [start for _b, _l, start, _m in clips(data, path)]
	assert found, "bundle has compressed transform clips but none were examined"
	assert all(s is not None for s in found), (
		f"{sum(s is None for s in found)} of {len(found)} clips have no container "
		f"at 0x80 or 0x88 before their ACL blob")


@pytest.mark.parametrize("path,data", ALL, ids=[os.path.basename(p) for p, _ in ALL])
def test_build_container_reproduces_vanilla_byte_for_byte(path, data):
	"""Given the clip's own flag, the container is fully determined by the masks.

	The flag is passed in rather than derived because it is the one field of the
	struct nothing in the clip predicts - see
	test_the_container_flag_is_one_except_for_a_known_handful.
	"""
	checked = 0
	for blob, layout, start, mani_info in clips(data, path):
		assert start is not None
		want = data[start:start + bonemask.CONTAINER_SIZE]
		flag, = struct.unpack_from("<I", want, bonemask.COUNT_AT + 4)
		got = bonemask.build_container(
			layout.num_tracks,
			rotation=posed(blob, layout, "rotation"),
			translation=posed(blob, layout, "translation"),
			scale=posed(blob, layout, "scale", mani_info),
			flag=flag)
		assert got == want, (
			f"{os.path.basename(path)} clip at {start}: container differs at "
			f"{[i for i, (a, b) in enumerate(zip(got, want)) if a != b][:8]}")
		checked += 1
	assert checked


@pytest.mark.parametrize("path,data", ALL, ids=[os.path.basename(p) for p, _ in ALL])
def test_build_record_is_the_tail_of_build_container_when_there_is_no_scale(path, data):
	"""They must agree wherever build_record CAN express the container.

	`build_record` writes the rotation mask, then zeroes everything up to the
	count pair - and the scale mask at 0x58 is inside that range. So the two
	agree only on a rig whose clips declare no scale channels, which is every rig
	except Deinosuchus in this corpus. See the next test for the consequence.
	"""
	for blob, layout, start, mani_info in clips(data, path):
		assert start is not None
		scale = posed(blob, layout, "scale", mani_info)
		if scale:
			continue
		container = bonemask.build_container(
			layout.num_tracks,
			rotation=posed(blob, layout, "rotation"),
			translation=posed(blob, layout, "translation"),
			scale=scale)
		record = bonemask.build_record(layout.num_tracks,
									   posed(blob, layout, "rotation"))
		assert container[bonemask.ROT_MASK_AT:] == record


@pytest.mark.parametrize("path,data", ALL, ids=[os.path.basename(p) for p, _ in ALL])
def test_build_record_would_clear_a_scale_mask(path, data):
	"""Pin the limitation, so nobody patches a scale rig's container with it.

	`build_record` does not merely omit the translation mask that sits before it -
	on a rig that declares scale channels it also CLEARS the scale mask, because
	it zero-fills from the rotation mask to the count pair and 0x58 is in the way.
	Splicing its 0x48 bytes over a Deinosuchus container would silently disable
	every scaled bone. `build_container` is the one to use.
	"""
	found = 0
	for blob, layout, start, mani_info in clips(data, path):
		scale = posed(blob, layout, "scale", mani_info)
		if not scale:
			continue
		found += 1
		record = bonemask.build_record(layout.num_tracks,
									   posed(blob, layout, "rotation"))
		lost = record[bonemask.SCL_MASK_AT - bonemask.ROT_MASK_AT:
					  bonemask.COUNT_AT - bonemask.ROT_MASK_AT]
		assert lost == bytes(len(lost)), "build_record grew a scale mask"
		live = data[start + bonemask.SCL_MASK_AT:start + bonemask.COUNT_AT]
		assert live != lost, "vanilla scale mask is empty; this clip proves nothing"
	if not found:
		pytest.skip("no clip on this rig declares scale channels")


@pytest.mark.parametrize("path,data", ALL, ids=[os.path.basename(p) for p, _ in ALL])
def test_read_container_round_trips(path, data):
	for blob, layout, start, mani_info in clips(data, path):
		assert start is not None
		got = bonemask.read_container(data, start, layout.num_tracks)
		assert got["num_tracks"] == layout.num_tracks
		assert got["rotation"] == posed(blob, layout, "rotation")
		assert got["translation"] == posed(blob, layout, "translation")
		assert got["scale"] == posed(blob, layout, "scale", mani_info)
		assert bonemask.build_container(
			layout.num_tracks, rotation=got["rotation"],
			translation=got["translation"], scale=got["scale"],
			flag=got["flag"]) == data[start:start + bonemask.CONTAINER_SIZE]


def test_the_container_flag_is_one_except_for_a_known_handful():
	"""The flag is 1 on 592 of 605 shipped clips and nothing in the clip predicts it.

	`baryonyx$partial_mouth01` carries 0 while `dimetrodon$partial_mouth01` -
	same role, same shape, same rig family - carries 1, so it is not derivable
	from the sub-track census, the dtype, the frame count or the wrap flag. This
	test pins the distribution so that a future model for it has something to
	beat, and so a corpus change that shifts it is noticed.
	"""
	ones = zeros = 0
	for _path, data in ALL:
		for _blob, layout, start, _mi in clips(data, _path):
			if start is None:
				continue
			flag, = struct.unpack_from("<I", data, start + bonemask.COUNT_AT + 4)
			assert flag in (0, 1), f"flag is {flag}, expected 0 or 1"
			ones += flag == 1
			zeros += flag == 0
	assert ones > zeros * 20, f"flag census shifted: {ones} ones, {zeros} zeros"


def test_build_container_rejects_a_bone_outside_the_rig():
	with pytest.raises(ValueError):
		bonemask.build_container(64, rotation=[64])
	with pytest.raises(ValueError):
		bonemask.build_container(64, translation=[-1])


def test_build_container_refuses_a_rig_too_wide_for_the_mask_field():
	"""Each mask field is 0x20 bytes, so 256 bones is the ceiling."""
	bonemask.build_container(256, rotation=[255])
	with pytest.raises(ValueError):
		bonemask.build_container(257, rotation=[0])
