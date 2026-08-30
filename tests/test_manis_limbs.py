"""The limb structure after each compressed ManiBlock sits at align 8, not 16.

cobra does not model it - `ManiBlock`'s `LimbTrackData` is `vercond="!#PC2#"` and
`KeysReader` keeps the bytes as opaque `inter_block_data` - so nothing else in
the toolchain notices when a re-encode leaves it a few bytes from where the
runtime computes it to be. The game does notice: it reads counts out of padding
and dies at `JWE3.exe+0x1697FBD`.

The regression these tests protect is that rebuilding a shipped bundle with its
own blobs is the identity. If the align8/align16 model is ever wrong again, that
equality breaks immediately and locally, instead of as a crash on spawn.
"""
import glob
import os

import pytest

from source.formats.manis.limbs import (
	block_layout, buffer_residue, rebuild_block, replace_clip_blob)
from source.formats.manis.splice import list_clip_blobs, read_blob_header

# any directory of shipped JWE3 bundles; the tests skip when none is present
CORPUS = [
	r"D:\JWE2 Stuff\Personal Mods\JWE3\Images and Models\Dinosaurs\Land (Base)\Indoraptor",
	r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Acro Female",
]


def bundles():
	for folder in CORPUS:
		if os.path.isdir(folder):
			for path in sorted(glob.glob(os.path.join(folder, "*.manis"))):
				with open(path, "rb") as fh:
					data = fh.read()
				if list_clip_blobs(data):
					yield path, data


ALL = list(bundles())
pytestmark = pytest.mark.skipif(not ALL, reason="no reference manis bundles present")


def parts(data):
	blobs = list_clip_blobs(data)
	headers = [read_blob_header(data, offset) for offset, _size in blobs]
	return blobs, headers, buffer_residue(blobs)


@pytest.mark.parametrize("path,data", ALL, ids=[os.path.basename(p) for p, _ in ALL])
def test_rebuilding_a_shipped_bundle_is_the_identity(path, data):
	blobs, headers, base = parts(data)
	out = data
	for block in reversed(block_layout(data, blobs, headers)):
		out = rebuild_block(out, block, base,
							[out[o:o + s] for o, s in block["blobs"]])
	assert out == data


@pytest.mark.parametrize("path,data", ALL, ids=[os.path.basename(p) for p, _ in ALL])
def test_every_limb_structure_ends_on_a_16_boundary(path, data):
	blobs, headers, base = parts(data)
	for block in block_layout(data, blobs, headers):
		if not block["limb"]:
			continue
		start, extent = block["limb"]
		# the runtime computes this address; it must be align8 from the last blob
		last = block["blobs"][-1]
		assert start == last[0] + last[1] + (-(last[0] + last[1] - base) % 8)
		# and only zero padding may separate its end from the next block
		end = start + extent
		slack = -(end - base) % 16
		assert data[end:end + slack] == b"\x00" * slack


@pytest.mark.parametrize("path,data", ALL, ids=[os.path.basename(p) for p, _ in ALL])
def test_replace_clip_blob_with_identical_bytes_is_a_noop(path, data):
	offset, size = list_clip_blobs(data)[0]
	assert replace_clip_blob(data, 0, data[offset:offset + size]) == data


@pytest.mark.parametrize("path,data", ALL, ids=[os.path.basename(p) for p, _ in ALL])
def test_a_resized_blob_keeps_the_limb_structure_at_align8(path, data):
	blobs, headers, base = parts(data)
	blocks = block_layout(data, blobs, headers)
	target = next((i for i, b in enumerate(blocks) if b["limb"]), None)
	if target is None:
		pytest.skip("no clip in this bundle carries limb data")
	index = sum(len(b["blobs"]) for b in blocks[:target])
	offset, size = blobs[index]
	# grow by 8, which is exactly the amount that flips align8 vs align16
	grown = data[offset:offset + size] + b"\x00" * 8
	# size lives in the ACL header, so patch it or the walk desynchronises
	grown = (size + 8).to_bytes(4, "little") + grown[4:]
	out = replace_clip_blob(data, index, grown)

	new_blobs = list_clip_blobs(out)
	new_headers = [read_blob_header(out, o) for o, _ in new_blobs]
	rebuilt = block_layout(out, new_blobs, new_headers)[target]
	assert rebuilt["limb"] is not None, "the limb structure was lost"
	start, extent = rebuilt["limb"]
	last = rebuilt["blobs"][-1]
	new_base = buffer_residue(new_blobs)
	assert start == last[0] + last[1] + (-(last[0] + last[1] - new_base) % 8)
	assert extent == blocks[target]["limb"][1], "the limb structure changed size"
	# the bytes themselves must survive verbatim
	old_start, old_extent = blocks[target]["limb"]
	assert out[start:start + extent] == data[old_start:old_start + old_extent]
