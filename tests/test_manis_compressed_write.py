"""`ManisFile.save()` CAN write a database-backed JWE3 bundle - with one limit.

`splice.py` says re-serialising a JWE3 manis drops the compressed_database, the
inter-block auxiliary data and the trailing bulk, so every ACL tool here works by
splicing bytes instead. Two of those three have since been fixed
(`CompressedHeaderReader` keeps the database blob verbatim, `KeysReader` keeps
`inter_block_data`), so the claim was re-measured rather than inherited.

Measured over 46 shipped JWE3 bundles:

    compression 0, has_list 0        4/4  keys region byte-identical
    compression 1, has_list 1        4/4  keys region byte-identical
    compression 1, has_list {1,3}    1/1  keys region byte-identical
    compression 1, has_list 3       0/37  truncated

The failure is precise and it is not the database: the output is a strict PREFIX
of the input, ending at `eoh`. A clip with `has_list > 1` carries a limb
structure AFTER its ManiBlock, and the last clip's trails past where the reader
stops, so it is never read and cannot be written back. Everything before it -
preamble, ManiInfo array, compressed_database, name buffer and every ACL
container - comes back exactly.

That makes save() a sound writer for a bundle whose clips are all `has_list = 1`,
which is what a from-scratch ACL bundle wants anyway: 17 shipped Acrocanthosaurus
clips attest that a compressed clip needs no limb structure, and the limb
structure is the thing behind the JWE3.exe+0x1697FBD spawn crash.

The trailing database bulk is genuinely not modelled and is still the caller's to
append; these tests compare the keys region only.
"""
import glob
import os

import pytest

from generated.formats.manis import ManisFile
from source.formats.manis.database import locate_bulk

CORPUS_GLOBS = [
	r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Acro Female\*.manis",
	r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Spino Female\*.manis",
	r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game\Dinosaur Files\Base Game Dinos\*\*\*.manis",
]


def bundles():
	seen = set()
	for pattern in CORPUS_GLOBS:
		for path in sorted(glob.glob(pattern)):
			key = os.path.realpath(path)
			if key not in seen:
				seen.add(key)
				yield path


ALL = list(bundles())
pytestmark = pytest.mark.skipif(not ALL, reason="no reference manis bundles present")


def reload_and_save(path, tmp_path):
	"""(original keys region, what save() produced, the set of has_list values)."""
	with open(path, "rb") as fh:
		raw = fh.read()
	manis = ManisFile()
	manis.game = "Jurassic World Evolution 3"
	manis.load(path)
	lists = {int(mi.dtype.has_list) for mi in manis.mani_infos}
	out = str(tmp_path / "out.manis")
	manis.save(out)
	with open(out, "rb") as fh:
		got = fh.read()
	found = locate_bulk(raw)
	keys_end = found["low_offset"] if found else len(raw)
	return raw[:keys_end], got, lists


@pytest.mark.parametrize("path", ALL, ids=[os.path.basename(p) for p in ALL])
def test_save_is_exact_whenever_the_last_clip_has_no_limb_structure(path, tmp_path):
	want, got, lists = reload_and_save(path, tmp_path)
	manis = ManisFile()
	manis.game = "Jurassic World Evolution 3"
	manis.load(path)
	last_has_limb = int(manis.mani_infos[-1].dtype.has_list) > 1
	if last_has_limb:
		pytest.skip(f"last clip has has_list={int(manis.mani_infos[-1].dtype.has_list)}")
	assert got == want, (
		f"has_list={sorted(lists)} but the keys region was not reproduced: "
		f"{len(got)} bytes out, {len(want)} in")


@pytest.mark.parametrize("path", ALL, ids=[os.path.basename(p) for p in ALL])
def test_when_save_is_not_exact_the_output_is_a_truncation_not_a_corruption(path, tmp_path):
	"""The lost bytes are the tail, never a changed byte in the middle.

	This is what makes the limitation safe to work around by choosing
	`has_list = 1` rather than by avoiding save() altogether. If a future change
	ever corrupts the body instead of dropping the tail, this fails even though
	the test above would still just skip.
	"""
	want, got, _lists = reload_and_save(path, tmp_path)
	assert len(got) <= len(want)
	assert want.startswith(got), "save() changed a byte rather than dropping the tail"


def test_at_least_one_database_backed_bundle_round_trips_exactly(tmp_path):
	"""Guards against the corpus silently losing the case that matters.

	A from-scratch writer leans on save() reproducing a bundle that has BOTH an
	ACL database and compressed clips. If no such bundle is present the tests
	above all skip or pass vacuously, and the property goes unmeasured.
	"""
	proven = []
	for path in ALL:
		with open(path, "rb") as fh:
			raw = fh.read()
		if locate_bulk(raw) is None:
			continue
		manis = ManisFile()
		manis.game = "Jurassic World Evolution 3"
		manis.load(path)
		if int(manis.mani_infos[-1].dtype.has_list) > 1:
			continue
		if not any(int(mi.dtype.compression) for mi in manis.mani_infos):
			continue
		want, got, _ = reload_and_save(path, tmp_path)
		if got == want:
			proven.append(os.path.basename(path))
	assert proven, ("no database-backed, compressed, limb-free bundle in the corpus "
					"reproduced exactly - the from-scratch writer's premise is unmeasured")
