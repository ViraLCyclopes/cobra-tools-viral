"""Fixed-width hash-preserving JWE3 family retargeting."""
import logging
from pathlib import Path

import pytest

from modules.formats.shared import djb2
from source.formats.ovl.retarget import (
	find_c_strings, retarget_family, validate_prefixes)


DONOR = "deinosuchus"
ALIAS = "sarcmimsaee"
BASELINE = Path(
	r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game"
	r"\Dinosaur Files\Animation Research\retest_20260823\baseline\Female"
)
SOURCE = BASELINE / "Deinosuchus_Female.ovl"
def test_known_alias_preserves_width_and_djb2_state():
	assert validate_prefixes(DONOR, ALIAS) == (DONOR, ALIAS)
	assert len(DONOR) == len(ALIAS)
	assert djb2(DONOR) == djb2(ALIAS) == 1549699023


@pytest.mark.parametrize("alias", ["sarco", "sarcmimsaef", "Deinosuchus", "déinosuchus"])
def test_invalid_aliases_are_refused(alias):
	with pytest.raises(ValueError):
		validate_prefixes(DONOR, alias)


def test_c_string_search_does_not_match_the_tail_of_another_string():
	blob = b"prefixdeinosuchus$bad\x00deinosuchus$good\x00"
	assert list(find_c_strings(blob, DONOR.encode())) == [
		(len(b"prefixdeinosuchus$bad\x00"), b"deinosuchus$good")]


@pytest.mark.skipif(not SOURCE.is_file(), reason="Deinosuchus reference family not present")
def test_deinosuchus_family_matches_the_game_verified_build(tmp_path):
	# OvlFile's reporter expects cobra's logging extension when called outside the CLI.
	if not hasattr(logging, "success"):
		logging.success = logging.info
	source_bytes = SOURCE.read_bytes()
	output = tmp_path / "SarcoViral_Female.ovl"
	report = retarget_family(SOURCE, output, DONOR, ALIAS)

	assert report.renamed_loaders == 355
	assert report.pool_strings == 221
	# 221 occurrences x the 11-byte fixed-width prefix.  This is the exact
	# decompressed STATIC delta audited on the build that loaded in game.
	assert report.pool_bytes == 2431
	assert len(report.output_files) == 18
	assert all(path.is_file() and path.stat().st_size for path in report.output_files)
	assert SOURCE.read_bytes() == source_bytes
	assert (tmp_path / "UKY2N3QKP0WPI_.aux").is_file()

	for source_ovs in BASELINE.glob("Deinosuchus_Female.ovs.*"):
		target = tmp_path / ("SarcoViral_Female" + source_ovs.name[len("Deinosuchus_Female"):])
		assert target.read_bytes() == source_ovs.read_bytes()
