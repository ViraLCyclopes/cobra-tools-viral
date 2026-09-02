"""Copy a species OVL family and rename it - file, companions, insides and aux.

Ported into cobra from `JWE 3 Luas/Base Game/Dinosaur Files/rename_dino_ovl.py` so
the GUI and the CLI share one implementation. The GUI's plain Rename Files / Rename
Contents actions do only two of the four jobs below and produce a broken species.

A dinosaur species is 18 files that all have to move together:

* `<Name>_Female.ovl`
* 16 `<Name>_Female.ovs.*` stream companions
* one `.aux` whose FILENAME IS DERIVED, not stored

That last one is what makes a hand rename fail. `TexelLoader.get_aux_name` hashes
`f"{ovl.basename}_{texel_name}"` with fnv64 and base32-encodes it, so renaming the
OVL changes the expected aux name and orphans the old file. The archive then saves
happily and dies on load inside `get_aux_data` with `KeyError: ''`.

So this does all four jobs in the right order:

1. rename the internal file basenames      (`ovl.rename`)
2. rename the strings inside them          (`ovl.rename_contents`)
3. write the archive out under the new name
4. rename the `.ovs` companions and recompute the `.aux` name to match

Nothing is written to the source. `CHECKLIST` covers what this CANNOT do - the
second donor token, the prefab's AssetPackages inheritance, and the FDB columns -
and callers should surface it.

DIFFERENT-LENGTH RENAMES WORK. Cobra's older `retarget-family` demands an
equal-width alias with the same djb2 hash state; that constraint was retired when
`sarcmimsaee` (11) -> `viralsarcosuchus` (16) was game-verified.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from generated.formats.ovl import OvlFile

# `logging.success` is installed per-module across this codebase rather than
# globally, so a caller that imports only this module would otherwise raise
# AttributeError on the success path.
if not hasattr(logging, "success"):
	logging.success = logging.info  # type: ignore[attr-defined]

DEFAULT_GAME = "Jurassic World Evolution 3"

CHECKLIST = """\
A DONOR-BASED SPECIES HAS TWO TOKENS. Renaming one is not enough:

  --from deinosuchus  ... files, models, clips, motiongraph, textures
  --from <donor>      ... the .ms2 MATERIAL name, e.g. Deinosuchus_Female

rename_contents is case-SENSITIVE; name_variants() expands each token into
lower/Capitalised/UPPER, which is why a lowercase token still reaches the
capitalised material. Run both passes and the species is self-contained.

Rename only the first and the material still points at the donor, which lives in
the donor's asset package. Then the prefab MUST use:

  AssetPackages = { Default = { '<Pkg>', __inheritance = 'Append' } }

'Overwrite' REPLACES the base prefab's package list and cuts the model off from
that material. Symptom: spawns, sounds play, no crash, model INVISIBLE.

THE FDB DOES NOT FOLLOW THE RENAME. This renames the .wsm world-space-motion files
to the new token, but SpeciesCosmeticSets still names the old one:

  WorldSpaceMotionSpeciesNameOverride   must equal the .wsm prefix shipped here
  RetargetSpeciesNameOverride           likewise

Leave them pointing at a species with no .wsm and the lookup that places two
animals relative to each other returns nothing. Symptom: no crash, no error, but
animals walk into each other and SIT ON THE SAME SPOT in social interactions.
Game-verified 2026-08-30.

Only the sex whose OVL you renamed becomes self-contained. Males and juveniles
reuse the donor's assets, so their rows must KEEP the donor name.

The FDB the game reads is PACKED INSIDE Main.ovl - loose copies in the mod tree are
ignored. Extract Main.ovl, patch that copy, inject it back with a plain
`inject --in-place` (`--update` is not supported for FdbLoader). Do NOT repack
Main.ovl from the Main/ folder without diffing: that source copy goes stale and a
repack can silently regress PackageName back to the donor.

AUDIO IS NOT RENAMED, deliberately. Wwise resolves events by ID = FNV-1 32-bit of
the lowercased name, so a renamed reference hashes to an ID no bank contains and
the species goes silent. It keeps the donor's events on purpose."""


def name_variants(old: str, new: str):
	"""Cover the casings the data actually uses: lower for file names, capitalised
	for the `Species$Clip` references inside the motiongraph."""
	pairs = [(old.lower(), new.lower()), (old.capitalize(), new.capitalize()),
			 (old.upper(), new.upper())]
	return list(dict.fromkeys(pairs))


def open_ovl(path: Path, game: str = DEFAULT_GAME) -> OvlFile:
	ovl = OvlFile()
	# Without this, save() raises "Unsupported game" from get_mime.
	ovl.load_hash_table()
	ovl.load(str(path), {"game": game, "update_aux": True})
	return ovl


def aux_names(ovl: OvlFile) -> dict:
	"""{texel loader name: expected aux filename} for the CURRENT basename."""
	out = {}
	for loader in ovl.loaders.values():
		if getattr(loader, "extension", None) == ".texel":
			try:
				out[loader.name] = loader.get_aux_name("")
			except Exception as exc:
				logging.warning(f"could not derive aux name for {loader.name}: {exc}")
	return out


def rename_family(source_ovl, out_dir, old: str, new: str, stem: str = None,
				  game: str = DEFAULT_GAME) -> dict:
	"""Copy the family at `source_ovl` into `out_dir` under the new token."""
	source_ovl = Path(source_ovl).resolve()
	out_dir = Path(out_dir).resolve()
	if out_dir == source_ovl.parent:
		raise ValueError("Refusing to write beside the source; use a separate folder")
	if not old or not new:
		raise ValueError("Both the old and the new token are required")
	out_dir.mkdir(parents=True, exist_ok=True)

	src_dir = source_ovl.parent
	src_stem = source_ovl.stem
	if stem is not None:
		# An explicit stem is what you want when rebuilding a mod from a donor: the
		# insides take the new species token while the archive keeps the filename the
		# mod's .assetpkg already declares. The aux hash follows the stem, so it is
		# recomputed from this, not from the token.
		new_stem = stem
	else:
		new_stem = src_stem.replace(old, new)
		for variant_old, variant_new in name_variants(old, new):
			new_stem = new_stem.replace(variant_old, variant_new)
	# A stem that does not carry the token is fine and common: a mod's archive is
	# often named for the package (SarcoViral_Female.ovl) while the assets inside
	# use a different token entirely. Then only the insides move, the basename is
	# unchanged, and the aux keeps its derived name.
	internals_only = new_stem == src_stem

	# Stage the whole family under the ORIGINAL names first, so the loader finds its
	# companions and its aux exactly as it expects them.
	staged = []
	for item in sorted(src_dir.iterdir()):
		if item.is_file() and (item.stem.startswith(src_stem) or item.suffix == ".aux"):
			shutil.copy2(item, out_dir / item.name)
			staged.append(item.name)
	if not staged:
		raise ValueError(f"No family files found beside {source_ovl.name}")

	ovl = open_ovl(out_dir / source_ovl.name, game)
	before_aux = aux_names(ovl)
	renamed_files = [n for n in ovl.loaders if old.lower() in n.lower()]

	pairs = name_variants(old, new)
	ovl.rename(pairs)
	ovl.rename_contents(pairs, None)

	# basename drives the aux hash, so set it by writing to the new path
	target = out_dir / f"{new_stem}{source_ovl.suffix}"
	ovl.save(str(target))

	after = open_ovl(target, game)
	after_aux = aux_names(after)

	# Put the .ovs companions on the new stem. save() derives each archive's ovs_path
	# from the output name, so it has usually written them already - in that case the
	# staged original is just leftover and gets removed.
	moved, dropped = [], []
	for name in staged:
		if internals_only:
			# Stem unchanged: the companions are already correctly named and save()
			# has rewritten them in place. Touching them here would delete the very
			# files it just wrote.
			break
		if not name.startswith(src_stem) or name == source_ovl.name:
			continue
		old_path = out_dir / name
		if not old_path.exists():
			continue
		new_path = out_dir / name.replace(src_stem, new_stem, 1)
		if new_path.exists():
			old_path.unlink()
			dropped.append(name)
		else:
			old_path.rename(new_path)
			moved.append((name, new_path.name))

	# Make each aux match what the NEW basename derives. Saving with update_aux
	# already writes it under the new name, so usually the only job left is deleting
	# the stale original - which would otherwise sit there looking valid while
	# nothing references it.
	aux_moves, aux_orphans = [], []
	for texel, wanted in after_aux.items():
		had = before_aux.get(texel)
		if not had or had == wanted:
			continue
		old_aux, new_aux = out_dir / had, out_dir / wanted
		if new_aux.exists():
			if old_aux.exists():
				old_aux.unlink()
				aux_orphans.append(had)
			aux_moves.append((had, wanted, "written by save"))
		elif old_aux.exists():
			old_aux.rename(new_aux)
			aux_moves.append((had, wanted, "renamed"))
	if (out_dir / source_ovl.name).exists() and target != out_dir / source_ovl.name:
		(out_dir / source_ovl.name).unlink()

	return {
		"output": target,
		"stem": f"{src_stem} -> {new_stem}" + (" (unchanged)" if internals_only else ""),
		"renamed_internal_files": len(renamed_files),
		"ovs_companions_moved": len(moved),
		"ovs_written_by_save": len(dropped),
		"aux_renames": aux_moves,
		"stale_aux_removed": aux_orphans,
		"aux_expected": after_aux,
		"checklist": CHECKLIST,
	}


EXPECTED_NOISE = """\
A `KeyError: ''` traceback from DDS.collect / get_aux_data during rename_contents is
EXPECTED and harmless. Texture loaders try to collect pixel data through
`aux_handles['']`, which is not populated on this path. Nothing is lost: only
strings are rewritten, and the aux is carried through verbatim - verified
byte-identical (md5 98c50f89366b81074c45a97ec4b61a3f) across a full Deinosuchus
rename, with 215 of 234 extracted files byte-identical and all 19 differences
explained by name lengthening. Do not "fix" it by touching BaseFormat: that is
shared machinery used by every format."""


def summary_text(report: dict) -> str:
	"""One human-readable block for a log pane or stdout."""
	lines = [
		f"wrote {report['output']}",
		f"  stem                  {report['stem']}",
		f"  internal files hit    {report['renamed_internal_files']}",
		f"  .ovs companions moved {report['ovs_companions_moved']}",
		f"  .ovs written by save  {report['ovs_written_by_save']}",
	]
	for had, wanted, how in report["aux_renames"]:
		lines.append(f"  aux {had} -> {wanted} ({how})")
	for orphan in report["stale_aux_removed"]:
		lines.append(f"  removed stale aux {orphan}")
	lines.append("")
	lines.append(EXPECTED_NOISE)
	return "\n".join(lines)
