"""Hash-preserving, fixed-width retargeting for JWE3 OVL asset families.

This is intentionally narrower than cobra's ordinary rename machinery.  JWE3 keeps
the same djb2 file hash in several linked tables and expects their ordering to remain
stable.  A replacement prefix with the same length and djb2 state lets us change the
visible names without changing any hash, index, offset, pool size, or archive topology.

The Deinosuchus -> ``sarcmimsaee`` path is game-verified.  Everything else is guarded
by the same invariants but remains tool-verified until tested in game.
"""
from __future__ import annotations

import logging
import os
import shutil
import struct
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generated.formats.ovl import OvlFile
from modules.formats.shared import djb2


if not hasattr(logging, "success"):
	logging.success = logging.info


COMPRESSED_SIZE_OFFSET = 44
JWE3 = "Jurassic World Evolution 3"


@dataclass(frozen=True)
class RetargetReport:
	donor: str
	alias: str
	renamed_loaders: int
	pool_strings: int
	pool_bytes: int
	pool_indices: tuple[int, ...]
	static_compressed_before: int
	static_compressed_after: int
	topology: tuple
	output_files: tuple[Path, ...]


def validate_prefixes(donor: str, alias: str) -> tuple[str, str]:
	"""Validate the fixed-width hash-preserving contract and return lowercase names."""
	try:
		donor.encode("ascii")
		alias.encode("ascii")
	except UnicodeEncodeError as exc:
		raise ValueError("donor and alias must contain ASCII characters only") from exc
	if not donor or not alias:
		raise ValueError("donor and alias must not be empty")
	if donor != donor.lower() or alias != alias.lower():
		raise ValueError("donor and alias must be lowercase")
	if "\x00" in donor or "\x00" in alias:
		raise ValueError("donor and alias must not contain NUL bytes")
	if len(donor) != len(alias):
		raise ValueError(
			f"alias must be exactly {len(donor)} characters (got {len(alias)}); "
			"fixed-width retargeting never moves suffixes or offsets")
	donor_hash = djb2(donor)
	alias_hash = djb2(alias)
	if donor_hash != alias_hash:
		raise ValueError(
			f"alias is not hash-preserving: djb2({donor!r})={donor_hash}, "
			f"djb2({alias!r})={alias_hash}")
	return donor, alias


def find_c_strings(blob: bytes, prefix: bytes):
	"""Yield null-terminated strings that begin at an allocation boundary."""
	start = 0
	while True:
		offset = blob.find(prefix, start)
		if offset < 0:
			return
		start = offset + 1
		if offset and blob[offset - 1] != 0:
			continue
		end = blob.find(b"\x00", offset)
		if end >= 0:
			yield offset, blob[offset:end]


def topology_signature(ovl: OvlFile) -> tuple:
	return (
		len(ovl.loaders),
		tuple(archive.name for archive in ovl.archives),
		tuple(len(archive.content.pools) for archive in ovl.archives),
		tuple(len(archive.content.fragments) for archive in ovl.archives),
	)


def _locate_files_array(raw: bytes, ovl: OvlFile) -> int:
	probe = ovl.files[:min(6, len(ovl.files))].tobytes()
	offset = raw.find(probe)
	if offset < 0:
		raise ValueError("could not locate the FileEntry array in the OVL header")
	if raw.find(probe, offset + 1) >= 0:
		raise ValueError("FileEntry signature is not unique; refusing to guess")
	return offset


def _replacement(text: str, donor: str, alias: str,
				 renamed_basenames: set[str]) -> str | None:
	low = text.lower()
	if not low.startswith(donor):
		return None
	rest = text[len(donor):]
	replacement = alias.capitalize() if text[:1].isupper() else alias
	if rest[:1] == "$" or low in renamed_basenames:
		return replacement + rest
	# Other matching strings are normally external audio events, not local assets.
	return None


def patch_ovl_bytes(raw: bytes, ovl: OvlFile, donor: str, alias: str):
	"""Patch a loaded OVL and return ``(new_bytes, details)``.

	Only the STATIC pool strings and fixed-width basename bytes are changed.  File
	hashes and every hash-bearing/index-bearing table remain byte-identical.
	"""
	donor, alias = validate_prefixes(donor, alias)
	static = next((archive for archive in ovl.archives if archive.name == "STATIC"), None)
	if static is None:
		raise ValueError("archive has no STATIC section")
	topology = topology_signature(ovl)

	renames: dict[str, str] = {}
	for name in ovl.loaders:
		if name.lower().startswith(donor):
			renames[name] = alias + name[len(donor):]
	if not renames:
		raise ValueError(f"archive contains no loader beginning with {donor!r}")

	# Every renamed filename must retain its exact existing hash.
	files_by_name = dict(zip(ovl.files_name, ovl.files))
	for old, new in renames.items():
		old_record = files_by_name[old]
		old_hash = int(old_record["file_hash"])
		new_base = new.rsplit(".", 1)[0]
		if djb2(new_base) != old_hash:
			raise ValueError(
				f"{old!r} -> {new!r} changes its FileEntry hash "
				f"({old_hash} -> {djb2(new_base)})")

	renamed_basenames = {
		name.rsplit(".", 1)[0].lower() for name in renames
	}
	static_pool_ids = {id(pool) for pool in static.content.pools}
	pool_edits = defaultdict(list)
	for pool_index, pool in enumerate(ovl.pools):
		if id(pool) not in static_pool_ids:
			continue
		blob = pool.data.getvalue()
		seen_offsets = set()
		for spelling in (donor, donor.capitalize()):
			for offset, old_bytes in find_c_strings(blob, spelling.encode("ascii")):
				if offset in seen_offsets:
					continue
				old_text = old_bytes.decode("ascii")
				new_text = _replacement(old_text, donor, alias, renamed_basenames)
				if new_text is None:
					continue
				new_bytes = new_text.encode("ascii")
				if len(new_bytes) != len(old_bytes):
					raise ValueError(
						f"pool string {old_text!r} is not a fixed-width replacement")
				pool_edits[pool_index].append((offset, old_bytes, new_bytes))
				seen_offsets.add(offset)

	for pool_index, edits in pool_edits.items():
		pool = ovl.pools[pool_index]
		for offset, old_bytes, new_bytes in edits:
			actual = pool.data.getvalue()[offset:offset + len(old_bytes)]
			if actual != old_bytes:
				raise ValueError(f"pool {pool_index}:{offset} changed during retarget")
			pool.data.seek(offset)
			pool.data.write(new_bytes)
		pool.data.seek(0)

	static.content.write_pools()
	uncompressed = static.content.write_archive()
	if len(uncompressed) != static.uncompressed_size:
		raise ValueError(
			f"STATIC grew from {static.uncompressed_size} to {len(uncompressed)} bytes")
	_, compressed_size, compressed = static.content.compress(uncompressed, True)
	header_size = len(raw) - static.compressed_size
	if header_size < 0 or raw[header_size:] == b"":
		raise ValueError("could not locate the trailing STATIC payload")
	out = bytearray(raw[:header_size])
	out.extend(compressed)
	struct.pack_into("<I", out, static.io_start + COMPRESSED_SIZE_OFFSET, compressed_size)

	# Fixed-width basename overwrite.  Duplicate FileEntries may share an offset.
	names_start = ovl.names.io_start
	names_size = ovl.names.io_size
	names_blob = bytearray(raw[names_start:names_start + names_size])
	files_at = _locate_files_array(raw, ovl)
	unique_name_edits = {}
	for index, record in enumerate(ovl.files):
		offset = int(record["basename"])
		end = names_blob.find(b"\x00", offset)
		old_bytes = bytes(names_blob[offset:end])
		old_name = old_bytes.decode("ascii")
		full_name = ovl.files_name[index]
		if full_name not in renames:
			continue
		new_bytes = alias.encode("ascii") + old_bytes[len(donor):]
		if len(new_bytes) != len(old_bytes):
			raise ValueError(f"basename {old_name!r} cannot be replaced in place")
		previous = unique_name_edits.setdefault(offset, (old_bytes, new_bytes))
		if previous != (old_bytes, new_bytes):
			raise ValueError(f"conflicting basename edits at offset {offset}")

	for offset, (old_bytes, new_bytes) in unique_name_edits.items():
		actual = names_blob[offset:offset + len(old_bytes)]
		if actual != old_bytes:
			raise ValueError(f"name buffer changed at offset {offset}")
		names_blob[offset:offset + len(old_bytes)] = new_bytes
	out[names_start:names_start + names_size] = names_blob

	# The FileEntry records themselves must remain byte-identical.
	files_size = ovl.files.nbytes
	if out[files_at:files_at + files_size] != raw[files_at:files_at + files_size]:
		raise ValueError("FileEntry array changed during a hash-preserving retarget")

	details = {
		"renames": renames,
		"pool_strings": sum(len(edits) for edits in pool_edits.values()),
		"pool_bytes": sum(
			sum(old != new for old, new in zip(old_bytes, new_bytes))
			for edits in pool_edits.values()
			for _, old_bytes, new_bytes in edits),
		"pool_indices": tuple(sorted(pool_edits)),
		"static_before": int(static.compressed_size),
		"static_after": int(compressed_size),
		"topology": topology,
	}
	return bytes(out), details


def _load(path: Path, game: str, keep_aux_open: bool = False) -> OvlFile:
	ovl = OvlFile()
	commands = {"game": game}
	if keep_aux_open:
		commands["update_aux"] = False
	ovl.load(str(path), commands)
	return ovl


def _close_aux(ovl: OvlFile) -> None:
	for loader in ovl.loaders.values():
		loader.close_aux_handles()


def _aux_copy_map(ovl: OvlFile, target_basename: str) -> dict[Path, str]:
	"""Map referenced source AUX paths to names required by the target OVL basename."""
	result = {}
	original_basename = ovl.basename
	try:
		for loader in ovl.loaders.values():
			for suffix, handle in loader.aux_handles.items():
				source_name = getattr(handle, "name", None)
				if not isinstance(source_name, str):
					raise ValueError(f"missing AUX file for {loader.name!r}")
				source_path = Path(source_name)
				ovl.basename = target_basename
				target_name = loader.get_aux_name(suffix, source_path.stat().st_size)
				ovl.basename = original_basename
				previous = result.setdefault(source_path, target_name)
				if previous != target_name:
					raise ValueError(f"AUX {source_path.name} maps to multiple target names")
	finally:
		ovl.basename = original_basename
	return result


def _companion_map(source: Path, target_stem: str) -> dict[Path, str]:
	prefix = (source.stem + ".ovs.").lower()
	result = {}
	for path in source.parent.iterdir():
		if path.is_file() and path.name.lower().startswith(prefix):
			result[path] = target_stem + path.name[len(source.stem):]
	return result


def retarget_family(source: str | os.PathLike, output: str | os.PathLike,
					donor: str, alias: str, game: str = JWE3,
					force: bool = False) -> RetargetReport:
	"""Build, verify, and publish a complete retargeted OVL/OVS/AUX family."""
	donor, alias = validate_prefixes(donor, alias)
	source = Path(source).resolve()
	output = Path(output).resolve()
	if not source.is_file():
		raise FileNotFoundError(source)
	if source.suffix.lower() != ".ovl" or output.suffix.lower() != ".ovl":
		raise ValueError("source and output must both be .ovl files")
	if source == output:
		raise ValueError("output must not overwrite the donor OVL")
	if game != JWE3:
		raise ValueError("hash-preserving family retargeting is only game-verified for JWE3")

	output.parent.mkdir(parents=True, exist_ok=True)
	ovl = _load(source, game, keep_aux_open=True)
	raw = source.read_bytes()
	original_file_records = ovl.files.tobytes()
	original_dependencies = ovl.dependencies.tobytes()
	original_aux_entries = ovl.aux_entries.tobytes()
	patched, details = patch_ovl_bytes(raw, ovl, donor, alias)
	companions = _companion_map(source, output.stem)
	aux_files = _aux_copy_map(ovl, output.stem.lower())
	_close_aux(ovl)

	if not companions:
		raise ValueError(f"no .ovs companions found beside {source.name}")
	name_to_source = {output.name: source}
	for path, name in companions.items():
		if name in name_to_source and name_to_source[name] != path:
			raise ValueError(f"multiple source files map to {name}")
		name_to_source[name] = path
	for path, name in aux_files.items():
		if name in name_to_source and name_to_source[name] != path:
			raise ValueError(f"multiple source files map to {name}")
		name_to_source[name] = path

	final_paths = tuple(output.parent / name for name in sorted(name_to_source))
	for path in final_paths:
		if path.exists() and not force:
			raise FileExistsError(f"output family file exists: {path} (use --force)")

	with tempfile.TemporaryDirectory(prefix=".retarget-", dir=output.parent) as temp_dir:
		temp = Path(temp_dir)
		stage_ovl = temp / output.name
		stage_ovl.write_bytes(patched)
		for target_name, source_path in name_to_source.items():
			if target_name == output.name:
				continue
			shutil.copy2(source_path, temp / target_name)

		check = _load(stage_ovl, game)
		expected_names = {
			alias + name[len(donor):] if name.lower().startswith(donor) else name
			for name in ovl.loaders
		}
		if set(check.loaders) != expected_names:
			missing = sorted(expected_names - set(check.loaders))[:5]
			extra = sorted(set(check.loaders) - expected_names)[:5]
			raise ValueError(f"loader verification failed; missing={missing}, extra={extra}")
		if topology_signature(check) != details["topology"]:
			raise ValueError("archive topology changed during retarget")
		if check.files.tobytes() != original_file_records:
			raise ValueError("FileEntry records changed during retarget")
		if check.dependencies.tobytes() != original_dependencies:
			raise ValueError("dependency records changed during retarget")
		if check.aux_entries.tobytes() != original_aux_entries:
			raise ValueError("AUX entry records changed during retarget")

		for target_name, source_path in name_to_source.items():
			if target_name == output.name:
				continue
			if (temp / target_name).read_bytes() != source_path.read_bytes():
				raise ValueError(f"companion copy changed bytes: {target_name}")

		# Verification is complete before a single destination file is replaced.
		for path in final_paths:
			os.replace(temp / path.name, path)

	return RetargetReport(
		donor=donor,
		alias=alias,
		renamed_loaders=len(details["renames"]),
		pool_strings=details["pool_strings"],
		pool_bytes=details["pool_bytes"],
		pool_indices=details["pool_indices"],
		static_compressed_before=details["static_before"],
		static_compressed_after=details["static_after"],
		topology=details["topology"],
		output_files=final_paths,
	)
