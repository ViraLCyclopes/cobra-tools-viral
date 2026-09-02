"""Diff two OVLs' STATIC pools into a replayable motiongraph patch plan.

Motiongraph edits cannot be recovered from an extracted `.motiongraph`: that file
is faithful XML (129,010 elements for a JWE3 land graph, and it does reflect edits)
but it is SEMANTIC - elements carry `pool_type="3"` and never a pool index or
offset - while every edit is addressed as pool:offset. `MotiongraphLoader.create()`
raises NotImplementedError, and the rebuilt-archive route is rejected by the game.

So recovery works the other way round: record what CHANGED, as bytes, and replay it
onto a fresh OVL with `apply_patch_plan`. That is the same mechanism that set 151
speed values back to 1.0, and it preserves every byte it does not name.

What this CANNOT capture, and says so loudly:

* fragment retargets - repointing a pointer changes the fragment table, not pool
  bytes, so `rename_repair` style fixes are invisible here
* topology growth - a grown pool changes sizes, and `apply_patch_plan` refuses any
  width change

Both are reported in the plan's `unreplayable` block so a recovery is never
silently partial. Those operations are scripted, so re-running the script is their
recovery path.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .edit import DEFAULT_GAME
from .surgical_growth import _load_quiet

PLAN_FORMAT = "cobra-motiongraph-patch-v1"
# merge two changed runs separated by fewer than this many identical bytes; each
# edit carries its own expected/replacement, so fewer, longer runs is cheaper
MERGE_GAP = 8


@dataclass
class DiffReport:
	plan: dict
	changed_pools: int = 0
	changed_bytes: int = 0
	edits: int = 0
	resized_pools: tuple = ()
	fragment_delta: int = 0
	fragments_retargeted: int = 0
	warnings: list = field(default_factory=list)


def _runs(old: bytes, new: bytes, merge_gap: int = MERGE_GAP):
	"""Yield (offset, length) spans covering every difference between two blobs."""
	if len(old) != len(new):
		raise ValueError("Blobs must be the same length to diff by run")
	spans = []
	start = None
	gap = 0
	for i, (a, b) in enumerate(zip(old, new)):
		if a != b:
			if start is None:
				start = i
			gap = 0
		elif start is not None:
			gap += 1
			if gap >= merge_gap:
				spans.append((start, i - gap + 1 - start))
				start, gap = None, 0
	if start is not None:
		spans.append((start, len(old) - gap - start))
	return spans


def diff_motiongraph(stock: Path, edited: Path, game: str = DEFAULT_GAME) -> DiffReport:
	"""Return a patch plan that turns `stock` into `edited`, plus what it cannot carry."""
	stock, edited = Path(stock).resolve(), Path(edited).resolve()
	stock_ovl, stock_static = _load_quiet(stock, game)
	edit_ovl, edit_static = _load_quiet(edited, game)

	stock_pools, edit_pools = stock_static.content.pools, edit_static.content.pools
	warnings = []
	if len(stock_pools) != len(edit_pools):
		raise ValueError(
			f"STATIC pool counts differ ({len(stock_pools)} vs {len(edit_pools)}); "
			f"these builds are not a value-edit apart")

	# global index lookup, because apply_patch_plan addresses pools globally
	stock_global = {id(pool): i for i, pool in enumerate(stock_ovl.pools)}

	edits = []
	changed_pools = changed_bytes = 0
	resized = []
	for local, (a, b) in enumerate(zip(stock_pools, edit_pools)):
		old, new = a.data.getvalue(), b.data.getvalue()
		if len(old) != len(new):
			resized.append((stock_global.get(id(a), -1), len(old), len(new)))
			continue
		if old == new:
			continue
		changed_pools += 1
		gi = stock_global.get(id(a), -1)
		for offset, length in _runs(old, new):
			changed_bytes += sum(
				1 for x, y in zip(old[offset:offset + length], new[offset:offset + length])
				if x != y)
			edits.append({
				"pool": gi, "offset": offset,
				"expected": old[offset:offset + length].hex(),
				"replacement": new[offset:offset + length].hex(),
				"note": f"STATIC pool {gi} +{offset} ({length} bytes)",
			})

	if resized:
		warnings.append(
			f"{len(resized)} pool(s) changed size - topology growth cannot be replayed "
			f"as a patch plan; re-run the growth script instead: {resized}")

	stock_frag, edit_frag = stock_static.content.fragments, edit_static.content.fragments
	fragment_delta = len(edit_frag) - len(stock_frag)
	retargeted = 0
	if fragment_delta:
		warnings.append(
			f"fragment count differs by {fragment_delta:+d} - new or removed pointers "
			f"cannot be replayed as a patch plan")
	else:
		# Compare by LINK SITE, never positionally. The fragment array is kept sorted
		# on (link_pool, struct_pool, link_offset, struct_offset), so retargeting a
		# pointer moves its row: a row-by-row comparison of 79 real retargets reported
		# 3,938 differences purely from the reordering.
		def by_site(rows):
			return {(int(r["link_pool"]), int(r["link_offset"])):
					(int(r["struct_pool"]), int(r["struct_offset"])) for r in rows}

		old_sites, new_sites = by_site(stock_frag), by_site(edit_frag)
		moved = [site for site, target in new_sites.items()
				 if old_sites.get(site, target) != target]
		appeared = set(new_sites) - set(old_sites)
		vanished = set(old_sites) - set(new_sites)
		retargeted = len(moved)
		if appeared or vanished:
			warnings.append(
				f"{len(appeared)} fragment link site(s) appeared and {len(vanished)} "
				f"vanished; the plan cannot carry either")
		if retargeted:
			warnings.append(
				f"{retargeted} fragment(s) point somewhere different - pointer "
				f"retargets live in the fragment table, not in pool bytes, so they "
				f"are NOT in this plan (re-run rename_repair or the clone script)")

	motiongraph = next(
		(str(loader.name) for loader in stock_ovl.loaders.values()
		 if str(loader.name).endswith(".motiongraph")), None)

	plan = {
		"format": PLAN_FORMAT,
		"source": str(stock),
		"source_sha256": hashlib.sha256(stock.read_bytes()).hexdigest(),
		"motiongraph": motiongraph,
		"field": None, "kind": "byte-diff", "value": None,
		"flags": None, "enum": None,
		"derived_from": str(edited),
		"derived_sha256": hashlib.sha256(edited.read_bytes()).hexdigest(),
		"unreplayable": {
			"resized_pools": resized,
			"fragment_delta": fragment_delta,
			"fragments_retargeted": retargeted,
			"warnings": warnings,
		},
		"edits": edits,
	}
	return DiffReport(
		plan=plan, changed_pools=changed_pools, changed_bytes=changed_bytes,
		edits=len(edits), resized_pools=tuple(resized),
		fragment_delta=fragment_delta, fragments_retargeted=retargeted,
		warnings=warnings)


def write_plan(report: DiffReport, path: Path) -> Path:
	path = Path(path)
	path.write_text(json.dumps(report.plan, indent=1), encoding="utf-8")
	return path
