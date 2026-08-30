"""Read-only topology-capacity audit for decoded JWE3 motiongraphs."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import gcd
from pathlib import Path
from typing import Any, Iterable

from generated.array import Array
from generated.formats.motiongraph.structs.AllocationArrayPointer import AllocationArrayPointer
from generated.formats.ovl_base.structs.ArrayPointer import ArrayPointer
from generated.formats.ovl_base.structs.MemStruct import MemStruct
from generated.formats.ovl_base.structs.Pointer import Pointer

from .report import analyze_states, build_deref, collect_subtree


def _nested_memstructs(value: Any) -> Iterable[MemStruct]:
    if isinstance(value, MemStruct):
        yield value
    elif isinstance(value, (Array, list, tuple)):
        for item in value:
            yield from _nested_memstructs(item)


def _iter_pointers(loader: Any):
    """Yield each decoded pointer object once, including pointers in array members."""
    seen_owners, seen_pointers = set(), set()
    roots = [loader.header, *loader.context.recursion.values()]
    for root in roots:
        for owner in _nested_memstructs(root):
            if id(owner) in seen_owners:
                continue
            seen_owners.add(id(owner))
            for pointer, field, _arguments in MemStruct.get_instances_recursive(owner, Pointer):
                if id(pointer) in seen_pointers:
                    continue
                seen_pointers.add(id(pointer))
                yield owner, field, pointer


def _used_bytes(pointer: Pointer, data: Any, allocation: bytes) -> int | None:
    if isinstance(data, str):
        terminator = allocation.find(b"\0")
        return terminator + 1 if terminator >= 0 else None
    if isinstance(data, (bytes, bytearray)):
        return len(data)
    size = getattr(data, "io_size", None)
    if size is not None:
        return int(size)
    return None


def _element_size(pointer: Pointer, data: Any, used: int | None) -> int | None:
    if not isinstance(pointer, ArrayPointer) or isinstance(pointer, AllocationArrayPointer):
        return None
    try:
        count = len(data)
    except TypeError:
        return None
    if count and used is not None and used % count == 0:
        return used // count
    template = getattr(pointer, "template", None)
    if template is None:
        return None
    try:
        sample = template(pointer.context, 0, None)
        return int(template.get_size(sample, pointer.context, 0, None))
    except Exception:
        return None


def audit_allocations(loader: Any) -> list[dict]:
    """Measure decoded use versus allocation boundaries for every pointer target."""
    recursion = loader.context.recursion
    by_target: dict[tuple[int, int], dict] = {}
    for owner, field, pointer in _iter_pointers(loader):
        pool = getattr(pointer, "target_pool", None)
        offset = getattr(pointer, "target_offset", None)
        if pool is None or offset is None or offset < 0:
            continue
        allocation_size = pool.size_map.get(offset)
        if allocation_size is None or allocation_size < 0:
            continue
        allocation_size = int(allocation_size)
        target = (int(pool.i), int(offset))
        allocation = pool.data.getvalue()[offset:offset + allocation_size]
        data = getattr(pointer, "data", None)
        if data is None:
            data = recursion.get((pool, offset))
        used = _used_bytes(pointer, data, allocation)
        slack = int(allocation_size - used) if used is not None else None
        if slack is not None and slack < 0:
            slack = None
        element_size = _element_size(pointer, data, used)
        zero_tail = bool(
            slack is not None and slack > 0 and not any(allocation[used:allocation_size])
        )
        reusable_slots = int(
            slack // element_size
            if zero_tail and element_size and isinstance(pointer, ArrayPointer)
            and not isinstance(pointer, AllocationArrayPointer)
            else 0
        )
        kind = (
            "allocation-array" if isinstance(pointer, AllocationArrayPointer)
            else "counted-array" if isinstance(pointer, ArrayPointer)
            else "string" if isinstance(data, str)
            else "struct" if isinstance(data, MemStruct)
            else type(data).__name__
        )
        record = by_target.get(target)
        reference = {
            "owner": type(owner).__name__, "field": str(field),
            "source_pool": int(pointer.src_pool.i) if pointer.src_pool is not None else None,
            "source_offset": int(pointer.io_start),
            "pointer_type": type(pointer).__name__,
        }
        if record is None:
            record = {
                "pool": target[0], "offset": target[1], "kind": kind,
                "target_type": type(data).__name__, "allocation_bytes": int(allocation_size),
                "used_bytes": used, "slack_bytes": slack, "zero_tail": zero_tail,
                "element_size": element_size, "reusable_slots": reusable_slots,
                "element_count": len(data) if isinstance(data, (Array, list, tuple)) else None,
                "string": data if isinstance(data, str) else None,
                "references": [],
            }
            by_target[target] = record
        record["references"].append(reference)
        # A shared target may first be encountered through a recursion-short-circuited
        # pointer. Prefer the record that carries actual decoded usage information.
        if record["used_bytes"] is None and used is not None:
            record.update({
                "kind": kind, "target_type": type(data).__name__, "used_bytes": used,
                "slack_bytes": slack, "zero_tail": zero_tail,
                "element_size": element_size, "reusable_slots": reusable_slots,
                "element_count": len(data) if isinstance(data, (Array, list, tuple)) else None,
                "string": data if isinstance(data, str) else None,
            })
    return sorted(by_target.values(), key=lambda row: (row["pool"], row["offset"]))


def audit_reachability(loader: Any) -> dict:
    """Find graph objects not reached through known state/activity routes."""
    deref, states, state_rows = analyze_states(loader)
    reachable_activities = set()
    for state in states:
        if state is None:
            continue
        for reference in (deref(state.activities) or []):
            for activity, _depth in collect_subtree(
                    deref(reference.activity), deref, reachable_activities):
                reachable_activities.add(id(activity))
    # Transition-local activity lists execute outside a state's main activity
    # tree. They are runtime roots too, as are legacy MRFMember2 transition lists.
    for root in loader.context.recursion.values():
        for value in _nested_memstructs(root):
            if type(value).__name__ not in {
                    "TransitionConditionRecord", "MRFMember2", "Transition"}:
                continue
            references = deref(getattr(value, "activities", None))
            try:
                references = list(references or [])
            except TypeError:
                references = [references] if references is not None else []
            for reference in references:
                activity_pointer = getattr(reference, "activity", None)
                for activity, _depth in collect_subtree(
                        deref(activity_pointer), deref, reachable_activities):
                    reachable_activities.add(id(activity))
    all_activities = {
        id(value): (pool, offset, value)
        for (pool, offset), value in loader.context.recursion.items()
        if type(value).__name__ == "Activity"
    }
    dormant_activities = []
    for identity, (pool, offset, activity) in all_activities.items():
        if identity in reachable_activities:
            continue
        dormant_activities.append({
            "pool": int(pool.i), "offset": int(offset),
            "activity_type": deref(activity.data_type),
            "name": deref(activity.name_b),
        })

    transition_inbound = Counter()
    for row in (row for row in state_rows if row):
        for edge in row["edges"]:
            for target in edge["targets"]:
                if target is not None:
                    transition_inbound[target] += 1
    no_inbound = []
    for row in (row for row in state_rows if row):
        decision_count = sum(row["inbound_decisions"].values())
        transition_count = transition_inbound[row["index"]]
        if not decision_count and not transition_count:
            no_inbound.append({
                "index": row["index"], "label": row["label"],
                "activity_nodes": row["activity_nodes"],
            })
    return {
        "states": sum(state is not None for state in states),
        "state_slots": len(states),
        "null_state_slots": sum(state is None for state in states),
        "activities": len(all_activities),
        "reachable_activities": len(reachable_activities),
        "dormant_activities": dormant_activities,
        "no_known_inbound_states": no_inbound,
    }


def census(loader: Any) -> dict:
    """Count decoded objects by type and record their exact addresses.

    Decoding reaches an object only through a live pointer chain, so an
    allocation that lost its last inbound reference simply stops appearing here.
    Comparing two censuses is therefore the only reliable way to prove an edit
    added what it intended and dropped nothing (see :func:`diff_census`).
    """
    counts: Counter = Counter()
    addresses: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for (pool, offset), value in loader.context.recursion.items():
        name = type(value).__name__
        counts[name] += 1
        # Some registry entries carry no resolved pool index or offset. They are
        # still decoded objects, so count them and mark the address unknown
        # rather than dropping them from the census.
        index = getattr(pool, "i", None)
        addresses[name].add((
            int(index) if index is not None else -1,
            int(offset) if offset is not None else -1,
        ))
    return {
        "motiongraph": loader.name,
        "total": sum(counts.values()),
        "counts": dict(counts),
        "addresses": {name: sorted(values) for name, values in addresses.items()},
    }


def diff_census(before: dict, after: dict) -> dict:
    """Report exactly which decoded objects an edit added and removed."""
    names = sorted(set(before["counts"]) | set(after["counts"]))
    changes = {}
    for name in names:
        old = set(before["addresses"].get(name, ()))
        new = set(after["addresses"].get(name, ()))
        added, removed = sorted(new - old), sorted(old - new)
        if added or removed:
            changes[name] = {
                "before": before["counts"].get(name, 0),
                "after": after["counts"].get(name, 0),
                "added": added,
                "removed": removed,
            }
    return {
        "total_before": before["total"],
        "total_after": after["total"],
        "changed_types": changes,
        "added": sum(len(row["added"]) for row in changes.values()),
        "removed": sum(len(row["removed"]) for row in changes.values()),
    }


def audit_dead_space(allocations: list[dict]) -> dict:
    """Measure bytes inside allocations that no decoded object accounts for.

    ``calc_size_map`` derives an allocation's size from the *next* registered
    pointer target, so a block nothing points at is not a short allocation - it
    is silently absorbed into the end of its predecessor. Non-zero trailing bytes
    are therefore the observable signature of an orphaned block, whether it was
    orphaned by an edit that redirected the last reference away or shipped that
    way. Zero-filled tails are excluded; those are ordinary padding.
    """
    rows = [
        row for row in allocations
        if row["slack_bytes"] and not row["zero_tail"]
    ]
    rows.sort(key=lambda row: (-row["slack_bytes"], row["pool"], row["offset"]))
    return {
        "allocations_with_dead_tail": len(rows),
        "dead_bytes": sum(row["slack_bytes"] for row in rows),
        "rows": rows,
    }


def build_capacity_audit(loader: Any, source: Path | None = None) -> dict:
    allocations = audit_allocations(loader)
    reusable = [row for row in allocations if row["reusable_slots"]]
    zero_slack = [row for row in allocations if row["zero_tail"]]
    strings = [row for row in allocations if row["kind"] == "string"]
    adjacent_candidates = audit_adjacent_growth(allocations)
    dead_space = audit_dead_space(allocations)
    return {
        "format": "cobra-motiongraph-capacity-v1",
        "source": str(source) if source else None,
        "motiongraph": loader.name,
        "summary": {
            "allocations": len(allocations),
            "zero_tail_allocations": len(zero_slack),
            "counted_arrays_with_reusable_slots": len(reusable),
            "reusable_array_slots": sum(row["reusable_slots"] for row in reusable),
            "string_allocations": len(strings),
            "adjacent_growth_candidates": len(adjacent_candidates),
            "allocations_with_dead_tail": dead_space["allocations_with_dead_tail"],
            "dead_bytes": dead_space["dead_bytes"],
        },
        "dead_space": dead_space,
        "reachability": audit_reachability(loader),
        "reusable_arrays": sorted(
            reusable, key=lambda row: (-row["reusable_slots"], -row["slack_bytes"])
        ),
        "zero_tail_allocations": sorted(
            zero_slack, key=lambda row: (-row["slack_bytes"], row["pool"], row["offset"])
        ),
        "adjacent_growth_candidates": adjacent_candidates,
        "allocations": allocations,
    }


def audit_adjacent_growth(allocations: list[dict]) -> list[dict]:
    """Find fixed-pool-size array growth that can consume adjacent zero padding.

    Prefix growth moves only the array target backward; aligned candidates keep
    the engine/tool convention that non-string targets begin at a 16-byte offset.
    Tail growth is currently limited to shifting a following string because it
    has no embedded pointer-source fragments and only requires byte alignment.
    """
    by_pool = defaultdict(list)
    for row in allocations:
        by_pool[row["pool"]].append(row)
    candidates = []
    for pool, rows in by_pool.items():
        rows.sort(key=lambda row: row["offset"])
        for index, array in enumerate(rows):
            stride = array.get("element_size")
            if array.get("kind") != "counted-array" or not stride:
                continue
            if index:
                previous = rows[index - 1]
                aligned_slots = 16 // gcd(int(stride), 16)
                aligned_bytes = aligned_slots * int(stride)
                available = previous.get("slack_bytes", 0)
                if previous.get("zero_tail") and available >= stride:
                    preserves_alignment = available >= aligned_bytes
                    new_slots = aligned_slots if preserves_alignment else 1
                    needed = new_slots * int(stride)
                    first = array["references"][0]
                    candidates.append({
                        "mode": "prefix", "pool": pool, "array_offset": array["offset"],
                        "owner": first["owner"], "field": first["field"],
                        "element_count": array["element_count"], "element_size": stride,
                        "new_slots": new_slots, "bytes": needed,
                        "preserves_16_byte_alignment": preserves_alignment,
                        "neighbor_offset": previous["offset"],
                        "neighbor_kind": previous["kind"],
                        "neighbor_slack": previous["slack_bytes"],
                        "array_references": len(array["references"]),
                        "neighbor_references": len(previous["references"]),
                    })
            if index + 1 < len(rows):
                following = rows[index + 1]
                if (following.get("kind") == "string" and following.get("zero_tail")
                        and following.get("slack_bytes", 0) >= stride):
                    first = array["references"][0]
                    candidates.append({
                        "mode": "tail-shift-string", "pool": pool,
                        "array_offset": array["offset"], "owner": first["owner"],
                        "field": first["field"], "element_count": array["element_count"],
                        "element_size": stride, "new_slots": 1, "bytes": stride,
                        "preserves_16_byte_alignment": True,
                        "neighbor_offset": following["offset"],
                        "neighbor_kind": following["kind"],
                        "neighbor_slack": following["slack_bytes"],
                        "array_references": len(array["references"]),
                        "neighbor_references": len(following["references"]),
                    })
    return sorted(
        candidates,
        key=lambda row: (
            row["bytes"], row["neighbor_references"], row["array_references"],
            row["pool"], row["array_offset"],
        ),
    )


def render_capacity_markdown(audit: dict) -> str:
    summary, reach = audit["summary"], audit["reachability"]
    lines = [
        f"# Motiongraph topology-capacity audit — {audit['motiongraph']}", "",
        f"- source: `{audit.get('source') or '(unknown)'}`",
        f"- pointer-target allocations: {summary['allocations']}",
        f"- zero-filled tail allocations: {summary['zero_tail_allocations']}",
        f"- counted arrays with whole reusable slots: "
        f"{summary['counted_arrays_with_reusable_slots']}",
        f"- total candidate array slots: {summary['reusable_array_slots']}",
        f"- fixed-pool adjacent-relocation candidates: "
        f"{summary['adjacent_growth_candidates']}",
        f"- allocations with a non-zero dead tail: "
        f"{summary['allocations_with_dead_tail']} ({summary['dead_bytes']} bytes)",
        f"- states: {reach['states']} in {reach['state_slots']} slots",
        f"- decoded activities: {reach['activities']}; known state-reachable: "
        f"{reach['reachable_activities']}",
        f"- dormant activity candidates: {len(reach['dormant_activities'])}",
        f"- states with no known decision/transition inbound edge: "
        f"{len(reach['no_known_inbound_states'])}", "",
        "`Reusable` means a counted array has enough verified zero tail bytes for one or more "
        "whole elements. It is a candidate, not yet proof that the engine accepts a larger count.",
        "", "## Candidate counted-array capacity", "",
        "| pool:offset | owner.field | elements | stride | slack | candidate slots | refs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit["reusable_arrays"][:100]:
        first = row["references"][0]
        lines.append(
            f"| {row['pool']}:{row['offset']} | {first['owner']}.{first['field']} | "
            f"{row['element_count']} | {row['element_size']} | {row['slack_bytes']} | "
            f"{row['reusable_slots']} | {len(row['references'])} |"
        )
    if not audit["reusable_arrays"]:
        lines.append("| — | No counted-array tail capacity found | — | — | — | — | — |")
    lines.extend(["", "## Fixed-pool adjacent-relocation candidates", "",
                  "| mode | pool:array | owner.field | elements +slots | stride | neighbor | "
                  "neighbor slack | 16-byte aligned | moved refs |",
                  "|---|---|---|---:|---:|---|---:|---|---:|"])
    for row in audit["adjacent_growth_candidates"][:100]:
        lines.append(
            f"| {row['mode']} | {row['pool']}:{row['array_offset']} | "
            f"{row['owner']}.{row['field']} | {row['element_count']} +{row['new_slots']} | "
            f"{row['element_size']} | {row['neighbor_kind']}@{row['neighbor_offset']} | "
            f"{row['neighbor_slack']} | {row['preserves_16_byte_alignment']} | "
            f"{row['neighbor_references']} |"
        )
    if not audit["adjacent_growth_candidates"]:
        lines.append("| — | — | No adjacent-padding candidate found | — | — | — | — | — | — |")
    lines.extend([
        "", "## Dead tail bytes (orphan signature)", "",
        "An allocation's size comes from the *next* registered pointer target, so a block "
        "nothing points at is absorbed into the end of its predecessor rather than "
        "disappearing. A non-zero tail is therefore the observable signature of an orphan. "
        "It is a candidate for reuse, not proof that the bytes are dead.", "",
        "| pool:offset | target type | allocation | used | dead |",
        "|---|---|---:|---:|---:|",
    ])
    for row in audit["dead_space"]["rows"][:100]:
        lines.append(
            f"| {row['pool']}:{row['offset']} | {row['target_type']} | "
            f"{row['allocation_bytes']} | {row['used_bytes']} | {row['slack_bytes']} |"
        )
    if not audit["dead_space"]["rows"]:
        lines.append("| — | No dead tail bytes found | — | — | — |")
    lines.extend(["", "## Dormant activity candidates", "",
                  "| pool:offset | type | name |", "|---|---|---|"])
    for row in reach["dormant_activities"][:100]:
        lines.append(
            f"| {row['pool']}:{row['offset']} | {row['activity_type']} | "
            f"{row['name'] or ''} |"
        )
    if not reach["dormant_activities"]:
        lines.append("| — | No dormant decoded activities found | — |")
    lines.extend(["", "## No-known-inbound state candidates", "",
                  "| state | derived label | activities |", "|---:|---|---:|"])
    for row in reach["no_known_inbound_states"]:
        lines.append(f"| {row['index']} | {row['label']} | {row['activity_nodes']} |")
    if not reach["no_known_inbound_states"]:
        lines.append("| — | None | — |")
    lines.append("")
    return "\n".join(lines)
