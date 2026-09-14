"""Safe, fixed-topology motiongraph inspection and patching.

JWE3 validates the existing STATIC pool layout.  Consequently this module only
supports value edits whose encoded width is unchanged.  It never rebuilds OVL
arrays, fragments, pools, or motiongraph allocations.
"""

from __future__ import annotations

import json
import hashlib
import logging
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from generated.base_enum import BaseEnum
from generated.bitfield import BasicBitfield, BitfieldMember
from generated.formats.motiongraph.imports import name_type_map
from generated.formats.motiongraph.structs.Activity import Activity
from generated.formats.ovl import OvlFile
from generated.formats.ovl_base.structs.MemStruct import MemStruct
from .staging import CandidatePublication, validate_staged_pair


if not hasattr(logging, "success"):
    logging.success = logging.info  # type: ignore[attr-defined]


DEFAULT_GAME = "Jurassic World Evolution 3"
COMPRESSED_SIZE_OFFSET = 44
CURVE_BIAS = 2.0
CURVE_FIELDS = {"y", "subsequent_curve_param", "subsequent_curve_param_b"}
CURVE_OWNER = "CurveDataPoint"
CURVE_TYPE = "CurveValue"
ARRAY_TYPES = ("ArrayPointer", "AllocationArrayPointer")
STORAGE_FORMATS = {1: ("<B", 1), 2: ("<H", 2), 4: ("<I", 4), 8: ("<Q", 8)}
SCALAR_FORMATS = {
    "Float": ("<f", 4),
    "Uint": ("<I", 4),
    "Int": ("<i", 4),
    "Uint64": ("<Q", 8),
    "Int64": ("<q", 8),
    "Ushort": ("<H", 2),
    "Short": ("<h", 2),
    "Ubyte": ("<B", 1),
    "Byte": ("<b", 1),
}
TYPE_NAMES = {id(value): name for name, value in name_type_map.items()}
LABEL_FIELDS = ("mani", "activity_name", "variable", "name", "ds_name")


@dataclass(frozen=True)
class PatchReport:
    output: Path
    edits: int
    pools: int
    changed_bytes: int
    static_pools: int
    static_fragments: int
    uncompressed_size: int
    compressed_before: int
    compressed_after: int


def curve_decode(raw_u16: int) -> float:
    """Decode Frontier's biased bfloat16 curve representation."""
    return struct.unpack("<f", struct.pack("<I", (raw_u16 & 0xFFFF) << 16))[0] - CURVE_BIAS


def curve_encode(value: float) -> int:
    """Encode a curve value, truncating exactly as the shipped files do."""
    bits = struct.unpack("<I", struct.pack("<f", float(value) + CURVE_BIAS))[0]
    return (bits >> 16) & 0xFFFF


def _is_bitfield(field_type: type) -> bool:
    return isinstance(field_type, type) and issubclass(field_type, BasicBitfield)


def _is_enum(field_type: type) -> bool:
    return isinstance(field_type, type) and issubclass(field_type, BaseEnum)


def _bitfield_members(field_type: type) -> dict[str, int]:
    result = {}
    for cls in reversed(field_type.__mro__):
        for name, member in vars(cls).items():
            if not name.startswith("_") and isinstance(member, BitfieldMember):
                result[name] = int(member.mask)
    return result


def _enum_options(field_type: type) -> dict[str, int]:
    return {member.name: int(member.value) for member in field_type}


def _type_name(field_type: type) -> str:
    return TYPE_NAMES.get(id(field_type), field_type.__name__)


def _type_size(field_type: type, instance: Any, context: Any, arg: Any, template: Any) -> int:
    try:
        return field_type.get_size(instance, context, arg, template)
    except TypeError:
        return field_type.get_size(instance, context)


def _walk_array(pointer: Any, context: Any, path: str, field_types: dict[str, type],
                stride_mismatches: dict[str, tuple[int, int]], depth: int,
                max_depth: int):
    if depth >= max_depth:
        return
    pool = getattr(pointer, "target_pool", None)
    base = getattr(pointer, "target_offset", None)
    elements = getattr(pointer, "data", None)
    if pool is None or base is None or base < 0 or not elements:
        return
    try:
        count = len(elements)
    except TypeError:
        return
    allocation_size = pool.size_map.get(base)
    if count <= 0 or allocation_size is None:
        return
    stride, remainder = divmod(allocation_size, count)
    if remainder:
        return
    first = elements[0]
    if not isinstance(first, MemStruct):
        return
    try:
        declared = _type_size(type(first), first, context, 0, None)
    except Exception:
        declared = None
    if declared is not None and declared != stride:
        stride_mismatches[type(first).__name__] = (declared, stride)
        return
    for index, element in enumerate(elements):
        if isinstance(element, MemStruct):
            yield from _walk_fields(
                element, context, pool, field_types, stride_mismatches,
                base + index * stride, f"{path}[{index}].", depth + 1, max_depth,
            )


def _walk_fields(instance: Any, context: Any, pool: Any,
                 field_types: dict[str, type],
                 stride_mismatches: dict[str, tuple[int, int]], base: int = 0,
                 prefix: str = "", depth: int = 0, max_depth: int = 8):
    offset = base
    attributes = type(instance)._get_filtered_attribute_list(instance, include_abstract=False)
    for name, field_type, args, _ in attributes:
        value = getattr(instance, name, None)
        arg, template = args[0], args[1] if len(args) > 1 else None
        try:
            size = _type_size(field_type, value, context, arg, template)
        except Exception:
            return
        path = f"{prefix}{name}"
        type_name = _type_name(field_type)
        if (type(instance).__name__ == CURVE_OWNER and name in CURVE_FIELDS
                and type_name in SCALAR_FORMATS):
            yield path, CURVE_TYPE, pool, offset, size, value
        elif type_name in SCALAR_FORMATS:
            yield path, type_name, pool, offset, size, value
        elif _is_bitfield(field_type) or _is_enum(field_type):
            field_types[type_name] = field_type
            yield path, type_name, pool, offset, size, value
        elif type_name in ARRAY_TYPES:
            yield from _walk_array(
                value, context, path, field_types, stride_mismatches, depth, max_depth,
            )
        elif isinstance(value, MemStruct) and type_name != "Pointer":
            yield from _walk_fields(
                value, context, pool, field_types, stride_mismatches,
                offset, f"{path}.", depth, max_depth,
            )
        offset += size


def load_motiongraph(ovl_path: Path, name: str | None = None,
                     game: str = DEFAULT_GAME):
    ovl = OvlFile()
    ovl.load(str(ovl_path), {"game": game})
    if name:
        loader = ovl.loaders.get(name.lower()) or ovl.loaders.get(name)
        if loader is None:
            matches = [loader for loader in ovl.loaders.values()
                       if loader.name.lower() == name.lower()]
            loader = matches[0] if matches else None
        if loader is None or not loader.name.lower().endswith(".motiongraph"):
            raise ValueError(f"Motiongraph not found: {name}")
    else:
        matches = [loader for loader in ovl.loaders.values()
                   if loader.name.lower().endswith(".motiongraph")]
        if len(matches) != 1:
            names = ", ".join(sorted(loader.name for loader in matches)) or "none"
            raise ValueError(
                f"Expected exactly one motiongraph, found {len(matches)}: {names}; use --name"
            )
        loader = matches[0]
    return ovl, loader


def _iter_activities(loader: Any) -> Iterable[tuple[Any, int, Activity]]:
    seen = set()
    for (pool, offset), value in loader.context.recursion.items():
        if isinstance(value, Activity) and id(value) not in seen:
            seen.add(id(value))
            yield pool, int(offset), value


def _activity_name(activity: Activity) -> str | None:
    for attr in ("name_b", "name"):
        value = getattr(getattr(activity, attr, None), "data", None)
        if value:
            return value
    payload = getattr(getattr(activity, "data", None), "data", None)
    for attr in LABEL_FIELDS:
        value = getattr(getattr(payload, attr, None), "data", None)
        if value:
            return value
    return None


def locate_fields(ovl_path: Path, name: str | None = None,
                  game: str = DEFAULT_GAME, activity: str | None = None,
                  activity_type: str | None = None, field: str | None = None,
                  activity_pool: int | None = None,
                  activity_offset: int | None = None,
                  activity_addresses: Iterable[tuple[int, int]] | None = None):
    """Return verified, concrete addresses for matching activity payload fields."""
    if (activity_pool is None) != (activity_offset is None):
        raise ValueError("Exact activity selection requires both pool and offset")
    if activity_pool is not None and activity_addresses is not None:
        raise ValueError("Choose either one exact activity or an activity-address set")
    address_filter = (
        {(int(activity_pool), int(activity_offset))}
        if activity_pool is not None else
        ({(int(pool), int(offset)) for pool, offset in activity_addresses}
         if activity_addresses is not None else None)
    )
    ovl_path = Path(ovl_path).resolve()
    source_hash = hashlib.sha256(ovl_path.read_bytes()).hexdigest()
    _, loader = load_motiongraph(ovl_path, name, game)
    if hashlib.sha256(ovl_path.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Source OVL changed while fields were being scanned; reload it")
    provenance = {"source": str(ovl_path), "source_sha256": source_hash,
                  "motiongraph": loader.name}
    context = loader.context
    field_types: dict[str, type] = {}
    stride_mismatches: dict[str, tuple[int, int]] = {}
    rows = []
    for item_pool, item_offset, item in _iter_activities(loader):
        if address_filter is not None and (int(item_pool.i), item_offset) not in address_filter:
            continue
        label = _activity_name(item)
        if activity and (label is None or activity.lower() not in label.lower()):
            continue
        item_type = item.data_type.data
        if activity_type and item_type != activity_type:
            continue
        pointer = item.data
        payload = getattr(pointer, "data", None)
        pool = getattr(pointer, "target_pool", None)
        offset = getattr(pointer, "target_offset", None)
        if not isinstance(payload, MemStruct) or pool is None or offset is None:
            continue
        matches = []
        for path, type_name, field_pool, absolute, size, value in _walk_fields(
                payload, context, pool, field_types, stride_mismatches, offset):
            field_type = field_types.get(type_name)
            if type_name != CURVE_TYPE and type_name not in SCALAR_FORMATS and field_type is None:
                continue
            if field and path != field:
                continue
            if type_name == CURVE_TYPE:
                kind, fmt, width = "curve", "<H", 2
            elif type_name in SCALAR_FORMATS:
                kind = "scalar"
                fmt, width = SCALAR_FORMATS[type_name]
            else:
                kind = "bitfield" if _is_bitfield(field_type) else "enum"
                if size not in STORAGE_FORMATS:
                    continue
                fmt, width = STORAGE_FORMATS[size]
            raw = field_pool.data.getvalue()[absolute:absolute + width]
            if len(raw) != width:
                continue
            decoded = struct.unpack(fmt, raw)[0]
            if kind == "curve":
                verified = decoded == (int(value) & 0xFFFF)
            elif type_name == "Float":
                verified = decoded == value or abs(decoded - value) < 1e-6
            else:
                try:
                    verified = decoded == int(value)
                except (TypeError, ValueError):
                    verified = True
            entry = {
                "path": path, "type": type_name, "kind": kind,
                "pool": int(field_pool.i), "offset": int(absolute),
                "hex": raw.hex(), "value": decoded, "verified": bool(verified),
            }
            if kind == "curve":
                entry["curve_value"] = curve_decode(decoded)
            elif kind == "bitfield":
                entry["members"] = _bitfield_members(field_type)
            elif kind == "enum":
                entry["options"] = _enum_options(field_type)
            matches.append(entry)
        if matches:
            rows.append({
                "activity": label, "activity_type": item_type,
                "pool": int(pool.i), "payload_offset": int(offset),
                "alloc_size": pool.size_map.get(offset), "fields": matches,
                "provenance": dict(provenance),
            })
    return rows, stride_mismatches


def build_patch_plan(rows: list[dict], source: Path, motiongraph: str | None,
                     field: str, value: float | int | None = None,
                     flag_ops: list[str] | None = None,
                     enum_name: str | None = None) -> dict:
    """Build a self-verifying, same-width patch plan from located rows."""
    source = Path(source).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = [row.get("provenance") for row in rows]
    if not provenance or any(not item for item in provenance):
        raise ValueError("Field rows have unknown source provenance; scan the loaded source again")
    for item in provenance:
        if Path(item.get("source", "")).resolve() != source:
            raise ValueError("Field rows belong to a different source OVL; scan again")
        if str(item.get("source_sha256", "")).lower() != source_hash.lower():
            raise ValueError("Field rows are stale because the source OVL bytes changed; scan again")
    resolved_names = {str(item.get("motiongraph") or "") for item in provenance}
    if len(resolved_names) != 1 or not next(iter(resolved_names)):
        raise ValueError("Field rows contain mixed or missing motiongraph provenance; scan again")
    resolved_name = next(iter(resolved_names))
    if motiongraph is None:
        motiongraph = resolved_name
    elif resolved_name.lower() != str(motiongraph).lower():
            raise ValueError("Field rows belong to a different motiongraph; scan again")
    fields = [(row, item) for row in rows for item in row["fields"]]
    if not fields:
        raise ValueError("No matching fields")
    if any(not item["verified"] for _, item in fields):
        raise ValueError("Refusing to plan: at least one field address failed verification")
    kinds = {item["kind"] for _, item in fields}
    types = {item["type"] for _, item in fields}
    if len(kinds) != 1 or len(types) != 1:
        raise ValueError(f"Matched mixed field layouts: kinds={sorted(kinds)}, types={sorted(types)}")
    kind = next(iter(kinds))
    if sum(option is not None for option in (value, flag_ops, enum_name)) != 1:
        raise ValueError("Choose exactly one of value, flag operations, or enum name")

    parsed_flags: dict[int, bool] = {}
    enum_value = None
    sample = fields[0][1]
    if flag_ops is not None:
        if kind != "bitfield":
            raise ValueError(f"Flag operations require a bitfield, not {kind}")
        members = sample["members"]
        for operation in flag_ops:
            name, separator, state = operation.partition("=")
            if not separator or name not in members or state not in ("0", "1"):
                raise ValueError(
                    f"Invalid flag operation {operation!r}; expected NAME=0|1, "
                    f"available: {', '.join(sorted(members))}"
                )
            parsed_flags[members[name]] = state == "1"
    elif enum_name is not None:
        if kind != "enum":
            raise ValueError(f"Enum selection requires an enum, not {kind}")
        options = sample["options"]
        match = next((name for name in options if name.lower() == enum_name.lower()), None)
        if match is None:
            raise ValueError(f"Unknown enum {enum_name!r}; available: {', '.join(options)}")
        enum_value = options[match]
    elif kind not in ("scalar", "curve"):
        raise ValueError(f"Numeric value is not valid for {kind}")

    edits, seen = [], set()
    for row, item in fields:
        if kind == "curve":
            fmt, replacement_value = "<H", curve_encode(float(value))
        elif kind == "scalar":
            fmt, _ = SCALAR_FORMATS[item["type"]]
            replacement_value = float(value) if item["type"] == "Float" else int(value)
        elif kind == "bitfield":
            fmt, _ = STORAGE_FORMATS[len(bytes.fromhex(item["hex"]))]
            replacement_value = int(item["value"])
            for mask, enabled in parsed_flags.items():
                replacement_value = (replacement_value | mask) if enabled else (replacement_value & ~mask)
        else:
            fmt, _ = STORAGE_FORMATS[len(bytes.fromhex(item["hex"]))]
            replacement_value = enum_value
        try:
            replacement = struct.pack(fmt, replacement_value).hex()
        except (struct.error, OverflowError, ValueError) as exc:
            raise ValueError(f"Value {replacement_value!r} does not fit {item['type']}") from exc
        key = (item["pool"], item["offset"])
        if replacement == item["hex"] or key in seen:
            continue
        seen.add(key)
        edits.append({
            "pool": item["pool"], "offset": item["offset"],
            "expected": item["hex"], "replacement": replacement,
            "note": f"{row['activity']}.{item['path']}",
        })
    if not edits:
        raise ValueError("All matching fields already have the requested value")
    return {
        "format": "cobra-motiongraph-patch-v1", "source": str(source),
        "source_sha256": source_hash,
        "motiongraph": motiongraph, "field": field, "kind": kind,
        "value": value, "flags": flag_ops, "enum": enum_name, "edits": edits,
    }


def save_plan(path: Path, plan: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def _plan_operations(plan: dict) -> list[dict]:
    operations = plan.get("operations")
    if operations:
        return [dict(operation) for operation in operations]
    return [{
        "field": plan.get("field"), "kind": plan.get("kind"),
        "value": plan.get("value"), "flags": plan.get("flags"),
        "enum": plan.get("enum"), "edit_count": len(plan.get("edits") or []),
    }]


def merge_patch_plans(plans: Iterable[dict]) -> dict:
    """Combine independently built edits into one safe, atomic patch plan.

    Exact duplicate edits are collapsed. Any incompatible source identity,
    motiongraph, or overlapping byte range is rejected before staging.
    """
    plans = list(plans)
    if not plans:
        raise ValueError("Choose at least one patch plan to merge")
    first = plans[0]
    expected_format = "cobra-motiongraph-patch-v1"
    source_hash = str(first.get("source_sha256") or "").lower()
    motiongraph = first.get("motiongraph")
    edits: list[dict] = []
    operations: list[dict] = []
    occupied: dict[tuple[int, int], int] = {}
    exact: dict[tuple[int, int, int], tuple[str, str]] = {}

    for plan in plans:
        if plan.get("format") not in (None, expected_format):
            raise ValueError(f"Unsupported patch-plan format: {plan.get('format')!r}")
        candidate_hash = str(plan.get("source_sha256") or "").lower()
        if candidate_hash != source_hash:
            raise ValueError("Patch plans were built from different source OVL bytes")
        candidate_motiongraph = plan.get("motiongraph")
        if str(candidate_motiongraph or "").lower() != str(motiongraph or "").lower():
            raise ValueError("Patch plans target different motiongraphs")
        plan_edits = plan.get("edits") or []
        if not plan_edits:
            raise ValueError("Patch plan contains no edits")
        plan_operations = _plan_operations(plan)
        new_edits = 0
        for edit in plan_edits:
            pool, offset = int(edit["pool"]), int(edit["offset"])
            expected = bytes.fromhex(edit["expected"])
            replacement = bytes.fromhex(edit["replacement"])
            if not expected or len(expected) != len(replacement):
                raise ValueError(f"Invalid or width-changing edit at pool {pool}:{offset}")
            key = (pool, offset, len(expected))
            payload = (expected.hex(), replacement.hex())
            if key in exact:
                if exact[key] != payload:
                    raise ValueError(f"Conflicting queued edits at pool {pool}:{offset}")
                continue
            collision = next(
                (occupied[(pool, byte)] for byte in range(offset, offset + len(expected))
                 if (pool, byte) in occupied),
                None,
            )
            if collision is not None:
                raise ValueError(
                    f"Overlapping queued edits at pool {pool}:{offset} and edit #{collision + 1}"
                )
            edit_index = len(edits)
            for byte in range(offset, offset + len(expected)):
                occupied[(pool, byte)] = edit_index
            exact[key] = payload
            edits.append(dict(edit))
            new_edits += 1
        if new_edits or not operations:
            operations.extend(plan_operations)

    if len(operations) == 1:
        operation = operations[0]
        field, kind = operation.get("field"), operation.get("kind")
        value, flags, enum_name = (
            operation.get("value"), operation.get("flags"), operation.get("enum")
        )
    else:
        field, kind, value, flags, enum_name = "multiple", "mixed", None, None, None
    return {
        "format": expected_format, "source": first.get("source"),
        "source_sha256": first.get("source_sha256"), "motiongraph": motiongraph,
        "field": field, "kind": kind, "value": value, "flags": flags,
        "enum": enum_name, "operations": operations, "edits": edits,
    }


def load_plan(path: Path) -> dict:
    """Load and structurally validate a saved patch plan."""
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load patch plan: {exc}") from exc
    if not isinstance(plan, dict):
        raise ValueError("Patch plan root must be a JSON object")
    return merge_patch_plans([plan])


def apply_patch_plan(source: Path, output: Path, plan: dict,
                     game: str = DEFAULT_GAME) -> PatchReport:
    """Apply a plan by recompressing STATIC only, preserving its topology."""
    source, output = validate_staged_pair(source, output)
    if plan.get("format") not in (None, "cobra-motiongraph-patch-v1"):
        raise ValueError(f"Unsupported patch-plan format: {plan.get('format')!r}")
    expected_hash = plan.get("source_sha256")
    if expected_hash:
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_hash.lower() != str(expected_hash).lower():
            raise ValueError(
                f"Patch plan was made for source SHA-256 {expected_hash}, found {actual_hash}"
            )
    edits = plan.get("edits") or []
    if not edits:
        raise ValueError("Patch plan contains no edits")

    ovl = OvlFile()
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        ovl.load(str(source), {"game": game})
    finally:
        logging.disable(previous_disable)
    static = next((archive for archive in ovl.archives if archive.name == "STATIC"), None)
    if static is None:
        raise ValueError("OVL has no STATIC archive")
    by_pool = defaultdict(list)
    for edit in edits:
        by_pool[int(edit["pool"])].append(edit)
    before = {}
    for pool_index in by_pool:
        if pool_index < 0 or pool_index >= len(ovl.pools):
            raise ValueError(f"Pool index out of range: {pool_index}")
        pool = ovl.pools[pool_index]
        if pool not in static.content.pools:
            raise ValueError(f"Pool {pool_index} is not in STATIC")
        before[pool_index] = pool.data.getvalue()

    intended = defaultdict(set)
    occupied = set()
    for pool_index, pool_edits in by_pool.items():
        blob = before[pool_index]
        for edit in pool_edits:
            offset = int(edit["offset"])
            expected = bytes.fromhex(edit["expected"])
            replacement = bytes.fromhex(edit["replacement"])
            if len(expected) != len(replacement):
                raise ValueError(f"Width change at pool {pool_index}:{offset}")
            if offset < 0 or offset + len(expected) > len(blob):
                raise ValueError(f"Edit outside pool {pool_index}: {offset}")
            span = {(pool_index, byte) for byte in range(offset, offset + len(expected))}
            if occupied & span:
                raise ValueError(f"Overlapping edits at pool {pool_index}:{offset}")
            occupied |= span
            actual = blob[offset:offset + len(expected)]
            if actual != expected:
                raise ValueError(
                    f"Stale plan at pool {pool_index}:{offset}; "
                    f"expected {expected.hex()}, found {actual.hex()}"
                )
            intended[pool_index].update(
                offset + i for i, (old, new) in enumerate(zip(expected, replacement)) if old != new
            )

    for pool_index, pool_edits in by_pool.items():
        pool = ovl.pools[pool_index]
        for edit in pool_edits:
            pool.data.seek(int(edit["offset"]))
            pool.data.write(bytes.fromhex(edit["replacement"]))
        pool.data.seek(0)
    for pool_index in by_pool:
        old, new = before[pool_index], ovl.pools[pool_index].data.getvalue()
        if len(old) != len(new):
            raise ValueError(f"Pool {pool_index} changed size")
        changed = {i for i, (left, right) in enumerate(zip(old, new)) if left != right}
        if changed != intended[pool_index]:
            raise ValueError(f"Pool {pool_index} changed outside the patch plan")

    static.content.write_pools()
    uncompressed = static.content.write_archive()
    if len(uncompressed) != static.uncompressed_size:
        raise ValueError("STATIC uncompressed size changed")
    _, compressed_size, compressed = static.content.compress(uncompressed, True)
    source_bytes = source.read_bytes()
    header_size = len(source_bytes) - static.compressed_size
    result = bytearray(source_bytes[:header_size])
    result.extend(compressed)
    struct.pack_into("<I", result, static.io_start + COMPRESSED_SIZE_OFFSET, compressed_size)
    publication = CandidatePublication(source, output)
    publication.path.write_bytes(result)

    # A valid main OVL is not sufficient: require the complete staged family to
    # reload and re-check all target bytes after decompression.
    check = OvlFile()
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        check.load(str(publication.path), {"game": game})
    finally:
        logging.disable(previous_disable)
    check_static = next((archive for archive in check.archives if archive.name == "STATIC"), None)
    if check_static is None:
        raise ValueError("Staged output reload has no STATIC archive")
    source_topology = (
        len(ovl.loaders), len(ovl.archives),
        sum(len(archive.content.pools) for archive in ovl.archives),
        sum(len(archive.content.fragments) for archive in ovl.archives),
    )
    output_topology = (
        len(check.loaders), len(check.archives),
        sum(len(archive.content.pools) for archive in check.archives),
        sum(len(archive.content.fragments) for archive in check.archives),
    )
    if output_topology != source_topology:
        raise ValueError(f"Staged output topology changed: {source_topology} -> {output_topology}")
    for edit in edits:
        pool_index, offset = int(edit["pool"]), int(edit["offset"])
        replacement = bytes.fromhex(edit["replacement"])
        actual = check.pools[pool_index].data.getvalue()[offset:offset + len(replacement)]
        if actual != replacement:
            raise ValueError(
                f"Staged reload lost edit at pool {pool_index}:{offset}; "
                f"expected {replacement.hex()}, found {actual.hex()}"
            )
    publication.commit()
    return PatchReport(
        output=output, edits=len(edits), pools=len(by_pool),
        changed_bytes=sum(len(items) for items in intended.values()),
        static_pools=static.num_pools, static_fragments=static.num_fragments,
        uncompressed_size=static.uncompressed_size,
        compressed_before=static.compressed_size, compressed_after=compressed_size,
    )
