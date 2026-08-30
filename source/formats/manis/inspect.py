"""Read-only inspection and verification for MANIS bundles, any supported game.

Nothing here assumes ACL. JWE1, JWE2 and Planet Zoo ship uncompressed clips and
report no blobs; JWE3 and PC2 clips carry an ACL ``compressed_tracks`` blob.
Capability is detected **per clip** - a compression flag plus a real blob - so a
caller can offer ACL-only actions exactly where they apply and fall back to the
uncompressed key arrays everywhere else.

Two jobs:

``bundle_report`` / ``clip_report``
    Describe what a bundle and its clips actually contain, including which ACL
    sub-tracks are animated, constant, or defaulted to the bind pose. A stripped
    sub-track holds no samples at all, so "the curve is flat" and "the curve is
    absent" are different answers and only this distinction tells them apart.

``verify_bundle``
    Gate a bundle before it is written or injected. The checks are deliberately
    conservative: they flag only what is wrong in every game, never what merely
    differs between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .acl_patch import (
    ScalarLayout,
    TransformLayout,
    parse_scalar_layout,
    parse_transform_layout,
    scalar_bit_rates,
    sub_track_types,
)
from .splice import list_clip_blobs, read_blob_header


QVVF_TRACK_TYPE = 12
SCALAR_TRACK_TYPE = 0
# ACL packs two bits per sub-track: 0 = animated, 1 = constant, 2 = default.
SUB_TRACK_ANIMATED = 0
SUB_TRACK_CONSTANT = 1
SUB_TRACK_DEFAULT = 2
SUB_TRACK_LABELS = {
    SUB_TRACK_ANIMATED: "animated",
    SUB_TRACK_CONSTANT: "constant",
    SUB_TRACK_DEFAULT: "default (bind)",
}

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass
class Issue:
    severity: str
    message: str
    clip: str | None = None

    def __str__(self) -> str:
        where = f" [{self.clip}]" if self.clip else ""
        return f"{self.severity.upper()}{where}: {self.message}"


@dataclass
class ClipReport:
    index: int
    name: str
    compressed: bool
    duration: float
    frame_count: int
    pos_bones: int
    ori_bones: int
    scl_bones: int
    floats: int
    target_bone_count: int
    acl: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_acl(self) -> bool:
        return self.acl is not None


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip_blob_map(data: bytes | None) -> list[dict]:
    """Headers for every ACL blob in file order, or [] for a non-ACL bundle."""
    if not data:
        return []
    blobs = []
    try:
        for offset, size in list_clip_blobs(data):
            header = read_blob_header(data, offset)
            header["offset"] = offset
            header["size"] = size
            blobs.append(header)
    except Exception:
        return []
    return blobs


def transform_blob_indices(blobs: list[dict]) -> list[int]:
    """Blob indices carrying transform tracks, in clip order.

    A clip's transform blob is the Nth ``track_type == 12`` blob, not blob
    ``2 * clip`` - scalar blobs are interleaved and not every clip has one.
    """
    return [i for i, header in enumerate(blobs)
            if header.get("track_type") == QVVF_TRACK_TYPE]


def describe_acl_blob(data: bytes, header: dict) -> dict:
    """Classify an ACL transform blob's sub-tracks without decoding samples."""
    blob = data[header["offset"]:header["offset"] + header["size"]]
    info: dict[str, Any] = {
        "offset": header["offset"],
        "size": header["size"],
        "version": header.get("version"),
        "num_tracks": header.get("num_tracks"),
        "num_samples": header.get("num_samples"),
        "track_type": header.get("track_type"),
    }
    for key in ("has_database", "wrap_optimized", "trivial_defaults", "has_scale"):
        if key in header:
            info[key] = header[key]
    try:
        layout = parse_transform_layout(blob)
    except Exception as exc:
        info["layout_error"] = str(exc)
        return info
    info["has_scale"] = bool(layout.has_scale)
    info["num_segments"] = getattr(layout, "num_segments", None)
    kinds = ["rotation", "translation"] + (["scale"] if layout.has_scale else [])
    breakdown = {}
    for kind in kinds:
        try:
            types = sub_track_types(blob, layout, kind)
        except Exception as exc:
            breakdown[kind] = {"error": str(exc)}
            continue
        counts = {label: 0 for label in SUB_TRACK_LABELS.values()}
        for value in types:
            counts[SUB_TRACK_LABELS.get(value, f"unknown ({value})")] = counts.get(
                SUB_TRACK_LABELS.get(value, f"unknown ({value})"), 0) + 1
        breakdown[kind] = {"counts": counts, "types": tuple(types)}
    info["sub_tracks"] = breakdown
    return info


def describe_scalar_blob(data: bytes, header: dict) -> dict:
    blob = data[header["offset"]:header["offset"] + header["size"]]
    info: dict[str, Any] = {
        "offset": header["offset"], "size": header["size"],
        "num_tracks": header.get("num_tracks"),
        "num_samples": header.get("num_samples"),
    }
    try:
        layout: ScalarLayout = parse_scalar_layout(blob)
        info["bit_rates"] = scalar_bit_rates(blob, layout)
    except Exception as exc:
        info["layout_error"] = str(exc)
    return info


def clip_blob_spans(data: bytes, clip_index: int) -> dict[str, tuple[int, int]]:
    """``{"transform": (offset, size), "scalar": (offset, size)}`` for one clip.

    Returns an empty mapping for a bundle with no ACL blobs, so callers in older
    games get "no ACL here" rather than an exception.
    """
    blobs = _clip_blob_map(data)
    transforms = transform_blob_indices(blobs)
    if clip_index >= len(transforms):
        return {}
    blob_index = transforms[clip_index]
    spans = {"transform": (blobs[blob_index]["offset"], blobs[blob_index]["size"])}
    following = blob_index + 1
    if following < len(blobs) and blobs[following].get("track_type") != QVVF_TRACK_TYPE:
        spans["scalar"] = (blobs[following]["offset"], blobs[following]["size"])
    return spans


def splice_blob(data: bytes, span: tuple[int, int], patched: bytes) -> bytes:
    """Put a patched blob back at its exact offset; refuses to change its size."""
    offset, size = span
    if len(patched) != size:
        raise ValueError(
            f"patched blob is {len(patched)} bytes, must stay {size} to keep every "
            "downstream offset valid")
    return data[:offset] + patched + data[offset + size:]


# Range scaling rewrites a vector's min/extent, so it has no meaning for a
# quaternion sub-track; only these kinds can be scaled.
RANGE_SCALABLE_KINDS = ("translation", "scale", "scalar")
# Where each kind's components sit in a decoded qvvf sample row.
SAMPLE_COMPONENTS = {"translation": (4, 7), "scale": (7, 10)}


def first_sample_pivot(blob: bytes, kind: str, track_index: int):
    """The curve's first decoded sample, for use as a range-scaling pivot.

    Scaling a curve about zero moves its starting value. On a track whose first
    sample is not zero that reads in game as a teleport - it is what made the
    coherent Patagotitan build (Test P) unusable - so anything that stretches a
    curve's travel should pivot about the value it starts at.
    """
    from .acl import decode_blob

    decoded = decode_blob(blob)
    if kind == "scalar":
        return float(decoded.values[0, track_index, 0])
    if kind not in SAMPLE_COMPONENTS:
        raise ValueError(f"{kind} cannot be range-scaled")
    low, high = SAMPLE_COMPONENTS[kind]
    return tuple(float(value) for value in decoded.values[0, track_index, low:high])


def zero_pivot(kind: str):
    return 0.0 if kind == "scalar" else (0.0, 0.0, 0.0)


def clip_report(manis, index: int, data: bytes | None = None,
                blobs: list[dict] | None = None) -> ClipReport:
    """Describe one clip. ``data`` is the raw bundle; omit it to skip ACL detail."""
    mani_info = manis.mani_infos[index]
    dtype = getattr(mani_info, "dtype", None)
    compressed = bool(getattr(dtype, "compression", 0)) if dtype is not None else False
    report = ClipReport(
        index=index,
        name=str(mani_info.name),
        compressed=compressed,
        duration=float(getattr(mani_info, "duration", 0.0) or 0.0),
        frame_count=_int(getattr(mani_info, "frame_count", 0)),
        pos_bones=_int(getattr(mani_info, "pos_bone_count", 0)),
        ori_bones=_int(getattr(mani_info, "ori_bone_count", 0)),
        scl_bones=_int(getattr(mani_info, "scl_bone_count", 0)),
        floats=_int(getattr(mani_info, "float_count", 0)),
        target_bone_count=_int(getattr(mani_info, "target_bone_count", 0)),
    )
    if not compressed:
        report.notes.append("uncompressed keys; ACL actions do not apply")
        return report
    if blobs is None:
        blobs = _clip_blob_map(data)
    if not blobs or not data:
        report.notes.append("marked compressed but no ACL blob was located")
        return report
    transforms = transform_blob_indices(blobs)
    if index >= len(transforms):
        report.notes.append("no transform blob for this clip index")
        return report
    blob_index = transforms[index]
    report.acl = describe_acl_blob(data, blobs[blob_index])
    following = blob_index + 1
    if following < len(blobs) and blobs[following].get("track_type") != QVVF_TRACK_TYPE:
        report.acl["scalar"] = describe_scalar_blob(data, blobs[following])
    return report


def bundle_report(manis, data: bytes | None = None) -> dict:
    blobs = _clip_blob_map(data)
    clips = [clip_report(manis, index, data, blobs)
             for index in range(len(manis.mani_infos))]
    return {
        "name": getattr(manis, "name", None),
        "clip_count": len(clips),
        "declared_mani_count": _int(getattr(manis, "mani_count", len(clips)), len(clips)),
        "acl_blobs": len(blobs),
        "acl_clips": sum(1 for clip in clips if clip.is_acl),
        "uncompressed_clips": sum(1 for clip in clips if not clip.compressed),
        "clips": clips,
    }


def verify_bundle(manis, data: bytes | None = None,
                  file_size: int | None = None) -> list[Issue]:
    """Conservative pre-write gate. Flags only what is wrong in every game.

    Deliberately NOT an error: ``eoh`` ending before the file size. JWE3 appends
    ACL database bulk past the parsed header region, so a short ``eoh`` is normal
    there and only ``eoh`` running past the file is a real fault.
    """
    issues: list[Issue] = []
    infos = list(manis.mani_infos)
    blobs = _clip_blob_map(data) if data else []

    declared = _int(getattr(manis, "mani_count", len(infos)), len(infos))
    if declared != len(infos):
        issues.append(Issue(
            ERROR, f"mani_count is {declared} but the bundle holds {len(infos)} clips"))

    names = [str(info.name) for info in infos]
    seen: dict[str, int] = {}
    for index, name in enumerate(names):
        if not name.strip():
            issues.append(Issue(ERROR, "clip has an empty name", f"#{index}"))
        if name in seen:
            issues.append(Issue(
                ERROR, f"duplicate clip name, also at #{seen[name]}", name))
        else:
            seen[name] = index

    declared_names = getattr(manis, "names", None)
    if declared_names is not None and len(declared_names) != len(infos):
        issues.append(Issue(
            ERROR,
            f"name list holds {len(declared_names)} entries for {len(infos)} clips"))

    for index, info in enumerate(infos):
        name = names[index]
        if _int(getattr(info, "frame_count", 0)) <= 0:
            issues.append(Issue(WARNING, "frame_count is zero", name))
        target = _int(getattr(info, "target_bone_count", 0))
        for attribute in ("pos_bone_count", "ori_bone_count", "scl_bone_count"):
            count = _int(getattr(info, attribute, 0))
            if target and count > target:
                issues.append(Issue(
                    ERROR,
                    f"{attribute} is {count} but target_bone_count is {target}", name))

    eoh = getattr(manis, "eoh", None)
    if eoh is not None and file_size is not None:
        if int(eoh) > int(file_size):
            issues.append(Issue(
                ERROR, f"parsed header end {int(eoh)} runs past the file ({file_size})"))
        elif int(eoh) < int(file_size):
            trailing = int(file_size) - int(eoh)
            reason = ("ACL database bulk is appended past the header region"
                      if blobs else "unparsed trailing data")
            issues.append(Issue(
                INFO, f"{trailing} bytes after the header region; {reason}"))

    if data:
        transforms = transform_blob_indices(blobs)
        compressed = [index for index, info in enumerate(infos)
                      if bool(getattr(getattr(info, "dtype", None), "compression", 0))]
        if blobs and len(transforms) < len(compressed):
            issues.append(Issue(
                ERROR,
                f"{len(compressed)} clips are marked compressed but only "
                f"{len(transforms)} ACL transform blobs were found"))
        if blobs and compressed and len(compressed) != len(infos):
            issues.append(Issue(
                WARNING,
                "bundle mixes compressed and uncompressed clips; writing it will be "
                "lossy for the compressed ones"))
    return issues


def worst_severity(issues: list[Issue]) -> str | None:
    for severity in (ERROR, WARNING, INFO):
        if any(issue.severity == severity for issue in issues):
            return severity
    return None
