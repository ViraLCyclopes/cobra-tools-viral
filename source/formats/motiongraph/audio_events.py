"""Audio-event helpers for the motiongraph tools.

A motiongraph fires sounds by NAME: it holds strings like ``Indoraptor_HeroRoarA``
and the engine hashes them. Giving a donor-based species its own voice therefore
means renaming those strings - but only the ones the donor's own events bank
actually owns.

Some names resolve from SHARED banks used by several species. Renaming one of
those orphans the sound in both directions: the shared bank no longer matches,
and the new bank never had it. The sound goes silent with no error anywhere,
which is why this module exists - so the UI can mark those names as unsafe
instead of leaving it to memory.
"""
from __future__ import annotations

import logging
import re
import struct
import tempfile
from pathlib import Path

# cobra's reporters call logging.success(); it only exists once the GUI has set
# logging up, so anything importable on its own has to provide the same shim.
if not hasattr(logging, "success"):
    logging.success = logging.info  # type: ignore[attr-defined]

EVENT = 4  # Wwise HIRC object type


def fnv1_32(name: str) -> int:
    """Wwise's id for a name: FNV-1 (not 1a), 32-bit, over the LOWERCASED bytes."""
    h = 0x811C9DC5
    for b in name.lower().encode():
        h = (h * 0x01000193) & 0xFFFFFFFF
        h ^= b
    return h


def _open_ovl(source: Path, game: str):
    from generated.formats.ovl import OvlFile

    previous = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        ovl = OvlFile()
        ovl.load(str(source), {"game": game})
    finally:
        logging.disable(previous)
    return ovl


def scan_graph_event_names(source: Path, donor: str, game: str) -> list[str]:
    """Audio event names the motiongraph fires, as ``<donor>_Something``.

    Read from the graph's ``<ds_name>`` elements specifically. Scanning the raw
    string pools instead would be faster but wrong: it also sweeps up material
    resources like ``<donor>_Variant_01_00`` and ``<donor>_VariantSet_Lux``,
    which are not sounds and must never appear in a rename list.
    """
    source = Path(source)
    ovl = _open_ovl(source, game)
    graphs = [name for name in ovl.loaders if name.lower().endswith(".motiongraph")]
    if not graphs:
        raise ValueError("This OVL contains no .motiongraph")

    previous = logging.root.manager.disable
    with tempfile.TemporaryDirectory(prefix="vl_graph_") as tmp:
        try:
            logging.disable(logging.CRITICAL)
            ovl.extract(tmp, only_names=graphs[:1])
        finally:
            logging.disable(previous)
        written = list(Path(tmp).glob("*.motiongraph"))
        if not written:
            raise ValueError(f"Could not extract {graphs[0]}")
        text = written[0].read_text(encoding="utf-8", errors="replace")

    pattern = re.compile(r"<ds_name>(" + re.escape(donor) + r"_[A-Za-z0-9_]+)</ds_name>")
    return sorted(set(pattern.findall(text)))


def _walk_chunks(data: bytes) -> dict[str, tuple[int, int]]:
    out, off = {}, 0
    while off + 8 <= len(data):
        tag = data[off:off + 4]
        if not tag.isalpha():
            break
        size = struct.unpack_from("<I", data, off + 4)[0]
        out[tag.decode()] = (off + 8, size)
        off += 8 + size
    return out


def event_ids_in_bank(aux_path: Path) -> set[int]:
    """EVENT object ids declared by a Wwise bank.

    The real bank is the ``..._bnk_b.aux`` companion, not cobra's 84-byte .bnk
    stub, and it must be read only as far as its chunks actually walk.
    """
    raw = aux_path.read_bytes()
    end, off = 0, 0
    while off + 8 <= len(raw):
        tag = raw[off:off + 4]
        if not tag.isalpha():
            break
        off += 8 + struct.unpack_from("<I", raw, off + 4)[0]
        end = off
    data = raw[:end]
    chunks = _walk_chunks(data)
    if "HIRC" not in chunks:
        return set()
    base, _size = chunks["HIRC"]
    count = struct.unpack_from("<I", data, base)[0]
    pos, ids = base + 4, set()
    for _ in range(count):
        obj_type = data[pos]
        obj_size = struct.unpack_from("<I", data, pos + 1)[0]
        if obj_type == EVENT:
            ids.add(struct.unpack_from("<I", data, pos + 5)[0])
        pos += 5 + obj_size
    return ids


def find_donor_events_ovl(game_root: Path, donor: str) -> Path | None:
    """Locate ``<Donor>_events.ovl``; it is not always in Content0."""
    name = f"{donor}_events"
    for pack in ("Content0", "ContentPDLC3", "ContentPDLC1", "ContentDeluxe",
                 "ContentPDLCRebirth"):
        candidate = game_root / "Win64" / "ovldata" / pack / "Audio" / name / f"{name}.ovl"
        if candidate.is_file():
            return candidate
    return None


def donor_event_ids(game_root: Path, donor: str, game: str) -> set[int]:
    """Event ids the donor's OWN events bank declares."""
    from generated.formats.ovl import OvlFile

    ovl_path = find_donor_events_ovl(Path(game_root), donor)
    if ovl_path is None:
        raise FileNotFoundError(
            f"Could not find {donor}_events.ovl in any content pack under {game_root}")
    ovl = OvlFile()
    ovl.load(str(ovl_path), {"game": game})
    with tempfile.TemporaryDirectory(prefix="vl_audio_") as tmp:
        ovl.extract(tmp)
        auxes = list(Path(tmp).glob("*_bnk_b.aux"))
        if not auxes:
            raise FileNotFoundError(
                f"{ovl_path.name} contained no '..._bnk_b.aux' - that companion IS the bank")
        return event_ids_in_bank(auxes[0])


def classify(names: list[str], donor: str, owned_ids: set[int]) -> list[tuple[str, bool]]:
    """Tag each graph name as safe to rename (donor owns it) or not (shared bank)."""
    return [(n, fnv1_32(n) in owned_ids) for n in names]


def rename_plan(names: list[str], donor: str, prefix: str) -> list[tuple[str, str]]:
    """(old, new) pairs, longest first.

    Longest-first is not strictly required - pool lookups match on the trailing
    NUL, so ``Indoraptor_Sniff`` cannot match inside ``Indoraptor_SniffGround``
    - but it keeps the order deterministic and matches how these renames have
    always been applied.
    """
    plan = []
    for name in sorted(names, key=len, reverse=True):
        suffix = name[len(donor) + 1:]
        plan.append((name, f"{prefix}_{suffix}"))
    return plan
