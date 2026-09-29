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

The reverse trip matters just as much. A modder who wants to publish a copy
WITHOUT the custom-audio dependency has to put the donor's names back, or the
graph fires ids no installed bank answers and the animal is mute. That direction
cannot ask "does the prefix own this sound" - the mod's bank may already be
deleted, and it never lived in an ``_events.ovl`` anyway: a mod ships the raw
``<prefix>_events.bnk`` and a ``.wmetasb.add`` fragment in ``<Mod>/Audio/``. So
reverting is verified against the STOCK REGISTRY instead, which declares every
vanilla event including the ones shared banks own. See ``classify_revert``.
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

    ``donor`` is whatever prefix the graph currently uses, so pass the MOD's
    prefix when scanning a renamed graph for a revert.
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

    Inside a packed ``<Species>_events.ovl`` the real bank is the
    ``..._bnk_b.aux`` companion, not cobra's 84-byte .bnk stub. A MOD ships the
    bank loose instead, as ``<Mod>/Audio/<prefix>_events.bnk``, and that file is
    a complete bank - ``BKHD`` then ``HIRC`` - so this reads either one. Only as
    far as the chunks actually walk, in both cases.
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


def find_mod_events_bank(mod_root: Path, prefix: str) -> Path | None:
    """Locate a mod's LOOSE events bank: ``<Mod>/Audio/<prefix>_events.bnk``.

    Mods do not ship an ``_events.ovl``; the raw bank and the ``.wmetasb.add``
    registry fragment go straight into the mod's ``Audio`` folder. Accepts either
    the mod root or the ``Audio`` folder itself, and matches case-insensitively
    because the built banks are lowercased while the prefix usually is not.
    """
    mod_root = Path(mod_root)
    wanted = f"{prefix}_events.bnk".lower()
    for folder in (mod_root / "Audio", mod_root):
        if not folder.is_dir():
            continue
        for candidate in folder.iterdir():
            if candidate.is_file() and candidate.name.lower() == wanted:
                return candidate
    return None


def stock_registry_path(game_root: Path) -> Path:
    game_root = Path(game_root)
    return (game_root / "Win64" / "ovldata" / "Content0" / "Audio" / "MetaData"
            / "audiometadata.ovl")


_REGISTRY_CACHE: dict[tuple[str, int, int], str] = {}


def read_stock_registry(game_root: Path, game: str) -> str:
    """The stock ``.wmetasb`` text, cached per file identity.

    Unpacking ``audiometadata.ovl`` costs seconds, and both the fragment builder
    and the revert classifier want the same text.
    """
    meta = stock_registry_path(game_root)
    if not meta.is_file():
        raise FileNotFoundError(f"Stock registry not found: {meta}")
    stat = meta.stat()
    key = (str(meta.resolve()), stat.st_size, int(stat.st_mtime))
    cached = _REGISTRY_CACHE.get(key)
    if cached is not None:
        return cached

    ovl = _open_ovl(meta, game)
    previous = logging.root.manager.disable
    with tempfile.TemporaryDirectory(prefix="vl_reg_") as tmp:
        try:
            logging.disable(logging.CRITICAL)
            ovl.extract(tmp)
        finally:
            logging.disable(previous)
        found = list(Path(tmp).glob("*.wmetasb"))
        if not found:
            raise FileNotFoundError("audiometadata.ovl held no .wmetasb")
        registry = found[0].read_text(encoding="utf-8")
    _REGISTRY_CACHE[key] = registry
    return registry


def stock_event_ids(game_root: Path, game: str) -> frozenset[int]:
    """Every event id the STOCK registry declares, across all banks.

    This is the oracle for "would the unmodded game post this sound". It is wider
    than one species' own bank on purpose: a revert target such as
    ``IndominusRex_SocialCall`` lives in a SHARED bank, and refusing to restore it
    because the donor's bank does not own it would be exactly backwards.
    """
    registry = read_stock_registry(game_root, game)
    return frozenset(
        int(value) for value in re.findall(
            r'(?:event_fnv|stop_start_fnv|start_fnv)="(\d+)"', registry)
    )


def classify(names: list[str], donor: str, owned_ids: set[int]) -> list[tuple[str, bool]]:
    """Tag each graph name as safe to rename (donor owns it) or not (shared bank)."""
    return [(n, fnv1_32(n) in owned_ids) for n in names]


def classify_revert(names: list[str], prefix: str, donor: str,
                    stock_ids: frozenset[int] | set[int] | None
                    ) -> list[tuple[str, str, bool]]:
    """Tag each ``<prefix>_*`` graph name with its stock target and whether it exists.

    Returns ``(current_name, stock_target, restorable)``. Restorable means the
    stock registry declares ``<donor>_<suffix>``, so putting the name back makes
    the graph fire a sound the unmodded game already answers.

    A name that is NOT restorable is one the modder invented - there is no stock
    event of that suffix to fall back to. Reverting it would swap one dead id for
    another, so the UI must not offer it as a fix.

    ``stock_ids`` of ``None`` skips verification and marks everything restorable,
    for the case where the game folder is not available.
    """
    plan = []
    for name in names:
        if not name.startswith(prefix + "_"):
            continue
        target = f"{donor}_{name[len(prefix) + 1:]}"
        restorable = True if stock_ids is None else fnv1_32(target) in stock_ids
        plan.append((name, target, restorable))
    return plan


def build_registry_fragment(game_root: Path, donor: str, prefix: str,
                            renamed_suffixes: list[str], game: str) -> tuple[str, int, int]:
    """This mod's contribution to the audio registry.

    ``wwisemetadatasoundbanks`` decides whether an event is ever POSTED - an
    event missing from it is never fired, with no error anywhere. But it is ONE
    GLOBAL FILE, so a mod must never ship its own whole copy: only one could win
    and every other audio mod would go silent. Each mod emits this fragment and a
    merge step folds them all into a single shared registry.

    Returns (xml, evententry_rows, remapped_fnvs).
    """
    registry = read_stock_registry(game_root, game)

    remap: dict[int, int] = {}
    for suffix in renamed_suffixes:
        for variant in (suffix, suffix + "_start", suffix + "_stop", suffix + "_oc",
                        suffix + "_oc_start", suffix + "_oc_stop"):
            remap[fnv1_32(f"{donor}_{variant}")] = fnv1_32(f"{prefix}_{variant}")

    donor_bank = fnv1_32(f"{donor.lower()}_events")
    match = re.search(r'(\t\t<bnkmetanew fnv="%d".*?</bnkmetanew>\n)' % donor_bank,
                      registry, re.S)
    if not match:
        raise ValueError(f"{donor}_events is not declared in the stock registry")

    def swap(mo):
        value = int(mo.group(2))
        return '%s="%d"' % (mo.group(1), remap.get(value, value))

    block = re.sub(r'(event_fnv|stop_start_fnv|start_fnv)="(\d+)"', swap, match.group(1))
    block = block.replace('fnv="%d"' % donor_bank,
                          'fnv="%d"' % fnv1_32(f"{prefix.lower()}_events"), 1)
    block += ('\t\t<bnkmetanew fnv="%d" flag="0" unk_2="0">\n'
              '\t\t\t<type_name>SFX</type_name>\n'
              '\t\t\t<events pool_type="4" />\n'
              '\t\t</bnkmetanew>\n' % fnv1_32(f"{prefix.lower()}_media"))
    rows = len(re.findall(r"<evententry", block))
    changed = sum(1 for v in re.findall(r'event_fnv="(\d+)"', match.group(1))
                  if int(v) in remap)
    return block, rows, changed


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


def revert_plan(names: list[str], prefix: str, donor: str) -> list[tuple[str, str]]:
    """(current, stock) pairs putting a renamed graph back on the donor's voice.

    The mirror of ``rename_plan``. Usually every target is SHORTER than what it
    replaces, so these land as in-place slot patches, but a shorter mod prefix
    than the donor's name is legal and those pairs relocate instead -
    ``apply_string_renames`` sorts that out.
    """
    return [(name, f"{donor}_{name[len(prefix) + 1:]}")
            for name in sorted(names, key=len, reverse=True)
            if name.startswith(prefix + "_")]
