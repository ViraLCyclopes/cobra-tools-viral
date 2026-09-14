"""Cheap, cached lookup of the clip names inside a .manis bundle.

manis file names are opaque hashes - a species ships a dozen files called things like
`motionextracted.maniset397ff974.manis` - so a file browser listing them tells you
nothing about which animations are inside. This backs the clip list drawn in the
import and export file dialogs.

Blender redraws a file browser panel constantly, so results are cached on
(path, mtime, size); the file is only parsed when it actually changes.
"""
from __future__ import annotations

import logging
import os

MAX_PARSE_BYTES = 64 * 1024 * 1024

# (path, mtime, size) -> list[str] | None on failure
_cache: dict[tuple, list | None] = {}


def _stat_key(filepath: str):
    st = os.stat(filepath)
    return filepath, st.st_mtime_ns, st.st_size


def clip_names(filepath: str) -> list | None:
    """Return the clip names in a .manis, or None if it cannot be read.

    None means 'unknown' - callers should say so rather than claim the file is empty.
    """
    if not filepath or not os.path.isfile(filepath):
        return None
    try:
        key = _stat_key(filepath)
    except OSError:
        return None
    if key in _cache:
        return _cache[key]

    names = None
    try:
        if key[2] > MAX_PARSE_BYTES:
            logging.debug(f"Not listing clips for {filepath}, over {MAX_PARSE_BYTES} bytes")
        else:
            # imported lazily: this module is imported by the operators at register time,
            # and the generated format package is heavy
            from generated.formats.manis import ManisFile
            manis = ManisFile()
            manis.load(filepath)
            names = [str(mani_info.name) for mani_info in manis.mani_infos]
    except Exception:
        logging.debug(f"Could not read clip names from {filepath}", exc_info=True)
        names = None

    # bound the cache; these lists are small but a bulk import can touch many files
    if len(_cache) > 32:
        _cache.clear()
    _cache[key] = names
    return names


# only a runaway guard - the file browser side region scrolls, so every clip in a real
# bundle is listed. The largest vanilla deinosuchus bundle holds 42.
MAX_LISTED = 500


def draw_splice_plan(layout, source_folder: str, action_source: str = "CHANGED") -> None:
    """Draw which actions will be spliced into which bundle.

    The splice exporter routes each action by the `manis` stamp its importer left
    on it, so unlike the import dialog the user never picks a bundle - which made
    the export dialog silent about where anything was going. This mirrors the
    candidate selection in `export_splice.save` so the panel shows the real plan.

    Deliberately cheap: a file browser redraws constantly, so this only does dict
    lookups, `os.path.isfile`, and the cached `clip_names`. It never samples an
    action, so it cannot show which clips actually DIFFER - that test requires
    sampling and only happens at export time.
    """
    import bpy

    box = layout.box()
    box.label(text="Splice plan", icon='EXPORT')

    folder = bpy.path.abspath(source_folder or "")
    if not folder or not os.path.isdir(folder):
        box.label(text="Set 'Source Bundles' to the vanilla .manis folder", icon='ERROR')
        return

    if action_source == "ACTIVE":
        ob = bpy.context.object
        anim = getattr(ob, "animation_data", None) if ob else None
        active = getattr(anim, "action", None)
        candidates = [active] if active else []
    elif action_source == "SELECTED":
        candidates = [a for a in bpy.data.actions if a.get("cobra_splice")]
    else:
        candidates = list(bpy.data.actions)

    stamped = [a for a in candidates if a is not None and a.get("manis")]
    unstamped = [a for a in candidates if a is not None and not a.get("manis")]

    if not stamped:
        box.label(text="No actions carry a bundle stamp - re-import them", icon='ERROR')
        return

    by_bundle: dict = {}
    for action in stamped:
        by_bundle.setdefault(action.get("manis"), []).append(action.name)

    for bundle in sorted(by_bundle):
        present = os.path.isfile(os.path.join(folder, bundle))
        row = box.row()
        row.label(text=bundle, icon='FILE' if present else 'ERROR')
        names = by_bundle[bundle]
        col = box.column(align=True)
        if not present:
            col.label(text="not in the source folder - these clips are SKIPPED")
        for name in sorted(names)[:MAX_LISTED]:
            col.label(text="    " + (name.split("$")[-1] if "$" in name else name))
        if len(names) > MAX_LISTED:
            col.label(text=f"    ... and {len(names) - MAX_LISTED} more")

    if unstamped:
        box.label(text=f"{len(unstamped)} action(s) have no stamp and are ignored",
                  icon='INFO')
    if action_source == "CHANGED":
        box.label(text="'Changed Only' writes just the clips that differ", icon='INFO')


def draw_clip_list(layout, filepath: str) -> None:
    """Draw the clip names of `filepath` into a file-browser side panel."""
    box = layout.box()
    name = os.path.basename(filepath) if filepath else ""
    if not name:
        box.label(text="No file selected", icon='INFO')
        return
    names = clip_names(filepath)
    if names is None:
        box.label(text="Could not read clips", icon='ERROR')
        return
    box.label(text=f"{len(names)} clips in {name}", icon='ANIM')
    col = box.column(align=True)
    for clip in sorted(names)[:MAX_LISTED]:
        # the species prefix is the same for every clip in a bundle, so drop it
        col.label(text=clip.split("$")[-1] if "$" in clip else clip)
    if len(names) > MAX_LISTED:
        col.label(text=f"... and {len(names) - MAX_LISTED} more (scroll to see the rest)")
