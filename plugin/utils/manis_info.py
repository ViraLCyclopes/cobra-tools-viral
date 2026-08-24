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
