"""Content-diff this repo into the installed Blender add-on copy.

Blender loads cobra-tools from its own addons folder, not from this repo, so an
edit here has no effect until it is copied across. This walks the source trees
that matter, compares by content rather than mtime, and copies only what differs.

`config.json` is never copied: the add-on's copy holds the user's own game paths
and overwriting it repoints their install.

    python sync_addon.py            # copy what differs
    python sync_addon.py --dry-run  # report what would be copied

Blender caches imported modules, so restart Blender after a sync that touches
anything already imported.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
ADDON = os.path.join(
    os.environ.get("APPDATA", ""),
    "Blender Foundation", "Blender", "4.5", "scripts", "addons", "cobra-tools-master",
)

# the trees Blender actually imports from; bin/ carries the ACL decoder exe
TREES = ("plugin", "source", "generated", "modules", "constants", "utils", "bin")
ROOT_FILES = ("__init__.py", "__version__.py")

SKIP_NAMES = {"config.json"}
SKIP_DIRS = {"__pycache__", ".git", "logs", "dumps"}
SKIP_EXTS = {".pyc", ".pyo", ".log"}


def should_skip(name: str) -> bool:
    if name in SKIP_NAMES or os.path.splitext(name)[1].lower() in SKIP_EXTS:
        return True
    # dated editor backups such as DDS.py.bak-20260814 are not part of the add-on
    return ".bak" in name.lower()


def sync_file(src: str, dst: str, dry_run: bool) -> str | None:
    """Copy src over dst if the contents differ. Returns a status word or None."""
    if os.path.isfile(dst):
        # shallow=False forces a content compare, not a stat compare
        if filecmp.cmp(src, dst, shallow=False):
            return None
        status = "update"
    else:
        status = "new"
    if not dry_run:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what differs without copying")
    parser.add_argument("--addon", default=ADDON,
                        help="destination add-on folder")
    args = parser.parse_args()

    if not os.path.isdir(args.addon):
        print(f"add-on folder not found: {args.addon}", file=sys.stderr)
        return 2

    changes = {"new": 0, "update": 0}
    skipped_config = False

    for name in ROOT_FILES:
        src = os.path.join(REPO, name)
        if not os.path.isfile(src):
            continue
        status = sync_file(src, os.path.join(args.addon, name), args.dry_run)
        if status:
            changes[status] += 1
            print(f"  {status:6s} {name}")

    for tree in TREES:
        src_root = os.path.join(REPO, tree)
        if not os.path.isdir(src_root):
            continue
        for dirpath, dirnames, filenames in os.walk(src_root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in sorted(filenames):
                if should_skip(filename):
                    if filename in SKIP_NAMES:
                        skipped_config = True
                    continue
                src = os.path.join(dirpath, filename)
                rel = os.path.relpath(src, REPO)
                status = sync_file(src, os.path.join(args.addon, rel), args.dry_run)
                if status:
                    changes[status] += 1
                    print(f"  {status:6s} {rel}")

    verb = "would copy" if args.dry_run else "copied"
    print(f"\n{verb}: {changes['new']} new, {changes['update']} updated")
    print(f"destination: {args.addon}")
    if skipped_config:
        print("skipped config.json (holds the add-on's own game paths)")
    if changes["new"] or changes["update"]:
        print("restart Blender - it caches imported modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
