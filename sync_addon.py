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
BLENDER_ROOT = os.path.join(os.environ.get("APPDATA", ""), "Blender Foundation", "Blender")


def installed_versions():
    """Blender versions that have a cobra-tools add-on folder, newest last."""
    if not os.path.isdir(BLENDER_ROOT):
        return []
    found = []
    for name in os.listdir(BLENDER_ROOT):
        path = os.path.join(BLENDER_ROOT, name, "scripts", "addons", "cobra-tools-master")
        if os.path.isdir(path):
            try:
                key = tuple(int(x) for x in name.split("."))
            except ValueError:
                key = (0,)
            found.append((key, name, path))
    return [(n, p) for _k, n, p in sorted(found)]


def addon_path(version=None):
    """Resolve which add-on copy to sync into.

    The version was hard-coded to 4.5, which silently synced into an install the
    user was not running - edits appeared to have no effect in Blender. Default to
    the NEWEST installed copy and print which one, so the target is never a guess.
    """
    installed = installed_versions()
    if version:
        for name, path in installed:
            if name == version:
                return path
        raise SystemExit(f"no cobra-tools add-on for Blender {version}; "
                         f"found: {', '.join(n for n, _ in installed) or 'none'}")
    if not installed:
        raise SystemExit(f"no cobra-tools add-on found under {BLENDER_ROOT}")
    return installed[-1][1]


ADDON = addon_path()

# the trees Blender actually imports from; bin/ carries the ACL decoder exe
TREES = ("plugin", "source", "generated", "modules", "constants", "utils", "bin")
# manis_database_cmd carries the game-verified bundle rebuild; the Blender splice
# exporter calls it, so it has to reach the add-on copy too.
ROOT_FILES = ("__init__.py", "__version__.py", "manis_database_cmd.py")

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
    parser.add_argument("--blender", metavar="VERSION",
                        help="Blender version to sync into, e.g. 5.2. Defaults to the "
                             "newest installed cobra-tools add-on.")
    parser.add_argument("--addon", default=None,
                        help="destination add-on folder (overrides --blender)")
    args = parser.parse_args()

    if args.addon is None:
        args.addon = addon_path(args.blender)
    versions = ", ".join(n for n, _ in installed_versions()) or "none"
    print(f"add-on installs found: {versions}")
    print(f"syncing into: {args.addon}")

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
