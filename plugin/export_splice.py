"""Splice authored actions back into a JWE3 .manis bundle.

The JWE3 path, and NOT "Export Manis": `ManisFile.save()` writes uncompressed
dtype 0 with no ACL blobs, dropping the database and limb data. Here each edited
action is re-encoded self-contained and spliced into the source bundle, so every
clip you did not touch stays byte-identical - rest03 included.

Bones vanilla stripped are kept when you actually animated them, and their bone
mask bits are set, which is what makes them move in game.
"""
import logging
import os

import bpy
import numpy as np

from generated.formats.manis import ManisFile
from generated.formats.manis.acl import decode_file
from plugin.modules_export.jacl import sample_action
from plugin.modules_export.armature import get_armature
from plugin.import_manis import anim_sys
from plugin.modules_export.jacl import write_jacl

CHANGED_EPS = 1e-5


def _curve_count(action):
    """How many fcurves the action actually holds, on either Blender API."""
    try:
        return len(anim_sys.get_data(action).fcurves)
    except Exception:
        return 0


def _clip_of(action):
    """The clip name. The importer names the action after the ManiInfo."""
    return action.name


def _bundle_of(action):
    """Which .manis file this clip came from - stamped by the importer.

    Bundle names are opaque hashes and a species ships a dozen of them, so this
    stamp is the only way to send an edited clip back to the right one.
    """
    return action.get("manis")


def save(reporter, filepath="", source_folder="", ms2_path="",
         action_source="CHANGED", unstrip=True):
    """Splice edited actions into their source bundles.

    `source_folder` holds the vanilla bundles; each action is routed back to the
    bundle its importer stamp names. The sources are never written to - results go
    to the folder `filepath` points at, so a bad export cannot destroy the
    originals.
    """
    scene = bpy.context.scene
    if scene.cobra.game != "Jurassic World Evolution 3":
        raise ValueError("this exporter is JWE3-only; the mask and ACL layout differ")
    source_folder = bpy.path.abspath(source_folder or "")
    if not os.path.isdir(source_folder):
        raise ValueError("set 'Source Bundles' to the folder holding the vanilla .manis files")
    ms2_path = bpy.path.abspath(ms2_path or "")
    if not os.path.isfile(ms2_path):
        raise ValueError("set 'MS2' to the species models.ms2 - it supplies the bind pose")
    out_dir = os.path.dirname(bpy.path.abspath(filepath)) or source_folder
    if os.path.normcase(out_dir) == os.path.normcase(source_folder):
        raise ValueError("choose an output folder that is not the source folder, "
                         "so the vanilla bundles are never overwritten")

    b_armature_ob = get_armature(scene.objects)
    if b_armature_ob is None:
        raise ValueError("no armature in the scene")
    anim = b_armature_ob.animation_data
    active = anim.action if anim else None

    if action_source == "ACTIVE":
        candidates = [active] if active else []
    elif action_source == "SELECTED":
        candidates = [a for a in bpy.data.actions if a.get("cobra_splice")]
        if not candidates:
            raise ValueError("no actions are ticked for export - use the Cobra panel, "
                             "or switch mode to Changed/Active")
    else:
        candidates = list(bpy.data.actions)
    candidates = [a for a in candidates if a is not None and _bundle_of(a)]
    if not candidates:
        raise ValueError("no actions carry a bundle stamp - re-import them")

    by_bundle = {}
    for action in candidates:
        by_bundle.setdefault(_bundle_of(action), []).append(action)

    os.makedirs(out_dir, exist_ok=True)
    total_clips = total_skipped = total_bones = 0
    for bundle_name, actions in sorted(by_bundle.items()):
        src = os.path.join(source_folder, bundle_name)
        if not os.path.isfile(src):
            reporter.show_warning(f"{bundle_name} not in the source folder - skipped")
            continue
        manis = ManisFile()
        manis.game = "Jurassic World Evolution 3"
        manis.load(src)
        names = [str(i.name) for i in manis.mani_infos]
        templates = dict(zip(names, [s for s in decode_file(src) if s.track_type == 12]))
        edits = {}
        for action in actions:
            clip = _clip_of(action)
            stream = templates.get(clip)
            if stream is None:
                continue
            template = stream.values
            # An action can exist but hold nothing - an import that raised part way
            # leaves the action created and empty. Sampling that yields one frame and
            # would abort the whole export over a clip the animator never touched.
            if not _curve_count(action):
                reporter.show_warning(f"{clip}: no animation data in this action "
                                      f"(failed import?) - skipped")
                total_skipped += 1
                continue
            try:
                values, _names, _unauth = sample_action(b_armature_ob, action, 30.0003)
            except Exception as err:
                reporter.show_warning(f"{clip}: could not be sampled ({err}) - skipped")
                total_skipped += 1
                continue
            if values.shape[0] == template.shape[0] + 1:
                values = values[:-1]      # wrap-optimised implicit last frame
            if action_source == "CHANGED" and values.shape == template.shape:
                # Only clips actually touched. Our encoder is not Frontier's, so
                # re-encoding an untouched clip is a real quality loss for nothing.
                both = ~np.isnan(template) & ~np.isnan(values)
                if both.any() and np.abs(template[both] - values[both]).max() < CHANGED_EPS:
                    total_skipped += 1
                    continue
            edits[clip] = values
        if not edits:
            continue
        out = os.path.join(out_dir, bundle_name)
        logging.info(f"splicing {len(edits)} clip(s) into {bundle_name}")
        total_bones += _rebuild(src, out, edits, ms2_path, unstrip)
        total_clips += len(edits)

    if not total_clips:
        reporter.show_info(f"nothing to export - {total_skipped} action(s) match their bundle")
        return
    reporter.show_info(
        f"Spliced {total_clips} clip(s) into {out_dir}; {total_skipped} unchanged and "
        f"left byte-identical"
        + (f"; {total_bones} stripped bone(s) enabled" if total_bones else ""))


def _rebuild(src, out, edits, ms2_path, unstrip):
    """Rebuild the bundle with the edited clips, via the game-verified path.

    Calls manis_database_cmd, which rebuilds the ACL database over every clip. That
    re-encodes clips we did not edit, which is a real cost - but the surgical
    alternative leaves the edited clip SELF-CONTAINED (has_database=0) while its
    original bulk is still in the database, and the engine then alternates between
    the two as it streams LOD tiers: the animal blinks between poses in game.
    Correctness wins. One clip per invocation, chained.
    """
    import sys, tempfile, shutil
    import manis_database_cmd

    enabled = 0
    with tempfile.TemporaryDirectory(prefix="cobra_splice_") as tmp:
        current = src
        for n, (clip, values) in enumerate(edits.items()):
            jacl = os.path.join(tmp, f"clip{n}.jacl")
            write_jacl(jacl, values, 30.0003)
            step = os.path.join(tmp, f"step{n}.manis")
            argv = [
                "manis_database_cmd.py", current, "--out", step,
                "--ms2", ms2_path,
                "--replace-clip", clip, "--jacl", jacl,
            ]
            if unstrip:
                argv.append("--unstrip")
            saved = sys.argv
            try:
                sys.argv = argv
                rc = manis_database_cmd.main()
            except SystemExit as err:
                raise RuntimeError(f"{clip}: rebuild failed - {err}") from err
            finally:
                sys.argv = saved
            if rc not in (0, None):
                raise RuntimeError(f"{clip}: rebuild returned {rc}")
            current = step
        shutil.copyfile(current, out)
    return enabled
