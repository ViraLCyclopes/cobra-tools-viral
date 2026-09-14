"""Splice authored actions back into a JWE3 .manis bundle.

The JWE3 path, and NOT "Export Manis": `ManisFile.save()` writes uncompressed
dtype 0 with no ACL blobs, dropping the database and limb data. Here each edited
action is re-encoded self-contained and spliced into the source bundle, so every
clip you did not touch stays byte-identical - rest03 included.

Bones vanilla stripped are kept when you actually animated them, and their bone
mask bits are set, which is what makes them move in game.
"""
import contextlib
import logging
import os
import tempfile

import bpy
import numpy as np

from generated.formats.manis import ManisFile
from generated.formats.manis.acl import decode_file
from source.formats.manis.channel_growth import rebuild as channel_rebuild
from plugin.modules_export.jacl import (
    bundle_track_index, missing_channels, sample_action, to_clip_tracks)
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


def _same_dir(a, b):
    """Whether two directory paths point at the same folder.

    `os.path.normcase` alone is not enough - one side comes from a Blender path
    property and can carry a trailing separator.
    """
    return (os.path.normcase(os.path.normpath(os.path.realpath(a)))
            == os.path.normcase(os.path.normpath(os.path.realpath(b))))


def _matches_template(template, values):
    """Whether sampled values are the clip's own animation, unchanged.

    Quaternions double-cover rotations: `q` and `-q` are the same orientation, and
    a round trip through Blender flips the sign of the odd one. Comparing raw
    components then reports a difference of up to 2.0 for an identical pose, which
    marked untouched clips as edited and re-encoded them for nothing - exactly what
    "Changed Only" exists to prevent. Measured on this bundle: one flipped
    quaternion in 9,072 made `eat03` look like a 1.79 change when it was 2.4e-07.
    """
    rot_t, rot_v = template[..., 0:4], values[..., 0:4]
    valid = (~np.isnan(rot_t) & ~np.isnan(rot_v)).all(axis=-1)
    if valid.any():
        a, b = rot_t[valid], rot_v[valid]
        flipped = (a * b).sum(axis=-1) < 0
        b = np.where(flipped[:, None], -b, b)
        if np.abs(a - b).max() >= CHANGED_EPS:
            return False
    rest_t, rest_v = template[..., 4:], values[..., 4:]
    both = ~np.isnan(rest_t) & ~np.isnan(rest_v)
    if both.any() and np.abs(rest_t[both] - rest_v[both]).max() >= CHANGED_EPS:
        return False
    return True


def _bundle_of(action):
    """Which .manis file this clip came from - stamped by the importer.

    Bundle names are opaque hashes and a species ships a dozen of them, so this
    stamp is the only way to send an edited clip back to the right one.
    """
    return action.get("manis")


def save(reporter, filepath="", source_folder="", ms2_path="",
         action_source="CHANGED", unstrip=True, grow_channels=True):
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
    # normpath as well as normcase: the file browser hands back "…\blender_in\"
    # with a trailing separator while dirname() never has one, so a bare normcase
    # compare called them different and the export OVERWROTE the source bundle.
    if _same_dir(out_dir, source_folder):
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
    total_clips = total_skipped = total_bones = total_channels = 0
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
        try:
            tracks = bundle_track_index(manis)
        except Exception as err:
            reporter.show_warning(f"{bundle_name}: {err} - channel growth disabled here")
            tracks = {}
        edits, raw, pending = {}, {}, {}
        bone_names = None      # armature order, identical for every action here
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
                values, bone_names, _unauth = sample_action(b_armature_ob, action, 30.0003)
            except Exception as err:
                reporter.show_warning(f"{clip}: could not be sampled ({err}) - skipped")
                total_skipped += 1
                continue
            if values.shape[0] == template.shape[0] + 1:
                values = values[:-1]      # wrap-optimised implicit last frame
            # A bone the animator posed that this clip has no channel for would be
            # dropped SILENTLY - the export succeeds and the bone never moves. Grow
            # the channel list instead (game-verified 2026-09-07, zero byte cost).
            if grow_channels:
                try:
                    add, unknown = missing_channels(
                        values, bone_names, manis.mani_infos[names.index(clip)], tracks)
                except Exception as err:
                    add, unknown = [], []
                    reporter.show_warning(f"{clip}: could not check channels ({err})")
                for group, bone in unknown:
                    reporter.show_warning(
                        f"{clip}: '{bone}' was animated but has no {group} track in this "
                        f"bundle - it cannot be added and will not move")
                if add:
                    pending.setdefault(clip, []).extend(add)
            raw[clip] = values

        # Grow every missing channel in one pass, then re-read the bundle so the
        # mapping below sees the new channel lists.
        if pending:
            grown = bytes(open(src, "rb").read())
            for clip, adds in pending.items():
                for group, bone, track in adds:
                    reloaded = ManisFile()
                    reloaded.game = "Jurassic World Evolution 3"
                    with tempfile.NamedTemporaryFile(suffix=".manis", delete=False) as tmp:
                        tmp.write(grown)
                        tmp_path = tmp.name
                    try:
                        reloaded.load(tmp_path)
                        grown = channel_rebuild(grown, reloaded, clip, (group, bone, track))
                    finally:
                        with contextlib.suppress(OSError):
                            os.remove(tmp_path)
                    total_channels += 1
                    reporter.show_info(f"{clip}: added a {group} channel for '{bone}'")
            src = os.path.join(out_dir, bundle_name + ".grown")
            with open(src, "wb") as stream:
                stream.write(grown)
            manis = ManisFile()
            manis.game = "Jurassic World Evolution 3"
            manis.load(src)
            names = [str(i.name) for i in manis.mani_infos]
            templates = dict(zip(names, [s for s in decode_file(src) if s.track_type == 12]))

        for action in actions:
            clip = _clip_of(action)
            if clip not in raw or clip not in templates:
                continue
            values, template = raw[clip], templates[clip].values
            # Sampling is in ARMATURE order; a clip's tracks are not. Reorder onto
            # this clip's own track list before anything compares or encodes it -
            # see `clip_track_bones`. Without this the encoder rejects the count
            # outright when they differ, and silently mis-assigns every bone when
            # they happen to match.
            try:
                values = to_clip_tracks(values, bone_names, manis.mani_infos[names.index(clip)])
            except Exception as err:
                reporter.show_warning(f"{clip}: could not map to its track order "
                                      f"({err}) - skipped")
                total_skipped += 1
                continue
            if action_source == "CHANGED" and values.shape == template.shape:
                # Only clips actually touched. Our encoder is not Frontier's, so
                # re-encoding an untouched clip is a real quality loss for nothing.
                if _matches_template(template, values):
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
        + (f"; {total_bones} stripped bone(s) enabled" if total_bones else "")
        + (f"; {total_channels} NEW channel(s) added" if total_channels else ""))


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
