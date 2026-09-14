"""Pinned-family integration checks; run explicitly because they copy large assets."""

import hashlib
from pathlib import Path

import pytest

import source.formats.motiongraph.clone as clone_module
from source.formats.motiongraph.clone import clone_complete_animation_activity
from source.formats.motiongraph.edit import apply_patch_plan, locate_fields
from source.formats.motiongraph.edit import load_motiongraph
from source.formats.motiongraph.report import build_deref
from source.formats.motiongraph.audio_events import scan_graph_event_names
from source.formats.motiongraph.static_patch import apply_string_renames
from source.formats.motiongraph.datastream_growth import (
    activities_with_stream,
    set_curve_everywhere,
)
from source.formats.motiongraph.staging import copy_motiongraph_family, family_hashes


FIXTURE = Path(
    r"D:\JWE2 Stuff\Cobra Tool Versions\Main Mod Kit\JWE 3 Luas\Base Game"
    r"\Dinosaur Files\Motiongraph Research\Appended Clip String Test X"
    r"\Source Test W\Acrocanthosaurus_Female.ovl"
)


def test_null_static_operation_is_byte_identical_and_preserves_family(tmp_path):
    if not FIXTURE.is_file():
        pytest.skip("Pinned Acrocanthosaurus fixture is unavailable")
    copy_motiongraph_family(FIXTURE, tmp_path / "source")
    source = tmp_path / "source" / FIXTURE.name
    copy_motiongraph_family(source, tmp_path / "stage")
    output = tmp_path / "stage" / FIXTURE.name
    source_family = family_hashes(source)
    stage_companions = {name: digest for name, digest in family_hashes(output).items()
                        if name != output.name.lower()}

    rows, _mismatches = locate_fields(source)
    field = next(item for row in rows for item in row["fields"] if item["verified"])
    plan = {
        "format": "cobra-motiongraph-patch-v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "motiongraph": None,
        "edits": [{"pool": field["pool"], "offset": field["offset"],
                   "expected": field["hex"], "replacement": field["hex"],
                   "note": "null-operation fidelity gate"}],
    }
    report = apply_patch_plan(source, output, plan)
    assert report.changed_bytes == 0
    assert hashlib.sha256(output.read_bytes()).hexdigest() == source_family[output.name.lower()]
    assert {name: digest for name, digest in family_hashes(output).items()
            if name != output.name.lower()} == stage_companions


def test_longer_only_audio_batch_relocates_in_one_private_operation(tmp_path):
    if not FIXTURE.is_file():
        pytest.skip("Pinned Acrocanthosaurus fixture is unavailable")
    copy_motiongraph_family(FIXTURE, tmp_path / "source")
    source = tmp_path / "source" / FIXTURE.name
    copy_motiongraph_family(source, tmp_path / "stage")
    output = tmp_path / "stage" / FIXTURE.name
    original_hashes = family_hashes(source)
    stage_companions = {name: digest for name, digest in family_hashes(output).items()
                        if name != output.name.lower()}
    names = scan_graph_event_names(source, "Acrocanthosaurus",
                                   "Jurassic World Evolution 3")[:2]
    pairs = [(name, "AcrocanthosaurusCE" + name[len("Acrocanthosaurus"):])
             for name in names]
    report = apply_string_renames(source, output, pairs)
    assert report.in_place == 0 and report.relocated == 2
    assert family_hashes(source) == original_hashes
    assert hashlib.sha256(output.read_bytes()).hexdigest() != original_hashes[output.name.lower()]
    assert {name: digest for name, digest in family_hashes(output).items()
            if name != output.name.lower()} == stage_companions


def test_batch_curves_mutate_all_targets_in_one_published_result(tmp_path):
    if not FIXTURE.is_file():
        pytest.skip("Pinned Acrocanthosaurus fixture is unavailable")
    copy_motiongraph_family(FIXTURE, tmp_path / "source")
    source = tmp_path / "source" / FIXTURE.name
    copy_motiongraph_family(source, tmp_path / "stage")
    output = tmp_path / "stage" / FIXTURE.name
    original_hashes = family_hashes(source)
    targets = activities_with_stream(source, "Acrocanthosaurus_BreathSlow")
    assert len(targets) == 4
    applied = set_curve_everywhere(
        source, output, "Acrocanthosaurus_BreathSlow",
        [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)],
        ds_type="AudioLoopingEvent", curve_type=0)
    assert len(applied) == 4
    assert family_hashes(source) == original_hashes
    assert {name: digest for name, digest in family_hashes(output).items()
            if name != output.name.lower()} == {
                name: digest for name, digest in original_hashes.items()
                if name != output.name.lower()}


@pytest.mark.parametrize("failure_gate", ["semantic", "census"])
def test_clone_final_failure_preserves_previous_stage(tmp_path, monkeypatch, failure_gate):
    if not FIXTURE.is_file():
        pytest.skip("Pinned Acrocanthosaurus fixture is unavailable")
    copy_motiongraph_family(FIXTURE, tmp_path / "source")
    source = tmp_path / "source" / FIXTURE.name
    copy_motiongraph_family(source, tmp_path / "stage")
    output = tmp_path / "stage" / FIXTURE.name
    before = family_hashes(output)

    _ovl, loader = load_motiongraph(source, None, "Jurassic World Evolution 3")
    deref = build_deref(loader)
    address = next(
        (int(pool.i), int(offset))
        for (pool, offset), activity in loader.context.recursion.items()
        if type(activity).__name__ == "Activity"
        and getattr(activity.data_type, "data", None) == "AnimationActivity"
        and deref(activity.data) is not None
    )
    original_activity_at = clone_module._activity_at

    def fail_final_candidate_reload(path, *args, **kwargs):
        if Path(path).parent.name.startswith("motiongraph-candidate-"):
            raise ValueError("injected final clone semantic validation failure")
        return original_activity_at(path, *args, **kwargs)

    if failure_gate == "semantic":
        monkeypatch.setattr(clone_module, "_activity_at", fail_final_candidate_reload)
    else:
        monkeypatch.setattr(
            clone_module, "diff_census",
            lambda *_args: (_ for _ in ()).throw(
                ValueError("injected final clone census validation failure")))
    with pytest.raises(ValueError, match="injected final clone"):
        clone_complete_animation_activity(source, output, *address)
    assert family_hashes(output) == before
