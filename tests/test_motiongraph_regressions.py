import hashlib
import os
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from source.formats.motiongraph import datastream_growth as growth
from source.formats.motiongraph.edit import build_patch_plan
from source.formats.motiongraph.staging import (
    CandidatePublication,
    copy_motiongraph_family,
    motiongraph_family,
    validate_staged_pair,
)
from source.formats.motiongraph import static_patch
from source.formats.motiongraph import staging


def test_batch_rejects_same_source_before_target_discovery(tmp_path, monkeypatch):
    source = tmp_path / "animal.ovl"
    source.write_bytes(b"original")
    discovered = False

    def discover(*_args):
        nonlocal discovered
        discovered = True
        return [{"activity": "Species$A"}]

    monkeypatch.setattr(growth, "activities_with_stream", discover)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        growth.set_curve_everywhere(source, source, "VFX", [(0, 0)])
    assert not discovered
    assert source.read_bytes() == b"original"


def test_batch_rejects_wrong_basename_and_missing_stage(tmp_path, monkeypatch):
    source = tmp_path / "animal.ovl"
    source.write_bytes(b"original")
    monkeypatch.setattr(growth, "activities_with_stream", lambda *_: pytest.fail("discovered"))
    other = tmp_path / "other.ovl"
    other.write_bytes(b"stage")
    with pytest.raises(ValueError, match="basenames"):
        growth.set_curve_everywhere(source, other, "VFX", [(0, 0)])
    missing = tmp_path / "stage" / source.name
    missing.parent.mkdir()
    with pytest.raises(ValueError, match="complete archive family"):
        growth.set_curve_everywhere(source, missing, "VFX", [(0, 0)])


def test_same_file_alias_is_rejected(tmp_path):
    source = tmp_path / "animal.ovl"
    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    alias = alias_dir / source.name
    source.write_bytes(b"original")
    try:
        os.link(source, alias)
    except OSError:
        pytest.skip("hard links are unavailable")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        validate_staged_pair(source, alias)


def test_family_copy_keeps_hash_aux_and_excludes_unrelated_files(tmp_path):
    source = tmp_path / "animal.ovl"
    source.write_bytes(b"ovl")
    (tmp_path / "animal.ovs.anim_l0").write_bytes(b"ovs")
    (tmp_path / "04b16c7f.aux").write_bytes(b"aux")
    (tmp_path / "notes.txt").write_text("unrelated")
    destination = tmp_path / "stage"
    copied = copy_motiongraph_family(source, destination)
    assert {p.name for p in copied} == {
        "animal.ovl", "animal.ovs.anim_l0", "04b16c7f.aux"}
    assert not (destination / "notes.txt").exists()


def test_validation_rejects_incomplete_staged_family(tmp_path):
    source_dir, stage_dir = tmp_path / "source", tmp_path / "stage"
    source_dir.mkdir(); stage_dir.mkdir()
    source, output = source_dir / "animal.ovl", stage_dir / "animal.ovl"
    source.write_bytes(b"source")
    output.write_bytes(b"stage")
    (source_dir / "hash.aux").write_bytes(b"aux")
    with pytest.raises(ValueError, match="incomplete.*hash.aux"):
        validate_staged_pair(source, output)


def _row(source, source_hash, graph="species.motiongraph"):
    return [{
        "activity": "Species$Idle", "activity_type": "AnimationActivity",
        "provenance": {"source": str(source.resolve()),
                       "source_sha256": source_hash, "motiongraph": graph},
        "fields": [{"path": "speed.float", "type": "Float", "kind": "scalar",
                    "pool": 1, "offset": 4, "hex": "0000803f", "value": 1.0,
                    "verified": True}],
    }]


def test_planner_rejects_unknown_and_foreign_scan_provenance(tmp_path):
    first = tmp_path / "first.ovl"
    second = tmp_path / "second.ovl"
    first.write_bytes(b"same-address-value")
    second.write_bytes(b"different source")
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    rows = _row(first, digest)
    with pytest.raises(ValueError, match="different source"):
        build_patch_plan(rows, second, "species.motiongraph", "speed.float", value=2)
    rows[0].pop("provenance")
    with pytest.raises(ValueError, match="unknown source provenance"):
        build_patch_plan(rows, first, "species.motiongraph", "speed.float", value=2)


def test_exact_clip_wins_before_substring_and_duplicate_exact_needs_identity():
    Activity = type("AnimationActivityData", (), {})
    short, long = Activity(), Activity()
    short.mani, long.mani = "Species$StandPreen", "Species$StandPreen02"
    Pool = type("Pool", (), {})
    p1, p2 = Pool(), Pool()
    p1.i, p2.i = 4, 5
    loader = SimpleNamespace(context=SimpleNamespace(recursion={(p1, 10): short, (p2, 20): long}))
    assert growth.find_activity(loader, lambda value: value, "Species$StandPreen")[:3] == (
        4, 10, "Species$StandPreen")
    duplicate = Activity()
    duplicate.mani = short.mani
    loader.context.recursion[(p2, 30)] = duplicate
    with pytest.raises(ValueError, match="ambiguous"):
        growth.find_activity(loader, lambda value: value, short.mani)
    assert growth.find_activity(loader, lambda value: value, short.mani, (5, 30))[1] == 30


def test_candidate_failure_preserves_previous_stage_and_companions(tmp_path):
    source_dir, stage_dir = tmp_path / "source", tmp_path / "stage"
    source_dir.mkdir(); stage_dir.mkdir()
    source, output = source_dir / "animal.ovl", stage_dir / "animal.ovl"
    source.write_bytes(b"source")
    output.write_bytes(b"last good")
    (source_dir / "hash.aux").write_bytes(b"aux")
    (stage_dir / "hash.aux").write_bytes(b"aux")
    publication = CandidatePublication(source, output)
    publication.path.write_bytes(b"bad candidate")
    del publication
    assert output.read_bytes() == b"last good"
    assert (stage_dir / "hash.aux").read_bytes() == b"aux"


def test_atomic_replace_failure_preserves_previous_stage(tmp_path, monkeypatch):
    source_dir, stage_dir = tmp_path / "source", tmp_path / "stage"
    source_dir.mkdir(); stage_dir.mkdir()
    source, output = source_dir / "animal.ovl", stage_dir / "animal.ovl"
    source.write_bytes(b"source")
    output.write_bytes(b"last good")
    publication = CandidatePublication(source, output)
    publication.path.write_bytes(b"verified candidate")
    monkeypatch.setattr(staging.os, "replace", lambda *_: (_ for _ in ()).throw(PermissionError("locked")))
    with pytest.raises(PermissionError, match="locked"):
        publication.commit()
    assert output.read_bytes() == b"last good"


def test_successful_candidate_atomically_replaces_only_ovl(tmp_path):
    source_dir, stage_dir = tmp_path / "source", tmp_path / "stage"
    source_dir.mkdir(); stage_dir.mkdir()
    source, output = source_dir / "animal.ovl", stage_dir / "animal.ovl"
    source.write_bytes(b"source")
    output.write_bytes(b"last good")
    publication = CandidatePublication(source, output)
    publication.path.write_bytes(b"verified candidate")
    publication.commit()
    assert source.read_bytes() == b"source"
    assert output.read_bytes() == b"verified candidate"


def test_candidate_rejects_stale_destination_companions(tmp_path):
    source_dir, stage_dir = tmp_path / "source", tmp_path / "stage"
    source_dir.mkdir(); stage_dir.mkdir()
    source, output = source_dir / "animal.ovl", stage_dir / "animal.ovl"
    source.write_bytes(b"source")
    output.write_bytes(b"last good")
    (source_dir / "animal.ovs.anim").write_bytes(b"source companion")
    (stage_dir / "animal.ovs.anim").write_bytes(b"stale companion")
    with pytest.raises(ValueError, match="companions do not match"):
        CandidatePublication(source, output)
    assert output.read_bytes() == b"last good"


def test_candidate_rejects_companion_change_before_commit(tmp_path):
    source_dir, stage_dir = tmp_path / "source", tmp_path / "stage"
    source_dir.mkdir(); stage_dir.mkdir()
    source, output = source_dir / "animal.ovl", stage_dir / "animal.ovl"
    source.write_bytes(b"source")
    output.write_bytes(b"last good")
    (source_dir / "hash.aux").write_bytes(b"matching")
    staged_aux = stage_dir / "hash.aux"
    staged_aux.write_bytes(b"matching")
    publication = CandidatePublication(source, output)
    publication.path.write_bytes(b"verified candidate")
    staged_aux.write_bytes(b"changed concurrently")
    with pytest.raises(ValueError, match="companions changed"):
        publication.commit()
    assert output.read_bytes() == b"last good"


@pytest.mark.parametrize("pairs,expected", [
    ([("A", "LongTarget")], ["relocate"]),
    ([("LongDonor", "B"), ("A", "LongTarget")], ["patch", "relocate"]),
])
def test_audio_rename_supports_relocation_only_and_mixed_batches(
        tmp_path, monkeypatch, pairs, expected):
    import numpy as np

    source_dir, stage_dir = tmp_path / "source", tmp_path / "stage"
    source_dir.mkdir(); stage_dir.mkdir()
    source, output = source_dir / "animal.ovl", stage_dir / "animal.ovl"
    source.write_bytes(b"source")
    output.write_bytes(b"last good")
    calls = []

    def patch(source_path, output_path, work, **_kwargs):
        calls.append("patch")
        output_path.write_bytes(b"short")
        return SimpleNamespace(skipped=[])

    def relocate(source_path, output_path, work, **_kwargs):
        calls.append("relocate")
        if "patch" in expected:
            assert source_path.read_bytes() == b"short"
        output_path.write_bytes(b"relocated")
        return SimpleNamespace(skipped=[], fragments_repointed=len(work))

    targets = b"\0".join(new.encode("ascii") for _old, new in pairs) + b"\0"
    dtype = np.dtype([("link_pool", "<u4"), ("link_offset", "<u4"),
                      ("struct_pool", "<u4"), ("struct_offset", "<u4")])
    offsets, cursor = [], 0
    for _old, new in pairs:
        offsets.append(cursor)
        cursor += len(new) + 1
    fragments = np.array([(0, index * 8, 0, offset)
                          for index, offset in enumerate(offsets)], dtype=dtype)
    fake_static = SimpleNamespace(content=SimpleNamespace(
        pools=[SimpleNamespace(type=2, data=io.BytesIO(targets))], fragments=fragments))

    monkeypatch.setattr(static_patch, "patch_string_slots", patch)
    monkeypatch.setattr(static_patch, "relocate_strings", relocate)
    monkeypatch.setattr(static_patch, "_load", lambda *_: (None, fake_static))
    report = static_patch.apply_string_renames(source, output, pairs)
    assert calls == expected
    assert output.read_bytes() == b"relocated"
    assert report.in_place == (1 if "patch" in expected else 0)
