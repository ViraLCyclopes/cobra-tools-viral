import struct
import hashlib
from types import SimpleNamespace

import pytest
import ovl_tool_cmd
import source.formats.motiongraph.edit as edit_module

from source.formats.motiongraph.edit import (
    apply_patch_plan,
    build_patch_plan,
    curve_decode,
    curve_encode,
    load_plan,
    locate_fields,
    merge_patch_plans,
    save_plan,
)
from source.formats.motiongraph.report import derive_label, strip_species


def field_row(source=None, kind="scalar", type_name="Float", value=1.0, raw=None):
    if raw is None:
        raw = struct.pack("<f", value).hex()
    row = {
        "activity": "Species$StandIdle01",
        "activity_type": "AnimationActivity",
        "pool": 1,
        "payload_offset": 0,
        "fields": [{
            "path": "speed.float",
            "type": type_name,
            "kind": kind,
            "pool": 10,
            "offset": 32,
            "hex": raw,
            "value": value,
            "verified": True,
        }],
    }
    if source is not None:
        source = source.resolve()
        row["provenance"] = {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "motiongraph": "species.motiongraph",
        }
    return [row]


@pytest.mark.parametrize("value, encoded", [(0.0, 0x4000), (1.0, 0x4040), (30.0, 0x4200)])
def test_curve_codec_matches_frontier_values(value, encoded):
    assert curve_encode(value) == encoded
    assert curve_decode(encoded) == value


def test_scalar_plan_is_fixed_width_and_self_verifying(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    plan = build_patch_plan(
        field_row(source), source, "species.motiongraph", "speed.float", value=1.25,
    )
    assert plan["format"] == "cobra-motiongraph-patch-v1"
    assert plan["kind"] == "scalar"
    assert plan["edits"] == [{
        "pool": 10,
        "offset": 32,
        "expected": struct.pack("<f", 1.0).hex(),
        "replacement": struct.pack("<f", 1.25).hex(),
        "note": "Species$StandIdle01.speed.float",
    }]
    assert len(bytes.fromhex(plan["edits"][0]["expected"])) == len(
        bytes.fromhex(plan["edits"][0]["replacement"])
    )


def test_plan_auto_resolves_unique_motiongraph_from_scan_provenance(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    plan = build_patch_plan(
        field_row(source), source, None, "speed.float", value=1.25,
    )
    assert plan["motiongraph"] == "species.motiongraph"


def test_plan_still_rejects_explicit_motiongraph_mismatch(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    with pytest.raises(ValueError, match="different motiongraph"):
        build_patch_plan(
            field_row(source), source, "other.motiongraph", "speed.float", value=1.25,
        )


@pytest.mark.parametrize("name", [None, "species.motiongraph"])
def test_cli_plan_accepts_omitted_or_matching_name(tmp_path, monkeypatch, name):
    source, plan_path = tmp_path / "stock.ovl", tmp_path / "plan.json"
    source.write_bytes(b"stock")
    monkeypatch.setattr(
        edit_module, "locate_fields", lambda *_args, **_kwargs: (field_row(source), {}))
    args = SimpleNamespace(
        ovl=str(source), game="Jurassic World Evolution 3", name=name,
        activity=None, activity_type=None, field="speed.float", json=None,
        set_value=1.25, set_flag=[], set_enum=None, plan=str(plan_path),
    )
    ovl_tool_cmd.cmd_motiongraph_fields(args)
    assert load_plan(plan_path)["motiongraph"] == "species.motiongraph"


def test_cli_plan_rejects_explicit_wrong_name(tmp_path, monkeypatch):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    monkeypatch.setattr(
        edit_module, "locate_fields", lambda *_args, **_kwargs: (field_row(source), {}))
    args = SimpleNamespace(
        ovl=str(source), game="Jurassic World Evolution 3", name="wrong.motiongraph",
        activity=None, activity_type=None, field="speed.float", json=None,
        set_value=1.25, set_flag=[], set_enum=None, plan=str(tmp_path / "plan.json"),
    )
    with pytest.raises(SystemExit):
        ovl_tool_cmd.cmd_motiongraph_fields(args)


def test_duplicate_addresses_are_patched_once(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    rows = field_row(source) + field_row(source)
    plan = build_patch_plan(
        rows, source, "species.motiongraph", "speed.float", value=2.0,
    )
    assert len(plan["edits"]) == 1


def test_mixed_value_plans_merge_into_one_atomic_queue(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    first = build_patch_plan(
        field_row(source), source, "species.motiongraph", "speed.float", value=1.25,
    )
    second_rows = field_row(source)
    second_rows[0]["fields"][0]["offset"] = 40
    second = build_patch_plan(
        second_rows, source, "species.motiongraph", "blend_time", value=2.5,
    )
    merged = merge_patch_plans([first, second])
    assert merged["field"] == "multiple"
    assert merged["kind"] == "mixed"
    assert [operation["value"] for operation in merged["operations"]] == [1.25, 2.5]
    assert [(edit["pool"], edit["offset"]) for edit in merged["edits"]] == [(10, 32), (10, 40)]


def test_merge_deduplicates_an_identical_queued_operation(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    plan = build_patch_plan(
        field_row(source), source, "species.motiongraph", "speed.float", value=1.25,
    )
    merged = merge_patch_plans([plan, plan])
    assert len(merged["operations"]) == 1
    assert len(merged["edits"]) == 1


def test_merge_rejects_conflicting_or_overlapping_edits(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    first = build_patch_plan(
        field_row(source), source, "species.motiongraph", "speed.float", value=1.25,
    )
    conflict = build_patch_plan(
        field_row(source), source, "species.motiongraph", "speed.float", value=2.5,
    )
    with pytest.raises(ValueError, match="Conflicting queued edits"):
        merge_patch_plans([first, conflict])
    overlap = {**conflict, "edits": [{**conflict["edits"][0], "offset": 34}]}
    with pytest.raises(ValueError, match="Overlapping queued edits"):
        merge_patch_plans([first, overlap])


def test_saved_composite_queue_loads_and_validates(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    plan = build_patch_plan(
        field_row(source), source, "species.motiongraph", "speed.float", value=1.25,
    )
    path = tmp_path / "queue.json"
    save_plan(path, merge_patch_plans([plan]))
    loaded = load_plan(path)
    assert loaded["source_sha256"] == plan["source_sha256"]
    assert loaded["edits"] == plan["edits"]


def test_unverified_address_refuses_a_plan(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    rows = field_row(source)
    rows[0]["fields"][0]["verified"] = False
    with pytest.raises(ValueError, match="failed verification"):
        build_patch_plan(
            rows, source, "species.motiongraph", "speed.float", value=2.0,
        )


def test_mixed_field_layouts_refuse_a_plan(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    rows = field_row(source)
    other = field_row(source, kind="curve", type_name="CurveValue", value=0x4000, raw="0040")
    rows.extend(other)
    with pytest.raises(ValueError, match="mixed field layouts"):
        build_patch_plan(
            rows, source, "species.motiongraph", "speed.float", value=2.0,
        )


def test_apply_never_overwrites_source(tmp_path):
    source = tmp_path / "source.ovl"
    source.write_bytes(b"not needed: refusal happens before parsing")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        apply_patch_plan(source, source, {"edits": [{"dummy": True}]})


def test_exact_activity_selector_requires_pool_and_offset(tmp_path):
    with pytest.raises(ValueError, match="both pool and offset"):
        locate_fields(tmp_path / "unused.ovl", activity_pool=4)


def test_activity_selector_rejects_mixed_exact_and_set_scope(tmp_path):
    with pytest.raises(ValueError, match="either one exact activity"):
        locate_fields(
            tmp_path / "unused.ovl", activity_pool=4, activity_offset=8,
            activity_addresses=[(4, 8)],
        )


def test_report_strips_only_the_species_prefix():
    assert strip_species("Acrocanthosaurus_Female$RestToStand") == "RestToStand"
    assert strip_species("RestToStand") == "RestToStand"


def test_placeholder_state_uses_a_distinctive_signal_label():
    label = derive_label(
        {0: ["Species$BindPose"]},
        {"DataStreamProducerActivity": 1},
        ["ChainDynamicsEnabled", "ClimbFence"],
        {"ChainDynamicsEnabled": 150, "ClimbFence": 1},
        193,
    )
    assert label == "~ClimbFence"
