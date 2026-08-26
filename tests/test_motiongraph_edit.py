import struct

import pytest

from source.formats.motiongraph.edit import (
    apply_patch_plan,
    build_patch_plan,
    curve_decode,
    curve_encode,
    locate_fields,
)
from source.formats.motiongraph.report import derive_label, strip_species


def field_row(kind="scalar", type_name="Float", value=1.0, raw=None):
    if raw is None:
        raw = struct.pack("<f", value).hex()
    return [{
        "activity": "Species$StandIdle01",
        "activity_type": "AnimationActivity",
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
    }]


@pytest.mark.parametrize("value, encoded", [(0.0, 0x4000), (1.0, 0x4040), (30.0, 0x4200)])
def test_curve_codec_matches_frontier_values(value, encoded):
    assert curve_encode(value) == encoded
    assert curve_decode(encoded) == value


def test_scalar_plan_is_fixed_width_and_self_verifying(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    plan = build_patch_plan(
        field_row(), source, "species.motiongraph", "speed.float", value=1.25,
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


def test_duplicate_addresses_are_patched_once(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    rows = field_row() + field_row()
    plan = build_patch_plan(
        rows, source, "species.motiongraph", "speed.float", value=2.0,
    )
    assert len(plan["edits"]) == 1


def test_unverified_address_refuses_a_plan(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    rows = field_row()
    rows[0]["fields"][0]["verified"] = False
    with pytest.raises(ValueError, match="failed verification"):
        build_patch_plan(
            rows, source, "species.motiongraph", "speed.float", value=2.0,
        )


def test_mixed_field_layouts_refuse_a_plan(tmp_path):
    source = tmp_path / "stock.ovl"
    source.write_bytes(b"stock")
    rows = field_row()
    other = field_row(kind="curve", type_name="CurveValue", value=0x4000, raw="0040")
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
