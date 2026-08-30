from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

from source.formats.motiongraph.capacity import (
    audit_adjacent_growth,
    audit_dead_space,
    census,
    diff_census,
)
from source.formats.motiongraph.clone import (
    clone_fragment_sources,
    exact_inbound_mask,
    inbound_source_mask,
    internal_fragment_mask,
)
from source.formats.motiongraph.surgical_growth import (
    append_tail_pool_bytes,
    plan_tail_pool_allocation,
    repoint_pool_end_fragments,
)


def allocation(offset, kind, *, slack=0, stride=None, count=None, field="value"):
    return {
        "pool": 4, "offset": offset, "kind": kind,
        "zero_tail": slack > 0, "slack_bytes": slack,
        "element_size": stride, "element_count": count,
        "references": [{"owner": "Owner", "field": field}],
    }


def test_prefix_growth_consumes_padding_but_preserves_array_alignment():
    rows = [
        allocation(0, "struct", slack=16),
        allocation(32, "counted-array", stride=8, count=3, field="activities"),
    ]
    candidates = audit_adjacent_growth(rows)
    assert candidates == [{
        "mode": "prefix", "pool": 4, "array_offset": 32,
        "owner": "Owner", "field": "activities", "element_count": 3,
        "element_size": 8, "new_slots": 2, "bytes": 16,
        "preserves_16_byte_alignment": True,
        "neighbor_offset": 0, "neighbor_kind": "struct", "neighbor_slack": 16,
        "array_references": 1, "neighbor_references": 1,
    }]


def test_tail_growth_only_moves_a_pointer_free_string_into_its_padding():
    rows = [
        allocation(0, "counted-array", stride=8, count=3, field="activities"),
        allocation(24, "string", slack=8),
        allocation(48, "struct", slack=8),
        allocation(64, "counted-array", stride=8, count=1),
        allocation(72, "struct", slack=8),
    ]
    candidates = audit_adjacent_growth(rows)
    tail = [candidate for candidate in candidates if candidate["mode"] == "tail-shift-string"]
    assert len(tail) == 1
    assert tail[0]["neighbor_offset"] == 24


def test_prefix_growth_reports_an_eight_aligned_experimental_candidate():
    rows = [
        allocation(0, "struct", slack=8),
        allocation(32, "counted-array", stride=8, count=3, field="activities"),
    ]
    candidate = audit_adjacent_growth(rows)[0]
    assert candidate["new_slots"] == 1
    assert candidate["bytes"] == 8
    assert candidate["preserves_16_byte_alignment"] is False


def dead_allocation(offset, target_type, allocation_bytes, used, dead):
    return {
        "pool": 7, "offset": offset, "kind": "struct", "target_type": target_type,
        "allocation_bytes": allocation_bytes, "used_bytes": used,
        "slack_bytes": dead, "zero_tail": False,
        "element_size": None, "element_count": None,
        "references": [{"owner": "Owner", "field": "value"}],
    }


def test_dead_space_ignores_zero_filled_padding_and_ranks_by_size():
    rows = [
        dead_allocation(0, "Activity", 96, 48, 48),
        dead_allocation(256, "Activity", 720, 48, 672),
        allocation(512, "struct", slack=16),  # zero_tail padding, not dead
    ]

    report = audit_dead_space(rows)

    assert report["allocations_with_dead_tail"] == 2
    assert report["dead_bytes"] == 720
    assert [row["offset"] for row in report["rows"]] == [256, 0]


class FakePool:
    """Hashable stand-in for MemPool; the registry is keyed by pool identity."""

    def __init__(self, index):
        self.i = index


def fake_loader(name, objects):
    """Build a loader stub whose recursion registry holds decoded objects."""
    registry = {
        (FakePool(pool_index), offset): type(type_name, (), {})()
        for pool_index, offset, type_name in objects
    }
    return SimpleNamespace(name=name, context=SimpleNamespace(recursion=registry))


def test_census_counts_decoded_objects_and_records_addresses():
    loader = fake_loader("acro", [(72, 2048, "Activity"), (72, 2096, "Activity"),
                                  (133, 14688, "AnimationActivityData")])

    result = census(loader)

    assert result["total"] == 3
    assert result["counts"] == {"Activity": 2, "AnimationActivityData": 1}
    assert result["addresses"]["Activity"] == [(72, 2048), (72, 2096)]


def test_diff_census_detects_a_silent_orphan_behind_a_flat_total():
    """A full-inbound redirect adds a clone and orphans its donor, netting zero."""
    before = census(fake_loader("acro", [(72, 2048, "Activity")]))
    after = census(fake_loader("acro", [(72, 2144, "Activity")]))

    difference = diff_census(before, after)

    assert difference["total_before"] == difference["total_after"] == 1
    assert difference["added"] == 1 and difference["removed"] == 1
    assert difference["changed_types"]["Activity"]["added"] == [(72, 2144)]
    assert difference["changed_types"]["Activity"]["removed"] == [(72, 2048)]


def test_diff_census_is_empty_when_nothing_moved():
    objects = [(72, 2048, "Activity"), (133, 14688, "AnimationActivityData")]

    difference = diff_census(census(fake_loader("acro", objects)),
                             census(fake_loader("acro", objects)))

    assert difference["changed_types"] == {}
    assert difference["added"] == difference["removed"] == 0


def pool(pool_type, size, num_files=0):
    return SimpleNamespace(
        type=pool_type,
        size=size,
        num_files=num_files,
        data=BytesIO(b"X" * size),
    )


def test_tail_pool_allocation_uses_partial_final_page_and_alignment():
    first = pool(2, 16416, 342)
    tail = pool(2, 2568, 68)
    other = pool(3, 64, 1)
    pools = [first, tail, other]

    plan = plan_tail_pool_allocation(pools, pools, 2, 48)

    assert plan.global_pool == 1
    assert plan.local_pool == 1
    assert plan.old_size == 2568
    assert plan.offset == 2576
    assert plan.padding == 8
    assert plan.new_size == 2624


def test_tail_pool_allocation_refuses_to_overflow_full_page():
    full = pool(2, 16416, 342)

    with pytest.raises(ValueError, match="creating a new page is not supported"):
        plan_tail_pool_allocation([full], [full], 2, 48)


def test_tail_pool_allocation_bound_comes_from_the_largest_pool_of_its_type():
    """Test W game-verified 17,600 in a type-3 pool, so the bound is data-derived."""
    big = pool(3, 24000, 900)
    tail = pool(3, 14976, 700)

    plan = plan_tail_pool_allocation([big, tail], [big, tail], 3, 96)

    assert plan.page_size == 24000
    assert plan.old_size == 14976
    assert plan.new_size == 15072


def test_tail_pool_allocation_still_refuses_to_exceed_what_the_archive_uses():
    tail = pool(3, 6240, 700)

    with pytest.raises(ValueError, match="has no room"):
        plan_tail_pool_allocation([tail], [tail], 3, 32)


def test_append_tail_pool_bytes_preserves_unknown_pool_counters():
    # A larger sibling supplies the bound; the tail itself is what grows.
    big = pool(2, 16416, 342)
    tail = pool(2, 2568, 68)
    payload = bytes(range(48))

    plan = append_tail_pool_bytes([big, tail], [big, tail], 2, payload)

    assert tail.num_files == 68
    assert tail.size == 2624
    assert tail.data.getvalue()[2568:2576] == b"\0" * 8
    assert tail.data.getvalue()[plan.offset:plan.new_size] == payload


def test_repoint_pool_end_fragments_preserves_shared_sentinel():
    fragments = np.array(
        [(1, 8, 7, 6240), (2, 16, 7, 6240), (3, 24, 8, 6240)],
        dtype=[("link_pool", "u4"), ("link_offset", "u4"),
               ("struct_pool", "u4"), ("struct_offset", "u4")],
    )

    moved = repoint_pool_end_fragments(fragments, 7, 6240, 6272)

    assert moved == 2
    assert fragments["struct_offset"].tolist() == [6272, 6272, 6240]


def test_complete_clone_helpers_select_and_translate_exact_sources():
    fragments = np.array(
        [(4, 100, 1, 8), (4, 108, 2, 16), (4, 148, 3, 24), (5, 100, 4, 100)],
        dtype=[("link_pool", "u4"), ("link_offset", "u4"),
               ("struct_pool", "u4"), ("struct_offset", "u4")],
    )

    selected = fragments[internal_fragment_mask(fragments, 4, 100, 48)]
    cloned = clone_fragment_sources(selected, 9, 100, 208)

    assert selected["link_offset"].tolist() == [100, 108]
    assert cloned["link_pool"].tolist() == [9, 9]
    assert cloned["link_offset"].tolist() == [208, 216]
    assert cloned["struct_pool"].tolist() == [1, 2]
    assert cloned["struct_offset"].tolist() == [8, 16]


def test_complete_clone_inbound_mask_matches_exact_target_only():
    fragments = np.array(
        [(1, 0, 7, 64), (2, 0, 7, 64), (3, 0, 7, 80), (4, 0, 8, 64)],
        dtype=[("link_pool", "u4"), ("link_offset", "u4"),
               ("struct_pool", "u4"), ("struct_offset", "u4")],
    )

    assert exact_inbound_mask(fragments, 7, 64).tolist() == [True, True, False, False]


def test_complete_clone_can_select_one_exact_inbound_source():
    fragments = np.array(
        [(1, 16, 7, 64), (2, 32, 7, 64), (2, 48, 7, 80), (3, 32, 8, 64)],
        dtype=[("link_pool", "u4"), ("link_offset", "u4"),
               ("struct_pool", "u4"), ("struct_offset", "u4")],
    )
    inbound = exact_inbound_mask(fragments, 7, 64)

    assert inbound_source_mask(fragments, inbound, ((2, 32),)).tolist() == [
        False, True, False, False,
    ]


def test_complete_clone_rejects_non_inbound_source():
    fragments = np.array(
        [(1, 16, 7, 64)],
        dtype=[("link_pool", "u4"), ("link_offset", "u4"),
               ("struct_pool", "u4"), ("struct_offset", "u4")],
    )
    inbound = exact_inbound_mask(fragments, 7, 64)

    with pytest.raises(ValueError, match="matched 0 exact references"):
        inbound_source_mask(fragments, inbound, ((2, 32),))
