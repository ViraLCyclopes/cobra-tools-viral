"""A full tail must not force growth when orphaned clip slots can be reused."""
import io
from types import SimpleNamespace

import pytest

from source.formats.motiongraph.static_patch import _reuse_unreferenced_clip_slot


def fixture(blob=b'LiveEvent\0Donor$LongUnusedClipName\0'):
    pool = SimpleNamespace(type=2, size=len(blob), data=io.BytesIO(blob))
    fragment = dict(link_pool=1, link_offset=0, struct_pool=0, struct_offset=0)
    ovs = SimpleNamespace(pools=[pool], fragments=[fragment], root_entries=[])
    static = SimpleNamespace(content=ovs)
    ovl = SimpleNamespace(pools=[pool], archives=[static], dependencies=[])
    return ovl, static


def test_reuses_an_orphan_without_moving_live_strings_or_growing():
    ovl, static = fixture()
    before = static.content.pools[0].data.getvalue()
    result = _reuse_unreferenced_clip_slot(ovl, static, b'LongerSpecies_Roar\0')
    assert result.offset == len(b'LiveEvent\0')
    assert result.old_size == result.new_size == len(before)
    after = static.content.pools[0].data.getvalue()
    assert after[:result.offset] == before[:result.offset]
    assert after[result.offset:].rstrip(b'\0') == b'LongerSpecies_Roar'
    assert len(after) == len(before)


@pytest.mark.parametrize('offset', [10, 16, 34])
def test_any_live_reference_including_suffix_and_terminator_protects_slot(offset):
    ovl, static = fixture()
    before = static.content.pools[0].data.getvalue()
    # 10 is the slot start; the final byte is its terminator.
    offset = len(before) - 1 if offset == 34 else offset
    static.content.fragments.append(dict(link_pool=1, link_offset=8, struct_pool=0, struct_offset=offset))
    assert _reuse_unreferenced_clip_slot(ovl, static, b'NewEvent\0') is None
    assert static.content.pools[0].data.getvalue() == before


@pytest.mark.parametrize('kind', ['root', 'dependency', 'source', 'binary', 'incomplete', 'plain'])
def test_uncertified_space_is_never_reclaimed(kind):
    ovl, static = fixture()
    pool = static.content.pools[0]
    if kind == 'root':
        static.content.root_entries.append({'struct_ptr': {'pool_index': 0, 'data_offset': 10}})
    elif kind == 'dependency':
        ovl.dependencies.append({'link_ptr': {'pool_index': 0, 'data_offset': 10}})
    elif kind == 'source':
        static.content.fragments[0]['link_pool'] = 0
    elif kind == 'binary':
        pool.data = io.BytesIO(pool.data.getvalue() + b'\xff')
    elif kind == 'incomplete':
        pool.data = io.BytesIO(pool.data.getvalue()[:-1])
    else:
        pool.data = io.BytesIO(b'LiveEvent\0SomeUnreferencedResourceName\0')
    before = pool.data.getvalue()
    assert _reuse_unreferenced_clip_slot(ovl, static, b'NewEvent\0') is None
    assert pool.data.getvalue() == before


def test_best_fit_preserves_larger_slot_and_live_reuse_is_not_allocated_twice():
    ovl, static = fixture(b'LiveEvent\0Donor$VeryLongUnusedClip\0Donor$ShortClip\0')
    first = _reuse_unreferenced_clip_slot(ovl, static, b'New$ShortClip\0')
    assert first.offset == len(b'LiveEvent\0Donor$VeryLongUnusedClip\0')
    static.content.fragments.append(dict(link_pool=1, link_offset=8, struct_pool=0, struct_offset=first.offset))
    second = _reuse_unreferenced_clip_slot(ovl, static, b'New$ShortClip\0')
    assert second.offset == len(b'LiveEvent\0')
