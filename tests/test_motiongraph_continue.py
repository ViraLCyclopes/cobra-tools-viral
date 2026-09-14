"""Exercise staging helpers without starting the desktop application."""
from pathlib import Path

import pytest


from source.formats.motiongraph.staging import copy_motiongraph_family


def prepare(load, source, name=None):
    import tempfile
    source = source.resolve()
    root = Path(tempfile.mkdtemp(prefix="motiongraph-step-", dir=source.parent.parent))
    copy_motiongraph_family(source, root / "source")
    snapshot = root / "source" / source.name
    report = load(snapshot, name)
    outputs = copy_motiongraph_family(snapshot, root / "output")
    return snapshot, root / "output", outputs, report


def test_continue_keeps_previous_edits_and_complete_family(tmp_path):
    stage = tmp_path / 'stage'
    stage.mkdir()
    source = stage / 'animal.ovl'
    source.write_bytes(b'edit one')
    (stage / 'animal.ovs.anim_l0').write_bytes(b'stream')
    (stage / 'animal.texture.aux').write_bytes(b'texture')
    advance = lambda path, name: prepare(lambda p, n: p.read_bytes(), path, name)
    snapshot, output, files, report = advance(source, None)
    assert report == b'edit one'
    assert snapshot.read_bytes() == (output / source.name).read_bytes() == b'edit one'
    assert {p.name for p in files} == {p.name for p in stage.iterdir()}
    assert (output / 'animal.ovs.anim_l0').read_bytes() == b'stream'
    assert (output / 'animal.texture.aux').read_bytes() == b'texture'
    (output / source.name).write_bytes(b'edit one plus edit two')
    snapshot2, output2, _, report2 = advance(output / source.name, None)
    assert report2 == b'edit one plus edit two'
    assert output2 != output
    assert snapshot2.read_bytes() == b'edit one plus edit two'
    assert source.read_bytes() == snapshot.read_bytes() == b'edit one'


def test_failed_analysis_does_not_overwrite_previous_stage(tmp_path):
    stage = tmp_path / 'stage'
    stage.mkdir()
    source = stage / 'animal.ovl'
    source.write_bytes(b'previous result')
    def fail(path, name):
        raise ValueError('invalid archive')
    with pytest.raises(ValueError, match='invalid archive'):
        prepare(fail, source, None)
    assert source.read_bytes() == b'previous result'
    assert not list(tmp_path.glob('motiongraph-step-*/output/animal.ovl'))
