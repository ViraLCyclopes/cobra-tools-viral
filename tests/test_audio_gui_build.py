"""Check GUI command construction and failure handling without launching tools."""
import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest


def functions(runner=None):
    source = Path(__file__).resolve().parents[1] / 'motiongraph_tool_gui.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('audio_build_command', 'run_audio_build')]
    env = dict(Path=Path, sys=sys, subprocess=SimpleNamespace(run=runner))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), env)
    return env


def test_gui_stock_custom_marker_and_baseline_arguments(tmp_path):
    (tmp_path/'ovl_tool_cmd.py').touch()
    soundmod = tmp_path/'replacement media.ovl'
    soundmod.touch()
    make = functions()['audio_build_command']
    args = (tmp_path, tmp_path, 'Indoraptor', 'VLIndo', str(tmp_path), str(tmp_path/'out'))
    stock = make(*args)
    assert stock[stock.index('--cobra')+1] == str(tmp_path)
    assert '--sounds' not in stock
    custom = make(*args, sound_mod=str(soundmod), sounds=str(tmp_path))
    assert custom[custom.index('--sounds')+1] == str(tmp_path)
    assert custom[custom.index('--sound-mod')+1] == str(soundmod)
    marker = make(*args, marker=str(soundmod))
    assert '--marker-from' in marker
    baseline = make(*args, sounds='missing', marker='missing', baseline=True)
    assert '--sounds' not in baseline and '--marker-from' not in baseline
    with pytest.raises(ValueError, match='Choose custom WEMs or a marker'):
        make(*args, sounds=str(tmp_path), marker=str(soundmod))


def test_failed_build_never_reports_stale_outputs_as_success(tmp_path):
    cmd = ['python', '--out', str(tmp_path), '--prefix', 'VLIndo']
    run = functions(lambda *a, **k: SimpleNamespace(returncode=1, stdout='', stderr='bad WEM'))['run_audio_build']
    (tmp_path/'vlindo_events.bnk').write_bytes(b'old output')
    with pytest.raises(ValueError, match='bad WEM'):
        run(cmd, tmp_path)


def test_success_requires_complete_outputs(tmp_path):
    cmd = ['python', '--out', str(tmp_path), '--prefix', 'VLIndo']
    run = functions(lambda *a, **k: SimpleNamespace(returncode=0, stdout='built', stderr=''))['run_audio_build']
    with pytest.raises(ValueError, match='outputs are missing'):
        run(cmd, tmp_path)
    for name in ('vlindo_events.bnk', 'vlindo_media.bnk', 'vlindo.wmetasb.add', 'report.json', 'sounds_available.txt'):
        (tmp_path/name).write_bytes(b'output')
    out, names, log = run(cmd, tmp_path)
    assert out == tmp_path and len(names) == 5 and 'built' in log
