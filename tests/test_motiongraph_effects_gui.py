import os
import hashlib
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

import motiongraph_tool_gui as gui

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class Status:
    def showMessage(self, *_args):
        pass


def controls(tmp_path):
    app = APP
    fake = SimpleNamespace()
    fake.effect_list = QtWidgets.QTreeWidget()
    fake.effect_list.setColumnCount(3)
    fake.effect_activity = QtWidgets.QComboBox()
    fake.effect_name = QtWidgets.QLineEdit()
    fake.effect_type = QtWidgets.QComboBox()
    fake.effect_type.setEditable(True)
    fake.effect_set_button = QtWidgets.QPushButton()
    fake.curve_editor = gui.CurveEditor()
    fake.effects_dirty = False
    fake.effect_draft = None
    fake.effect_loaded_target = None
    fake.effect_held = QtWidgets.QLabel()
    fake.status_bar = Status()
    fake.showerror = lambda error: (_ for _ in ()).throw(AssertionError(error))
    source = tmp_path / "animal.ovl"
    source.write_bytes(b"placeholder")
    fake.source_path = lambda: source
    fake.current_effect_activity = lambda: gui.MainWindow.current_effect_activity(fake)
    fake.effect_activity.addItem("Species$Idle", (1, 10, "Species$Idle"))
    fake.effect_activity.addItem("Species$Roar", (2, 20, "Species$Roar"))
    return app, fake


def test_usage_row_selects_its_activity_and_retains_stream_identity(tmp_path, monkeypatch):
    _app, fake = controls(tmp_path)
    item = QtWidgets.QTreeWidgetItem(fake.effect_list, ["Species$Roar", "VFXToggle", "0"])
    item.setData(0, QtCore.Qt.UserRole, "usage")
    item.setData(0, QtCore.Qt.UserRole + 1, (2, 20, "Species$Roar"))
    item.setData(0, QtCore.Qt.UserRole + 2, "VFX_Fire")
    item.setSelected(True)
    monkeypatch.setattr(gui, "read_data_stream_curve", lambda *a, **k: [(0, 0), (1, 1)])
    gui.MainWindow.load_effect_curve(fake)
    assert fake.effect_activity.currentIndex() == 1
    assert fake.effect_name.text() == "VFX_Fire"
    assert fake.effect_loaded_target == ((2, 20, "Species$Roar"), "VFX_Fire")
    assert fake.effect_set_button.isEnabled()


def test_failed_curve_lookup_clears_target_and_disables_write(tmp_path, monkeypatch):
    _app, fake = controls(tmp_path)
    item = QtWidgets.QTreeWidgetItem(fake.effect_list, ["VFX_Missing", "VFXEnable", "65537"])
    item.setData(0, QtCore.Qt.UserRole, "stream")
    item.setData(0, QtCore.Qt.UserRole + 1, (1, 10, "Species$Idle"))
    item.setData(0, QtCore.Qt.UserRole + 2, "VFX_Missing")
    item.setSelected(True)
    fake.effect_loaded_target = ((1, 10, "Species$Idle"), "old")
    monkeypatch.setattr(gui, "read_data_stream_curve", lambda *a, **k: (_ for _ in ()).throw(ValueError("missing")))
    gui.MainWindow.load_effect_curve(fake)
    assert fake.effect_loaded_target is None
    assert not fake.effect_set_button.isEnabled()
    assert fake.curve_editor.points == [(0.0, 0.0), (1.0, 0.0)]


def test_dirty_curve_blocks_selection_change_and_preserves_undo_scope(tmp_path, monkeypatch):
    _app, fake = controls(tmp_path)
    errors = []
    fake.showerror = errors.append
    first = QtWidgets.QTreeWidgetItem(fake.effect_list, ["VFX_First", "VFXEnable", "0"])
    first.setData(0, QtCore.Qt.UserRole + 1, (1, 10, "Species$Idle"))
    first.setData(0, QtCore.Qt.UserRole + 2, "VFX_First")
    second = QtWidgets.QTreeWidgetItem(fake.effect_list, ["VFX_Second", "VFXToggle", "0"])
    second.setData(0, QtCore.Qt.UserRole + 1, (2, 20, "Species$Roar"))
    second.setData(0, QtCore.Qt.UserRole + 2, "VFX_Second")
    curves = {
        "VFX_First": [(0, 0), (1, 1)],
        "VFX_Second": [(0, 0), (1, 0)],
    }
    monkeypatch.setattr(
        gui, "read_data_stream_curve", lambda _s, _clip, name, *_a, **_k: curves[name])
    first.setSelected(True)
    gui.MainWindow.load_effect_curve(fake)
    fake.curve_editor.set_points([(0, 0), (0.5, 1), (1, 1)])
    fake.effects_dirty = True
    dirty_points = list(fake.curve_editor.points)
    first.setSelected(False)
    second.setSelected(True)
    gui.MainWindow.load_effect_curve(fake)
    assert errors and "unsaved edits" in errors[-1]
    assert first.isSelected() and not second.isSelected()
    assert fake.effect_loaded_target == ((1, 10, "Species$Idle"), "VFX_First")
    assert fake.curve_editor.points == dirty_points


def test_discard_reloads_curve_and_clears_undo_history(tmp_path, monkeypatch):
    _app, fake = controls(tmp_path)
    item = QtWidgets.QTreeWidgetItem(fake.effect_list, ["VFX_First", "VFXEnable", "0"])
    item.setData(0, QtCore.Qt.UserRole + 1, (1, 10, "Species$Idle"))
    item.setData(0, QtCore.Qt.UserRole + 2, "VFX_First")
    item.setSelected(True)
    monkeypatch.setattr(gui, "read_data_stream_curve", lambda *_a, **_k: [(0, 0), (1, 1)])
    gui.MainWindow.load_effect_curve(fake)
    fake.curve_editor.set_points([(0, 0), (1, 0)])
    fake.effects_dirty = True
    gui.MainWindow.discard_effect_curve_edits(fake)
    assert fake.curve_editor.points == [(0.0, 0.0), (1.0, 1.0)]
    assert not fake.effects_dirty
    assert fake.curve_editor._undo == [] and fake.curve_editor._redo == []


def test_existing_stage_result_requires_explicit_continue(tmp_path):
    source_dir, stage_dir = tmp_path / "source", tmp_path / "stage"
    source_dir.mkdir(); stage_dir.mkdir()
    source = source_dir / "animal.ovl"
    output = stage_dir / "animal.ovl"
    source.write_bytes(b"source")
    output.write_bytes(b"source")
    fake = SimpleNamespace(
        source_edit=QtWidgets.QLineEdit(str(source)),
        name_edit=QtWidgets.QLineEdit("species.motiongraph"),
        stage_edit=QtWidgets.QLineEdit(str(stage_dir)),
        loaded_identity=(source.resolve(), "species.motiongraph",
                         hashlib.sha256(source.read_bytes()).hexdigest(), 1),
    )
    fake.requested_source_path = lambda: gui.MainWindow.requested_source_path(fake)
    fake.source_path = lambda: gui.MainWindow.source_path(fake)
    assert gui.MainWindow.output_path(fake) == output
    output.write_bytes(b"result")
    try:
        gui.MainWindow.output_path(fake)
    except ValueError as exc:
        assert "Continue from staged result" in str(exc)
    else:
        raise AssertionError("a second operation was not blocked")
    assert gui.MainWindow.output_path(fake, allow_existing_result=True) == output
