"""Edit world-space motion (.wsm) - the baked trajectory that PLACES an actor.

Extract a .wsm from an OVL, open it here, transform the path, save, inject it back.
The format is per-frame position + heading, so the useful view is top-down: X across,
Z up the screen, with a tick every N frames and an arrow showing which way the animal
faces.

The single most valuable feature is the OVERLAY. A social or a fight is two actors
who must meet - `socialinteractiona` and `socialinteractionb` - and if their paths
disagree they end up standing in each other. Seeing both at once makes that visual
instead of arithmetic.

NOTHING HERE IS GAME-VERIFIED. Load/save round-trips byte-identical across all 146
shipped files and the transforms are self-consistent (360 degrees returns to 1.7e-06
degrees), but no modified .wsm has been injected and watched to play. Treat every
edit as an experiment.
"""
import logging
import math
import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from gui import GuiOptions, startup, widgets
from gui.widgets import window
from source.formats.wsm.edit import (
	Wsm, load_wsm, recentre, resample_path, reverse, rotate_y, save_wsm, scale,
	translate, trim)

# how often to draw a heading arrow along the path
ARROW_EVERY = 15


class PathView(QtWidgets.QGraphicsView):
	"""Top-down X/Z plot of one or two trajectories."""

	# emitted with a list of (x, z) in world metres when a stroke finishes
	drawn = QtCore.pyqtSignal(list)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setScene(QtWidgets.QGraphicsScene(self))
		self.setRenderHint(QtGui.QPainter.Antialiasing)
		self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
		self.setBackgroundBrush(QtGui.QColor(32, 32, 36))
		self.draw_mode = False
		self._stroke = []
		self._preview = []

	def set_draw_mode(self, on: bool):
		self.draw_mode = on
		# panning and drawing both use the left button, so they cannot both be live
		self.setDragMode(QtWidgets.QGraphicsView.NoDrag if on
						 else QtWidgets.QGraphicsView.ScrollHandDrag)
		self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)

	def wheelEvent(self, event):
		factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
		self.scale(factor, factor)

	# ------------------------------------------------------------ freehand draw

	def mousePressEvent(self, event):
		if self.draw_mode and event.button() == QtCore.Qt.LeftButton:
			self._stroke = [self._world(event.pos())]
			self._preview = []
			return
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event):
		if self.draw_mode and self._stroke:
			point = self._world(event.pos())
			# drop samples closer than a centimetre; a jittery mouse otherwise
			# produces thousands of near-duplicate points and a lumpy arc length
			if math.dist(point, self._stroke[-1]) >= 0.01:
				self._stroke.append(point)
				pen = QtGui.QPen(QtGui.QColor(150, 255, 150), 0)
				a, b = self._stroke[-2], self._stroke[-1]
				self._preview.append(self.scene().addLine(a[0], a[1], b[0], b[1], pen))
			return
		super().mouseMoveEvent(event)

	def mouseReleaseEvent(self, event):
		if self.draw_mode and self._stroke:
			stroke, self._stroke = self._stroke, []
			for item in self._preview:
				self.scene().removeItem(item)
			self._preview = []
			if len(stroke) >= 2:
				self.drawn.emit(stroke)
			return
		super().mouseReleaseEvent(event)

	def _world(self, pos):
		"""Widget point -> (x, z) in metres. mapToScene undoes the zoom and Y flip."""
		scene_point = self.mapToScene(pos)
		return (scene_point.x(), scene_point.y())

	def draw(self, primary: Wsm, overlay: Wsm = None):
		scene = self.scene()
		scene.clear()
		if primary is None:
			return
		tracks = [(primary, QtGui.QColor(120, 200, 255), "primary")]
		if overlay is not None:
			tracks.append((overlay, QtGui.QColor(255, 170, 90), "overlay"))

		# a metre grid, so distances are readable rather than relative
		xs = [p[0] for w, _c, _n in tracks for p in w.locs]
		zs = [p[2] for w, _c, _n in tracks for p in w.locs]
		if not xs:
			return
		lo_x, hi_x, lo_z, hi_z = min(xs) - 2, max(xs) + 2, min(zs) - 2, max(zs) + 2
		grid = QtGui.QPen(QtGui.QColor(58, 58, 64), 0)
		for gx in range(int(math.floor(lo_x)), int(math.ceil(hi_x)) + 1):
			scene.addLine(gx, lo_z, gx, hi_z, grid)
		for gz in range(int(math.floor(lo_z)), int(math.ceil(hi_z)) + 1):
			scene.addLine(lo_x, gz, hi_x, gz, grid)
		axis = QtGui.QPen(QtGui.QColor(90, 90, 100), 0)
		scene.addLine(lo_x, 0, hi_x, 0, axis)
		scene.addLine(0, lo_z, 0, hi_z, axis)

		for wsm, colour, _name in tracks:
			pen = QtGui.QPen(colour, 0)
			for a, b in zip(wsm.locs, wsm.locs[1:]):
				scene.addLine(a[0], a[2], b[0], b[2], pen)
			# start marker (filled) and end marker (hollow)
			if wsm.locs:
				r = 0.12
				sx, _sy, sz = wsm.locs[0]
				scene.addEllipse(sx - r, sz - r, r * 2, r * 2, pen, QtGui.QBrush(colour))
				ex, _ey, ez = wsm.locs[-1]
				scene.addEllipse(ex - r, ez - r, r * 2, r * 2, pen)
			# heading arrows, from the quaternion's forward vector
			arrow = QtGui.QPen(colour.darker(130), 0)
			for i in range(0, len(wsm.locs), ARROW_EVERY):
				if i >= len(wsm.quats):
					break
				x, _y, z = wsm.locs[i]
				fx, fz = _forward(wsm.quats[i])
				scene.addLine(x, z, x + fx * 0.6, z + fz * 0.6, arrow)

		self.setSceneRect(scene.itemsBoundingRect())
		self.fitInView(scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
		# Qt's Y axis points down; flip so +Z is up like a map
		self.scale(1, -1)


def _forward(quat):
	"""Rotate +Z by the quaternion and return its (x, z) - the facing direction."""
	x, y, z, w = quat
	fx = 2 * (x * z + w * y)
	fz = 1 - 2 * (x * x + y * y)
	length = math.hypot(fx, fz) or 1.0
	return fx / length, fz / length


class MainWindow(window.MainWindow):

	def __init__(self, opts: GuiOptions):
		window.MainWindow.__init__(self, "WSM Tool", opts=opts)
		self.setAcceptDrops(True)
		self.wsm = None
		self.overlay = None
		self.path = None
		# The base closeEvent guards unsaved work through self.file_widget.dirty, but
		# this tool has its own open/save dialogs and never creates a file_widget, so
		# that guard can never fire here. Track it ourselves and override closeEvent.
		self.unsaved = False

		self.view = PathView(self)
		self.view.drawn.connect(self.apply_drawn_path)
		self.info = QtWidgets.QLabel("Open a .wsm extracted from an OVL")
		self.info.setWordWrap(True)
		self.overlay_label = QtWidgets.QLabel("overlay: none")

		# --- transforms
		box = QtWidgets.QGroupBox("Transform the whole path")
		form = QtWidgets.QFormLayout(box)
		self.dx = QtWidgets.QDoubleSpinBox(); self.dx.setRange(-500, 500); self.dx.setDecimals(3)
		self.dz = QtWidgets.QDoubleSpinBox(); self.dz.setRange(-500, 500); self.dz.setDecimals(3)
		self.deg = QtWidgets.QDoubleSpinBox(); self.deg.setRange(-360, 360); self.deg.setDecimals(2)
		self.factor = QtWidgets.QDoubleSpinBox(); self.factor.setRange(0.01, 20.0)
		self.factor.setValue(1.0); self.factor.setDecimals(3); self.factor.setSingleStep(0.05)
		form.addRow("Move X (m)", self.dx)
		form.addRow("Move Z (m)", self.dz)
		form.addRow("Rotate (deg)", self.deg)
		form.addRow("Scale", self.factor)
		row = QtWidgets.QHBoxLayout()
		for label, slot in (("Apply", self.apply_transform), ("Reverse", self.do_reverse),
							("Recentre", self.do_recentre)):
			button = QtWidgets.QPushButton(label)
			button.clicked.connect(slot)
			row.addWidget(button)
		form.addRow(row)

		trim_box = QtWidgets.QGroupBox("Trim frames (inclusive)")
		trim_form = QtWidgets.QFormLayout(trim_box)
		self.first = QtWidgets.QSpinBox(); self.first.setRange(0, 999999)
		self.last = QtWidgets.QSpinBox(); self.last.setRange(0, 999999)
		trim_button = QtWidgets.QPushButton("Trim")
		trim_button.clicked.connect(self.do_trim)
		trim_form.addRow("First", self.first)
		trim_form.addRow("Last", self.last)
		trim_form.addRow(trim_button)

		draw_box = QtWidgets.QGroupBox("Draw a new path")
		draw_form = QtWidgets.QVBoxLayout(draw_box)
		self.draw_toggle = QtWidgets.QCheckBox("Draw mode (drag in the view)")
		self.draw_toggle.toggled.connect(self.view.set_draw_mode)
		self.keep_headings = QtWidgets.QCheckBox("Keep original headings")
		self.keep_headings.setToolTip(
			"Off: the animal turns to follow the new path. On: positions move but "
			"facing is untouched, which leaves it walking sideways unless you meant it.")
		hint = QtWidgets.QLabel(
			"The stroke is resampled by arc length to the SAME frame count, so the "
			"animal keeps its timing and just takes a different route.")
		hint.setWordWrap(True)
		draw_form.addWidget(self.draw_toggle)
		draw_form.addWidget(self.keep_headings)
		draw_form.addWidget(hint)

		left = QtWidgets.QWidget()
		column = QtWidgets.QVBoxLayout(left)
		column.addWidget(self.info)
		column.addWidget(self.overlay_label)
		column.addWidget(box)
		column.addWidget(trim_box)
		column.addWidget(draw_box)
		column.addStretch(1)

		grid = QtWidgets.QGridLayout()
		self.create_main_splitter(grid, left, self.view)

		self.build_menus({
			widgets.FILE_MENU: [
				widgets.MenuItem("Open .wsm", self.open_wsm, shortcut="CTRL+O", icon="dir"),
				widgets.MenuItem("Open overlay (the paired actor)", self.open_overlay),
				widgets.MenuItem("Clear overlay", self.clear_overlay),
				widgets.SeparatorMenuItem(),
				widgets.MenuItem("Save", self.save, shortcut="CTRL+S", icon="save"),
				widgets.MenuItem("Save As...", self.save_as, shortcut="CTRL+SHIFT+S"),
			],
			widgets.HELP_MENU: self.help_menu_items,
		})

	# ------------------------------------------------------------------ files

	def _ask(self, title):
		return QtWidgets.QFileDialog.getOpenFileName(
			self, title, self.cfg.get("dir_extract", "C://"), "WSM files (*.wsm)")[0]

	def open_wsm(self):
		chosen = self._ask("Open a .wsm")
		if not chosen:
			return
		try:
			self.wsm = load_wsm(chosen)
			self.path = chosen
		except Exception:
			self.handle_error("Could not read that .wsm, see log!")
			return
		self.first.setValue(0)
		self.last.setValue(max(0, self.wsm.frame_count - 1))
		self.unsaved = False
		self.refresh()

	def open_overlay(self):
		chosen = self._ask("Open the paired .wsm to overlay")
		if not chosen:
			return
		try:
			self.overlay = load_wsm(chosen)
		except Exception:
			self.handle_error("Could not read that .wsm, see log!")
			return
		self.overlay_label.setText(f"overlay: {os.path.basename(chosen)}")
		self.refresh()

	def clear_overlay(self):
		self.overlay = None
		self.overlay_label.setText("overlay: none")
		self.refresh()

	def save(self):
		if not self.wsm or not self.path:
			return
		self._write(self.path)

	def save_as(self):
		if not self.wsm:
			return
		chosen = QtWidgets.QFileDialog.getSaveFileName(
			self, "Save .wsm", self.path or "", "WSM files (*.wsm)")[0]
		if chosen:
			self._write(chosen)
			self.path = chosen

	def _write(self, target):
		problems = self.wsm.problems()
		if problems:
			# these are all silent in game, so refuse rather than warn
			self.showwarning("Refusing to save - fix these first:\n  " + "\n  ".join(problems))
			return
		try:
			save_wsm(self.wsm, target)
		except Exception:
			self.handle_error("Saving failed, see log!")
			return
		logging.info(f"wrote {target}")
		self.unsaved = False
		self.refresh()
		self.info.setText(self.info.text() + f"\n\nsaved to {target}")

	# ------------------------------------------------------------- transforms

	def _guard(self):
		if self.wsm is None:
			self.showwarning("Open a .wsm first")
			return False
		return True

	def apply_transform(self):
		if not self._guard():
			return
		if self.dx.value() or self.dz.value():
			translate(self.wsm, dx=self.dx.value(), dz=self.dz.value())
		if self.deg.value():
			rotate_y(self.wsm, self.deg.value())
		if self.factor.value() != 1.0:
			scale(self.wsm, self.factor.value())
		self.mark_dirty()

	def do_reverse(self):
		if self._guard():
			reverse(self.wsm)
			self.mark_dirty()

	def do_recentre(self):
		if self._guard():
			recentre(self.wsm)
			self.mark_dirty()

	def do_trim(self):
		if not self._guard():
			return
		try:
			trim(self.wsm, self.first.value(), self.last.value())
		except Exception as exc:
			self.showwarning(str(exc))
			return
		self.last.setValue(self.wsm.frame_count - 1)
		self.first.setValue(0)
		self.mark_dirty()

	def apply_drawn_path(self, points):
		"""A finished stroke becomes the trajectory, at the SAME frame count."""
		if self.wsm is None:
			self.showwarning("Open a .wsm before drawing a path")
			return
		try:
			resample_path(self.wsm, points,
						  keep_headings=self.keep_headings.isChecked())
		except Exception as exc:
			self.showwarning(str(exc))
			return
		logging.info(f"redrew the path from {len(points)} stroke points")
		self.mark_dirty()

	def mark_dirty(self):
		self.unsaved = True
		self.refresh()

	def closeEvent(self, event):
		"""Offer to save before quitting.

		super() still has to run - it cancels workers, shuts the log splitter down
		and stops the status timer - so on every path that proceeds we hand the event
		on rather than accepting it here.
		"""
		if self.wsm is not None and self.unsaved:
			box = QtWidgets.QMessageBox(self)
			box.setWindowTitle("Unsaved changes")
			box.setText(f"{os.path.basename(self.path or 'this .wsm')} has unsaved changes.")
			box.setInformativeText("Save before closing?")
			box.setStandardButtons(QtWidgets.QMessageBox.Save
								   | QtWidgets.QMessageBox.Discard
								   | QtWidgets.QMessageBox.Cancel)
			box.setDefaultButton(QtWidgets.QMessageBox.Save)
			choice = box.exec_()
			if choice == QtWidgets.QMessageBox.Cancel:
				event.ignore()
				return
			if choice == QtWidgets.QMessageBox.Save:
				self.save() if self.path else self.save_as()
				# a refused save (inconsistent file) or a cancelled Save As leaves it
				# dirty - do not quit and silently lose the work
				if self.unsaved:
					event.ignore()
					return
		super().closeEvent(event)

	def refresh(self):
		if self.wsm is None:
			return
		lo_x, hi_x, lo_z, hi_z = self.wsm.bounds()
		problems = self.wsm.problems()
		text = [
			os.path.basename(self.path or "(unsaved)"),
			f"{self.wsm.frame_count} frames, {self.wsm.duration:.4f} s",
			f"path length {self.wsm.path_length():.2f} m",
			f"extent  X {lo_x:.2f}..{hi_x:.2f}   Z {lo_z:.2f}..{hi_z:.2f}",
		]
		if self.overlay is not None:
			gap = math.dist(self.wsm.locs[-1], self.overlay.locs[-1]) if (
				self.wsm.locs and self.overlay.locs) else float("nan")
			text.append(f"end-to-end gap between the two actors: {gap:.2f} m")
		text.append("PROBLEMS: " + "; ".join(problems) if problems else "consistent")
		text.append("UNSAVED CHANGES" if self.unsaved else "saved")
		self.info.setText("\n".join(text))
		name = os.path.basename(self.path) if self.path else "WSM Tool"
		self.setWindowTitle(f"WSM Tool - {name}{' *' if self.unsaved else ''}")
		self.view.draw(self.wsm, self.overlay)


if __name__ == '__main__':
	startup(MainWindow, GuiOptions(log_name="wsm_tool_gui", size=(1300, 800)))
