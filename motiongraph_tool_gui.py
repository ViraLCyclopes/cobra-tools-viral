"""Safe, fixed-topology JWE3 motiongraph browser and patch-plan editor."""

from __future__ import annotations

import math
import logging
import subprocess
import sys
import tempfile
import hashlib
from pathlib import Path

from gui import GuiOptions, startup, widgets
from gui.widgets import window
from PyQt5 import QtCore, QtGui, QtWidgets

from source.formats.motiongraph.edit import (
	DEFAULT_GAME,
	apply_patch_plan,
	build_patch_plan,
	load_plan,
	load_motiongraph,
	locate_fields,
	merge_patch_plans,
	save_plan,
)
from source.formats.motiongraph.clone import clone_complete_animation_activity
from source.formats.motiongraph.capacity import (
	audit_adjacent_growth,
	audit_allocations,
	audit_dead_space,
	build_capacity_audit,
	census,
	diff_census,
	render_capacity_markdown,
)
from source.formats.motiongraph.decision_growth import (
	add_state_and_route,
	grow_decision_chooser,
	inbound_count,
	list_decision_choosers,
	list_states,
	set_activity_clip,
)
from source.formats.motiongraph.chooser_growth import (
	grow_random_animation_chooser,
	list_choosers,
	set_chooser_weights,
)
from source.formats.motiongraph.rename_repair import (
	repoint_stale_species_strings,
	survey_stale_references,
)
from source.formats.motiongraph.report import (
	build_activity_tree,
	build_decision_graph,
	build_decision_report,
	build_state_report,
)
from source.formats.motiongraph.static_patch import (apply_string_renames, patch_string_slot, patch_string_slots,
													 relocate_strings,
                                                     repoint_existing_string)
from source.formats.motiongraph import audio_events
from source.formats.motiongraph.datastream_growth import (
	activities_with_stream, curve_type_for, describe_activity, grow_data_stream,
	is_held, list_activity_records, read_data_stream_curve, set_curve_everywhere,
	set_data_stream_curve)
from source.formats.motiongraph.staging import (
	copy_motiongraph_family, motiongraph_family)


# Audio Events tab directions. Renaming gives a species its own voice; reverting
# puts the donor's stock names back so a published copy needs no custom-audio
# chain at all - no proxy dll, no loader mod, no banks in the mod's Audio folder.
AUDIO_RENAME = "rename"
AUDIO_REVERT = "revert"


def describe_motiongraph_field(path: str, kind: str = "") -> str:
	"""Return a user-facing explanation without overstating unknown semantics."""
	name = (path or "").lower()
	if name.endswith("mani") or name == "mani":
		return "Animation clip: the MANI animation this activity plays."
	if "speed" in name or "playbackrate" in name:
		return "Playback speed: 1.0 is normal, 0.5 is half speed, and 2.0 is double speed."
	if "weight" in name:
		return "Blend weight: how strongly this activity contributes when animations are combined."
	if "animationflags" in name or kind == "bitfield":
		return (
			"Playback switches such as Looping, Additive, Mirrored, Affects Motion, and "
			"data-stream suppression. Change named switches rather than the raw number."
		)
	if "priorit" in name:
		return (
			"Animation priority/routing value. It affects how competing contributions are selected, "
			"but JWE3's individual numeric/bit meanings are not yet proven. Clone it from a compatible donor."
		)
	if "propthrough" in name or "sync" in name:
		return "Synchronization link used to keep animation phase/progress aligned with another activity or variable."
	if "datastream" in name:
		return "Extra curves/events carried with the animation, such as gameplay, bone, or audio-blend data."
	if kind == "pointer":
		return "Link to another string, object, or list. The pool/offset is an address, not a gameplay value."
	if kind == "curve":
		return "Time-varying curve input rather than one constant value."
	if kind == "enum":
		return "Named mode stored as a number. Choose a known name instead of guessing a raw value."
	return "Fixed-width activity setting. Keep donor values unless its gameplay meaning is understood."


def _load_reports(source: Path, name: str | None):
	ovl, loader = load_motiongraph(source, name or None, DEFAULT_GAME)
	state_text, states, state_stats = build_state_report(loader)
	decision_text, decision_stats = build_decision_report(loader)
	decision_graph = build_decision_graph(loader, states)
	return (
		ovl, loader, state_text, states, state_stats,
		decision_text, decision_stats, decision_graph,
	)


def _load_reports_identity(source: Path, name: str | None, generation: int):
	source = Path(source).resolve()
	before = hashlib.sha256(source.read_bytes()).hexdigest()
	reports = _load_reports(source, name)
	after = hashlib.sha256(source.read_bytes()).hexdigest()
	if before != after:
		raise ValueError("Source OVL changed while it was loading; try again")
	return generation, source, reports[1].name, before, reports


def _scan_fields_identity(generation, identity, *args):
	return generation, identity, locate_fields(*args)


def prepare_next_stage(source: Path, name: str | None):
	"""Snapshot a staged result and validate it before advancing editor state."""
	source = source.resolve()
	if not source.is_file():
		raise ValueError("The staged result is missing")
	# Unique directories preserve every prior result and never overwrite a family.
	root = Path(tempfile.mkdtemp(prefix="motiongraph-step-", dir=source.parent.parent))
	copy_motiongraph_family(source, root / "source")
	snapshot = root / "source" / source.name
	reports = _load_reports(snapshot, name)
	outputs = copy_motiongraph_family(snapshot, root / "output")
	return snapshot, root / "output", outputs, reports


def audio_build_command(kit, cobra, donor, prefix, game, out, sound_mod="", sounds="", marker="", baseline=False):
	missing = [n for n, v in (("donor species", donor), ("new prefix", prefix),
		("game folder", game), ("output folder", out)) if not v]
	if missing:
		raise ValueError("Fill in: " + ", ".join(missing))
	if not (Path(cobra) / "ovl_tool_cmd.py").is_file():
		raise ValueError("This Cobra checkout has no ovl_tool_cmd.py")
	if not baseline and sounds and marker:
		raise ValueError("Choose custom WEMs or a marker build; clear the other field first")
	cmd = [sys.executable, str(Path(kit) / "build_species_audio.py"), "--cobra", str(cobra),
		"--donor", donor, "--prefix", prefix, "--game", str(Path(game).resolve()), "--out", str(Path(out).resolve())]
	for flag, value, directory in (("--sound-mod", sound_mod, False),
		("--sounds", "" if baseline else sounds, True),
		("--marker-from", "" if baseline else marker, False)):
		if value:
			path = Path(value)
			if not (path.is_dir() if directory else path.is_file()):
				raise ValueError(f"{flag}: path does not exist: {value}")
			cmd += [flag, str(path.resolve())]
	return cmd


def run_audio_build(cmd, kit):
	result = subprocess.run(cmd, cwd=str(kit), capture_output=True, text=True, errors="replace")
	log = (result.stdout or "") + "\n" + (result.stderr or "")
	if result.returncode:
		raise ValueError(f"Audio builder failed (exit {result.returncode}):\n{log[-6000:]}")
	out = Path(cmd[cmd.index("--out") + 1])
	prefix = cmd[cmd.index("--prefix") + 1].lower()
	names = [prefix + suffix for suffix in ("_events.bnk", "_media.bnk", ".wmetasb.add")]
	names += ["report.json", "sounds_available.txt"]
	if any(not (out / name).is_file() for name in names):
		raise ValueError("Builder exited successfully but expected outputs are missing")
	return out, names, log


def hierarchical_neighborhood_positions(center, visible, edge_pairs,
										  x_step=245, y_step=105):
	"""Lay out a directed local neighborhood around one selected graph node.

	The selected node occupies level zero. Following an outgoing edge moves one
	column right; following an incoming edge moves one column left. Cycles and
	mixed-direction paths keep the first shortest assignment, which is stable
	because both nodes and neighbors are traversed in sorted order.
	"""
	visible = set(visible)
	if center not in visible:
		raise ValueError("The center node must be visible")
	adjacent = {node: [] for node in visible}
	for source, target in edge_pairs:
		if source in visible and target in visible and source != target:
			adjacent[source].append((target, 1))
			adjacent[target].append((source, -1))
	levels, queue = {center: 0}, [center]
	while queue:
		node = queue.pop(0)
		for neighbor, direction in sorted(adjacent[node], key=lambda item: item[0]):
			if neighbor not in levels:
				levels[neighbor] = levels[node] + direction
				queue.append(neighbor)
	# Defensive fallback for isolated nodes supplied by a caller.
	for node in sorted(visible - levels.keys()):
		levels[node] = 0
	columns = {}
	for node, level in levels.items():
		columns.setdefault(level, []).append(node)
	positions = {}
	for level, nodes in columns.items():
		nodes.sort()
		if center in nodes:
			positions[center] = (level * x_step, 0)
			others = [node for node in nodes if node != center]
			for row, node in enumerate(others):
				distance = (row // 2 + 1) * y_step
				positions[node] = (level * x_step, -distance if row % 2 == 0 else distance)
		else:
			start_y = -(len(nodes) - 1) * y_step / 2
			for row, node in enumerate(nodes):
				positions[node] = (level * x_step, start_y + row * y_step)
	return positions


class GraphNode(QtWidgets.QGraphicsRectItem):
	WIDTH = 170
	HEIGHT = 62

	def __init__(self, state, position, open_callback=None):
		super().__init__(0, 0, self.WIDTH, self.HEIGHT)
		self.state = state
		self.open_callback = open_callback
		self.setPos(*position)
		self.setFlag(self.ItemIsSelectable, True)
		self.setToolTip(
			f"STATE[{state['index']}] {state['label']}\n"
			f"{state['activity_nodes']} activities\n"
			f"{', '.join(state['clips'][:5]) or '(no clips)'}"
		)
		self.setPen(QtGui.QPen(QtGui.QColor("#708090"), 1.5))
		self.setBrush(QtGui.QBrush(QtGui.QColor("#27313d")))
		index = QtWidgets.QGraphicsSimpleTextItem(f"STATE {state['index']}", self)
		index.setAcceptedMouseButtons(QtCore.Qt.NoButton)
		index.setBrush(QtGui.QBrush(QtGui.QColor("#8ab4f8")))
		index.setPos(7, 5)
		label = QtWidgets.QGraphicsTextItem("≈ " + state["label"], self)
		label.setAcceptedMouseButtons(QtCore.Qt.NoButton)
		label.setDefaultTextColor(QtGui.QColor("#f1f3f4"))
		label.setTextWidth(self.WIDTH - 12)
		label.setPos(3, 21)

	def paint(self, painter, option, widget=None):
		if self.isSelected():
			self.setPen(QtGui.QPen(QtGui.QColor("#ffd75f"), 3))
		else:
			self.setPen(QtGui.QPen(QtGui.QColor("#708090"), 1.5))
		super().paint(painter, option, widget)

	def mouseDoubleClickEvent(self, event):
		if self.open_callback:
			self.open_callback(self.state["index"])
			event.accept()
			return
		super().mouseDoubleClickEvent(event)


class GraphView(QtWidgets.QGraphicsView):
	WORKSPACE_MARGIN = 100000.0

	def __init__(self, scene, parent=None):
		super().__init__(scene, parent)
		self._forced_pan = False
		self._space_down = False
		self._pan_position = QtCore.QPoint()
		self._content_rect = QtCore.QRectF()
		self.setRenderHint(QtGui.QPainter.Antialiasing)
		self.setDragMode(self.ScrollHandDrag)
		self.setTransformationAnchor(self.AnchorUnderMouse)
		self.setBackgroundBrush(QtGui.QColor("#161b22"))
		self.setFocusPolicy(QtCore.Qt.StrongFocus)

	def wheelEvent(self, event):
		factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
		self.scale(factor, factor)

	def keyPressEvent(self, event):
		if event.key() == QtCore.Qt.Key_H and not event.isAutoRepeat():
			self.fit_content()
			event.accept()
			return
		if event.key() == QtCore.Qt.Key_Space and not event.isAutoRepeat():
			self._space_down = True
			self.setCursor(QtCore.Qt.OpenHandCursor)
			event.accept()
			return
		super().keyPressEvent(event)

	def keyReleaseEvent(self, event):
		if event.key() == QtCore.Qt.Key_Space and not event.isAutoRepeat():
			self._space_down = False
			if not self._forced_pan:
				self.unsetCursor()
			event.accept()
			return
		super().keyReleaseEvent(event)

	def mousePressEvent(self, event):
		force_pan = event.button() == QtCore.Qt.MiddleButton or (
			event.button() == QtCore.Qt.LeftButton and self._space_down
		)
		if force_pan:
			self._forced_pan = True
			self._pan_position = event.pos()
			self.setCursor(QtCore.Qt.ClosedHandCursor)
			event.accept()
			return
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event):
		if self._forced_pan:
			delta = event.pos() - self._pan_position
			self._pan_position = event.pos()
			self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
			self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
			event.accept()
			return
		super().mouseMoveEvent(event)

	def mouseReleaseEvent(self, event):
		if self._forced_pan and event.button() in (QtCore.Qt.MiddleButton, QtCore.Qt.LeftButton):
			self._forced_pan = False
			self.setCursor(QtCore.Qt.OpenHandCursor if self._space_down else QtCore.Qt.ArrowCursor)
			event.accept()
			return
		super().mouseReleaseEvent(event)

	def set_content_rect(self, rect):
		"""Keep a tight fit target inside a much larger pannable workspace."""
		self._content_rect = QtCore.QRectF(rect)
		margin = self.WORKSPACE_MARGIN
		self.scene().setSceneRect(self._content_rect.adjusted(-margin, -margin, margin, margin))

	def fit_content(self):
		if not self._content_rect.isEmpty():
			self.fitInView(self._content_rect, QtCore.Qt.KeepAspectRatio)
			self.centerOn(self._content_rect.center())


class DecisionNode(QtWidgets.QGraphicsRectItem):
	WIDTH = 205
	HEIGHT = 78

	def __init__(self, node, position, open_callback=None):
		super().__init__(0, 0, self.WIDTH, self.HEIGHT)
		self.node = node
		self.open_callback = open_callback
		self.setPos(*position)
		self.setFlag(self.ItemIsSelectable, True)
		self.setBrush(QtGui.QBrush(QtGui.QColor("#302b3f")))
		self.setPen(QtGui.QPen(QtGui.QColor("#9b72cf"), 1.5))
		opcode = (node["opcode"] or "?").split(".")[-1]
		title = QtWidgets.QGraphicsSimpleTextItem(f"#{node['index']}  {opcode}", self)
		title.setAcceptedMouseButtons(QtCore.Qt.NoButton)
		title.setBrush(QtGui.QBrush(QtGui.QColor("#d7b7ff")))
		title.setPos(7, 5)
		body_parts = list(node["fields"])
		if node["target_state"] is not None:
			body_parts.append(
				f"→ STATE[{node['target_state']}] {node['target_label'] or ''}".rstrip()
			)
		body = QtWidgets.QGraphicsTextItem("\n".join(body_parts) or "(no parameters)", self)
		body.setAcceptedMouseButtons(QtCore.Qt.NoButton)
		body.setDefaultTextColor(QtGui.QColor("#f1f3f4"))
		body.setTextWidth(self.WIDTH - 12)
		body.setPos(3, 23)

	def paint(self, painter, option, widget=None):
		color, width = ("#ffd75f", 3) if self.isSelected() else ("#9b72cf", 1.5)
		self.setPen(QtGui.QPen(QtGui.QColor(color), width))
		super().paint(painter, option, widget)

	def mouseDoubleClickEvent(self, event):
		if self.open_callback:
			self.open_callback(self.node["index"])
			event.accept()
			return
		super().mouseDoubleClickEvent(event)


class CurveEditor(QtWidgets.QWidget):
	"""Click-and-drag editor for a datastream's on/off curve.

	x is normalised clip time 0..1, y the signal value. Keys use
	SubCurveType.Constant, which holds its value until the next key, so the curve
	is drawn as a STEP - drawing it as a slope would misrepresent when the effect
	actually switches.
	"""

	changed = QtCore.pyqtSignal()

	MARGIN = 34
	HANDLE = 9           # drawn size
	GRAB = 16            # click tolerance - generous, or a near-miss ADDS a point

	def __init__(self, parent=None):
		super().__init__(parent)
		self.points = [(0.0, 0.0), (0.35, 1.0), (0.75, 0.0), (1.0, 0.0)]
		self.snap = True
		self._drag = None
		# undo history: snapshots pushed BEFORE each mutation, so one Ctrl+Z
		# reverts a whole drag rather than each mouse-move sample
		self._undo = []
		self._redo = []
		self.setMinimumHeight(190)
		self.setMouseTracking(True)
		self.setFocusPolicy(QtCore.Qt.StrongFocus)
		self.setToolTip(
			"Click a gap to add a key, drag a key to move it, right-click one to delete.\n"
			"The first key must stay at x=0. Snap keeps y at exactly 0 or 1.")

	def _push_undo(self):
		self._undo.append(list(self.points))
		del self._undo[:-64]
		self._redo.clear()

	def undo(self):
		if not self._undo:
			return False
		self._redo.append(list(self.points))
		self.points = self._undo.pop()
		self.update()
		self.changed.emit()
		return True

	def redo(self):
		if not self._redo:
			return False
		self._undo.append(list(self.points))
		self.points = self._redo.pop()
		self.update()
		self.changed.emit()
		return True

	def keyPressEvent(self, event):
		if event.matches(QtGui.QKeySequence.Undo):
			self.undo()
		elif event.matches(QtGui.QKeySequence.Redo) or (
				event.key() == QtCore.Qt.Key_Y and event.modifiers() & QtCore.Qt.ControlModifier):
			self.redo()
		else:
			super().keyPressEvent(event)

	def set_points(self, points, record=True):
		if record:
			self._push_undo()
		self.points = [(float(x), float(y)) for x, y in points] or [(0.0, 0.0), (1.0, 0.0)]
		self.update()
		self.changed.emit()

	def _plot_rect(self):
		return QtCore.QRectF(self.MARGIN, 12,
							 max(10, self.width() - self.MARGIN - 18),
							 max(10, self.height() - self.MARGIN - 12))

	def _to_px(self, x, y):
		r = self._plot_rect()
		return QtCore.QPointF(r.left() + x * r.width(), r.bottom() - y * r.height())

	def _from_px(self, pos):
		r = self._plot_rect()
		x = (pos.x() - r.left()) / r.width() if r.width() else 0.0
		y = (r.bottom() - pos.y()) / r.height() if r.height() else 0.0
		return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)

	def paintEvent(self, _event):
		painter = QtGui.QPainter(self)
		painter.setRenderHint(QtGui.QPainter.Antialiasing)
		r = self._plot_rect()
		text_colour = self.palette().color(QtGui.QPalette.WindowText)

		grid = QtGui.QColor(text_colour)
		grid.setAlpha(46)
		painter.setPen(QtGui.QPen(grid, 1))
		for step in range(11):
			x = r.left() + r.width() * step / 10.0
			painter.drawLine(QtCore.QPointF(x, r.top()), QtCore.QPointF(x, r.bottom()))
		for value in (0.0, 0.5, 1.0):
			y = r.bottom() - r.height() * value
			painter.drawLine(QtCore.QPointF(r.left(), y), QtCore.QPointF(r.right(), y))

		painter.setPen(QtGui.QPen(text_colour, 1))
		painter.drawRect(r)
		font = painter.font()
		font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
		painter.setFont(font)
		painter.drawText(QtCore.QRectF(4, r.bottom() - 9, self.MARGIN - 8, 16),
						 QtCore.Qt.AlignRight, "off 0")
		painter.drawText(QtCore.QRectF(4, r.top() - 7, self.MARGIN - 8, 16),
						 QtCore.Qt.AlignRight, "on 1")
		painter.drawText(QtCore.QRectF(r.left() - 12, r.bottom() + 4, 40, 16),
						 QtCore.Qt.AlignLeft, "start")
		painter.drawText(QtCore.QRectF(r.right() - 30, r.bottom() + 4, 40, 16),
						 QtCore.Qt.AlignRight, "end")

		if not self.points:
			return
		accent = QtGui.QColor("#C6402A")
		painter.setPen(QtGui.QPen(accent, 2))
		path = QtGui.QPainterPath(self._to_px(*self.points[0]))
		for index in range(1, len(self.points)):
			previous, current = self.points[index - 1], self.points[index]
			corner = self._to_px(current[0], previous[1])
			path.lineTo(corner)
			path.lineTo(self._to_px(*current))
		last = self.points[-1]
		if last[0] < 1.0:
			path.lineTo(self._to_px(1.0, last[1]))
		painter.drawPath(path)

		painter.setBrush(QtGui.QBrush(accent))
		for x, y in self.points:
			painter.drawRect(QtCore.QRectF(self._to_px(x, y).x() - self.HANDLE / 2,
										   self._to_px(x, y).y() - self.HANDLE / 2,
										   self.HANDLE, self.HANDLE))

	def _hit(self, pos):
		"""Index of the nearest key within GRAB pixels, else None.

		Nearest rather than first-within-range: keys can sit close together (an
		off key right after an on key), and picking the first match grabs the
		wrong one.
		"""
		best, best_d = None, None
		for index, (x, y) in enumerate(self.points):
			delta = self._to_px(x, y) - pos
			d = (delta.x() ** 2 + delta.y() ** 2) ** 0.5
			if d <= self.GRAB and (best_d is None or d < best_d):
				best, best_d = index, d
		return best

	def mousePressEvent(self, event):
		index = self._hit(event.pos())
		if event.button() == QtCore.Qt.RightButton:
			# the first key anchors x=0; a curve also needs two points to exist
			if index is not None and index > 0 and len(self.points) > 2:
				self._push_undo()
				del self.points[index]
				self.update()
				self.changed.emit()
			return
		if event.button() != QtCore.Qt.LeftButton:
			return
		if index is None:
			x, y = self._from_px(event.pos())
			if self.snap:
				y = 1.0 if y >= 0.5 else 0.0
			self._push_undo()
			self.points.append((x, y))
			self.points.sort(key=lambda point: point[0])
			index = self.points.index((x, y))
			self.changed.emit()
		else:
			self._push_undo()
		self._drag = index
		self.update()

	def mouseMoveEvent(self, event):
		if self._drag is None:
			self.setCursor(QtCore.Qt.SizeAllCursor if self._hit(event.pos()) is not None
						   else QtCore.Qt.CrossCursor)
			return
		x, y = self._from_px(event.pos())
		if self.snap:
			y = 1.0 if y >= 0.5 else 0.0
		if self._drag == 0:
			x = 0.0                                  # the first key must stay at the clip start
		else:
			x = max(x, self.points[self._drag - 1][0])
		if self._drag < len(self.points) - 1:
			x = min(x, self.points[self._drag + 1][0])
		self.points[self._drag] = (x, y)
		self.update()
		self.changed.emit()

	def mouseReleaseEvent(self, _event):
		self._drag = None


class MainWindow(window.MainWindow):
	def __init__(self, opts: GuiOptions, initial_path: str = ""):
		self.body = QtWidgets.QWidget()
		super().__init__("Motiongraph Editor", opts=opts, central_widget=self.body)
		self.setAcceptDrops(True)
		self.ovl = None
		self.loader = None
		self.state_rows = []
		self.field_rows = []
		self.plan = None
		self.activity_address_filter = None
		self.activity_tree_state = None
		self.clone_redirect_source = None
		self.loaded_identity = None
		self.editor_generation = 0
		self.scan_generation = None
		self.effect_loaded_target = None
		self.effect_draft = None
		self.stage_has_result = False
		self.audio_scan_identity = None

		root = QtWidgets.QVBoxLayout(self.body)
		banner = QtWidgets.QLabel(
			"SAFE MODE: staged output only. Topology growth is limited to the game-verified complete-activity clone. "
			"The source OVL is never overwritten."
		)
		banner.setWordWrap(True)
		banner.setStyleSheet("color: #ffe075; font-weight: bold; padding: 5px;")
		root.addWidget(banner)

		source_row = QtWidgets.QHBoxLayout()
		self.source_edit = QtWidgets.QLineEdit(initial_path)
		self.source_edit.setPlaceholderText("Source OVL containing one motiongraph")
		browse = QtWidgets.QPushButton("Browse…")
		browse.clicked.connect(self.browse_source)
		self.load_button = QtWidgets.QPushButton("Load")
		self.load_button.clicked.connect(self.load_source)
		source_row.addWidget(QtWidgets.QLabel("Source OVL"))
		source_row.addWidget(self.source_edit, 1)
		source_row.addWidget(browse)
		source_row.addWidget(self.load_button)
		root.addLayout(source_row)

		name_row = QtWidgets.QHBoxLayout()
		self.name_edit = QtWidgets.QLineEdit()
		self.name_edit.setPlaceholderText("Auto-detect when the OVL contains exactly one motiongraph")
		self.loaded_label = QtWidgets.QLabel("Not loaded")
		name_row.addWidget(QtWidgets.QLabel("Motiongraph"))
		name_row.addWidget(self.name_edit, 1)
		name_row.addWidget(self.loaded_label)
		root.addLayout(name_row)

		self.tabs = QtWidgets.QTabWidget()
		root.addWidget(self.tabs, 1)
		self._build_graph_tab()
		self._build_decision_graph_tab()
		self._build_states_tab()
		self._build_decisions_tab()
		self._build_fields_tab()
		self._build_clone_tab()
		self._build_chooser_tab()
		self._build_decision_tab()
		self._build_retarget_tab()
		self._build_rename_repair_tab()
		self._build_capacity_tab()
		self._build_audio_tab()
		self._build_effects_tab()
		self._build_stage_tab()

		# The body was initially inserted directly by MainWindow. Reparent it into
		# Cobra's standard collapsible content/logger splitter, as used by OVL Tool.
		self.central_layout.removeWidget(self.body)
		if self.cfg.get("enable_logger_widget", True) and self.opts.logger_enabled:
			self.layout_logger(self.body)
		else:
			self.central_layout.addWidget(self.body)
		self.build_menus({widgets.VIEW_MENU: self.view_menu_items})
		self.status_bar.showMessage("Open a donor OVL to begin")
		logging.info("Motiongraph Editor ready; source archives are read-only")

	def _build_graph_tab(self):
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)
		controls = QtWidgets.QHBoxLayout()
		self.graph_search = QtWidgets.QLineEdit()
		self.graph_search.setPlaceholderText("Find state label or clip")
		self.graph_search.returnPressed.connect(self.find_graph_node)
		find_button = QtWidgets.QPushButton("Find next")
		find_button.clicked.connect(self.find_graph_node)
		fit_button = QtWidgets.QPushButton("Fit graph")
		fit_button.clicked.connect(self.fit_graph)
		self.graph_hops = QtWidgets.QSpinBox()
		self.graph_hops.setRange(1, 4)
		self.graph_hops.setValue(1)
		self.graph_hops.setPrefix("Hops: ")
		self.graph_hops.valueChanged.connect(self.refocus_state_graph)
		self.graph_layout_mode = QtWidgets.QComboBox()
		self.graph_layout_mode.addItem("Hierarchical focus", "hierarchical")
		self.graph_layout_mode.addItem("Original grid", "grid")
		self.graph_layout_mode.setToolTip(
			"Hierarchical focus puts incoming states left, the selected state in the middle, "
			"and outgoing states right. Show All remains a compact overview grid."
		)
		self.graph_layout_mode.currentIndexChanged.connect(self.relayout_state_graph)
		focus_button = QtWidgets.QPushButton("Focus selected")
		focus_button.clicked.connect(self.focus_selected_state)
		all_button = QtWidgets.QPushButton("Show all")
		all_button.clicked.connect(self.show_all_state_graph)
		self.edges_toggle = QtWidgets.QCheckBox("Show transition edges")
		self.edges_toggle.setChecked(True)
		self.edges_toggle.toggled.connect(self.toggle_graph_edges)
		pan_hint = QtWidgets.QLabel("Pan: background/middle/Space + drag · H: recenter")
		pan_hint.setStyleSheet("color: #9aa0a6;")
		controls.addWidget(self.graph_search, 1)
		controls.addWidget(find_button)
		controls.addWidget(fit_button)
		controls.addWidget(self.graph_hops)
		controls.addWidget(self.graph_layout_mode)
		controls.addWidget(focus_button)
		controls.addWidget(all_button)
		controls.addWidget(self.edges_toggle)
		controls.addWidget(pan_hint)
		layout.addLayout(controls)
		self.graph_breadcrumb = QtWidgets.QLabel("All states")
		self.graph_breadcrumb.setStyleSheet("color: #8ab4f8; padding-left: 3px;")
		layout.addWidget(self.graph_breadcrumb)
		explanation = QtWidgets.QLabel(
			"≈ labels are inferred from reachable clips or streams; Frontier states are anonymous runtime containers. "
			"In Hierarchical focus mode, select any state to isolate and lay out its neighborhood."
		)
		explanation.setStyleSheet("color: #9aa0a6; padding: 0 2px 4px 2px;")
		layout.addWidget(explanation)
		self.graph_scene = QtWidgets.QGraphicsScene(self)
		self.graph_scene.selectionChanged.connect(self.graph_selection_changed)
		self.graph_view = GraphView(self.graph_scene)
		layout.addWidget(self.graph_view, 1)
		self.graph_nodes = {}
		self.graph_edges = []
		self.graph_edge_records = []
		self.graph_grid_positions = {}
		self.graph_focus_state = None
		self.graph_match_index = -1
		self.tabs.addTab(page, "Graph")

	def _build_decision_graph_tab(self):
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)
		controls = QtWidgets.QHBoxLayout()
		self.decision_graph_search = QtWidgets.QLineEdit()
		self.decision_graph_search.setPlaceholderText("Find opcode, variable, parameter, or target state")
		self.decision_graph_search.returnPressed.connect(self.find_decision_node)
		find_button = QtWidgets.QPushButton("Find next")
		find_button.clicked.connect(self.find_decision_node)
		fit_button = QtWidgets.QPushButton("Fit graph")
		fit_button.clicked.connect(self.fit_decision_graph)
		self.decision_graph_hops = QtWidgets.QSpinBox()
		self.decision_graph_hops.setRange(1, 2)
		self.decision_graph_hops.setValue(1)
		self.decision_graph_hops.setPrefix("Hops: ")
		focus_button = QtWidgets.QPushButton("Focus selected")
		focus_button.clicked.connect(self.focus_selected_decision)
		all_button = QtWidgets.QPushButton("Show all")
		all_button.clicked.connect(self.show_all_decision_graph)
		controls.addWidget(self.decision_graph_search, 1)
		controls.addWidget(find_button)
		controls.addWidget(fit_button)
		controls.addWidget(self.decision_graph_hops)
		controls.addWidget(focus_button)
		controls.addWidget(all_button)
		controls.addWidget(QtWidgets.QLabel("Pan: background/middle/Space + drag · H: recenter"))
		layout.addLayout(controls)
		self.decision_graph_breadcrumb = QtWidgets.QLabel("All decision roots")
		self.decision_graph_breadcrumb.setStyleSheet("color: #d7b7ff; padding-left: 3px;")
		layout.addWidget(self.decision_graph_breadcrumb)
		self.decision_graph_scene = QtWidgets.QGraphicsScene(self)
		self.decision_graph_scene.selectionChanged.connect(self.decision_graph_selection_changed)
		self.decision_graph_view = GraphView(self.decision_graph_scene)
		layout.addWidget(self.decision_graph_view, 1)
		self.decision_node_details = QtWidgets.QPlainTextEdit()
		self.decision_node_details.setReadOnly(True)
		self.decision_node_details.setMaximumHeight(125)
		layout.addWidget(self.decision_node_details)
		self.decision_graph_nodes = {}
		self.decision_graph_edges = []
		self.decision_edge_records = []
		self.decision_graph_data = {"nodes": [], "edges": [], "roots": []}
		self.decision_match_index = -1
		self.tabs.addTab(page, "Decision Graph")

	def _build_states_tab(self):
		page = QtWidgets.QWidget()
		layout = QtWidgets.QHBoxLayout(page)
		self.state_table = QtWidgets.QTableWidget(0, 5)
		self.state_table.setHorizontalHeaderLabels(
			["Index", "Derived label", "Activities", "Clips", "Outgoing"]
		)
		self.state_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
		self.state_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
		self.state_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
		self.state_table.horizontalHeader().setStretchLastSection(True)
		self.state_table.itemSelectionChanged.connect(self.show_state_details)
		self.state_details = QtWidgets.QPlainTextEdit()
		self.state_details.setReadOnly(True)
		self.state_details.setPlaceholderText("Select a state to inspect its composition and transitions")
		self.activity_search = QtWidgets.QLineEdit()
		self.activity_search.setPlaceholderText("Find activity type, relationship, label, or clip")
		self.activity_search.returnPressed.connect(self.find_activity)
		self.inspect_activity_button = QtWidgets.QPushButton("Inspect selected activity fields")
		self.inspect_activity_button.setToolTip(
			"Open Fields / Edit and scan for byte-verified properties of the selected activity"
		)
		self.inspect_activity_button.clicked.connect(self.inspect_activity_fields)
		self.clone_activity_button = QtWidgets.QPushButton("Clone selected complete activity")
		self.clone_activity_button.setToolTip(
			"Open the Clone Activity tab with this exact activity instance selected"
		)
		self.clone_activity_button.clicked.connect(self.select_activity_for_clone)
		self.activity_tree = QtWidgets.QTreeWidget()
		self.activity_tree.setHeaderLabels(["Relationship", "Activity type", "Label / clips"])
		self.activity_tree.setAlternatingRowColors(True)
		self.activity_tree.itemDoubleClicked.connect(self.open_activity_reference)
		self.activity_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
		self.activity_tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
		self.activity_tree.header().setStretchLastSection(True)
		right = QtWidgets.QWidget()
		right_layout = QtWidgets.QVBoxLayout(right)
		right_layout.setContentsMargins(0, 0, 0, 0)
		right_layout.addWidget(self.state_details, 2)
		right_layout.addWidget(self.activity_search)
		right_layout.addWidget(self.inspect_activity_button)
		right_layout.addWidget(self.clone_activity_button)
		right_layout.addWidget(self.activity_tree, 5)
		split = QtWidgets.QSplitter()
		split.addWidget(self.state_table)
		split.addWidget(right)
		split.setSizes([650, 450])
		layout.addWidget(split)
		self.tabs.addTab(page, "States")

	def _build_decisions_tab(self):
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)
		self.decision_search = QtWidgets.QLineEdit()
		self.decision_search.setPlaceholderText("Find text in decision report")
		self.decision_search.returnPressed.connect(self.find_decision)
		self.decision_text = QtWidgets.QPlainTextEdit()
		self.decision_text.setReadOnly(True)
		self.decision_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
		layout.addWidget(self.decision_search)
		layout.addWidget(self.decision_text, 1)
		self.tabs.addTab(page, "Decisions")

	def _build_fields_tab(self):
		self.fields_page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(self.fields_page)
		guide = QtWidgets.QGroupBox("Activity settings — plain-language guide")
		guide_layout = QtWidgets.QVBoxLayout(guide)
		guide_text = QtWidgets.QLabel(
			"<b>Clip</b> chooses the animation; <b>Speed</b> changes playback rate; "
			"<b>Weight</b> controls blend strength; <b>Flags</b> are named playback switches; "
			"<b>Priority</b> affects competing animation contributions; and "
			"<b>Pointers</b> are links to strings, objects, or lists—not values to guess. "
			"Hover a field row for its specific explanation."
		)
		guide_text.setWordWrap(True)
		guide_layout.addWidget(guide_text)
		safety = QtWidgets.QLabel(
			"Safe authoring rule: clone the <b>complete activity settings</b> from a compatible donor. "
			"Changing only the clip keeps the old donor's flags, layer/priority behavior, sync links, "
			"and data streams, which can produce partial-body or T-like results."
		)
		safety.setWordWrap(True)
		safety.setStyleSheet("color: #ffe075; padding-top: 3px;")
		guide_layout.addWidget(safety)
		layout.addWidget(guide)
		scope_row = QtWidgets.QHBoxLayout()
		self.field_scope = QtWidgets.QComboBox()
		self.field_scope.addItem("Entire motiongraph", "all")
		self.field_scope.addItem("Selected state(s)", "states")
		self.field_scope.addItem("Exact selected activity", "activity")
		self.field_scope.currentIndexChanged.connect(self.update_field_scope_label)
		self.field_scope_label = QtWidgets.QLabel("All activity instances in the motiongraph")
		self.field_scope_label.setStyleSheet("color: #9aa0a6;")
		scope_row.addWidget(QtWidgets.QLabel("Scope"))
		scope_row.addWidget(self.field_scope)
		scope_row.addWidget(self.field_scope_label, 1)
		layout.addLayout(scope_row)
		filters = QtWidgets.QHBoxLayout()
		self.activity_filter = QtWidgets.QLineEdit()
		self.activity_filter.setPlaceholderText("Activity label contains (optional)")
		self.type_filter = QtWidgets.QLineEdit()
		self.type_filter.setPlaceholderText("Exact activity type (optional)")
		self.field_filter = QtWidgets.QLineEdit()
		self.field_filter.setPlaceholderText("Exact field path (recommended)")
		self.scan_button = QtWidgets.QPushButton("Scan fields")
		self.scan_button.clicked.connect(self.scan_fields)
		for widget in (self.activity_filter, self.type_filter, self.field_filter, self.scan_button):
			filters.addWidget(widget)
		layout.addLayout(filters)
		self.field_table = QtWidgets.QTableWidget(0, 8)
		self.field_table.setHorizontalHeaderLabels(
			["Activity", "Activity type", "Field", "Kind", "Value", "Pool", "Offset", "Verified"]
		)
		self.field_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
		self.field_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
		self.field_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
		self.field_table.horizontalHeader().setStretchLastSection(True)
		self.show_field_addresses = QtWidgets.QCheckBox("Advanced: show pool/offset addresses")
		self.show_field_addresses.setToolTip(
			"Pool and offset identify where bytes are stored; they are not animation settings."
		)
		self.show_field_addresses.toggled.connect(
			lambda checked: (
				self.field_table.setColumnHidden(5, not checked),
				self.field_table.setColumnHidden(6, not checked),
			)
		)
		self.field_table.setColumnHidden(5, True)
		self.field_table.setColumnHidden(6, True)
		layout.addWidget(self.show_field_addresses)
		edit_row = QtWidgets.QHBoxLayout()
		self.edit_mode = QtWidgets.QComboBox()
		self.edit_mode.addItems(["Numeric value", "Enum name", "Flag operations"])
		self.value_edit = QtWidgets.QLineEdit()
		self.value_edit.setPlaceholderText("Replacement: e.g. 0.25, enum name, or FlagA=1,FlagB=0")
		self.plan_button = QtWidgets.QPushButton("Add selected edit to queue")
		self.plan_button.clicked.connect(self.create_plan)
		edit_row.addWidget(QtWidgets.QLabel("Replacement"))
		edit_row.addWidget(self.edit_mode)
		edit_row.addWidget(self.value_edit, 1)
		edit_row.addWidget(self.plan_button)
		layout.addLayout(edit_row)
		queue_controls = QtWidgets.QHBoxLayout()
		self.queue_label = QtWidgets.QLabel("Patch queue: empty")
		self.queue_label.setStyleSheet("color: #8ab4f8;")
		self.save_queue_button = QtWidgets.QPushButton("Save queue...")
		self.save_queue_button.clicked.connect(self.save_queue)
		self.load_queue_button = QtWidgets.QPushButton("Load queue...")
		self.load_queue_button.clicked.connect(self.load_queue)
		self.clear_queue_button = QtWidgets.QPushButton("Clear queue")
		self.clear_queue_button.clicked.connect(self.clear_queue)
		queue_controls.addWidget(self.queue_label, 1)
		queue_controls.addWidget(self.save_queue_button)
		queue_controls.addWidget(self.load_queue_button)
		queue_controls.addWidget(self.clear_queue_button)
		layout.addLayout(queue_controls)
		self.queue_table = QtWidgets.QTableWidget(0, 4)
		self.queue_table.setHorizontalHeaderLabels(["Field", "Kind", "Replacement", "Addresses"])
		self.queue_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
		self.queue_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
		self.queue_table.horizontalHeader().setStretchLastSection(True)
		self.queue_table.setMaximumHeight(130)
		layout.addWidget(self.queue_table)
		layout.addWidget(self.field_table, 1)
		self.refresh_patch_queue()
		self.tabs.addTab(self.fields_page, "Fields / Edit")

	def _build_clone_tab(self):
		self.clone_page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(self.clone_page)
		info = QtWidgets.QLabel(
			"Clone one exact <b>AnimationActivity</b> together with its full settings and pointer layout, "
			"then redirect that instance's current inbound references to the clone. This is the complete "
			"activity/payload operation verified in game by Test L."
		)
		info.setWordWrap(True)
		layout.addWidget(info)
		limits = QtWidgets.QLabel(
			"This does <b>not</b> duplicate a MANI, create a new clip name, add another state entry, or add "
			"random selection. It creates an independent copy of the selected activity on its existing edges."
		)
		limits.setWordWrap(True)
		limits.setStyleSheet("color: #ffe075; padding: 4px 0;")
		layout.addWidget(limits)
		form = QtWidgets.QFormLayout()
		self.clone_selection_label = QtWidgets.QLabel(
			"No activity selected — choose one in States and click Clone selected complete activity"
		)
		self.clone_selection_label.setWordWrap(True)
		form.addRow("Exact donor", self.clone_selection_label)
		self.clone_usage_label = QtWidgets.QLabel("Not selected")
		self.clone_usage_label.setWordWrap(True)
		form.addRow("Used by states", self.clone_usage_label)
		self.clone_redirect_scope = QtWidgets.QComboBox()
		self.clone_redirect_scope.addItem("Only selected state occurrence", "occurrence")
		self.clone_redirect_scope.addItem(
			"Every inbound reference to this shared activity", "all"
		)
		self.clone_redirect_scope.setToolTip(
			"Occurrence redirects only the exact activity pointer clicked in the state tree. "
			"All redirects every pointer that currently reaches the shared donor."
		)
		form.addRow("Redirect scope", self.clone_redirect_scope)
		self.clone_speed_override = QtWidgets.QCheckBox("Override playback speed on the clone")
		self.clone_speed_override.setChecked(False)
		form.addRow("Optional edit", self.clone_speed_override)
		self.clone_speed = QtWidgets.QDoubleSpinBox()
		self.clone_speed.setDecimals(4)
		self.clone_speed.setRange(0.0, 1000.0)
		self.clone_speed.setSingleStep(0.25)
		self.clone_speed.setValue(1.0)
		self.clone_speed.setEnabled(False)
		self.clone_speed_override.toggled.connect(self.clone_speed.setEnabled)
		form.addRow("Clone speed", self.clone_speed)
		layout.addLayout(form)
		apply_button = QtWidgets.QPushButton("Clone complete activity into staged family")
		apply_button.clicked.connect(self.apply_complete_activity_clone)
		layout.addWidget(apply_button)
		workflow = QtWidgets.QLabel(
			"Workflow: load the untouched source OVL → select an activity in States → copy the full family "
			"in Stage / Apply → run this once. The staged OVL is reloaded and verified automatically."
		)
		workflow.setWordWrap(True)
		workflow.setStyleSheet("color: #9aa0a6;")
		layout.addWidget(workflow)
		layout.addStretch(1)
		self.tabs.addTab(self.clone_page, "Clone Activity")

	def _build_retarget_tab(self):
		page = QtWidgets.QWidget()
		form = QtWidgets.QFormLayout(page)
		info = QtWidgets.QLabel(
			"Clip retarget repoints existing references to another existing same-pool string. "
			"String-slot replacement fits a new shorter/equal ASCII name into one existing allocation. "
			"Important: clip retargeting does not copy flags, priority/layer behavior, sync links, or data "
			"streams. Use it only between behaviorally compatible clips; complete activity cloning is safer."
		)
		info.setWordWrap(True)
		form.addRow(info)
		self.from_clip = QtWidgets.QLineEdit()
		self.to_clip = QtWidgets.QLineEdit()
		self.expected_count = QtWidgets.QSpinBox()
		self.expected_count.setRange(1, 1000000)
		self.expected_count.setValue(1)
		clip_button = QtWidgets.QPushButton("Apply clip retarget to staged family")
		clip_button.clicked.connect(self.apply_clip_retarget)
		form.addRow("Existing source string", self.from_clip)
		form.addRow("Existing target string", self.to_clip)
		form.addRow("Expected references", self.expected_count)
		form.addRow(clip_button)
		line = QtWidgets.QFrame()
		line.setFrameShape(QtWidgets.QFrame.HLine)
		form.addRow(line)
		self.slot_from = QtWidgets.QLineEdit()
		self.slot_to = QtWidgets.QLineEdit()
		slot_button = QtWidgets.QPushButton("Apply string-slot replacement to staged family")
		slot_button.clicked.connect(self.apply_string_slot)
		form.addRow("Allocated source string", self.slot_from)
		form.addRow("Replacement string", self.slot_to)
		form.addRow(slot_button)

		line2 = QtWidgets.QFrame()
		line2.setFrameShape(QtWidgets.QFrame.HLine)
		form.addRow(line2)
		single = QtWidgets.QLabel(
			"<b>Set the clip on ONE activity.</b> The two operations above move EVERY reference to a "
			"string, or overwrite a string in place. This moves a single activity's clip pointer and "
			"ALLOCATES the name if the archive does not have it, so a clone can be given its own clip. "
			"Select the activity in <b>States</b> first (the same selection Clone Activity uses)."
		)
		single.setWordWrap(True)
		form.addRow(single)
		self.single_clip_label = QtWidgets.QLabel("No activity selected")
		self.single_clip_label.setWordWrap(True)
		form.addRow("Selected activity", self.single_clip_label)
		self.single_clip_name = QtWidgets.QLineEdit()
		self.single_clip_name.setPlaceholderText("Full clip name, e.g. Species$StandPreen03")
		form.addRow("New clip name", self.single_clip_name)
		single_button = QtWidgets.QPushButton("Set clip on selected activity in staged family")
		single_button.clicked.connect(self.apply_single_clip)
		form.addRow(single_button)
		warn = QtWidgets.QLabel(
			"Writes the REFERENCE, not the animation. The clip must exist in the .manis bundles or "
			"the activity resolves to nothing."
		)
		warn.setWordWrap(True)
		warn.setStyleSheet("color: #ffe075; padding: 4px 0;")
		form.addRow(warn)
		self.tabs.addTab(page, "Clip Retarget")

	def apply_single_clip(self):
		"""Point one activity at a clip name, allocating it if it is new.

		Chaining note: like every tab here this reads `source` and writes the
		stage. To clone an activity and then give the clone its own clip, run the
		clone, then point `source` at the staged file and re-stage before this.
		"""
		try:
			selection = getattr(self, "activity_address_filter", None)
			if not selection:
				raise ValueError(
					"Select an activity in the States tab first - the same selection "
					"Clone Activity uses")
			pool, offset = selection
			clip = self.single_clip_name.text().strip()
			if not clip:
				raise ValueError("Enter the full clip name, e.g. Species$StandPreen03")
			report = set_activity_clip(self.source_path(), self.output_path(), pool, offset,
									   clip, name=self.name_edit.text().strip() or None,
									   game=DEFAULT_GAME)
			self.status_bar.showMessage(
				f"{report.old_clip} -> {report.new_clip} on activity {pool}:{offset}", 12000)
			logging.info(f"Single-activity clip retarget: {report}")
			# Allocating a name grows a type-2 pool but adds no decoded object;
			# reusing an existing one changes nothing at all.
			self.census_guard("single clip retarget", expect_added=0, expect_removed=0)
			if report.allocated:
				self.showerror(
					f"'{report.new_clip}' was ALLOCATED in the type-2 tail at "
					f"{report.string_at[0]}:{report.string_at[1]}. It must exist as a real clip in "
					f"the .manis bundles or the activity will resolve to nothing. Test IN GAME."
				)
		except Exception as exc:
			self.showerror(str(exc))

	def _build_chooser_tab(self):
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)
		info = QtWidgets.QLabel(
			"A <b>RandomAnimationActivity</b> picks among clips BY NAME, so adding one needs no "
			"activity wrapper, payload, state or edge - far cheaper than cloning. Weights are "
			"relative; the engine draws from their sum, so the share is what actually matters."
		)
		info.setWordWrap(True)
		layout.addWidget(info)
		note = QtWidgets.QLabel(
			"Weights are reachable as animations[N].weight in Fields / Edit, but nobody finds them "
			"that way. Adding a clip RELOCATES the entry array into a tail pool - game-verified, but "
			"it is topology growth, so re-load the staged file and check the log."
		)
		note.setWordWrap(True)
		note.setStyleSheet("color: #ffe075; padding: 4px 0;")
		layout.addWidget(note)

		filter_row = QtWidgets.QHBoxLayout()
		self.chooser_filter = QtWidgets.QLineEdit()
		self.chooser_filter.setPlaceholderText("Filter by clip substring, e.g. Rest or Eat")
		refresh = QtWidgets.QPushButton("List choosers")
		refresh.clicked.connect(self.refresh_choosers)
		filter_row.addWidget(QtWidgets.QLabel("Match"))
		filter_row.addWidget(self.chooser_filter, 1)
		filter_row.addWidget(refresh)
		layout.addLayout(filter_row)

		self.chooser_tree = QtWidgets.QTreeWidget()
		self.chooser_tree.setHeaderLabels(["Chooser / clip", "Weight", "Share", "Detail"])
		self.chooser_tree.setColumnWidth(0, 340)
		self.chooser_tree.itemSelectionChanged.connect(self.chooser_selected)
		layout.addWidget(self.chooser_tree, 1)

		form = QtWidgets.QFormLayout()
		self.chooser_target = QtWidgets.QLabel("No chooser selected")
		form.addRow("Selected", self.chooser_target)
		self.chooser_weights = QtWidgets.QLineEdit()
		self.chooser_weights.setPlaceholderText("Comma-separated weights for every clip, in order")
		form.addRow("Weights", self.chooser_weights)
		weight_button = QtWidgets.QPushButton("Apply weights to staged family")
		weight_button.clicked.connect(self.apply_chooser_weights)
		form.addRow(weight_button)
		line = QtWidgets.QFrame()
		line.setFrameShape(QtWidgets.QFrame.HLine)
		form.addRow(line)
		self.chooser_add_clip = QtWidgets.QLineEdit()
		self.chooser_add_clip.setPlaceholderText("Full clip name, e.g. Species$Rest03")
		form.addRow("Add clip", self.chooser_add_clip)
		self.chooser_add_weight = QtWidgets.QSpinBox()
		self.chooser_add_weight.setRange(1, 10000)
		self.chooser_add_weight.setValue(1)
		form.addRow("Its weight", self.chooser_add_weight)
		self.chooser_add_weights = QtWidgets.QLineEdit()
		self.chooser_add_weights.setPlaceholderText(
			"Optional: all weights after the add, comma separated (one per clip, including the new one)")
		self.chooser_add_weights.setToolTip(
			"Re-weight in the SAME pass as the add. Applying weights as a second, separate "
			"operation re-reads the original source and throws the added clip away.")
		form.addRow("Re-weight all", self.chooser_add_weights)
		add_button = QtWidgets.QPushButton("Add clip to chooser in staged family")
		add_button.clicked.connect(self.apply_chooser_add)
		form.addRow(add_button)
		layout.addLayout(form)
		self.tabs.addTab(page, "Choosers")

	def _build_decision_tab(self):
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)
		info = QtWidgets.QLabel(
			"Some clips look like they are NOT chooser-driven - StandPreen / StandPreen02 are each "
			"a lone AnimationActivity the decision layer routes to - but they ARE, at the "
			"<b>decision</b> layer. Both are results of a <b>RandomChoiceEndDecisionScope</b> whose "
			"<tt>children</tt> is an ArrayPointer, so it relocates and grows like any other array. "
			"<b>Add a state</b> gives you somewhere to route to; <b>add a result</b> is the route."
		)
		info.setWordWrap(True)
		layout.addWidget(info)
		note = QtWidgets.QLabel(
			"Both raise topology. Adding a state RELOCATES the whole state array and bumps the "
			"state count - game-verified 196 -> 197, but re-load the staged file and check the "
			"census. A new state is DORMANT until something routes to it. Nodes marked SHARED are "
			"referenced by more than one place (often a transition-layer decision block as well as "
			"this chooser), which is why a result always gets its own new StateOutput node rather "
			"than reusing one."
		)
		note.setWordWrap(True)
		note.setStyleSheet("color: #ffe075; padding: 4px 0;")
		layout.addWidget(note)

		button_row = QtWidgets.QHBoxLayout()
		refresh = QtWidgets.QPushButton("List decision routes")
		refresh.clicked.connect(self.refresh_decisions)
		button_row.addWidget(refresh)
		button_row.addStretch(1)
		layout.addLayout(button_row)

		self.decision_tree = QtWidgets.QTreeWidget()
		self.decision_tree.setHeaderLabels(["Chooser / result", "Weight", "Share", "Node"])
		self.decision_tree.setColumnWidth(0, 340)
		self.decision_tree.itemSelectionChanged.connect(self.decision_selected)
		layout.addWidget(self.decision_tree, 1)

		form = QtWidgets.QFormLayout()
		self.decision_target = QtWidgets.QLabel("No chooser selected")
		form.addRow("Selected", self.decision_target)
		self.decision_route_state = QtWidgets.QSpinBox()
		self.decision_route_state.setRange(0, 9999)
		form.addRow("Route to state index", self.decision_route_state)
		self.decision_route_weight = QtWidgets.QSpinBox()
		self.decision_route_weight.setRange(1, 100000)
		self.decision_route_weight.setValue(1)
		form.addRow("Its weight", self.decision_route_weight)
		route_button = QtWidgets.QPushButton("Add result to chooser in staged family")
		route_button.clicked.connect(self.apply_decision_route)
		form.addRow(route_button)

		line = QtWidgets.QFrame()
		line.setFrameShape(QtWidgets.QFrame.HLine)
		form.addRow(line)

		self.decision_twin_state = QtWidgets.QSpinBox()
		self.decision_twin_state.setRange(0, 9999)
		form.addRow("New state twinned on", self.decision_twin_state)
		self.decision_new_weight = QtWidgets.QSpinBox()
		self.decision_new_weight.setRange(1, 100000)
		self.decision_new_weight.setValue(1)
		form.addRow("Its weight", self.decision_new_weight)
		state_button = QtWidgets.QPushButton(
			"Add new state AND route to it, in the selected chooser")
		state_button.clicked.connect(self.apply_add_state)
		form.addRow(state_button)
		layout.addLayout(form)
		self.tabs.addTab(page, "Decision Routes")

	def refresh_decisions(self):
		try:
			source = self.source_path()
			rows = list_decision_choosers(source, self.name_edit.text().strip() or None,
										  DEFAULT_GAME)
		except Exception as exc:
			self.showerror(str(exc))
			return
		self.decision_tree.clear()
		for row in rows:
			parent = QtWidgets.QTreeWidgetItem([
				f"chooser {row['pool']}:{row['offset']}", "", "",
				f"{row['num_children']} results  "
				f"array {row['children_at'][0]}:{row['children_at'][1]}"])
			parent.setData(0, QtCore.Qt.UserRole, (row["pool"], row["offset"]))
			for result in row["results"]:
				node = result["node_at"]
				try:
					shared = inbound_count(source, node[0], node[1], DEFAULT_GAME) > 1
				except Exception:
					shared = False
				parent.addChild(QtWidgets.QTreeWidgetItem([
					f"STATE[{result['state']}] {result['label']}",
					str(result["weight"]), f"{result['share']:.1f}%",
					f"{node[0]}:{node[1]}" + ("  SHARED" if shared else "")]))
			self.decision_tree.addTopLevelItem(parent)
			parent.setExpanded(True)
		self.status_bar.showMessage(f"Listed {len(rows)} decision chooser(s)", 8000)
		if not rows:
			logging.info("No RandomChoiceEndDecisionScope nodes in this graph")

	def decision_selected(self):
		items = self.decision_tree.selectedItems()
		if not items:
			return
		item = items[0]
		if item.parent() is not None:
			item = item.parent()
		data = item.data(0, QtCore.Qt.UserRole)
		if data:
			self.decision_target.setText(f"chooser {data[0]}:{data[1]}")

	def _selected_decision(self):
		items = self.decision_tree.selectedItems()
		if not items:
			raise ValueError("Select a chooser in the list first")
		item = items[0]
		if item.parent() is not None:
			item = item.parent()
		data = item.data(0, QtCore.Qt.UserRole)
		if not data:
			raise ValueError("Select a chooser row, not a result row")
		return data

	def apply_decision_route(self):
		try:
			pool, offset = self._selected_decision()
			state = self.decision_route_state.value()
			weight = self.decision_route_weight.value()
			report = grow_decision_chooser(
				self.source_path(), self.output_path(), pool, offset, state,
				weight=weight, name=self.name_edit.text().strip() or None, game=DEFAULT_GAME)
			self.status_bar.showMessage(
				f"Chooser {pool}:{offset} now has {report.new_results} results", 12000)
			logging.info(f"Decision route added: {report}")
			# A new result adds a node, a Something and a ResultParam, and orphans
			# the donor children array - so objects rise and one may disappear.
			self.census_guard("decision route growth", expect_added=None, expect_removed=None)
			self.showerror(
				"Decision growth is TOPOLOGY GROWTH, and cobra reload is not proof of engine "
				"acceptance. Check the census line in the log, then test IN GAME."
			)
		except Exception as exc:
			self.showerror(str(exc))

	def apply_add_state(self):
		"""Add a state AND its inbound route in ONE pass.

		Not two operations: every tab reads `source` and writes the stage, so
		adding the state and then routing to it separately would have the second
		call discard the first. A dormant state is useless anyway.
		"""
		try:
			pool, offset = self._selected_decision()
			twin = self.decision_twin_state.value()
			report = add_state_and_route(
				self.source_path(), self.output_path(), twin, pool, offset,
				weight=self.decision_new_weight.value(),
				name=self.name_edit.text().strip() or None, game=DEFAULT_GAME)
			self.status_bar.showMessage(
				f"States {report.old_count} -> {report.new_count}; "
				f"new STATE[{report.new_state_index}] routed from {pool}:{offset}", 15000)
			logging.info(f"New state added and routed: {report}")
			self.census_guard("new state", expect_added=None, expect_removed=None)
			self.showerror(
				f"STATE[{report.new_state_index}] added (twin of [{twin}]) and routed from "
				f"chooser {pool}:{offset} at weight {self.decision_new_weight.value()}.\n\n"
				f"It currently plays the SAME clips as its twin - that is what makes the "
				f"change visible. Give it its own AnimationActivity to make it a distinct "
				f"behaviour.\n\nThis raises the state count, which is topology growth. Cobra "
				f"reload is not proof of engine acceptance - test IN GAME."
			)
		except Exception as exc:
			self.showerror(str(exc))

	def _build_capacity_tab(self):
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)
		info = QtWidgets.QLabel(
			"Where the room is: tail space, allocations with adjacent padding that could absorb one "
			"more element, and what is decoded but unreachable. Consult this BEFORE attempting growth."
		)
		info.setWordWrap(True)
		layout.addWidget(info)
		warn = QtWidgets.QLabel(
			"A reported 'dead tail' has already turned out to hold LIVE STRINGS. Read the bytes "
			"before reusing any space this panel offers."
		)
		warn.setWordWrap(True)
		warn.setStyleSheet("color: #ffe075; padding: 4px 0;")
		layout.addWidget(warn)
		run = QtWidgets.QPushButton("Run capacity audit on the loaded motiongraph")
		run.clicked.connect(self.run_capacity_audit)
		layout.addWidget(run)
		self.capacity_text = QtWidgets.QPlainTextEdit()
		self.capacity_text.setReadOnly(True)
		self.capacity_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
		layout.addWidget(self.capacity_text, 1)
		self.tabs.addTab(page, "Capacity")

	def _build_rename_repair_tab(self):
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)
		info = QtWidgets.QLabel(
			"Renaming a species rewrites clip names inside the .manis, but graph references can be "
			"left pointing at the OLD <b>&lt;Token&gt;$Clip</b> strings. Such a reference names a clip "
			"that exists in no bundle, so whatever plays it has NO ANIMATION - and nothing reports an "
			"error, at load or at runtime. Survey first; the repair is a pure fragment repoint."
		)
		info.setWordWrap(True)
		layout.addWidget(info)
		form = QtWidgets.QFormLayout()
		self.rename_old = QtWidgets.QLineEdit()
		self.rename_old.setPlaceholderText("Pre-rename token, e.g. Sarcmimsaee")
		self.rename_new = QtWidgets.QLineEdit()
		self.rename_new.setPlaceholderText("Current token, e.g. Viralsarcosuchus")
		form.addRow("Old token", self.rename_old)
		form.addRow("New token", self.rename_new)
		survey_button = QtWidgets.QPushButton("Survey stale references (read-only)")
		survey_button.clicked.connect(self.run_rename_survey)
		form.addRow(survey_button)
		repair_button = QtWidgets.QPushButton("Repoint stale references in staged family")
		repair_button.clicked.connect(self.apply_rename_repair)
		form.addRow(repair_button)
		layout.addLayout(form)
		self.rename_text = QtWidgets.QPlainTextEdit()
		self.rename_text.setReadOnly(True)
		layout.addWidget(self.rename_text, 1)
		hint = QtWidgets.QLabel(
			"Note: a whole-OVL extract diff is BLIND to this edit - the extracted .motiongraph is the "
			"source XML and already carries the correct names. Verify by re-surveying, not by diffing."
		)
		hint.setWordWrap(True)
		hint.setStyleSheet("color: #9aa0a6;")
		layout.addWidget(hint)
		self.tabs.addTab(page, "Rename Repair")

	def _build_audio_tab(self):
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)
		info = QtWidgets.QLabel(
			"A motiongraph fires sounds BY NAME. To give a donor-based species its own voice, "
			"rename those strings to your own prefix and ship a bank that answers them.<br><br>"
			"<b>Not every name is safe to rename.</b> Some resolve from SHARED banks used by "
			"several species; renaming one orphans the sound in both directions and it goes "
			"silent with no error. Scan first - those are shown greyed out and cannot be ticked."
			"<br><br><b>Revert</b> goes the other way, for publishing a copy that needs none of "
			"the custom-audio chain - no dll, no loader mod, no banks in <i>Audio/</i>. Stock "
			"names resolve straight out of the game's own audio OVLs."
		)
		info.setWordWrap(True)
		layout.addWidget(info)

		self.audio_direction_box = QtWidgets.QComboBox()
		self.audio_direction_box.addItem(
			"Give this species its own voice  (donor -> your prefix)", AUDIO_RENAME)
		self.audio_direction_box.addItem(
			"Revert to the donor's stock sounds  (your prefix -> donor)", AUDIO_REVERT)
		self.audio_direction_box.setToolTip(
			"Revert reads only the graph and the game's shipped audio registry. It needs "
			"neither the mod's banks nor the audio kit, so it still works after you have "
			"deleted them.")
		self.audio_direction_box.currentIndexChanged.connect(self.audio_direction_changed)
		direction_row = QtWidgets.QFormLayout()
		direction_row.addRow("Direction", self.audio_direction_box)
		layout.addLayout(direction_row)

		form = QtWidgets.QFormLayout()
		self.audio_game = QtWidgets.QLineEdit()
		self.audio_game.setPlaceholderText("Game folder, e.g. C:/.../Jurassic World Evolution 3")
		game_row = QtWidgets.QHBoxLayout()
		game_row.addWidget(self.audio_game, 1)
		game_browse = QtWidgets.QPushButton("Browse...")
		game_browse.clicked.connect(self.browse_audio_game)
		game_row.addWidget(game_browse)
		game_holder = QtWidgets.QWidget()
		game_holder.setLayout(game_row)
		self.audio_donor = QtWidgets.QLineEdit()
		self.audio_donor.setPlaceholderText("Donor species, e.g. Indoraptor")
		self.audio_prefix = QtWidgets.QLineEdit()
		self.audio_prefix.setPlaceholderText("Your prefix, e.g. Indocapi - may be longer than the donor")
		form.addRow("Game folder", game_holder)
		form.addRow("Donor species", self.audio_donor)
		form.addRow("New prefix", self.audio_prefix)
		layout.addLayout(form)

		build_box = QtWidgets.QGroupBox("1. Build this species' own sound banks")
		build_form = QtWidgets.QFormLayout(build_box)
		self.audio_soundmod = QtWidgets.QLineEdit()
		self.audio_soundmod.setPlaceholderText(
			"optional: a replacement-sound mod's <Donor>_media.ovl - blank uses stock audio")
		sm_row = QtWidgets.QHBoxLayout()
		sm_row.addWidget(self.audio_soundmod, 1)
		sm_browse = QtWidgets.QPushButton("Browse...")
		sm_browse.clicked.connect(self.browse_audio_soundmod)
		sm_row.addWidget(sm_browse)
		sm_holder = QtWidgets.QWidget(); sm_holder.setLayout(sm_row)

		self.audio_out = QtWidgets.QLineEdit()
		self.audio_out.setPlaceholderText("output folder for the built .bnk files")
		out_row = QtWidgets.QHBoxLayout()
		out_row.addWidget(self.audio_out, 1)
		out_browse = QtWidgets.QPushButton("Browse...")
		out_browse.clicked.connect(self.browse_audio_out)
		out_row.addWidget(out_browse)
		out_holder = QtWidgets.QWidget(); out_holder.setLayout(out_row)

		build_form.addRow("Replacement sound mod", sm_holder)
		self.audio_sounds = QtWidgets.QLineEdit()
		self.audio_sounds.setPlaceholderText("optional: folder of <EventSuffix>.wem files")
		self.audio_marker = QtWidgets.QLineEdit()
		self.audio_marker.setPlaceholderText("optional: media OVL for a distinctive test sound")
		for label, field, pick in (("Custom WEM folder", self.audio_sounds, self.browse_audio_sounds),
			("Marker media OVL", self.audio_marker, self.browse_audio_marker)):
			row = QtWidgets.QHBoxLayout()
			row.addWidget(field, 1)
			button = QtWidgets.QPushButton("Browse...")
			button.clicked.connect(pick)
			row.addWidget(button)
			holder = QtWidgets.QWidget(); holder.setLayout(row)
			build_form.addRow(label, holder)
		build_form.addRow("Output folder", out_holder)
		baseline = QtWidgets.QPushButton("Build baseline / discover WEM names")
		baseline.setToolTip("Build donor audio and sounds_available.txt; ignore Custom WEM and Marker fields for this build. Uses the selected output folder.")
		baseline.clicked.connect(lambda: self.build_species_audio(baseline=True))
		build_form.addRow(baseline)
		self.audio_build = QtWidgets.QPushButton("Build species audio banks")
		self.audio_build.setToolTip(
			"Re-IDs the donor's banks onto a new prefix so this species has its own "
			"audio. A replacement-sound mod ships the SAME ids as stock, so installing "
			"one changes every animal of the donor species; this makes a scoped, "
			"non-replacement mod instead.")
		self.audio_build.clicked.connect(self.build_species_audio)
		build_form.addRow(self.audio_build)
		outputs_row = QtWidgets.QHBoxLayout()
		for label, filename in (("Open WEM suffix list", "sounds_available.txt"),
			("Open build report", "report.json"), ("Open output folder", "")):
			button = QtWidgets.QPushButton(label)
			button.clicked.connect(lambda _checked=False, name=filename: self.open_audio_output(name))
			outputs_row.addWidget(button)
		build_form.addRow(outputs_row)
		hint = QtWidgets.QLabel("Custom recordings must already be compatible WEMs. Build a baseline, open the suffix list, name your WEMs accordingly, then select their folder and build. Use a fresh output folder for each build. Copy BOTH banks and the fragment; then merge.")
		hint.setWordWrap(True)
		build_form.addRow(hint)
		layout.addWidget(build_box)

		scan_button = QtWidgets.QPushButton("2. Scan graph for audio events")
		scan_button.clicked.connect(self.scan_audio_events)
		layout.addWidget(scan_button)

		self.audio_list = QtWidgets.QTreeWidget()
		self.audio_list.setHeaderLabels(["Event name in graph", "Your name for it", "Rename?"])
		self.audio_list.setRootIsDecorated(False)
		self.audio_list.setAlternatingRowColors(True)
		layout.addWidget(self.audio_list, 1)

		button_row = QtWidgets.QHBoxLayout()
		tick_all = QtWidgets.QPushButton("Tick all safe")
		tick_all.clicked.connect(lambda: self.set_audio_ticks(True))
		untick_all = QtWidgets.QPushButton("Untick all")
		untick_all.clicked.connect(lambda: self.set_audio_ticks(False))
		button_row.addWidget(tick_all)
		button_row.addWidget(untick_all)
		button_row.addStretch(1)
		layout.addLayout(button_row)

		self.audio_apply = QtWidgets.QPushButton("Rename ticked events in staged family")
		self.audio_apply.clicked.connect(self.apply_audio_rename)
		layout.addWidget(self.audio_apply)

		self.audio_fragment = QtWidgets.QPushButton(
			"Write registry fragment (.wmetasb.add) for the ticked events")
		self.audio_fragment.clicked.connect(self.write_audio_fragment)
		layout.addWidget(self.audio_fragment)

		note = QtWidgets.QLabel(
			"Renaming the graph is only one of three pieces. The prefab still needs "
			"<b>MotionGraphName</b> pointed at your renamed graph, <b>AudioCore.Name</b> left on "
			"the DONOR (it prefixes engine-generated footstep and death events that live in "
			"shared banks), and the registry fragment below merged in - an undeclared event is "
			"never posted at all."
		)
		note.setWordWrap(True)
		layout.addWidget(note)
		self.audio_direction_changed()
		scroll = QtWidgets.QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setWidget(page)
		self.tabs.addTab(scroll, "Audio Events")

	def _build_effects_tab(self):
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)
		info = QtWidgets.QLabel(
			"An activity fires effects through its <b>datastreams</b>. Add one here and the "
			"animation can switch a particle on - the prefab entity must be named EXACTLY the "
			"same, because that name is the only link between the two.<br><br>"
			"<b>The curve is the on/off switch.</b> Every stock VFX curve goes on and never "
			"goes off, which is fine for a one-shot dust puff and wrong for a continuous "
			"emitter like fire - it would burn forever. Give it a key back down to 0."
		)
		info.setWordWrap(True)
		layout.addWidget(info)

		pick_row = QtWidgets.QHBoxLayout()
		self.effect_activity = QtWidgets.QComboBox()
		self.effect_activity.setEditable(True)
		self.effect_activity.setMinimumWidth(260)
		scan = QtWidgets.QPushButton("List activities")
		scan.clicked.connect(self.scan_effect_activities)
		show = QtWidgets.QPushButton("Show its datastreams")
		show.clicked.connect(self.show_effect_streams)
		pick_row.addWidget(QtWidgets.QLabel("Activity"))
		pick_row.addWidget(self.effect_activity, 1)
		pick_row.addWidget(scan)
		pick_row.addWidget(show)
		layout.addLayout(pick_row)

		self.effect_list = QtWidgets.QTreeWidget()
		self.effect_list.setHeaderLabels(["Datastream", "Type", "curve_type"])
		self.effect_list.setRootIsDecorated(False)
		self.effect_list.setAlternatingRowColors(True)
		self.effect_list.setMaximumHeight(130)
		self.effect_list.itemSelectionChanged.connect(self.load_effect_curve)
		layout.addWidget(self.effect_list)
		self.effect_held = QtWidgets.QLabel()
		self.effect_held.setWordWrap(True)
		layout.addWidget(self.effect_held)

		form = QtWidgets.QFormLayout()
		self.effect_name = QtWidgets.QLineEdit()
		self.effect_name.setPlaceholderText("VFX_Roar_Fire - and name the prefab entity this too")
		self.effect_type = QtWidgets.QComboBox()
		self.effect_type.setEditable(True)
		self.effect_type.addItems(["VFXEnable", "VFXToggle", "AudioEvent",
								   "AudioLoopingEvent", "AudioRTPC", "General"])
		form.addRow("Datastream name", self.effect_name)
		form.addRow("Type", self.effect_type)
		layout.addLayout(form)

		self.curve_editor = CurveEditor()
		self.effects_dirty = False
		self.curve_editor.changed.connect(self._mark_effects_dirty)
		layout.addWidget(self.curve_editor, 1)
		self.curve_readout = QtWidgets.QLabel()
		self.curve_readout.setWordWrap(True)
		self.curve_editor.changed.connect(self.update_curve_readout)
		layout.addWidget(self.curve_readout)

		preset_row = QtWidgets.QHBoxLayout()
		snap = QtWidgets.QCheckBox("Snap to on/off")
		snap.setChecked(True)
		snap.toggled.connect(lambda on: setattr(self.curve_editor, "snap", on))
		preset_row.addWidget(snap)
		burst = QtWidgets.QPushButton("Burst preset")
		burst.setToolTip("What vanilla ships: on part-way through and never off. "
						 "Correct ONLY for one-shot effects.")
		burst.clicked.connect(
			lambda: self.curve_editor.set_points([(0.0, 0.0), (0.35, 1.0), (1.0, 1.0)]))
		window_preset = QtWidgets.QPushButton("On/off window preset")
		window_preset.setToolTip("On, then explicitly off again - use for continuous emitters.")
		window_preset.clicked.connect(
			lambda: self.curve_editor.set_points(
				[(0.0, 0.0), (0.35, 1.0), (0.75, 0.0), (1.0, 0.0)]))
		preset_row.addWidget(burst)
		preset_row.addWidget(window_preset)
		preset_row.addStretch(1)
		layout.addLayout(preset_row)

		action_row = QtWidgets.QHBoxLayout()
		self.effect_add_button = QtWidgets.QPushButton("Add datastream to staged family")
		self.effect_add_button.clicked.connect(self.apply_effect_growth)
		self.effect_set_button = QtWidgets.QPushButton("Write curve to staged family")
		self.effect_set_button.clicked.connect(self.apply_effect_curve)
		self.effect_set_button.setEnabled(False)
		self.effect_all_button = QtWidgets.QPushButton("Write to ALL activities firing this name")
		self.effect_all_button.setToolTip("Applies this curve and type to every activity that already "
							  "fires this datastream, chaining the rewrites for you.")
		self.effect_all_button.clicked.connect(self.apply_effect_curve_everywhere)
		self.effect_discard_button = QtWidgets.QPushButton("Discard curve edits")
		self.effect_discard_button.setToolTip(
			"Reload the selected curve and clear its undo/redo history.")
		self.effect_discard_button.clicked.connect(self.discard_effect_curve_edits)
		usage_button = QtWidgets.QPushButton("Where is it used?")
		usage_button.clicked.connect(self.show_effect_usage)
		action_row.addWidget(self.effect_add_button)
		action_row.addWidget(self.effect_set_button)
		action_row.addWidget(self.effect_all_button)
		action_row.addWidget(self.effect_discard_button)
		action_row.addWidget(usage_button)
		layout.addLayout(action_row)

		note = QtWidgets.QLabel(
			"Adding needs room in the last string pool for the new name; if that pool is full "
			"you will be told so. Adding a name the archive ALREADY holds reuses it and needs "
			"no such room. Writing a curve never allocates a name, and never edits the borrowed "
			"array in place - curves are shared, so it always allocates a fresh one."
		)
		note.setWordWrap(True)
		layout.addWidget(note)
		self.update_curve_readout()
		self.tabs.addTab(page, "Effects")

	def _mark_effects_dirty(self):
		self.effects_dirty = True

	def closeEvent(self, event):
		"""Warn about curve edits that were never written to the staged family.

		The base class guards `file_widget.dirty`, but this tool never uses
		file_widget - it has its own source/stage paths - so nothing was guarding
		the Effects tab and an edited curve could be lost silently on quit.
		"""
		if getattr(self, "effects_dirty", False):
			if not self.showconfirmation(
					"Quit? The curve in the Effects tab has not been written to a "
					"staged family, and those edits will be lost.", title="Quit"):
				event.ignore()
				return
		super().closeEvent(event)

	def update_curve_readout(self):
		points = self.curve_editor.points
		text = "  ".join("(%.2f, %g)" % (x, y) for x, y in points)
		if points and points[-1][1] != 0.0:
			text += "   -  ends ON: correct for a burst, wrong for a continuous emitter"
		self.curve_readout.setText(text)

	def scan_effect_activities(self):
		try:
			records = list_activity_records(self.source_path(), DEFAULT_GAME)
			self.effect_activity.clear()
			counts = {}
			for record in records:
				counts[record["activity"]] = counts.get(record["activity"], 0) + 1
			for record in records:
				label = record["activity"]
				if counts[label] > 1:
					label += "  [%d:%d]" % (record["activity_pool"], record["activity_offset"])
				self.effect_activity.addItem(label, (record["activity_pool"], record["activity_offset"], record["activity"]))
			self.status_bar.showMessage("%d activities" % len(records), 8000)
		except Exception as error:
			self.showerror(str(error))

	def show_effect_streams(self):
		try:
			identity = self.current_effect_activity()
			described = describe_activity(
				self.source_path(), identity[2], DEFAULT_GAME,
				activity_address=identity[:2])
			self.effect_list.clear()
			self.effect_list.setHeaderLabels(["Datastream", "Type", "curve_type"])
			for stream in described["streams"]:
				item = QtWidgets.QTreeWidgetItem(
					self.effect_list, [stream["ds_name"], stream["type"],
									   str(stream.get("curve_type", ""))])
				item.setData(0, QtCore.Qt.UserRole, "stream")
				item.setData(0, QtCore.Qt.UserRole + 1, identity)
				item.setData(0, QtCore.Qt.UserRole + 2, stream["ds_name"])
			held = described.get("held")
			note = (" - HELD activity (flags=%d): a VFXEnable here NEVER drops, use VFXToggle"
					% described.get("animation_flags", 0)) if held else " - one-shot activity"
			self.effect_held.setText(
				("<b style='color:#C6402A'>HELD</b> (flags=%d) - an effect here must be "
				 "<b>VFXToggle</b> or it will never switch off"
				 % described.get("animation_flags", 0)) if held else
				"one-shot activity (flags=%d)" % described.get("animation_flags", 0))
			self.status_bar.showMessage(
				"%s fires %d datastreams%s" % (described["activity"], described["count"], note),
				15000)
		except Exception as error:
			self.showerror(str(error))

	def current_effect_activity(self):
		data = self.effect_activity.currentData()
		if isinstance(data, (tuple, list)) and len(data) == 3:
			return int(data[0]), int(data[1]), str(data[2])
		text = self.effect_activity.currentText().strip()
		return None, None, text

	def load_effect_curve(self):
		items = self.effect_list.selectedItems()
		if not items:
			return
		item = items[0]
		kind = item.data(0, QtCore.Qt.UserRole) or "stream"
		identity = item.data(0, QtCore.Qt.UserRole + 1) or self.current_effect_activity()
		name = item.data(0, QtCore.Qt.UserRole + 2) or item.text(0)
		requested_target = (identity, name)
		if self.effects_dirty:
			if self.effect_loaded_target != requested_target:
				# Selection has already moved by the time this signal runs. Put it
				# back on the curve whose dirty points are still displayed.
				self.effect_list.blockSignals(True)
				try:
					self.effect_list.clearSelection()
					iterator = QtWidgets.QTreeWidgetItemIterator(self.effect_list)
					while iterator.value():
						candidate = iterator.value()
						candidate_identity = (
							candidate.data(0, QtCore.Qt.UserRole + 1)
							or self.current_effect_activity())
						candidate_name = (
							candidate.data(0, QtCore.Qt.UserRole + 2)
							or candidate.text(0))
						if (candidate_identity, candidate_name) == self.effect_loaded_target:
							candidate.setSelected(True)
							self.effect_list.setCurrentItem(candidate)
							break
						iterator += 1
				finally:
					self.effect_list.blockSignals(False)
				self.showerror(
					"This curve has unsaved edits. Write it or click Discard curve edits "
					"before selecting another datastream.")
			return
		if kind == "usage":
			for index in range(self.effect_activity.count()):
				if self.effect_activity.itemData(index) == identity:
					self.effect_activity.setCurrentIndex(index)
					break
		self.effect_name.setText(name)
		self.effect_type.setEditText(item.text(1))
		self.effect_loaded_target = None
		self.audio_scan_identity = None
		self.effect_set_button.setEnabled(False)
		try:
			points = read_data_stream_curve(
				self.source_path(), identity[2], name, DEFAULT_GAME,
				activity_address=identity[:2] if identity[0] is not None else None)
			if points:
				self.curve_editor.set_points(points, record=False)
				self.curve_editor._undo.clear()
				self.curve_editor._redo.clear()
				self.effects_dirty = False
				self.effect_loaded_target = (identity, name)
				self.effect_set_button.setEnabled(True)
		except Exception as error:
			self.curve_editor.set_points([], record=False)
			self.curve_editor._undo.clear()
			self.curve_editor._redo.clear()
			self.effects_dirty = False
			logging.warning("Could not read curve for %s: %s" % (name, error))

	def discard_effect_curve_edits(self):
		"""Explicitly discard the displayed draft before navigation is allowed."""
		if not self.effects_dirty:
			return
		self.effects_dirty = False
		self.curve_editor._undo.clear()
		self.curve_editor._redo.clear()
		MainWindow.load_effect_curve(self)

	def apply_effect_growth(self):
		try:
			identity = self.current_effect_activity()
			report = grow_data_stream(
				self.source_path(), self.output_path(),
				identity[2],
				self.effect_name.text().strip(),
				self.effect_type.currentText().strip(),
				curve_points=list(self.curve_editor.points),
				curve_type=curve_type_for(self.effect_type.currentText().strip()),
				game=DEFAULT_GAME,
				activity_address=identity[:2] if identity[0] is not None else None)

			message = ("Added %s to %s: %d -> %d datastreams, +%d fragments"
					   % (report.ds_name, report.activity, report.old_count,
						  report.new_count, report.fragments_added))
			self.effects_dirty = False
			self.status_bar.showMessage(message, 15000)
			logging.info(message)
		except Exception as error:
			self.showerror(str(error))

	def show_effect_usage(self):
		try:
			rows = activities_with_stream(self.source_path(),
										  self.effect_name.text().strip(), DEFAULT_GAME)
			self.effect_list.clear()
			self.effect_list.setHeaderLabels(["Activity", "Type", "curve_type / held"])
			for row in rows:
				item = QtWidgets.QTreeWidgetItem(
					self.effect_list,
					[row["activity"], row["type"],
					 "%d  %s" % (row["curve_type"], "HELD" if row["held"] else "one-shot")])
				identity = (row["activity_pool"], row["activity_offset"], row["activity"])
				item.setData(0, QtCore.Qt.UserRole, "usage")
				item.setData(0, QtCore.Qt.UserRole + 1, identity)
				item.setData(0, QtCore.Qt.UserRole + 2, self.effect_name.text().strip())
				if row["held"] and row["type"] != "VFXToggle":
					item.setForeground(2, QtGui.QBrush(QtGui.QColor("#C6402A")))
			bad = [r for r in rows if r["held"] and r["type"] != "VFXToggle"]
			message = "%d activities fire %s" % (len(rows), self.effect_name.text().strip())
			if bad:
				message += " - %d are HELD but not VFXToggle and will never switch off" % len(bad)
			self.status_bar.showMessage(message, 20000)
		except Exception as error:
			self.showerror(str(error))

	def apply_effect_curve_everywhere(self):
		try:
			ds_type = self.effect_type.currentText().strip()
			applied = set_curve_everywhere(
				self.source_path(), self.output_path(),
				self.effect_name.text().strip(), list(self.curve_editor.points),
				DEFAULT_GAME, ds_type=ds_type, curve_type=curve_type_for(ds_type))
			message = "Applied to %d activities as %s: %s" % (
				len(applied), ds_type, ", ".join(a["activity"].split("$")[-1] for a in applied))
			self.effects_dirty = False
			self.status_bar.showMessage(message, 20000)
			logging.info(message)
		except Exception as error:
			self.showerror(str(error))

	def apply_effect_curve(self):
		try:
			if self.effect_loaded_target is None:
				raise ValueError("Select and successfully load a datastream curve before writing")
			identity, loaded_name = self.effect_loaded_target
			if loaded_name != self.effect_name.text().strip():
				raise ValueError("The datastream name differs from the loaded curve target")
			ds_type = self.effect_type.currentText().strip()
			result = set_data_stream_curve(
				self.source_path(), self.output_path(), identity[2],
				self.effect_name.text().strip(),
				list(self.curve_editor.points), DEFAULT_GAME,
				ds_type=ds_type, curve_type=curve_type_for(ds_type),
				activity_address=identity[:2])
			message = ("Wrote a %d-point curve onto %s / %s as %s"
					   % (len(result["points"]), result["activity"],
						  result["ds_name"], result["type"]))
			self.effects_dirty = False
			self.status_bar.showMessage(message, 15000)
			logging.info(message)
		except Exception as error:
			self.showerror(str(error))

	def _build_stage_tab(self):
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)
		stage_row = QtWidgets.QHBoxLayout()
		self.stage_edit = QtWidgets.QLineEdit()
		self.stage_edit.setPlaceholderText("Separate directory for a complete copied OVL family")
		browse = QtWidgets.QPushButton("Browse…")
		browse.clicked.connect(self.browse_stage)
		copy_button = QtWidgets.QPushButton("Copy full family")
		copy_button.clicked.connect(self.copy_family)
		stage_row.addWidget(QtWidgets.QLabel("Stage directory"))
		stage_row.addWidget(self.stage_edit, 1)
		stage_row.addWidget(browse)
		stage_row.addWidget(copy_button)
		layout.addLayout(stage_row)
		self.stage_files = QtWidgets.QPlainTextEdit()
		self.stage_files.setReadOnly(True)
		layout.addWidget(self.stage_files, 1)
		self.apply_plan_button = QtWidgets.QPushButton("Apply queued patch plan to staged family")
		self.apply_plan_button.clicked.connect(self.apply_plan)
		layout.addWidget(self.apply_plan_button)
		self.continue_button = QtWidgets.QPushButton("Continue from staged result")
		self.continue_button.setToolTip(
			"Keep the staged result, copy it into a new editing step, and reload. "
			"Clears the old patch queue and selections; apply desired edits first.")
		self.continue_button.clicked.connect(self.continue_from_staged)
		layout.addWidget(self.continue_button)
		warning = QtWidgets.QLabel(
			"Apply writes the complete queued edit from the current source. To add another operation "
			"without losing that result, click Continue from staged result first. This preserves the "
			"previous files and reloads a fresh editing step, clearing old queued edits and selections. "
			"This editor does not install into the live game."
		)
		warning.setWordWrap(True)
		layout.addWidget(warning)
		self.tabs.addTab(page, "Stage / Apply")

	def census_guard(self, label: str, expect_added: int | None = None,
					 expect_removed: int = 0) -> dict | None:
		"""Census the staged output against the source after a write.

		This is a GUARD, not a report. Cloning can take over every inbound
		reference of its donor, leaving the donor unreachable: it is never decoded
		again and its bytes are absorbed into the preceding allocation. Raw AND
		semantic reload both pass on that build, so a census diff is the only
		detector. Anything unexpected is surfaced loudly rather than logged
		quietly, because the failure is silent everywhere else.

		Returns the diff, or None if it could not be taken - a guard that cannot
		run must say so rather than imply a pass.
		"""
		try:
			name = self.name_edit.text().strip() or None
			_ovl_a, loader_a = load_motiongraph(self.source_path(), name, DEFAULT_GAME)
			_ovl_b, loader_b = load_motiongraph(
				self.output_path(allow_existing_result=True), name, DEFAULT_GAME)
			delta = diff_census(census(loader_a), census(loader_b))
		except Exception as exc:
			logging.warning(f"census guard could not run after {label}: {exc}")
			self.status_bar.showMessage(
				f"{label}: census guard COULD NOT RUN ({exc}) - verify manually", 15000)
			return None
		added, removed = delta["added"], delta["removed"]
		logging.info(
			f"census guard after {label}: total {delta['total_before']} -> "
			f"{delta['total_after']} (added={added} removed={removed})")
		for type_name, row in delta["changed_types"].items():
			logging.info(
				f"    {type_name}: {row['before']} -> {row['after']} "
				f"(+{len(row['added'])} / -{len(row['removed'])})")
		problems = []
		if removed != expect_removed:
			problems.append(
				f"{removed} decoded object(s) DISAPPEARED (expected {expect_removed}). "
				"That is the orphaned-donor signature: reload will still pass.")
		if expect_added is not None and added != expect_added:
			problems.append(f"added {added} object(s), expected {expect_added}")
		if problems:
			self.showerror(
				f"Census guard failed after {label}:\n\n" + "\n\n".join(problems)
				+ "\n\nThe staged family is suspect. Do not install it."
			)
		else:
			self.status_bar.showMessage(
				f"{label}: census guard OK (+{added} / -{removed})", 10000)
		return delta

	def requested_source_path(self) -> Path:
		path = Path(self.source_edit.text().strip())
		if not path.is_file():
			raise ValueError("Choose an existing source OVL")
		return path.resolve()

	def source_path(self) -> Path:
		requested = self.requested_source_path()
		if self.loaded_identity is None:
			raise ValueError("Load and analyse the selected source first")
		loaded, graph, expected_hash, _generation = self.loaded_identity
		if requested != loaded or self.name_edit.text().strip().lower() != graph.lower():
			raise ValueError("The requested source differs from the loaded analysis; click Load")
		actual_hash = hashlib.sha256(loaded.read_bytes()).hexdigest()
		if actual_hash != expected_hash:
			raise ValueError("The loaded source changed outside the editor; reload before editing")
		return loaded

	def output_path(self, allow_existing_result=False) -> Path:
		stage = Path(self.stage_edit.text().strip())
		if not stage.is_dir():
			raise ValueError("Copy the full family to a stage directory first")
		output = stage / self.source_path().name
		if not output.is_file():
			raise ValueError(f"Staged main OVL is missing: {output}")
		if not allow_existing_result and self.loaded_identity is not None:
			if hashlib.sha256(output.read_bytes()).hexdigest() != self.loaded_identity[2]:
				raise ValueError(
					"The stage already contains a result newer than the loaded source; "
					"use Continue from staged result before another operation"
				)
		return output

	def browse_source(self):
		path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open donor OVL", "", "OVL files (*.ovl)")
		if path:
			self.source_edit.setText(path)

	def browse_stage(self):
		path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose stage directory")
		if path:
			self.stage_edit.setText(path)

	def load_source(self):
		if self.plan or getattr(self, "effects_dirty", False):
			self.showerror("Apply/save and clear queued edits, or discard the dirty Effects draft, before loading another source")
			return
		try:
			source = self.requested_source_path()
		except Exception as exc:
			self.showerror(str(exc))
			return
		self.load_button.setEnabled(False)
		self.status_bar.showMessage("Loading and analysing motiongraph…")
		logging.info(f"Loading and analysing motiongraph family: {source}")
		self.editor_generation += 1
		generation = self.editor_generation
		worker = self.run_background_task(
			_load_reports_identity, self.loaded_source, source,
			self.name_edit.text().strip() or None, generation
		)
		worker.signals.finished.connect(lambda: self.load_button.setEnabled(True))

	def loaded_source(self, result):
		generation, source, graph, source_hash, reports = result
		if generation != self.editor_generation:
			logging.info("Ignored a completed load from editor generation %d", generation)
			return
		self.invalidate_source_dependent_state()
		(self.ovl, self.loader, _state_text, self.state_rows, state_stats,
		 decision_text, decision_stats, self.decision_graph_data) = reports
		self.loaded_identity = (Path(source).resolve(), graph, source_hash, generation)
		self.source_edit.setText(str(source))
		self.name_edit.setText(self.loader.name)
		self.loaded_label.setText(
			f"{state_stats.states} states / {decision_stats.decision_nodes} decision nodes"
		)
		self.decision_text.setPlainText(decision_text)
		self.populate_graph()
		self.populate_decision_graph()
		self.state_table.setRowCount(0)
		for state in (row for row in self.state_rows if row):
			row_index = self.state_table.rowCount()
			self.state_table.insertRow(row_index)
			values = [
				state["index"], state["label"], state["activity_nodes"],
				", ".join(state["clips"]), sum(len(edge["targets"]) for edge in state["edges"]),
			]
			for column, value in enumerate(values):
				self.state_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
		self.state_table.resizeColumnsToContents()
		self.status_bar.showMessage(f"Loaded {self.loader.name}", 6000)
		logging.info(
			f"Loaded {self.loader.name}: {state_stats.states} states, "
			f"{decision_stats.decision_nodes} decision nodes"
		)

	def invalidate_source_dependent_state(self):
		"""Clear every address-bearing result when a new source is accepted."""
		self.field_rows = []
		if hasattr(self, "field_table"):
			self.field_table.setRowCount(0)
		self.activity_address_filter = None
		self.activity_tree_state = None
		self.clone_redirect_source = None
		self.scan_generation = None
		self.effect_loaded_target = None
		self.effect_draft = None
		self.stage_has_result = False
		self.audio_scan_identity = None
		self.activity_node_items = {}
		for name in ("activity_tree", "chooser_tree", "decision_tree", "audio_list", "effect_list"):
			widget = getattr(self, name, None)
			if widget is not None:
				widget.clear()
		if hasattr(self, "effect_activity"):
			self.effect_activity.clear()
		self.effect_loaded_target = None

	def populate_graph(self):
		self.graph_scene.clear()
		self.graph_nodes = {}
		self.graph_edges = []
		self.graph_edge_records = []
		self.graph_grid_positions = {}
		self.graph_focus_state = None
		valid = [state for state in self.state_rows if state]
		columns, x_step, y_step = 12, 225, 105
		for order, state in enumerate(valid):
			position = ((order % columns) * x_step, (order // columns) * y_step)
			self.graph_grid_positions[state["index"]] = position
			node = GraphNode(state, position, self.focus_state_neighborhood)
			self.graph_scene.addItem(node)
			self.graph_nodes[state["index"]] = node
		pen = QtGui.QPen(QtGui.QColor(111, 168, 220, 90), 1.1)
		pen.setCosmetic(True)
		arrow_brush = QtGui.QBrush(QtGui.QColor(111, 168, 220, 130))
		for state in valid:
			source_node = self.graph_nodes[state["index"]]
			for edge in state["edges"]:
				for target_index in edge["targets"]:
					target_node = self.graph_nodes.get(target_index)
					if target_node is None or target_node is source_node:
						continue
					start = source_node.sceneBoundingRect().center()
					end = target_node.sceneBoundingRect().center()
					line = QtCore.QLineF(start, end)
					if not line.length():
						continue
					# Stop at the approximate target-node boundary, then add an arrowhead.
					trim = min(92.0, line.length() * 0.35)
					unit_x, unit_y = line.dx() / line.length(), line.dy() / line.length()
					end = QtCore.QPointF(end.x() - unit_x * trim, end.y() - unit_y * trim)
					line.setP2(end)
					line_item = self.graph_scene.addLine(line, pen)
					line_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
					line_item.setZValue(-2)
					angle = math.atan2(line.dy(), line.dx())
					left = end - QtCore.QPointF(
						math.cos(angle - 0.5) * 9, math.sin(angle - 0.5) * 9
					)
					right = end - QtCore.QPointF(
						math.cos(angle + 0.5) * 9, math.sin(angle + 0.5) * 9
					)
					arrow = self.graph_scene.addPolygon(
						QtGui.QPolygonF([end, left, right]), QtGui.QPen(QtCore.Qt.NoPen), arrow_brush
					)
					arrow.setAcceptedMouseButtons(QtCore.Qt.NoButton)
					arrow.setZValue(-1)
					self.graph_edges.extend((line_item, arrow))
					self.graph_edge_records.append(
						(state["index"], target_index, line_item, arrow)
					)
		self.graph_view.set_content_rect(
			self.graph_scene.itemsBoundingRect().adjusted(-40, -40, 40, 40)
		)
		self.fit_graph()

	@staticmethod
	def position_graph_edge(source_node, target_node, line_item, arrow, trim=92.0):
		"""Reposition one straight directed edge after a graph layout change."""
		start = source_node.sceneBoundingRect().center()
		end = target_node.sceneBoundingRect().center()
		line = QtCore.QLineF(start, end)
		if not line.length():
			line_item.setLine(line)
			arrow.setPolygon(QtGui.QPolygonF())
			return
		cut = min(trim, line.length() * 0.35)
		unit_x, unit_y = line.dx() / line.length(), line.dy() / line.length()
		end = QtCore.QPointF(end.x() - unit_x * cut, end.y() - unit_y * cut)
		line.setP2(end)
		line_item.setLine(line)
		angle = math.atan2(line.dy(), line.dx())
		left = end - QtCore.QPointF(
			math.cos(angle - 0.5) * 9, math.sin(angle - 0.5) * 9
		)
		right = end - QtCore.QPointF(
			math.cos(angle + 0.5) * 9, math.sin(angle + 0.5) * 9
		)
		arrow.setPolygon(QtGui.QPolygonF([end, left, right]))

	def update_state_graph_edges(self):
		for source, target, line, arrow in self.graph_edge_records:
			self.position_graph_edge(
				self.graph_nodes[source], self.graph_nodes[target], line, arrow
			)

	def apply_state_graph_layout(self):
		mode = self.graph_layout_mode.currentData()
		if mode == "hierarchical" and self.graph_focus_state is not None:
			visible = {
				index for index, item in self.graph_nodes.items() if item.isVisible()
			}
			positions = hierarchical_neighborhood_positions(
				self.graph_focus_state,
				visible,
				[(source, target) for source, target, *_ in self.graph_edge_records],
			)
		else:
			positions = self.graph_grid_positions
		for index, position in positions.items():
			self.graph_nodes[index].setPos(*position)
		self.update_state_graph_edges()

	def relayout_state_graph(self, *_args):
		if not self.graph_nodes:
			return
		self.apply_state_graph_layout()
		self.graph_view.set_content_rect(self.visible_node_bounds(self.graph_nodes.values()))
		self.fit_graph()

	def refocus_state_graph(self, *_args):
		if self.graph_focus_state in self.graph_nodes:
			self.focus_state_neighborhood(self.graph_focus_state)

	def populate_decision_graph(self):
		self.decision_graph_scene.clear()
		self.decision_graph_nodes = {}
		self.decision_graph_edges = []
		self.decision_edge_records = []
		data = self.decision_graph_data
		root_ids = {root["node"] for root in data["roots"]}
		columns, x_step, y_step = 10, 270, 120
		for order, node in enumerate(data["nodes"]):
			item = DecisionNode(
				node, ((order % columns) * x_step, (order // columns) * y_step),
				self.focus_decision_neighborhood,
			)
			if node["index"] in root_ids:
				item.setBrush(QtGui.QBrush(QtGui.QColor("#263b38")))
			item.setToolTip(
				(node["opcode"] or "?") + "\n" + "\n".join(node["fields"])
			)
			self.decision_graph_scene.addItem(item)
			self.decision_graph_nodes[node["index"]] = item
		pen = QtGui.QPen(QtGui.QColor(196, 145, 255, 105), 1.1)
		pen.setCosmetic(True)
		brush = QtGui.QBrush(QtGui.QColor(196, 145, 255, 150))
		for edge in data["edges"]:
			source = self.decision_graph_nodes[edge["source"]]
			target = self.decision_graph_nodes[edge["target"]]
			start = source.sceneBoundingRect().center()
			end = target.sceneBoundingRect().center()
			line = QtCore.QLineF(start, end)
			if not line.length():
				continue
			trim = min(108.0, line.length() * 0.35)
			unit_x, unit_y = line.dx() / line.length(), line.dy() / line.length()
			end = QtCore.QPointF(end.x() - unit_x * trim, end.y() - unit_y * trim)
			line.setP2(end)
			line_item = self.decision_graph_scene.addLine(line, pen)
			line_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
			line_item.setZValue(-2)
			angle = math.atan2(line.dy(), line.dx())
			left = end - QtCore.QPointF(math.cos(angle - 0.5) * 9, math.sin(angle - 0.5) * 9)
			right = end - QtCore.QPointF(math.cos(angle + 0.5) * 9, math.sin(angle + 0.5) * 9)
			arrow = self.decision_graph_scene.addPolygon(
				QtGui.QPolygonF([end, left, right]), QtGui.QPen(QtCore.Qt.NoPen), brush
			)
			arrow.setAcceptedMouseButtons(QtCore.Qt.NoButton)
			arrow.setZValue(-1)
			self.decision_graph_edges.extend((line_item, arrow))
			self.decision_edge_records.append(
				(edge["source"], edge["target"], line_item, arrow)
			)
		self.decision_graph_view.set_content_rect(
			self.decision_graph_scene.itemsBoundingRect().adjusted(-40, -40, 40, 40)
		)
		self.fit_decision_graph()

	def fit_decision_graph(self):
		self.decision_graph_view.fit_content()

	@staticmethod
	def graph_neighborhood(center, edge_records, hops):
		adjacent = {}
		for source, target, *_ in edge_records:
			adjacent.setdefault(source, set()).add(target)
			adjacent.setdefault(target, set()).add(source)
		visible, frontier = {center}, {center}
		for _ in range(hops):
			frontier = set().union(*(adjacent.get(node, set()) for node in frontier)) - visible
			visible.update(frontier)
		return visible

	@staticmethod
	def visible_node_bounds(items):
		rect = QtCore.QRectF()
		for item in items:
			if item.isVisible():
				rect = rect.united(item.sceneBoundingRect())
		return rect.adjusted(-40, -40, 40, 40)

	def focus_selected_decision(self):
		items = [item for item in self.decision_graph_scene.selectedItems()
		         if isinstance(item, DecisionNode)]
		if not items:
			self.showerror("Select a decision node first")
			return
		self.focus_decision_neighborhood(items[0].node["index"])

	def focus_decision_neighborhood(self, node_index):
		visible = self.graph_neighborhood(
			node_index, self.decision_edge_records, self.decision_graph_hops.value()
		)
		for index, item in self.decision_graph_nodes.items():
			item.setVisible(index in visible)
		for source, target, line, arrow in self.decision_edge_records:
			shown = source in visible and target in visible
			line.setVisible(shown)
			arrow.setVisible(shown)
		center = self.decision_graph_nodes[node_index]
		self.decision_graph_scene.clearSelection()
		center.setSelected(True)
		opcode = (center.node["opcode"] or "?").split(".")[-1]
		self.decision_graph_breadcrumb.setText(
			f"All decisions  ›  #{node_index} {opcode}  ›  {self.decision_graph_hops.value()} hop(s)"
		)
		self.decision_graph_view.set_content_rect(
			self.visible_node_bounds(self.decision_graph_nodes.values())
		)
		self.fit_decision_graph()

	def show_all_decision_graph(self):
		for item in self.decision_graph_nodes.values():
			item.setVisible(True)
		for _, _, line, arrow in self.decision_edge_records:
			line.setVisible(True)
			arrow.setVisible(True)
		self.decision_graph_breadcrumb.setText("All decision roots")
		self.decision_graph_view.set_content_rect(
			self.visible_node_bounds(self.decision_graph_nodes.values())
		)
		self.fit_decision_graph()

	def find_decision_node(self):
		query = self.decision_graph_search.text().strip().lower()
		if not query:
			return
		matches = []
		for _, item in sorted(self.decision_graph_nodes.items()):
			node = item.node
			haystack = " ".join([
				node["opcode"] or "", *node["fields"], node["target_label"] or "",
				str(node["target_state"] if node["target_state"] is not None else ""),
			]).lower()
			if query in haystack:
				matches.append(item)
		if not matches:
			self.status_bar.showMessage(f"No decision nodes match {query!r}", 5000)
			return
		self.decision_match_index = (self.decision_match_index + 1) % len(matches)
		item = matches[self.decision_match_index]
		self.focus_decision_neighborhood(item.node["index"])

	def decision_graph_selection_changed(self):
		items = [
			item for item in self.decision_graph_scene.selectedItems()
			if isinstance(item, DecisionNode)
		]
		if not items:
			return
		node = items[0].node
		lines = [f"Decision node #{node['index']}", f"Opcode: {node['opcode'] or '?'}"]
		lines.extend(f"Field: {field}" for field in node["fields"])
		if node["target_state"] is not None:
			lines.append(f"Output: STATE[{node['target_state']}] ≈ {node['target_label'] or '?'}")
		for edge in self.decision_graph_data["edges"]:
			if edge["source"] == node["index"]:
				lines.append(
					f"result {edge['result']} → node #{edge['target']}"
					+ (f" [{edge['parameters']}]" if edge["parameters"] else "")
				)
		self.decision_node_details.setPlainText("\n".join(lines))

	def fit_graph(self):
		self.graph_view.fit_content()

	def focus_selected_state(self):
		items = [item for item in self.graph_scene.selectedItems() if isinstance(item, GraphNode)]
		if not items:
			self.showerror("Select a state node first")
			return
		self.focus_state_neighborhood(items[0].state["index"])

	def focus_state_neighborhood(self, state_index):
		visible = self.graph_neighborhood(state_index, self.graph_edge_records, self.graph_hops.value())
		for index, item in self.graph_nodes.items():
			item.setVisible(index in visible)
		for source, target, line, arrow in self.graph_edge_records:
			shown = self.edges_toggle.isChecked() and source in visible and target in visible
			line.setVisible(shown)
			arrow.setVisible(shown)
		center = self.graph_nodes[state_index]
		self.graph_focus_state = state_index
		self.apply_state_graph_layout()
		self.graph_scene.clearSelection()
		center.setSelected(True)
		self.graph_breadcrumb.setText(
			f"All states  ›  STATE[{state_index}] ≈ {center.state['label']}  ›  "
			f"{self.graph_hops.value()} hop(s)"
		)
		self.graph_view.set_content_rect(self.visible_node_bounds(self.graph_nodes.values()))
		self.fit_graph()

	def show_all_state_graph(self):
		self.graph_focus_state = None
		self.graph_scene.clearSelection()
		for item in self.graph_nodes.values():
			item.setVisible(True)
		for _, _, line, arrow in self.graph_edge_records:
			line.setVisible(self.edges_toggle.isChecked())
			arrow.setVisible(self.edges_toggle.isChecked())
		self.graph_breadcrumb.setText("All states")
		self.apply_state_graph_layout()
		self.graph_view.set_content_rect(self.visible_node_bounds(self.graph_nodes.values()))
		self.fit_graph()

	def toggle_graph_edges(self, visible):
		for source, target, line, arrow in self.graph_edge_records:
			shown = visible and self.graph_nodes[source].isVisible() and self.graph_nodes[target].isVisible()
			line.setVisible(shown)
			arrow.setVisible(shown)

	def find_graph_node(self):
		query = self.graph_search.text().strip().lower()
		if not query:
			return
		matches = [node for _, node in sorted(self.graph_nodes.items()) if query in (
			node.state["label"] + " " + " ".join(node.state["clips"])
		).lower()]
		if not matches:
			self.status_bar.showMessage(f"No graph states match {query!r}", 5000)
			return
		self.graph_match_index = (self.graph_match_index + 1) % len(matches)
		node = matches[self.graph_match_index]
		self.focus_state_neighborhood(node.state["index"])

	def graph_selection_changed(self):
		nodes = [item for item in self.graph_scene.selectedItems() if isinstance(item, GraphNode)]
		if not nodes:
			return
		state_index = nodes[0].state["index"]
		self.render_state_details(state_index)
		for row in range(self.state_table.rowCount()):
			if int(self.state_table.item(row, 0).text()) == state_index:
				self.state_table.selectRow(row)
				break
		if (self.graph_layout_mode.currentData() == "hierarchical"
				and self.graph_focus_state != state_index):
			self.focus_state_neighborhood(state_index)

	def show_state_details(self):
		self.update_field_scope_label()
		selected = self.state_table.selectionModel().selectedRows()
		if not selected:
			return
		state_index = int(self.state_table.item(selected[0].row(), 0).text())
		self.render_state_details(state_index)

	def render_state_details(self, state_index):
		state = self.state_rows[state_index]
		lines = [f"STATE[{state_index}] {state['label']}", ""]
		lines.append("Composition: " + ", ".join(
			f"{name} x{count}" for name, count in sorted(state["types"].items())
		))
		lines.append("Clips: " + (", ".join(state["clips"]) or "(none)"))
		if state["streams"]:
			lines.append("Streams:")
			lines.extend(f"  {item['name']} ({item['type']})" for item in state["streams"])
		if state["inbound_decisions"]:
			lines.append("Inbound decisions:")
			lines.extend(f"  {name}: {count}" for name, count in state["inbound_decisions"].items())
		if state["edges"]:
			lines.append("Transitions:")
			for edge in state["edges"]:
				lines.append(f"  -> {edge['targets']}")
				for condition in edge["conditions"]:
					lines.append(
						f"     tier={condition['tier']} trigger={condition['trigger']!r} "
						f"curve={condition['curve_length']}"
					)
		self.state_details.setPlainText("\n".join(lines))
		self.populate_activity_tree(state_index)

	def populate_activity_tree(self, state_index):
		self.activity_tree.clear()
		self.activity_node_items = {}
		self.activity_tree_state = state_index
		if self.loader is None:
			return
		try:
			roots, count = build_activity_tree(self.loader, state_index)
		except Exception as exc:
			self.activity_tree.addTopLevelItem(QtWidgets.QTreeWidgetItem(["error", "", str(exc)]))
			return

		def add_node(parent, node):
			clips = [name.split("$", 1)[-1] for name in node["clips"]]
			label = node.get("label") or ""
			if clips:
				label += (" — " if label else "") + ", ".join(clips)
			item = QtWidgets.QTreeWidgetItem([
				node["relationship"], node["activity_type"], label,
			])
			node_number = node.get("node")
			item.setData(0, QtCore.Qt.UserRole, node_number)
			item.setData(1, QtCore.Qt.UserRole, node.get("target_node"))
			item.setData(2, QtCore.Qt.UserRole, node.get("label"))
			item.setData(2, QtCore.Qt.UserRole + 1, node.get("activity_type"))
			item.setData(2, QtCore.Qt.UserRole + 2, node.get("address"))
			item.setData(2, QtCore.Qt.UserRole + 3, node.get("inbound_source"))
			if node_number is not None:
				self.activity_node_items[node_number] = item
			if node.get("shared"):
				item.setForeground(0, QtGui.QBrush(QtGui.QColor("#9aa0a6")))
			if parent is None:
				self.activity_tree.addTopLevelItem(item)
			else:
				parent.addChild(item)
			for child in node["children"]:
				add_node(item, child)

		for root in roots:
			add_node(None, root)
		self.activity_tree.expandToDepth(1)
		self.status_bar.showMessage(
			f"STATE[{state_index}] contains {count} unique reachable activity nodes", 6000
		)

	def inspect_activity_fields(self):
		resolved = self.selected_activity()
		if resolved is None:
			return
		_activity_type = resolved["activity_type"]
		label = resolved["label"]
		address = resolved["address"]
		self.activity_address_filter = address
		self.field_scope.setCurrentIndex(self.field_scope.findData("activity"))
		self.activity_filter.setText(label)
		self.type_filter.setText(_activity_type)
		self.field_filter.clear()
		self.tabs.setCurrentWidget(self.fields_page)
		message = (
			f"Scanning exact {_activity_type} {label!r} at "
			f"pool {address[0]} offset {address[1]}"
		)
		self.status_bar.showMessage(message, 8000)
		logging.info(message)
		self.scan_fields()

	def selected_activity(self):
		"""Resolve the selected tree row to one exact editable Activity address."""
		item = self.activity_tree.currentItem()
		if item is None:
			self.showerror("Select an activity-tree node first")
			return None
		# Preserve the clicked edge before resolving a shared-reference display row
		# to the concrete activity. This identifies one exact state occurrence.
		inbound_source = item.data(2, QtCore.Qt.UserRole + 3)
		relationship = item.text(0)
		target = item.data(1, QtCore.Qt.UserRole)
		if target is not None:
			item = self.activity_node_items.get(target, item)
		activity_type = item.data(2, QtCore.Qt.UserRole + 1)
		if not activity_type or activity_type in ("shared reference", "limit"):
			self.showerror("The selected row is not an editable activity")
			return None
		label = item.data(2, QtCore.Qt.UserRole) or ""
		address = item.data(2, QtCore.Qt.UserRole + 2)
		if not address or len(address) != 2:
			self.showerror("Could not resolve the selected activity's exact STATIC address")
			return None
		if inbound_source and len(inbound_source) == 2:
			inbound_source = tuple(int(value) for value in inbound_source)
		else:
			inbound_source = None
		return {
			"activity_type": activity_type,
			"label": label,
			"address": tuple(int(value) for value in address),
			"inbound_source": inbound_source,
			"relationship": relationship,
			"state": self.activity_tree_state,
		}

	def select_activity_for_clone(self):
		resolved = self.selected_activity()
		if resolved is None:
			return
		activity_type = resolved["activity_type"]
		label = resolved["label"]
		address = resolved["address"]
		if activity_type != "AnimationActivity":
			self.showerror(
				f"Complete cloning currently supports AnimationActivity only, not {activity_type}"
			)
			return
		self.activity_address_filter = address
		self.clone_redirect_source = resolved["inbound_source"]
		usage = [
			row["index"] for row in self.state_rows
			if address in row.get("activity_addresses", ())
		]
		self.clone_selection_label.setText(
			f"{label or '(unnamed)'} — {activity_type} "
			f"(pool {address[0]}, offset {address[1]})"
		)
		self.clone_usage_label.setText(", ".join(str(index) for index in usage) or "None")
		# the Clip Retarget tab's single-activity control reads the same selection
		self.single_clip_label.setText(
			f"{label or '(unnamed)'} — {activity_type} "
			f"(pool {address[0]}, offset {address[1]})"
		)
		preferred_scope = "occurrence" if self.clone_redirect_source else "all"
		self.clone_redirect_scope.setCurrentIndex(
			self.clone_redirect_scope.findData(preferred_scope)
		)
		self.tabs.setCurrentWidget(self.clone_page)
		self.status_bar.showMessage("Exact activity selected for complete cloning", 6000)

	def selected_state_indices(self):
		return sorted({
			int(self.state_table.item(index.row(), 0).text())
			for index in self.state_table.selectionModel().selectedRows()
		})

	def activity_addresses_for_states(self, state_indices):
		addresses = set()
		for state_index in state_indices:
			roots, _count = build_activity_tree(self.loader, state_index)
			stack = list(roots)
			while stack:
				node = stack.pop()
				address = node.get("address")
				if address:
					addresses.add(tuple(int(value) for value in address))
				stack.extend(node.get("children", []))
		return sorted(addresses)

	def update_field_scope_label(self, _index=None):
		scope = self.field_scope.currentData()
		if scope == "activity":
			if self.activity_address_filter:
				pool, offset = self.activity_address_filter
				text = f"One activity at pool {pool}, offset {offset}"
			else:
				text = "No exact activity selected; use Inspect selected activity fields"
		elif scope == "states":
			states = self.selected_state_indices()
			text = f"Reachable activities in states {states}" if states else "No states selected"
		else:
			text = "All activity instances in the motiongraph"
		self.field_scope_label.setText(text)

	def open_activity_reference(self, item, _column):
		target = item.data(1, QtCore.Qt.UserRole)
		if target is None:
			item.setExpanded(not item.isExpanded())
			return
		destination = self.activity_node_items.get(target)
		if destination is None:
			return
		parent = destination.parent()
		while parent:
			parent.setExpanded(True)
			parent = parent.parent()
		self.activity_tree.setCurrentItem(destination)
		self.activity_tree.scrollToItem(destination)
		self.status_bar.showMessage(f"Jumped to shared activity node {target}", 4000)

	def find_activity(self):
		query = self.activity_search.text().strip().lower()
		if not query:
			return
		iterator = QtWidgets.QTreeWidgetItemIterator(self.activity_tree)
		matches = []
		while iterator.value():
			item = iterator.value()
			if query in " ".join(item.text(column) for column in range(3)).lower():
				matches.append(item)
			iterator += 1
		if not matches:
			self.status_bar.showMessage(f"No activities match {query!r}", 5000)
			return
		current = self.activity_tree.currentItem()
		try:
			index = (matches.index(current) + 1) % len(matches)
		except ValueError:
			index = 0
		item = matches[index]
		parent = item.parent()
		while parent:
			parent.setExpanded(True)
			parent = parent.parent()
		self.activity_tree.setCurrentItem(item)
		self.activity_tree.scrollToItem(item)

	def find_decision(self):
		query = self.decision_search.text()
		if query and not self.decision_text.find(query):
			cursor = self.decision_text.textCursor()
			cursor.movePosition(cursor.Start)
			self.decision_text.setTextCursor(cursor)
			self.decision_text.find(query)

	def scan_fields(self):
		if self.loader is None:
			self.showerror("Load a motiongraph first")
			return
		self.scan_button.setEnabled(False)
		self.status_bar.showMessage("Locating and verifying fixed-width fields…")
		scope = self.field_scope.currentData()
		activity_pool = activity_offset = None
		activity_addresses = None
		if scope == "activity":
			if not self.activity_address_filter:
				self.scan_button.setEnabled(True)
				self.showerror("No exact activity selected; inspect an activity first or change Scope")
				return
			activity_pool, activity_offset = self.activity_address_filter
			scope_message = f"exact activity at pool {activity_pool}, offset {activity_offset}"
		elif scope == "states":
			state_indices = self.selected_state_indices()
			if not state_indices:
				self.scan_button.setEnabled(True)
				self.showerror("Select one or more rows in the States tab first")
				return
			activity_addresses = self.activity_addresses_for_states(state_indices)
			scope_message = f"states {state_indices} ({len(activity_addresses)} unique activities)"
		else:
			scope_message = "entire motiongraph"
		logging.info(f"Locating and byte-verifying fields across {scope_message}")
		generation = self.editor_generation
		identity = self.loaded_identity
		worker = self.run_background_task(
			_scan_fields_identity, self.scanned_fields, generation, identity,
			self.source_path(), self.name_edit.text().strip(),
			DEFAULT_GAME, self.activity_filter.text().strip() or None,
			self.type_filter.text().strip() or None, self.field_filter.text().strip() or None,
			activity_pool, activity_offset, activity_addresses,
		)
		worker.signals.finished.connect(lambda: self.scan_button.setEnabled(True))

	def scanned_fields(self, result):
		generation, identity, scan_result = result
		if generation != self.editor_generation or identity != self.loaded_identity:
			logging.info("Ignored stale field scan from editor generation %d", generation)
			return
		rows, mismatches = scan_result
		self.field_rows = []
		self.field_table.setRowCount(0)
		for owner in rows:
			for field in owner["fields"]:
				self.field_rows.append((owner, field))
				row = self.field_table.rowCount()
				self.field_table.insertRow(row)
				value = field.get("curve_value", field["value"])
				values = [owner["activity"], owner["activity_type"], field["path"], field["kind"],
				          value, field["pool"], field["offset"], field["verified"]]
				explanation = describe_motiongraph_field(field["path"], field["kind"])
				for column, item in enumerate(values):
					cell = QtWidgets.QTableWidgetItem(str(item))
					cell.setToolTip(explanation)
					self.field_table.setItem(row, column, cell)
		self.field_table.resizeColumnsToContents()
		message = f"Found {len(self.field_rows)} fields"
		if mismatches:
			message += f"; skipped {len(mismatches)} stride-mismatched layouts"
		self.status_bar.showMessage(message, 8000)
		logging.info(message)

	@staticmethod
	def operation_replacement(operation):
		if operation.get("flags") is not None:
			return ", ".join(operation["flags"])
		if operation.get("enum") is not None:
			return str(operation["enum"])
		return str(operation.get("value"))

	def refresh_patch_queue(self):
		operations = (self.plan or {}).get("operations") or []
		edits = (self.plan or {}).get("edits") or []
		self.queue_table.setRowCount(0)
		for operation in operations:
			row = self.queue_table.rowCount()
			self.queue_table.insertRow(row)
			values = [
				operation.get("field"), operation.get("kind"),
				self.operation_replacement(operation), operation.get("edit_count", "?"),
			]
			for column, value in enumerate(values):
				self.queue_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
		self.queue_table.resizeColumnsToContents()
		if edits:
			self.queue_label.setText(
				f"Patch queue: {len(operations)} operations / {len(edits)} unique addresses"
			)
		else:
			self.queue_label.setText("Patch queue: empty")
		enabled = bool(edits)
		self.save_queue_button.setEnabled(enabled)
		self.clear_queue_button.setEnabled(enabled)

	def save_queue(self):
		if not self.plan:
			self.showerror("Add at least one edit to the patch queue first")
			return
		path, _ = QtWidgets.QFileDialog.getSaveFileName(
			self, "Save patch queue", "motiongraph_patch.json", "JSON files (*.json)"
		)
		if not path:
			return
		try:
			save_plan(Path(path), self.plan)
			self.status_bar.showMessage(f"Saved patch queue to {path}", 8000)
			logging.info(f"Saved motiongraph patch queue: {path}")
		except Exception as exc:
			self.showerror(str(exc))

	def load_queue(self):
		path, _ = QtWidgets.QFileDialog.getOpenFileName(
			self, "Load patch queue", "", "JSON files (*.json)"
		)
		if not path:
			return
		if self.plan and QtWidgets.QMessageBox.question(
			self, "Replace patch queue", "Replace the current queued edits?",
		) != QtWidgets.QMessageBox.Yes:
			return
		try:
			self.plan = load_plan(Path(path))
			self.refresh_patch_queue()
			self.status_bar.showMessage(
				f"Loaded {len(self.plan['edits'])} queued addresses from {path}", 8000
			)
			logging.info(f"Loaded motiongraph patch queue: {path}")
		except Exception as exc:
			self.showerror(str(exc))

	def clear_queue(self):
		self.plan = None
		self.refresh_patch_queue()
		self.status_bar.showMessage("Cleared patch queue", 5000)
		logging.info("Cleared motiongraph patch queue")

	def create_plan(self):
		selected = sorted({index.row() for index in self.field_table.selectionModel().selectedRows()})
		if not selected:
			self.showerror("Select one or more matching field rows")
			return
		try:
			grouped = []
			for index in selected:
				owner, field = self.field_rows[index]
				grouped.append({**owner, "fields": [field]})
			field_paths = {field["path"] for _, field in (self.field_rows[index] for index in selected)}
			if len(field_paths) != 1:
				raise ValueError("Selected rows must have the same field path")
			text = self.value_edit.text().strip()
			mode = self.edit_mode.currentIndex()
			kwargs = {}
			if mode == 0:
				kwargs["value"] = float(text)
			elif mode == 1:
				kwargs["enum_name"] = text
			else:
				kwargs["flag_ops"] = [item.strip() for item in text.split(",") if item.strip()]
			addition = build_patch_plan(
				grouped, self.source_path(), self.name_edit.text().strip(), field_paths.pop(), **kwargs
			)
			before = len((self.plan or {}).get("edits") or [])
			self.plan = merge_patch_plans([self.plan, addition] if self.plan else [addition])
			added = len(self.plan["edits"]) - before
			self.refresh_patch_queue()
			self.status_bar.showMessage(
				f"Queued {added} new addresses; {len(self.plan['edits'])} total", 8000
			)
			logging.info(
				f"Queued motiongraph operation for {len(addition['edits'])} addresses; "
				f"{len(self.plan['edits'])} unique addresses total"
			)
		except Exception as exc:
			self.showerror(str(exc))

	def copy_family(self):
		try:
			outputs = copy_motiongraph_family(self.source_path(), Path(self.stage_edit.text().strip()))
			self.stage_files.setPlainText("\n".join(str(path) for path in outputs))
			self.status_bar.showMessage(f"Copied {len(outputs)} family files", 8000)
			logging.info(f"Copied {len(outputs)} archive-family files to staging: {outputs[0].parent}")
		except Exception as exc:
			self.showerror(str(exc))

	def continue_from_staged(self):
		if self.active_workers:
			self.showerror("Wait for the current operation to finish first")
			return
		if self.plan or getattr(self, "effects_dirty", False):
			self.showerror("Apply/save and clear queued edits, or discard the dirty Effects draft, before continuing")
			return
		try:
			result = self.output_path(allow_existing_result=True)
			name = self.name_edit.text().strip() or None
		except Exception as exc:
			self.showerror(str(exc))
			return
		self.body.setEnabled(False)
		self.status_bar.showMessage("Copying and checking the next editing step...")
		worker = self.run_background_task(prepare_next_stage, self.continued_stage, result, name)
		worker.signals.finished.connect(lambda: self.body.setEnabled(True))

	def continued_stage(self, result):
		source, stage, outputs, reports = result
		self.source_edit.setText(str(source))
		self.stage_edit.setText(str(stage))
		self.stage_files.setPlainText("\n".join(str(path) for path in outputs))
		# Old addresses and queue hashes belong to the previous source generation.
		self.plan = None
		self.refresh_patch_queue()
		self.field_rows = []
		self.field_table.setRowCount(0)
		self.activity_address_filter = None
		self.activity_tree_state = None
		self.clone_redirect_source = None
		self.scan_generation = None
		self.effect_loaded_target = None
		self.effect_draft = None
		self.stage_has_result = False
		self.audio_scan_identity = None
		self.activity_node_items = {}
		for tree in (self.activity_tree, self.chooser_tree, self.decision_tree, self.audio_list):
			tree.clear()
		self.effect_activity.clear()
		self.effect_list.clear()
		self.effect_name.clear()
		self.activity_filter.clear()
		self.type_filter.clear()
		self.field_filter.clear()
		self.effects_dirty = False
		self.editor_generation += 1
		generation = self.editor_generation
		source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
		self.loaded_source((generation, source, reports[1].name, source_hash, reports))
		self.status_bar.showMessage("Ready for the next edit. Previous result preserved; select and scan again.", 15000)
		logging.info(f"Continued editing from {source}; next output: {stage}")

	def apply_plan(self):
		if not self.plan:
			self.showerror("Add at least one edit to the patch queue first")
			return
		try:
			report = apply_patch_plan(self.source_path(), self.output_path(), self.plan, DEFAULT_GAME)
			self.status_bar.showMessage(
				f"Verified {report.edits} edits / {report.changed_bytes} changed bytes in staged family", 10000
			)
			logging.info(
				f"Applied and reloaded staged patch: {report.edits} edits, "
				f"{report.changed_bytes} changed bytes"
			)
			# Fixed-width value edits inside existing slots: the object set must be
			# untouched. A change here means the plan hit something structural.
			delta = self.census_guard("patch plan", expect_added=0, expect_removed=0)
			if delta is not None and not delta["added"] and not delta["removed"]:
				self.plan = None
				self.refresh_patch_queue()
		except Exception as exc:
			self.showerror(str(exc))

	def apply_complete_activity_clone(self):
		if not self.activity_address_filter:
			self.showerror(
				"Select an AnimationActivity in the States tab and click "
				"Clone selected complete activity first"
			)
			return
		try:
			pool, offset = self.activity_address_filter
			speed = self.clone_speed.value() if self.clone_speed_override.isChecked() else None
			redirect_sources = None
			if self.clone_redirect_scope.currentData() == "occurrence":
				if self.clone_redirect_source is None:
					raise ValueError(
						"The selected tree row has no exact inbound source; choose All inbound "
						"references or select the activity through a state occurrence"
					)
				redirect_sources = (self.clone_redirect_source,)
			report = clone_complete_animation_activity(
				self.source_path(), self.output_path(), pool, offset,
				name=self.name_edit.text().strip() or None,
				speed=speed, redirect_sources=redirect_sources, game=DEFAULT_GAME,
			)
			self.status_bar.showMessage(
				f"Cloned complete activity and redirected {report.inbound_references} references; "
				"staged archive reloaded successfully",
				12000,
			)
			logging.info(
				f"Complete activity clone verified: donor={report.donor_wrapper}, "
				f"clone={report.clone_wrapper}, data={report.clone_data}, "
				f"inbound={report.inbound_references}, fragments=+{report.cloned_fragments}, "
				f"sentinels={report.moved_end_sentinels}, speed={report.speed}"
			)
			# clone.py census-guards internally, but the GUI never surfaced the one
			# outcome a user cannot see any other way: taking over ALL of a donor's
			# inbound references leaves it unreachable, and both raw and semantic
			# reload still pass on that build.
			if getattr(report, "orphaned_donor", False):
				self.showerror(
					"Clone succeeded, but it took over EVERY inbound reference of its "
					f"donor at {report.donor_wrapper}.\n\n"
					"The donor is now unreachable: it will never be decoded again and "
					"its bytes are absorbed into the preceding allocation. Reload and "
					"validation both still pass, so nothing else will warn you.\n\n"
					"If you meant to keep the original behaviour on some edges, redo "
					"this with Redirect scope = 'Only selected state occurrence'."
				)
		except Exception as exc:
			self.showerror(str(exc))

	def apply_clip_retarget(self):
		try:
			report = repoint_existing_string(
				self.source_path(), self.output_path(), self.from_clip.text(), self.to_clip.text(),
				expected_count=self.expected_count.value(), game=DEFAULT_GAME,
			)
			self.status_bar.showMessage(
				f"Retargeted {report.references} references; staged family reloaded successfully", 10000
			)
			logging.info(
				f"Retargeted {report.references} references and reloaded the staged archive family"
			)
			# A pure repoint must not change the decoded object set at all.
			self.census_guard("clip retarget", expect_added=0, expect_removed=0)
		except Exception as exc:
			self.showerror(str(exc))

	def apply_string_slot(self):
		try:
			report = patch_string_slot(
				self.source_path(), self.output_path(), self.slot_from.text(), self.slot_to.text(),
				game=DEFAULT_GAME,
			)
			self.status_bar.showMessage(
				f"Replaced string slot ({report.changed_bytes} changed bytes); staged family verified", 10000
			)
			logging.info(
				f"Replaced string slot and verified staged family: {report.changed_bytes} changed bytes"
			)
			self.census_guard("string-slot replacement", expect_added=0, expect_removed=0)
		except Exception as exc:
			self.showerror(str(exc))

	# ---- Audio Events ---------------------------------------------------

	def browse_audio_game(self):
		dirpath = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose the game folder")
		if dirpath:
			self.audio_game.setText(dirpath)

	def set_audio_ticks(self, ticked):
		state = QtCore.Qt.Checked if ticked else QtCore.Qt.Unchecked
		for index in range(self.audio_list.topLevelItemCount()):
			item = self.audio_list.topLevelItem(index)
			if item.flags() & QtCore.Qt.ItemIsUserCheckable:
				item.setCheckState(0, state)

	def browse_audio_soundmod(self):
		path, _ = QtWidgets.QFileDialog.getOpenFileName(
			self, "Replacement sound mod's <Donor>_media.ovl", "", "OVL (*.ovl)")
		if path:
			self.audio_soundmod.setText(path)

	def browse_audio_out(self):
		path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder for the banks")
		if path:
			self.audio_out.setText(path)

	def browse_audio_sounds(self):
		path = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder containing custom WEM files")
		if path:
			self.audio_sounds.setText(path)

	def browse_audio_marker(self):
		path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Marker media OVL", "", "OVL (*.ovl)")
		if path:
			self.audio_marker.setText(path)

	def open_audio_output(self, filename):
		folder = self.audio_out.text().strip()
		path = Path(folder) / filename
		if not folder or not path.exists():
			self.showerror("Choose an output folder and complete a build first")
			return
		QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path.resolve())))

	def _audio_kit_dir(self):
		"""Where JWE3 Audio Kit lives, relative to this cobra checkout."""
		here = Path(__file__).resolve().parent
		for base in (here.parent.parent, here.parent, here):
			candidate = base / "JWE3 Audio Kit"
			if (candidate / "build_species_audio.py").is_file():
				return candidate
		return None

	def build_species_audio(self, _checked=False, baseline=False):
		"""Run the audio kit's builder for the donor/prefix already entered above.

		Driven as a subprocess rather than imported: the builder is a verified
		pipeline with an argparse entry point, and shelling out keeps it that way.
		"""
		if self.active_workers:
			self.showerror("Wait for the current operation to finish first")
			return
		try:
			kit = self._audio_kit_dir()
			if kit is None:
				raise ValueError(
					"Could not find 'JWE3 Audio Kit' beside this cobra-tools checkout")
			donor = self.audio_donor.text().strip()
			prefix = self.audio_prefix.text().strip()
			game = self.audio_game.text().strip()
			out = self.audio_out.text().strip()
			cmd = audio_build_command(kit, Path(__file__).resolve().parent, donor, prefix, game, out,
				self.audio_soundmod.text().strip(), self.audio_sounds.text().strip(),
				self.audio_marker.text().strip(), baseline)
			logging.info("Building audio banks: %s" % " ".join(cmd))
			self.body.setEnabled(False)
			self.status_bar.showMessage("Building audio banks; output will appear in the log when finished...")
			worker = self.run_background_task(run_audio_build, self.audio_build_finished, cmd, kit)
			worker.signals.finished.connect(lambda: self.body.setEnabled(True))
		except Exception as error:
			self.body.setEnabled(True)
			self.showerror(str(error))

	def audio_build_finished(self, result):
		out, names, log = result
		logging.info(log)
		message = f"Built {', '.join(names)} in {out}. Copy both banks AND .wmetasb.add into <Mod>/Audio/, then merge."
		self.status_bar.showMessage(message, 25000)
		logging.info(message)

	def audio_direction(self):
		return self.audio_direction_box.currentData()

	def audio_direction_changed(self):
		"""A direction change invalidates the preview - the list means something else now."""
		self.audio_scan_identity = None
		self.audio_list.clear()
		reverting = self.audio_direction() == AUDIO_REVERT
		self.audio_list.setHeaderLabels([
			"Event name in graph",
			"Will become",
			"Restore?" if reverting else "Rename?",
		])
		self.audio_apply.setText(
			"Revert ticked events to stock in staged family" if reverting
			else "Rename ticked events in staged family")
		# A fragment declares events a mod's OWN bank answers. Reverting ships no
		# bank, so there is nothing to declare.
		self.audio_fragment.setEnabled(not reverting)
		self.audio_fragment.setToolTip(
			"Not used when reverting - a stock-sound copy ships no bank and needs no "
			"registry fragment." if reverting else "")

	def scan_audio_events(self):
		try:
			donor = self.audio_donor.text().strip()
			prefix = self.audio_prefix.text().strip()
			reverting = self.audio_direction() == AUDIO_REVERT
			if not donor:
				raise ValueError("Enter the donor species, e.g. Indoraptor")
			if not prefix:
				raise ValueError("Enter your new prefix, e.g. Indocapi")
			game_root = Path(self.audio_game.text().strip())
			if not (game_root / "Win64" / "ovldata").is_dir():
				raise ValueError("Choose the game folder (the one containing Win64/ovldata)")

			# The graph currently speaks whichever prefix we are moving AWAY from.
			current = prefix if reverting else donor
			names = audio_events.scan_graph_event_names(
				self.source_path(), current, DEFAULT_GAME)
			if not names:
				raise ValueError(
					"No '%s_*' audio event names in this motiongraph%s"
					% (current, " - has it been renamed yet?" if reverting else ""))

			if reverting:
				# Verified against the STOCK registry, not the mod's bank: the bank
				# may be gone already, and shared-bank sounds are legitimate targets.
				stock_ids = audio_events.stock_event_ids(game_root, DEFAULT_GAME)
				rows = [(name, target, ok, "Yes - the stock game answers it" if ok
							else "No - no stock event of that name to go back to")
						for name, target, ok
						in audio_events.classify_revert(names, prefix, donor, stock_ids)]
			else:
				owned_ids = audio_events.donor_event_ids(game_root, donor, DEFAULT_GAME)
				rows = [(name, "%s_%s" % (prefix, name[len(donor) + 1:]), owned,
							"Yes - your bank answers it" if owned
							else "No - a SHARED bank owns this sound")
						for name, owned in audio_events.classify(names, donor, owned_ids)]

			self.audio_scan_identity = (self.loaded_identity, donor, prefix,
				str(game_root.resolve()), self.audio_direction())

			self.audio_list.clear()
			safe = 0
			for name, target, usable, why in rows:
				item = QtWidgets.QTreeWidgetItem(
					[name, target if usable else "stays as-is", why])
				if usable:
					item.setData(0, QtCore.Qt.UserRole, target)
					item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
					item.setCheckState(0, QtCore.Qt.Checked)
					safe += 1
				else:
					item.setFlags(item.flags() & ~QtCore.Qt.ItemIsUserCheckable)
					item.setDisabled(True)
				self.audio_list.addTopLevelItem(item)
			for column in range(3):
				self.audio_list.resizeColumnToContents(column)
			self.status_bar.showMessage(
				("%d audio events: %d can go back to stock, %d are this mod's own "
				 "invention and have no stock equivalent" if reverting
				 else "%d audio events: %d safe to rename, %d owned by shared banks")
				% (len(rows), safe, len(rows) - safe), 15000
			)
		except Exception as exc:
			self.showerror(str(exc))

	def ticked_audio_pairs(self):
		"""(current, target) for every ticked row, longest source first.

		The target is read off the item rather than recomputed, so one code path
		serves both directions and the applied pairs are exactly what the user was
		shown in the preview.
		"""
		pairs = []
		for index in range(self.audio_list.topLevelItemCount()):
			item = self.audio_list.topLevelItem(index)
			if not (item.flags() & QtCore.Qt.ItemIsUserCheckable):
				continue
			if item.checkState(0) != QtCore.Qt.Checked:
				continue
			target = item.data(0, QtCore.Qt.UserRole)
			if target:
				pairs.append((item.text(0), target))
		pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
		return pairs

	def apply_audio_rename(self):
		try:
			donor = self.audio_donor.text().strip()
			prefix = self.audio_prefix.text().strip()
			direction = self.audio_direction()
			game_root = Path(self.audio_game.text().strip()).resolve()
			if self.audio_scan_identity != (self.loaded_identity, donor, prefix,
					str(game_root), direction):
				raise ValueError("Audio preview is stale; scan again after changing source, donor, prefix, direction, or game folder")
			pairs = self.ticked_audio_pairs()
			if not pairs:
				raise ValueError("Scan first, then tick at least one event")

			report = apply_string_renames(
				self.source_path(), self.output_path(), pairs, game=DEFAULT_GAME)
			verb = "Reverted" if direction == AUDIO_REVERT else "Renamed"
			message = ("%s %d audio events in place and relocated %d; "
						   "%d fragments repointed; staged family verified"
						   % (verb, report.in_place, report.relocated,
							  report.fragments_repointed))
			self.status_bar.showMessage(message, 15000)
			logging.info(message)
			if direction == AUDIO_REVERT:
				logging.info(
					"The graph is back on stock names. Drop the mod's Audio/*.bnk and "
					"*.wmetasb.add as well, then re-run the registry merge - a stock-sound "
					"copy needs no bank, no fragment and no loader.")
			self.census_guard("audio event rename", expect_added=0, expect_removed=0)
		except Exception as exc:
			self.showerror(str(exc))

	def write_audio_fragment(self):
		try:
			if self.audio_direction() == AUDIO_REVERT:
				raise ValueError(
					"A registry fragment declares the events YOUR bank answers. A reverted "
					"copy ships no bank, so it needs no fragment - delete the mod's existing "
					"one and re-run the merge instead.")
			donor = self.audio_donor.text().strip()
			prefix = self.audio_prefix.text().strip()
			suffixes = []
			for index in range(self.audio_list.topLevelItemCount()):
				item = self.audio_list.topLevelItem(index)
				if not (item.flags() & QtCore.Qt.ItemIsUserCheckable):
					continue
				if item.checkState(0) != QtCore.Qt.Checked:
					continue
				suffixes.append(item.text(0)[len(donor) + 1:])
			if not suffixes:
				raise ValueError("Scan first, then tick the events your bank will answer")

			game_root = Path(self.audio_game.text().strip())
			xml, rows, remapped = audio_events.build_registry_fragment(
				game_root, donor, prefix, suffixes, DEFAULT_GAME)

			suggested = "%s.wmetasb.add" % prefix.lower()
			chosen, _filter = QtWidgets.QFileDialog.getSaveFileName(
				self, "Save registry fragment into your mod's Audio folder",
				suggested, "Registry fragment (*.wmetasb.add)")
			if not chosen:
				return
			Path(chosen).write_text(xml, encoding="utf-8", newline="")
			message = ("Wrote %s - %d event rows, %d ids remapped. Put it in <Mod>/Audio/ "
					   "and run the registry merge." % (Path(chosen).name, rows, remapped))
			self.status_bar.showMessage(message, 20000)
			logging.info(message)
		except Exception as exc:
			self.showerror(str(exc))

	# ---- Choosers -------------------------------------------------------

	def refresh_choosers(self):
		try:
			# `name` is the motiongraph name, NOT the clip filter - the clip filter
			# is applied below, the way ovl_tool_cmd's --match does.
			rows = list_choosers(self.source_path(),
								 self.name_edit.text().strip() or None, DEFAULT_GAME)
		except Exception as exc:
			self.showerror(str(exc))
			return
		self.chooser_tree.clear()
		match = self.chooser_filter.text().strip().lower()
		shown = 0
		for row in rows:
			clips = row.get("clips", [])
			if match and not any(match in str(c.get("name", "")).lower() for c in clips):
				continue
			shown += 1
			address = f"{row['pool']}:{row['offset']}"
			total = sum(int(c.get("weight", 0)) for c in clips) or 1
			parent = QtWidgets.QTreeWidgetItem([
				f"chooser {address}", "", "",
				f"{len(clips)} clips  blend {row.get('blend_time', 0.0):.2f}  "
				f"flags {row.get('flags', '?')}"
				+ ("  (flags 8 ignores blend_time)" if row.get("flags") == 8 else "")])
			parent.setData(0, QtCore.Qt.UserRole, (row["pool"], row["offset"],
												   [int(c.get("weight", 0)) for c in clips]))
			for clip in clips:
				w = int(clip.get("weight", 0))
				parent.addChild(QtWidgets.QTreeWidgetItem(
					[str(clip.get("name", "?")), str(w), f"{100.0 * w / total:.1f}%", ""]))
			self.chooser_tree.addTopLevelItem(parent)
			parent.setExpanded(True)
		self.status_bar.showMessage(f"Listed {shown} chooser(s)", 8000)
		if not shown:
			logging.info("No choosers matched that filter")

	def chooser_selected(self):
		items = self.chooser_tree.selectedItems()
		if not items:
			return
		item = items[0]
		if item.parent() is not None:
			item = item.parent()
		data = item.data(0, QtCore.Qt.UserRole)
		if not data:
			return
		pool, offset, weights = data
		self.chooser_target.setText(f"chooser {pool}:{offset}")
		self.chooser_weights.setText(",".join(str(w) for w in weights))

	def _selected_chooser(self):
		items = self.chooser_tree.selectedItems()
		if not items:
			raise ValueError("Select a chooser in the list first")
		item = items[0]
		if item.parent() is not None:
			item = item.parent()
		data = item.data(0, QtCore.Qt.UserRole)
		if not data:
			raise ValueError("Select a chooser row, not a clip row")
		return data[0], data[1]

	def apply_chooser_weights(self):
		try:
			pool, offset = self._selected_chooser()
			weights = [int(x) for x in self.chooser_weights.text().split(",") if x.strip()]
			if not weights:
				raise ValueError("Enter one weight per clip, comma separated")
			row = set_chooser_weights(self.source_path(), self.output_path(), pool, offset,
									  weights, name=self.name_edit.text().strip() or None,
									  game=DEFAULT_GAME)
			total = sum(weights) or 1
			shares = "  ".join(f"{100.0 * w / total:.1f}%" for w in weights)
			self.status_bar.showMessage(f"Weights applied - shares {shares}", 12000)
			logging.info(f"Chooser {pool}:{offset} weights -> {weights} ({shares})")
			# A weight edit is fixed-width inside an existing slot.
			self.census_guard("chooser re-weight", expect_added=0, expect_removed=0)
		except Exception as exc:
			self.showerror(str(exc))

	def apply_chooser_add(self):
		try:
			pool, offset = self._selected_chooser()
			clip = self.chooser_add_clip.text().strip()
			if not clip:
				raise ValueError("Enter the full clip name to add, e.g. Species$Rest03")
			# Re-weight in the SAME pass. As a separate operation it would re-read
			# `source` and discard the clip we just added.
			text = self.chooser_add_weights.text().strip()
			weights = [int(x) for x in text.replace(" ", "").split(",") if x] if text else None
			report = grow_random_animation_chooser(
				self.source_path(), self.output_path(), pool, offset, clip,
				weight=self.chooser_add_weight.value(),
				name=self.name_edit.text().strip() or None, game=DEFAULT_GAME,
				weights=weights)
			self.status_bar.showMessage(f"Added '{clip}' to chooser {pool}:{offset}", 12000)
			logging.info(f"Chooser {pool}:{offset} grown with '{clip}': {report}")
			# Growth relocates the entry array and may append a string; the object
			# count should rise, and nothing may disappear.
			self.census_guard("chooser growth", expect_added=None, expect_removed=0)
			self.showerror(
				"Chooser growth is TOPOLOGY GROWTH, and cobra reload is not proof of "
				"engine acceptance. Every operation of this class has needed a game "
				"launch to verify. Check the census line in the log, then test in game."
			)
		except Exception as exc:
			self.showerror(str(exc))

	# ---- Capacity -------------------------------------------------------

	def run_capacity_audit(self):
		if self.loader is None:
			self.showerror("Load a source OVL first")
			return
		try:
			audit = build_capacity_audit(self.loader, self.source_path())
			self.capacity_text.setPlainText(render_capacity_markdown(audit))
			self.status_bar.showMessage("Capacity audit complete", 8000)
		except Exception as exc:
			self.showerror(str(exc))

	# ---- Rename repair --------------------------------------------------

	def _rename_tokens(self):
		old = self.rename_old.text().strip()
		new = self.rename_new.text().strip()
		if not old or not new:
			raise ValueError("Enter both the old and the current species token")
		return old, new

	def run_rename_survey(self):
		try:
			old, new = self._rename_tokens()
			repointable, missing = survey_stale_references(
				self.source_path(), old, new, DEFAULT_GAME)
			total = sum(n for _o, _n, n in repointable)
			lines = [f"{total} repointable fragment(s) across {len(repointable)} name(s)", ""]
			for o, n, count in repointable:
				lines.append(f"  {count:4d}  {o}  ->  {n}")
			lines.append("")
			lines.append(f"UNREPAIRABLE - no renamed counterpart exists: {len(missing)}")
			for m in missing:
				lines.append(f"        {m}")
			if missing:
				lines.append("")
				lines.append("A name with no counterpart is dead under BOTH names. Repointing it "
							 "would be inventing a target; it is left alone.")
			self.rename_text.setPlainText("\n".join(lines))
			self.status_bar.showMessage(
				f"{total} repointable, {len(missing)} unrepairable", 10000)
		except Exception as exc:
			self.showerror(str(exc))

	def apply_rename_repair(self):
		try:
			old, new = self._rename_tokens()
			report = repoint_stale_species_strings(
				self.source_path(), self.output_path(), old, new, DEFAULT_GAME)
			self.rename_text.setPlainText(
				f"Repointed {report.repointed} fragment(s) across "
				f"{len(report.distinct_names)} name(s).\n\n"
				+ "\n".join(f"  {n}" for n in report.distinct_names)
				+ (f"\n\nLeft alone (no counterpart): {', '.join(report.missing)}"
				   if report.missing else ""))
			self.status_bar.showMessage(
				f"Repointed {report.repointed} stale references", 12000)
			logging.info(
				f"Rename repair: {report.repointed} fragments, "
				f"{len(report.distinct_names)} names, missing={report.missing}")
			# A pure fragment repoint must not change the decoded object set.
			self.census_guard("rename repair", expect_added=0, expect_removed=0)
		except Exception as exc:
			self.showerror(str(exc))

	def open(self, filepath: str):
		self.source_edit.setText(filepath)
		self.load_source()

	def open_dir(self, dirpath: str):
		self.stage_edit.setText(dirpath)

	def save(self, filepath: str):
		if self.plan:
			save_plan(Path(filepath), self.plan)


if __name__ == "__main__":
	initial = sys.argv[1] if len(sys.argv) > 1 else ""
	startup(
		MainWindow,
		GuiOptions(
			log_name="motiongraph_tool_gui", log_to_file=False,
			size=(1200, 780), check_update=False,
		),
		initial_path=initial,
	)
