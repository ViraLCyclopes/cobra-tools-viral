"""Safe, fixed-topology JWE3 motiongraph browser and patch-plan editor."""

from __future__ import annotations

import math
import logging
import shutil
import sys
from pathlib import Path

from gui import GuiOptions, startup, widgets
from gui.widgets import window
from PyQt5 import QtCore, QtGui, QtWidgets

from source.formats.motiongraph.edit import (
	DEFAULT_GAME,
	apply_patch_plan,
	build_patch_plan,
	load_motiongraph,
	locate_fields,
	save_plan,
)
from source.formats.motiongraph.report import (
	build_activity_tree,
	build_decision_graph,
	build_decision_report,
	build_state_report,
)
from source.formats.motiongraph.static_patch import patch_string_slot, repoint_existing_string


def motiongraph_family(source: Path) -> list[Path]:
	"""Return the main OVL and its same-basename OVS/AUX companions."""
	source = source.resolve()
	if source.suffix.lower() != ".ovl" or not source.is_file():
		raise ValueError("Choose an existing .ovl file")
	prefix = source.stem.lower() + "."
	result = []
	for candidate in source.parent.iterdir():
		name = candidate.name.lower()
		# JWE3 AUX companions are hash-named rather than OVL-basename-prefixed.
		# Include every AUX in the source archive directory along with the named
		# OVL/OVS streams so "full family" really is load-complete.
		if candidate.is_file() and (
			(name.startswith(prefix) and (name.endswith(".ovl") or ".ovs" in name))
			or name.endswith(".aux")
		):
			result.append(candidate)
	if source not in result:
		result.append(source)
	return sorted(set(result), key=lambda path: path.name.lower())


def copy_motiongraph_family(source: Path, stage_dir: Path) -> list[Path]:
	"""Copy a complete archive family without modifying the source directory."""
	source, stage_dir = source.resolve(), stage_dir.resolve()
	if source.parent == stage_dir:
		raise ValueError("The stage directory must differ from the source directory")
	stage_dir.mkdir(parents=True, exist_ok=True)
	outputs = []
	for member in motiongraph_family(source):
		output = stage_dir / member.name
		shutil.copy2(member, output)
		outputs.append(output)
	return outputs


def _load_reports(source: Path, name: str | None):
	ovl, loader = load_motiongraph(source, name or None, DEFAULT_GAME)
	state_text, states, state_stats = build_state_report(loader)
	decision_text, decision_stats = build_decision_report(loader)
	decision_graph = build_decision_graph(loader, states)
	return (
		ovl, loader, state_text, states, state_stats,
		decision_text, decision_stats, decision_graph,
	)


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

		root = QtWidgets.QVBoxLayout(self.body)
		banner = QtWidgets.QLabel(
			"SAFE MODE: fixed topology and fixed-width edits only. The source OVL is never overwritten."
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
		self._build_retarget_tab()
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
		self.graph_hops.setRange(1, 2)
		self.graph_hops.setValue(1)
		self.graph_hops.setPrefix("Hops: ")
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
		controls.addWidget(focus_button)
		controls.addWidget(all_button)
		controls.addWidget(self.edges_toggle)
		controls.addWidget(pan_hint)
		layout.addLayout(controls)
		self.graph_breadcrumb = QtWidgets.QLabel("All states")
		self.graph_breadcrumb.setStyleSheet("color: #8ab4f8; padding-left: 3px;")
		layout.addWidget(self.graph_breadcrumb)
		explanation = QtWidgets.QLabel(
			"≈ labels are inferred from reachable clips or streams; Frontier states are anonymous runtime containers."
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
		edit_row = QtWidgets.QHBoxLayout()
		self.edit_mode = QtWidgets.QComboBox()
		self.edit_mode.addItems(["Numeric value", "Enum name", "Flag operations"])
		self.value_edit = QtWidgets.QLineEdit()
		self.value_edit.setPlaceholderText("Replacement: e.g. 0.25, enum name, or FlagA=1,FlagB=0")
		self.plan_button = QtWidgets.QPushButton("Create / save patch plan…")
		self.plan_button.clicked.connect(self.create_plan)
		edit_row.addWidget(QtWidgets.QLabel("Replacement"))
		edit_row.addWidget(self.edit_mode)
		edit_row.addWidget(self.value_edit, 1)
		edit_row.addWidget(self.plan_button)
		layout.addLayout(edit_row)
		layout.addWidget(self.field_table, 1)
		self.tabs.addTab(self.fields_page, "Fields / Edit")

	def _build_retarget_tab(self):
		page = QtWidgets.QWidget()
		form = QtWidgets.QFormLayout(page)
		info = QtWidgets.QLabel(
			"Clip retarget repoints existing references to another existing same-pool string. "
			"String-slot replacement fits a new shorter/equal ASCII name into one existing allocation."
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
		self.tabs.addTab(page, "Clip Retarget")

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
		self.apply_plan_button = QtWidgets.QPushButton("Apply current patch plan to staged family")
		self.apply_plan_button.clicked.connect(self.apply_plan)
		layout.addWidget(self.apply_plan_button)
		warning = QtWidgets.QLabel(
			"This editor does not install into the live game. Each Apply rebuilds the staged main OVL "
			"from the source, replacing an earlier staged edit. Use separate stage directories (or an "
			"alternating source/stage pipeline) for sequential operations."
		)
		warning.setWordWrap(True)
		layout.addWidget(warning)
		self.tabs.addTab(page, "Stage / Apply")

	def source_path(self) -> Path:
		path = Path(self.source_edit.text().strip())
		if not path.is_file():
			raise ValueError("Choose an existing source OVL")
		return path

	def output_path(self) -> Path:
		stage = Path(self.stage_edit.text().strip())
		if not stage.is_dir():
			raise ValueError("Copy the full family to a stage directory first")
		output = stage / self.source_path().name
		if not output.is_file():
			raise ValueError(f"Staged main OVL is missing: {output}")
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
		try:
			source = self.source_path()
		except Exception as exc:
			self.showerror(str(exc))
			return
		self.load_button.setEnabled(False)
		self.status_bar.showMessage("Loading and analysing motiongraph…")
		logging.info(f"Loading and analysing motiongraph family: {source}")
		worker = self.run_background_task(
			_load_reports, self.loaded_source, source, self.name_edit.text().strip() or None
		)
		worker.signals.finished.connect(lambda: self.load_button.setEnabled(True))

	def loaded_source(self, result):
		(self.ovl, self.loader, _state_text, self.state_rows, state_stats,
		 decision_text, decision_stats, self.decision_graph_data) = result
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

	def populate_graph(self):
		self.graph_scene.clear()
		self.graph_nodes = {}
		self.graph_edges = []
		self.graph_edge_records = []
		valid = [state for state in self.state_rows if state]
		columns, x_step, y_step = 12, 225, 105
		for order, state in enumerate(valid):
			position = ((order % columns) * x_step, (order // columns) * y_step)
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
		self.graph_scene.clearSelection()
		center.setSelected(True)
		self.graph_breadcrumb.setText(
			f"All states  ›  STATE[{state_index}] ≈ {center.state['label']}  ›  "
			f"{self.graph_hops.value()} hop(s)"
		)
		self.graph_view.set_content_rect(self.visible_node_bounds(self.graph_nodes.values()))
		self.fit_graph()

	def show_all_state_graph(self):
		for item in self.graph_nodes.values():
			item.setVisible(True)
		for _, _, line, arrow in self.graph_edge_records:
			line.setVisible(self.edges_toggle.isChecked())
			arrow.setVisible(self.edges_toggle.isChecked())
		self.graph_breadcrumb.setText("All states")
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
		item = self.activity_tree.currentItem()
		if item is None:
			self.showerror("Select an activity-tree node first")
			return
		# Resolve a displayed shared-reference row to the concrete activity row.
		target = item.data(1, QtCore.Qt.UserRole)
		if target is not None:
			item = self.activity_node_items.get(target, item)
		activity_type = item.data(2, QtCore.Qt.UserRole + 1)
		if not activity_type or activity_type in ("shared reference", "limit"):
			self.showerror("The selected row is not an editable activity")
			return
		label = item.data(2, QtCore.Qt.UserRole) or ""
		address = item.data(2, QtCore.Qt.UserRole + 2)
		if not address or len(address) != 2:
			self.showerror("Could not resolve the selected activity's exact STATIC address")
			return
		self.activity_address_filter = tuple(int(value) for value in address)
		self.field_scope.setCurrentIndex(self.field_scope.findData("activity"))
		self.activity_filter.setText(label)
		self.type_filter.setText(activity_type)
		self.field_filter.clear()
		self.tabs.setCurrentWidget(self.fields_page)
		message = (
			f"Scanning exact {activity_type} {label!r} at "
			f"pool {address[0]} offset {address[1]}"
		)
		self.status_bar.showMessage(message, 8000)
		logging.info(message)
		self.scan_fields()

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
		worker = self.run_background_task(
			locate_fields, self.scanned_fields, self.source_path(), self.name_edit.text().strip(),
			DEFAULT_GAME, self.activity_filter.text().strip() or None,
			self.type_filter.text().strip() or None, self.field_filter.text().strip() or None,
			activity_pool, activity_offset, activity_addresses,
		)
		worker.signals.finished.connect(lambda: self.scan_button.setEnabled(True))

	def scanned_fields(self, result):
		rows, mismatches = result
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
				for column, item in enumerate(values):
					self.field_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(item)))
		self.field_table.resizeColumnsToContents()
		message = f"Found {len(self.field_rows)} fields"
		if mismatches:
			message += f"; skipped {len(mismatches)} stride-mismatched layouts"
		self.status_bar.showMessage(message, 8000)
		logging.info(message)

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
			self.plan = build_patch_plan(
				grouped, self.source_path(), self.name_edit.text().strip(), field_paths.pop(), **kwargs
			)
			path, _ = QtWidgets.QFileDialog.getSaveFileName(
				self, "Save patch plan", "motiongraph_patch.json", "JSON files (*.json)"
			)
			if not path:
				return
			save_plan(Path(path), self.plan)
			self.status_bar.showMessage(f"Saved {len(self.plan['edits'])} verified edits to {path}", 8000)
			logging.info(f"Saved {len(self.plan['edits'])} verified edits to patch plan: {path}")
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

	def apply_plan(self):
		if not self.plan:
			self.showerror("Create a patch plan first")
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
