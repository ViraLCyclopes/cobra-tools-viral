import argparse
import contextlib
import os
import sys
import time
import logging

import numpy as np

if __name__ == "__main__":
	# Guard to hide from pytest or other imports
	from gui.auto_updater import run_update_check
	run_update_check("manis_tool_gui")

from gui import widgets, startup, GuiOptions  # Import widgets before everything except built-ins!
from gui.widgets import window, MenuItem, SeparatorMenuItem, get_icon
from generated.formats.manis import ManisFile
from generated.formats.manis.versions import games
from generated.formats.ms2 import Ms2File
from generated.formats.wsm.structs.WsmHeader import WsmHeader
from source.formats.manis import inspect as manis_inspect
from source.formats.manis.acl_patch import (
	patch_constant,
	patch_scalar_constant,
	read_animated_range,
	read_constant,
	read_scalar,
	scale_animated_range,
	scale_scalar_range,
)
from utils.logs import logging_setup
from typing import Optional
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt

from matplotlib import pyplot as plt
plt.set_loglevel(level='warning')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar


class AclEditDialog(QtWidgets.QDialog):
	"""Pick one ACL sub-track and either set its constant or scale its range.

	Both operations are size-preserving. Scaling a range multiplies the clip-wide
	min/extent that Frontier's normalised samples are decoded against, so the
	motion keeps its shape. ``pivot`` matters whenever a curve does not start at
	zero: scaling about zero moves the first sample and reads in game as a
	teleport, while scaling about the decoded first sample preserves it.
	"""

	KINDS = ("rotation", "translation", "scale", "scalar")

	def __init__(self, parent, clip_name):
		super().__init__(parent)
		self.setWindowTitle(f"Edit ACL Value - {clip_name}")

		self.kind_combo = QtWidgets.QComboBox()
		self.kind_combo.addItems(self.KINDS)
		self.track_spin = QtWidgets.QSpinBox()
		self.track_spin.setRange(0, 65535)

		self.mode_combo = QtWidgets.QComboBox()
		self.mode_combo.addItems(("Scale animated range", "Set constant value"))

		self.value_edits = [QtWidgets.QLineEdit("1.0") for _ in range(4)]
		for edit in self.value_edits:
			edit.setValidator(QtGui.QDoubleValidator())
		values_row = QtWidgets.QHBoxLayout()
		for edit in self.value_edits:
			values_row.addWidget(edit)

		self.pivot_combo = QtWidgets.QComboBox()
		self.pivot_combo.addItems(("first", "zero"))
		self.pivot_combo.setToolTip(
			"'first' preserves the curve's starting value and only stretches its "
			"travel; 'zero' scales about the origin and will move the start.")

		form = QtWidgets.QFormLayout()
		form.addRow("Kind", self.kind_combo)
		form.addRow("Track index", self.track_spin)
		form.addRow("Operation", self.mode_combo)
		form.addRow("Values (x y z w)", values_row)
		form.addRow("Range pivot", self.pivot_combo)

		self.hint = QtWidgets.QLabel()
		self.hint.setWordWrap(True)
		form.addRow(self.hint)

		buttons = QtWidgets.QDialogButtonBox(
			QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
		buttons.accepted.connect(self.accept)
		buttons.rejected.connect(self.reject)

		layout = QtWidgets.QVBoxLayout(self)
		layout.addLayout(form)
		layout.addWidget(buttons)

		self.kind_combo.currentTextChanged.connect(self.sync)
		self.mode_combo.currentTextChanged.connect(self.sync)
		self.sync()

	def sync(self, *_args):
		kind = self.kind()
		range_mode = self.is_range_mode()
		# Range scaling rewrites a vector min/extent, which a quaternion sub-track
		# does not have, so rotation is constant-only.
		scalable = kind in manis_inspect.RANGE_SCALABLE_KINDS
		if range_mode and not scalable:
			self.mode_combo.setCurrentIndex(1)
			return
		used = 1 if kind == "scalar" else (4 if kind == "rotation" else 3)
		for index, edit in enumerate(self.value_edits):
			edit.setEnabled(index < used)
		self.pivot_combo.setEnabled(range_mode)
		if not scalable:
			self.hint.setText(
				"Rotation has no range to scale - a quaternion sub-track stores no "
				"min/extent. Constant editing only.")
		elif range_mode:
			self.hint.setText(
				"Multiplies the stored range. Keep pivot 'first' unless the curve "
				"genuinely starts at zero.")
		else:
			self.hint.setText(
				"Overwrites a constant sub-track. Only sub-tracks reported as "
				"'constant' in the details panel have storage to write to.")

	def kind(self):
		return self.kind_combo.currentText()

	def track(self):
		return int(self.track_spin.value())

	def is_range_mode(self):
		return self.mode_combo.currentIndex() == 0

	def pivot(self):
		return self.pivot_combo.currentText()

	def values(self):
		out = []
		for edit in self.value_edits:
			if not edit.isEnabled():
				continue
			try:
				out.append(float(edit.text()))
			except ValueError:
				out.append(0.0)
		return out


class MainWindow(window.MainWindow):

	def __init__(self, opts: GuiOptions):
		window.MainWindow.__init__(self, "Manis Editor", opts=opts)
		self.setAcceptDrops(True)

		self.manis_file = ManisFile()

		self.filter = "Supported files (*manis)"

		self.file_widget = self.make_file_widget(ftype="MANIS")

		self.game_choice = widgets.LabelCombo("Game", [g.value for g in games], editable=False,
											  changed_fn=self.game_changed)

		self.stream_entry = QtWidgets.QLineEdit()
		self.stream_entry.setPlaceholderText("Stream")
		self.stream_entry.setToolTip("OVS stream that holds this manis' data")
		# create the table
		self.header_labels = ["Name", "Num", "Compressed", "Duration", "Frames"]

		# tree
		self.tree = QtWidgets.QTreeWidget(self)
		self.tree.setColumnCount(2)
		self.tree.setHeaderLabels(self.header_labels)
		self.tree.setAlternatingRowColors(True)
		self.tree.itemChanged.connect(self.edit_handle)
		self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
		self.tree.customContextMenuRequested.connect(self.context_menu)
		self.tree.itemSelectionChanged.connect(self.selection_change)

		# Raw file bytes, kept so ACL blobs can be read and patched in place. A
		# compressed clip cannot survive ManisFile.save(), so ACL edits are byte
		# edits against this buffer rather than changes to the parsed structs.
		self.raw_data = b""
		self.raw_path = ""
		self.raw_modified = False

		self.details = QtWidgets.QPlainTextEdit()
		self.details.setReadOnly(True)
		self.details.setPlaceholderText("Select a clip to inspect it")
		self.details.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

		self.setup_plot()
		splitter = QtWidgets.QSplitter()
		splitter.addWidget(self.tree)

		toolbar = NavigationToolbar(self.fig.canvas, self)

		plot = widgets.pack_in_box(toolbar, self.fig.canvas)
		self.fig.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		plot.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		right = QtWidgets.QSplitter(Qt.Vertical)
		right.addWidget(plot)
		right.addWidget(self.details)
		right.setSizes([60, 40])
		splitter.addWidget(right)
		splitter.setSizes([30, 50])
		splitter.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		hbox = QtWidgets.QHBoxLayout()
		hbox.addWidget(self.file_widget)
		hbox.addWidget(self.stream_entry)
		hbox.addWidget(self.game_choice)
		self.qgrid = QtWidgets.QVBoxLayout()
		self.qgrid.addLayout(hbox)
		self.qgrid.addWidget(splitter)

		self.qgrid.addWidget(self.progress)
		self.central_widget.setLayout(self.qgrid)

		self.build_menus({
			widgets.FILE_MENU: self.file_menu_items,
			widgets.EDIT_MENU: [
				MenuItem("Append", self.append, icon="append"),
				MenuItem("Duplicate Selected", self.duplicate, shortcut="SHIFT+D", icon="duplicate_mesh"),
				MenuItem("Remove Selected", self.remove, shortcut="DEL", icon="delete_mesh"),
				MenuItem("Resize", self.resize_popup, shortcut="CTRL+R"),
				SeparatorMenuItem(),
				MenuItem("Verify Bundle", self.verify_popup, shortcut="CTRL+E"),
				MenuItem("Edit ACL Value", self.acl_edit_popup),
			],
			widgets.HELP_MENU: self.help_menu_items
		})

	def setup_plot(self):
		# a figure instance to plot on
		self.setPalette(self.get_palette_from_cfg())
		text_color = self.get_palette_color(self.palette().text())
		text2_color = self.get_palette_color(self.palette().dark())
		self.fig, self.ax = plt.subplots(nrows=1, ncols=1, layout="tight", facecolor=text_color)
		# self.ax.set_xlabel('Frame', color=text_color)
		# self.ax.set_ylabel('Value', color=text_color)
		self.ax.spines['bottom'].set_color(text2_color)
		self.ax.spines['top'].set_color(text2_color)
		self.ax.spines['right'].set_color(text2_color)
		self.ax.spines['left'].set_color(text2_color)
		self.ax.tick_params(axis='x', colors=text2_color)
		self.ax.tick_params(axis='y', colors=text2_color)

		# the range is not automatically fixed
		self.fig.patch.set_facecolor(self.get_palette_color(self.palette().base()))
		self.ax.set_facecolor(self.get_palette_color(self.palette().alternateBase()))

	def get_palette_color(self, input_color):
		col = input_color.color().toRgb()
		col_rgb = (col.red() / 255, col.green() / 255, col.blue() / 255)
		return col_rgb

	@contextlib.contextmanager
	def update_plot(self):
		# discards the old graph
		self.ax.clear()
		yield
		# self.ax.set_xlabel('Frame')
		# self.ax.set_ylabel('Value')
		# refresh canvas
		self.fig.canvas.draw()

	def selection_change(self):
		for item in self.tree.selectedItems():
			names = self.get_parents(item)
			if len(names) == 1:
				mani_name, = names
				self.update_details(mani_name)
			elif len(names) == 3:
				mani_name, dtype, bone_name = names
				self.update_details(mani_name)
				self.plot_channel(mani_name, dtype, bone_name)

	def plot_channel(self, mani_name, dtype, bone_name):
		"""Draw one channel, from ACL samples when the clip is compressed.

		``show_keys_by_dtype`` reads the uncompressed key arrays, which an ACL
		clip does not have - its curves only exist inside the blob. Without this
		the plot is silently blank for every JWE3 clip.
		"""
		index = self.clip_index(mani_name)
		if index is None:
			return
		report = manis_inspect.clip_report(self.manis_file, index, self.raw_data)
		with self.update_plot():
			if not report.is_acl:
				try:
					self.manis_file.show_keys_by_dtype(mani_name, dtype, bone_name, self.ax)
				except:
					logging.exception("failed to plot uncompressed keys")
				return
			try:
				self.plot_acl_channel(index, dtype, bone_name)
			except Exception as exc:
				logging.exception("failed to plot ACL channel")
				self.ax.set_title(f"Could not plot: {exc}", fontsize=8)

	def plot_acl_channel(self, index, dtype, bone_name):
		from source.formats.manis.acl import decode_blob

		mani_info = self.manis_file.mani_infos[index]
		spans = manis_inspect.clip_blob_spans(self.raw_data, index)
		scalar = dtype == "floats"
		span = spans.get("scalar" if scalar else "transform")
		if span is None:
			raise ValueError(f"clip has no {'scalar' if scalar else 'transform'} blob")
		blob = self.raw_data[span[0]:span[0] + span[1]]
		names = [str(n) for n in getattr(mani_info.keys, f"{dtype}_names")]
		if bone_name not in names:
			raise ValueError(f"{bone_name} is not a {dtype} channel")
		channel = names.index(bone_name)
		if scalar:
			track, labels, columns = channel, ("value",), (0,)
		else:
			# ACL track order is the target skeleton's, so a channel must be
			# mapped through its channel_to_bone table before indexing samples.
			bare = dtype.replace("_bones", "")
			lut = getattr(mani_info.keys, f"{bare}_channel_to_bone", None)
			track = int(lut[channel]) if lut is not None and len(lut) > channel else channel
			labels, columns = {
				"pos": (("x", "y", "z"), (4, 5, 6)),
				"ori": (("x", "y", "z", "w"), (0, 1, 2, 3)),
				"scl": (("x", "y", "z"), (7, 8, 9)),
			}[bare]
		values = decode_blob(blob).values
		if track >= values.shape[1]:
			raise ValueError(f"track {track} is outside the {values.shape[1]} decoded tracks")
		frames = range(values.shape[0])
		drawn = 0
		for label, column in zip(labels, columns):
			series = values[:, track, column]
			# The decoder pre-fills NaN and ACL overwrites only what it stored, so
			# an all-NaN component is a stripped sub-track: it equals the bind
			# pose. That is data, not a decode failure, and must not read as zero.
			if np.isnan(series).all():
				continue
			self.ax.plot(frames, series, label=label)
			drawn += 1
		if not drawn:
			self.ax.set_title(
				f"{bone_name} [{dtype}] - stripped sub-track, equals the bind pose",
				fontsize=8)
			return
		self.ax.legend(loc="upper right", fontsize=7)
		self.ax.set_title(
			f"{bone_name} [{dtype}] - ACL track {track}, {values.shape[0]} samples",
			fontsize=8)

	def clip_index(self, mani_name):
		for index, mani_info in enumerate(self.manis_file.mani_infos):
			if str(mani_info.name) == mani_name:
				return index
		return None

	def update_details(self, mani_name):
		"""Describe the selected clip, including ACL sub-track classification.

		A stripped ACL sub-track stores no samples and falls back to the bind
		pose, which is not the same as a flat curve. Only this panel tells the
		two apart, and it is what says whether an ACL edit has anything to
		target. Bundles from JWE1/JWE2/PZ report as uncompressed and simply
		show no ACL section.
		"""
		index = self.clip_index(mani_name)
		if index is None:
			self.details.setPlainText("")
			return
		try:
			report = manis_inspect.clip_report(self.manis_file, index, self.raw_data)
		except Exception:
			logging.exception("clip inspection failed")
			self.details.setPlainText("Inspection failed, see log.")
			return
		lines = [
			f"{report.name}",
			f"  index          {report.index}",
			f"  storage        {'ACL compressed' if report.is_acl else ('compressed (no blob found)' if report.compressed else 'uncompressed keys')}",
			f"  duration       {report.duration:.4f} s over {report.frame_count} frames",
			f"  channels       pos {report.pos_bones} / ori {report.ori_bones} / "
			f"scl {report.scl_bones} / floats {report.floats}",
			f"  target rig     {report.target_bone_count} bones",
		]
		for note in report.notes:
			lines.append(f"  note           {note}")
		if report.acl:
			acl = report.acl
			lines += [
				"",
				"ACL transform blob",
				f"  at             offset {acl['offset']}, {acl['size']} bytes",
				f"  tracks         {acl.get('num_tracks')} over {acl.get('num_samples')} samples",
				f"  scale payload  {'yes' if acl.get('has_scale') else 'no'}",
			]
			for key in ("has_database", "wrap_optimized", "trivial_defaults"):
				if key in acl:
					lines.append(f"  {key:<14} {acl[key]}")
			if "layout_error" in acl:
				lines.append(f"  layout error   {acl['layout_error']}")
			for kind, entry in (acl.get("sub_tracks") or {}).items():
				if "counts" in entry:
					summary = ", ".join(f"{k} {v}" for k, v in entry["counts"].items() if v)
					lines.append(f"  {kind:<14} {summary}")
				else:
					lines.append(f"  {kind:<14} {entry.get('error')}")
			scalar = acl.get("scalar")
			if scalar:
				lines += [
					"",
					"ACL scalar blob",
					f"  at             offset {scalar['offset']}, {scalar['size']} bytes",
					f"  tracks         {scalar.get('num_tracks')}",
				]
				if "bit_rates" in scalar:
					lines.append(f"  bit rates      {scalar['bit_rates']}")
		self.details.setPlainText("\n".join(lines))

	def verify_popup(self):
		"""Run the pre-write gate and show what it found."""
		if not self.manis_file.mani_infos:
			self.showwarning("Open a MANIS first.")
			return
		try:
			issues = manis_inspect.verify_bundle(
				self.manis_file, self.raw_data or None,
				len(self.raw_data) if self.raw_data else None)
		except Exception:
			self.handle_error("Verification failed, see log!")
			return
		worst = manis_inspect.worst_severity(issues)
		body = "\n".join(str(issue) for issue in issues) or "No issues found."
		header = {
			manis_inspect.ERROR: "Errors found - do not inject this bundle.",
			manis_inspect.WARNING: "Warnings found.",
			manis_inspect.INFO: "Clean, with notes.",
			None: "Clean.",
		}[worst]
		box = QtWidgets.QMessageBox(self)
		box.setWindowTitle("Verify Bundle")
		box.setText(header)
		box.setDetailedText(body)
		box.setIcon(QtWidgets.QMessageBox.Critical if worst == manis_inspect.ERROR
					else QtWidgets.QMessageBox.Information)
		box.exec()

	def selected_acl_clip(self):
		"""The selected clip if it is ACL-compressed, else None with a reason shown."""
		items = self.tree.selectedItems()
		if not items:
			self.showwarning("Select a clip first.")
			return None
		mani_name = self.get_parents(items[0])[0]
		index = self.clip_index(mani_name)
		if index is None:
			self.showwarning("Could not resolve the selected clip.")
			return None
		report = manis_inspect.clip_report(self.manis_file, index, self.raw_data)
		if not report.is_acl:
			self.showwarning(
				f"{mani_name} is not an ACL clip.\n\n"
				"In-place ACL editing applies to compressed clips only - JWE1, JWE2 "
				"and Planet Zoo bundles store uncompressed keys, which the tree and "
				"plot already edit directly.")
			return None
		return index, mani_name, report

	def acl_edit_popup(self):
		"""Edit one ACL value in place, without re-encoding the clip.

		Re-encoding is not an option: a faithful rebuild of a vanilla blob is
		rejected by JWE3 even with no edit at all. These patches keep every
		offset, size, segment and bit rate identical and only rewrite stored
		constants or the clip-wide range metadata.
		"""
		selected = self.selected_acl_clip()
		if not selected:
			return
		index, mani_name, _report = selected
		dialog = AclEditDialog(self, mani_name)
		if not dialog.exec():
			return
		try:
			spans = manis_inspect.clip_blob_spans(self.raw_data, index)
			kind = dialog.kind()
			span = spans.get("scalar" if kind == "scalar" else "transform")
			if span is None:
				self.showwarning(f"No {kind} blob for this clip.")
				return
			blob = self.raw_data[span[0]:span[0] + span[1]]
			track = dialog.track()
			if dialog.is_range_mode():
				# Match the CLI exactly: 'first' means the decoded first sample,
				# which keeps the curve's start put while stretching its travel.
				pivot = (manis_inspect.first_sample_pivot(blob, kind, track)
						 if dialog.pivot() == "first" else manis_inspect.zero_pivot(kind))
			if kind == "scalar":
				before = read_scalar(blob, track)
				if dialog.is_range_mode():
					patched = scale_scalar_range(
						blob, track, dialog.values()[0], pivot)
				else:
					patched = patch_scalar_constant(blob, track, dialog.values()[0])
				after = read_scalar(patched, track)
			elif dialog.is_range_mode():
				before = read_animated_range(blob, kind, track)
				patched = scale_animated_range(
					blob, kind, track, tuple(dialog.values()), pivot)
				after = read_animated_range(patched, kind, track)
			else:
				before = read_constant(blob, kind, track)
				patched = patch_constant(blob, kind, track, tuple(dialog.values()))
				after = read_constant(patched, kind, track)
			self.raw_data = manis_inspect.splice_blob(self.raw_data, span, patched)
		except Exception as exc:
			self.handle_error(f"ACL edit failed: {exc}")
			return
		self.raw_modified = True
		self.set_file_modified(True)
		self.update_details(mani_name)
		self.set_progress_message(f"{mani_name} {kind} track {track}: {before} -> {after}")
		logging.info(f"ACL edit {mani_name} {kind} track {track}: {before} -> {after}")

	def context_menu(self, pos):
		item = self.tree.itemAt(pos)
		if item:
			names = self.get_parents(item)
			if len(names) == 1:
				mani_name, = names
			elif len(names) == 3:
				mani_name, dtype, bone_name = names
				show_keys = QtWidgets.QAction("Show Keys")

				def show_keys_cb(checked):
					with self.update_plot():
						try:
							self.manis_file.show_keys_by_dtype(mani_name, dtype, bone_name, self.ax)
						except:
							logging.exception(f"failed")

				def delet_bone_cb(checked):
					try:
						self.manis_file.remove_bone(mani_name, dtype, bone_name)
					except:
						logging.exception(f"failed")
				show_keys.triggered.connect(show_keys_cb)
				delete = QtWidgets.QAction(f"Delete {bone_name} [{dtype}]")
				delete.triggered.connect(delet_bone_cb)

				menu = QtWidgets.QMenu("Context", self.tree)
				menu.addAction(show_keys)
				menu.addAction(delete)
				menu.exec(self.tree.mapToGlobal(pos))

	def get_parents(self, item):
		names = [item.text(0), ]
		while item.parent():
			item = item.parent()
			names.insert(0, item.text(0))
		return names

	def game_changed(self, game: Optional[str] = None):
		if game is None:
			game = self.game_choice.entry.currentText()
		logging.info(f"Setting Manis version to {game}")
		self.manis_file.game = game

	def edit_handle(self, item, col):
		ix = self.tree.indexOfTopLevelItem(item)
		mani_info = self.manis_file.mani_infos[ix]
		new_val = item.text(col)
		dtype = self.header_labels[col]
		logging.info(f"Editing {mani_info.name}.{dtype} = {new_val}")
		try:
			if dtype == "Name":
				# force new name to be lowercase
				new_name = new_val.lower()
				try:
					if self.manis_file.name_used(new_name):
						self.showwarning(f"Model {new_name} already exists in MANIS!")
					# new name is new
					else:
						self.manis_file.rename_file(mani_info.name, new_name)
						self.set_file_modified(True)
				except:
					self.handle_error("Renaming failed, see log!")
				self.update_gui_table()
			elif dtype == "Compressed":
				if mani_info.dtype.compression == 1 and new_val == "0":
					logging.info(f"Decompressing")
					self.manis_file.decompress(mani_info, dump=False)
			elif dtype == "Duration":
				logging.info(f"Changing duration to {new_val}")
				mani_info.duration = float(new_val)
		except:
			logging.exception("edit_handle")

	def remove(self):
		for item in self.tree.selectedItems():
			names = self.get_parents(item)
			if len(names) == 1:
				mani_name, = names
				try:
					self.manis_file.remove((mani_name, ))
					self.set_file_modified(True)
				except:
					self.handle_error("Removing file failed, see log!")
			elif len(names) == 3:
				mani_name, dtype, bone_name = names
				try:
					self.manis_file.remove_bone(mani_name, dtype, bone_name)
					self.set_file_modified(True)
				except:
					self.handle_error("Removing file failed, see log!")
		self.update_gui_table()

	def duplicate(self):
		for item in self.tree.selectedItems():
			names = self.get_parents(item)
			if len(names) == 1:
				mani_name, = names
				try:
					self.manis_file.duplicate((mani_name, ))
					self.set_file_modified(True)
				except:
					self.handle_error("Duplicating file failed, see log!")
			elif len(names) == 3:
				mani_name, dtype, bone_name = names
		self.update_gui_table()

	def open(self, filepath):
		if filepath:
			self.set_file_modified(False)
			self.raw_modified = False
			self.raw_data = b""
			self.raw_path = filepath
			try:
				# Keep the bytes as well as the parsed structs; ACL blobs are read
				# and patched from these, and ManisFile cannot round-trip them.
				with open(filepath, "rb") as stream:
					self.raw_data = stream.read()
			except:
				logging.exception("could not read raw bundle bytes")
			try:
				self.manis_file.load(filepath)
				# print(self.manis_file)
			except:
				self.handle_error("Loading failed, see log!")
			self.update_gui_table()
			self.report_open_state()

	def report_open_state(self):
		"""Summarise the bundle and surface any errors as soon as it is opened."""
		try:
			report = manis_inspect.bundle_report(self.manis_file, self.raw_data or None)
			issues = manis_inspect.verify_bundle(
				self.manis_file, self.raw_data or None,
				len(self.raw_data) if self.raw_data else None)
		except Exception:
			logging.exception("bundle inspection failed")
			return
		errors = [i for i in issues if i.severity == manis_inspect.ERROR]
		summary = (f"{report['clip_count']} clips, {report['acl_clips']} ACL, "
				   f"{report['uncompressed_clips']} uncompressed")
		if errors:
			self.set_progress_message(f"{summary} - {len(errors)} error(s), see Verify Bundle")
		else:
			self.set_progress_message(summary)

	def append(self):
		if self.file_widget.is_open():
			append_path = self.file_widget.get_open_file_name(f'Append MANIS')
			if append_path:
				try:
					other_manis_file = ManisFile()
					other_manis_file.load(append_path)
					# ensure that there are no name collisions
					for mani_info in other_manis_file.mani_infos:
						self.manis_file.make_name_unique(mani_info)
						self.manis_file.mani_infos.append(mani_info)
						# update context reference on everything do that indexing happens using the correct reference
						mani_info.set_context(self.manis_file.context)
						mani_info.keys.set_context(self.manis_file.context)
					self.set_file_modified(True)
				except:
					self.handle_error("Appending failed, see log!")
				self.update_gui_table()

	def update_gui_table(self, ):
		start_time = time.time()
		self.tree.clear()
		self.tree.itemChanged.disconnect(self.edit_handle)
		try:
			logging.info(f"Loading {len(self.manis_file.mani_infos)} files into GUI")
			# addition data to the tree
			for m in self.manis_file.mani_infos:
				mani_item = QtWidgets.QTreeWidgetItem(self.tree)
				mani_item.setText(0, m.name)
				mani_item.setIcon(0, get_icon("mani"))
				mani_item.setText(2, str(m.dtype.compression))
				mani_item.setText(3, f"{m.duration:.4f}")
				mani_item.setText(4, str(m.frame_count))
				mani_item.setFlags(mani_item.flags() | Qt.ItemIsEditable)
				for dtype in ("pos_bones", "ori_bones", "scl_bones", "floats"):
					if not hasattr(m, "keys"):
						continue
					dtype_array = getattr(m.keys, f"{dtype}_names")
					if len(dtype_array):
						dtype_item = QtWidgets.QTreeWidgetItem(mani_item)
						dtype_item.setIcon(0, get_icon(dtype))
						dtype_item.setText(0, dtype)
						dtype_item.setText(1, str(len(dtype_array)))
						for i, bone_name in enumerate(dtype_array):
							bone_item = QtWidgets.QTreeWidgetItem(dtype_item)
							bone_item.setIcon(0, get_icon(dtype))
							bone_item.setText(0, bone_name)
							bone_item.setText(1, str(i))
			header = self.tree.header()
			header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
			self.game_choice.entry.setText(self.manis_file.game)
			self.stream_entry.setText(self.manis_file.stream)
			logging.info(f"Loaded GUI in {time.time() - start_time:.2f} seconds")
			self.set_progress_message("Operation completed!")
		except:
			self.handle_error("GUI update failed, see log!")
		self.tree.itemChanged.connect(self.edit_handle)

	def save(self, filepath) -> None:
		"""Write the bundle, choosing the path that does not destroy it.

		Two writers, and picking the wrong one is silent data loss:

		* an ACL edit is a byte patch against the original file, so it is written
		  by copying those bytes out verbatim;
		* ``ManisFile.save`` re-serialises from the parsed structs, which cannot
		  reproduce a compressed clip - every ACL blob would be lost.

		So a raw patch is written raw, and a structural save over a bundle that
		holds ACL clips asks first.
		"""
		try:
			issues = manis_inspect.verify_bundle(
				self.manis_file, self.raw_data or None,
				len(self.raw_data) if self.raw_data else None)
			errors = [i for i in issues if i.severity == manis_inspect.ERROR]
			if errors:
				listed = "\n".join(str(issue) for issue in errors[:8])
				if not self.ask_yes_no(
						"Verification found errors",
						f"{listed}\n\nSave anyway?"):
					return

			if self.raw_modified:
				with open(filepath, "wb") as stream:
					stream.write(self.raw_data)
				self.raw_path = filepath
				self.set_file_modified(False)
				self.set_progress_message(
					f"Saved ACL-patched {os.path.basename(filepath)} "
					f"({len(self.raw_data):,} bytes, byte-identical apart from the patch)")
				return

			acl_clips = sum(
				1 for info in self.manis_file.mani_infos
				if bool(getattr(getattr(info, "dtype", None), "compression", 0)))
			if acl_clips and not self.ask_yes_no(
					"This will discard compressed animation data",
					f"{acl_clips} clip(s) in this bundle are compressed. Saving through "
					"the MANIS writer re-serialises from the parsed structs and cannot "
					"reproduce a compressed clip, so their animation data will be lost.\n\n"
					"Use Edit ACL Value for in-place changes instead.\n\nSave anyway?"):
				return

			self.manis_file.stream = self.stream_entry.text()
			self.manis_file.save(filepath)
			self.set_file_modified(False)
			self.set_progress_message(f"Saved {self.manis_file.name}")
		except:
			self.handle_error("Saving MANIS failed, see log!")

	def ask_yes_no(self, title, body) -> bool:
		box = QtWidgets.QMessageBox(self)
		box.setWindowTitle(title)
		box.setText(title)
		box.setInformativeText(body)
		box.setIcon(QtWidgets.QMessageBox.Warning)
		box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
		box.setDefaultButton(QtWidgets.QMessageBox.No)
		return box.exec() == QtWidgets.QMessageBox.Yes

	def resize_popup(self):
		dialog = window.ResizeManisDialog(self, "Resize MS2 amd MANIS")
		if dialog.exec():
			self.run_in_threadpool(self.resize_manis, (), dialog.folder, dialog.factor)

	@classmethod
	def resize_manis(cls, in_folder, fac=1.0):
		"""Change size by fac for all manis and wsm files in folder"""
		# create output folder
		out_folder = os.path.join(in_folder, "resized")
		os.makedirs(out_folder, exist_ok=True)
		manis = ManisFile()
		ms2 = Ms2File()
		for filename in os.listdir(in_folder):
			file_path = os.path.join(in_folder, filename)
			out_path = os.path.join(out_folder, filename)

			if filename.endswith(".manis"):
				manis.load(file_path)
				if manis.mani_version >= 282:
					logging.warning(f"Unsupported MANI version {manis.mani_version}")
					continue
				manis.resize(fac)
				manis.save(out_path)
			elif filename.endswith(".wsm"):
				wsm = WsmHeader.from_xml_file(file_path, manis.context)
				for vec in wsm.locs.data:
					vec.x *= fac
					vec.y *= fac
					vec.z *= fac
				with WsmHeader.to_xml_file(wsm, out_path):
					pass
			elif filename.endswith(".ms2"):
				ms2.load(file_path, read_editable=True)
				ms2.resize(fac)
				ms2.save(out_path)
		logging.info(f"Finished resizing from {in_folder} to {out_folder}")


if __name__ == '__main__':

	if len(sys.argv) == 1:
		startup(MainWindow, GuiOptions(
		log_name="manis_tool",
		size=(900, 600),
		check_update=False  # Check update happens at top now
		))
	else:
		logging_setup("manis_tool")
		parser = argparse.ArgumentParser()
		parser.add_argument('dir', nargs='?', help='Folder containing all ms2, manis and wsm files')
		parser.add_argument('fac', nargs='?', default=1.0, type=float, help='Scale factor to scale by, as used in Blender')
		args = parser.parse_args()
		MainWindow.resize_manis(args.dir, args.fac)