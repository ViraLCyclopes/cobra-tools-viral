"""Read, transform and write the world-space motion (.wsm) XML cobra extracts.

`.wsm` is the baked trajectory the game uses to PLACE an actor - where it stands
and which way it faces, per frame. Frontier only bakes one where an animation must
land on an exact spot relative to something else, which is why a species ships 146
of them and none are locomotion:

    fightfinish 124, fightattack 8, huntkill 6, social 3, hatcheryexit 2,
    ambush 2, vehicleattack 1     (measured on Deinosuchus/SarcoViral)

Walking has no such constraint - the navigation controller owns it - which is why
scaling a walk's motion track got corrected in game.

The format is refreshingly plain:

    <WsmHeader duration="17.2666" frame_count="519" game="...">
        <unknowns>0.0 0.0 0.0 -0.0 0.0 -0.0 1.0 0.0</unknowns>
        <locs>  <vector3 x=".." y=".." z=".." /> x frame_count  </locs>
        <quats> <vector4 x=".." y=".." z=".." w=".." /> x frame_count </quats>
    </WsmHeader>

`WsmLoader` is a `MemStructLoader`, which has extract/collect/create, so an edited
file goes back in through the normal inject path.

`unknowns` is UNDECODED. The values look like a position followed by a quaternion
(0,0,0 then -0,0,-0,1), so probably an origin or reference pose. It is carried
through verbatim and never touched.

NOTHING HERE IS GAME-VERIFIED. The schema, the loader's create path and the XML
round trip are confirmed; no modified .wsm has been injected and watched to play.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# .wsm timing is exactly 30 fps - NOT the MANIS sample rate of 30.0003. Verified
# across the shipped files: duration == (frame_count - 1) / 30.0 to 8 decimals.
WSM_FPS = 30.0


@dataclass
class Wsm:
	duration: float
	frame_count: int
	game: str
	unknowns: str                      # verbatim, undecoded
	locs: list = field(default_factory=list)    # [[x, y, z], ...]
	quats: list = field(default_factory=list)   # [[x, y, z, w], ...]
	source: Path = None

	def problems(self):
		"""Consistency checks that are silent in game if wrong."""
		out = []
		if len(self.locs) != len(self.quats):
			out.append(f"locs ({len(self.locs)}) and quats ({len(self.quats)}) differ")
		if self.frame_count != len(self.locs):
			out.append(f"frame_count is {self.frame_count} but there are {len(self.locs)} locs")
		if self.duration <= 0:
			out.append(f"duration is {self.duration}")
		for i, q in enumerate(self.quats):
			length = math.sqrt(sum(c * c for c in q))
			if abs(length - 1.0) > 1e-3:
				out.append(f"quat {i} is not unit length ({length:.4f})")
				break
		return out

	def path_length(self):
		"""Total ground distance travelled, in metres."""
		total = 0.0
		for a, b in zip(self.locs, self.locs[1:]):
			total += math.dist(a, b)
		return total

	def bounds(self):
		"""(min_x, max_x, min_z, max_z) for a top-down view."""
		if not self.locs:
			return (0.0, 0.0, 0.0, 0.0)
		xs = [p[0] for p in self.locs]
		zs = [p[2] for p in self.locs]
		return (min(xs), max(xs), min(zs), max(zs))


def _fmt(value: float) -> str:
	"""Match cobra's own float formatting so an untouched round trip is identical."""
	return repr(float(value))


def load_wsm(path) -> Wsm:
	path = Path(path)
	root = ET.parse(path).getroot()
	if root.tag != "WsmHeader":
		raise ValueError(f"{path.name} is not a WsmHeader (root is <{root.tag}>)")
	locs, quats = [], []
	unknowns = ""
	for child in root:
		if child.tag == "unknowns":
			unknowns = (child.text or "").strip()
		elif child.tag == "locs":
			locs = [[float(v.get("x")), float(v.get("y")), float(v.get("z"))]
					for v in child]
		elif child.tag == "quats":
			quats = [[float(v.get("x")), float(v.get("y")),
					  float(v.get("z")), float(v.get("w"))] for v in child]
	return Wsm(duration=float(root.get("duration")),
			   frame_count=int(root.get("frame_count")),
			   game=root.get("game", "Jurassic World Evolution 3"),
			   unknowns=unknowns, locs=locs, quats=quats, source=path)


def save_wsm(wsm: Wsm, path) -> Path:
	"""Write the XML back in cobra's exact layout (tabs, attribute order, trailing \\n)."""
	path = Path(path)
	lines = [f'<WsmHeader duration="{_fmt(wsm.duration)}" '
			 f'frame_count="{int(wsm.frame_count)}" game="{wsm.game}">',
			 f'\t<unknowns>{wsm.unknowns}</unknowns>',
			 '\t<locs>']
	for x, y, z in wsm.locs:
		lines.append(f'\t\t<vector3 x="{_fmt(x)}" y="{_fmt(y)}" z="{_fmt(z)}" />')
	lines.append('\t</locs>')
	lines.append('\t<quats>')
	for x, y, z, w in wsm.quats:
		lines.append(f'\t\t<vector4 x="{_fmt(x)}" y="{_fmt(y)}" '
					 f'z="{_fmt(z)}" w="{_fmt(w)}" />')
	lines.append('\t</quats>')
	lines.append('</WsmHeader>')
	# newline="" - the files are LF, and on Windows the default would translate
	# every \n to \r\n, which silently breaks a byte-identical round trip
	with open(path, "w", encoding="utf-8", newline="") as stream:
		stream.write("\n".join(lines) + "\n")
	return path


# ---------------------------------------------------------------- transforms

def translate(wsm: Wsm, dx=0.0, dy=0.0, dz=0.0):
	"""Move the whole path. Rotation is untouched."""
	wsm.locs = [[x + dx, y + dy, z + dz] for x, y, z in wsm.locs]


def rotate_y(wsm: Wsm, degrees: float, about=None):
	"""Rotate the path and every heading about a vertical axis.

	Positions and quaternions must turn together or the actor walks its new path
	still facing the old way.
	"""
	radians = math.radians(degrees)
	cos, sin = math.cos(radians), math.sin(radians)
	cx, cz = about if about else (0.0, 0.0)
	wsm.locs = [[cx + (x - cx) * cos - (z - cz) * sin, y,
				 cz + (x - cx) * sin + (z - cz) * cos] for x, y, z in wsm.locs]
	# a yaw of `degrees` about Y, applied on the left: q_new = q_yaw * q_old
	half = radians / 2.0
	qy, qw = math.sin(half), math.cos(half)
	turned = []
	for x, y, z, w in wsm.quats:
		turned.append([
			qw * x + qy * z,
			qw * y + qy * w,
			qw * z - qy * x,
			qw * w - qy * y,
		])
	wsm.quats = turned


def scale(wsm: Wsm, factor: float, about=None):
	"""Scale the path about a point. Headings are unchanged - only distance is."""
	cx, cz = about if about else (0.0, 0.0)
	wsm.locs = [[cx + (x - cx) * factor, y, cz + (z - cz) * factor]
				for x, y, z in wsm.locs]


def reverse(wsm: Wsm):
	"""Play the trajectory backwards."""
	wsm.locs = list(reversed(wsm.locs))
	wsm.quats = list(reversed(wsm.quats))


def trim(wsm: Wsm, first: int, last: int, sample_rate: float = WSM_FPS):
	"""Keep frames [first, last] inclusive and fix frame_count and duration.

	frame_count must equal len(locs) or the runtime reads past the end, and the
	duration has to follow the new length at the SAME sample rate - editing the
	rate itself is what made every animal in the park spasm.

	NOTE the rate: .wsm uses exactly 30.0, NOT the MANIS rate of 30.0003. Measured
	across the shipped files, duration == (frame_count - 1) / 30.0 to 8 decimals
	every time (187 -> 6.19999981, 12 -> 0.36666599, 52 -> 1.70000005).
	"""
	if not 0 <= first <= last < len(wsm.locs):
		raise ValueError(f"range {first}..{last} outside 0..{len(wsm.locs) - 1}")
	wsm.locs = wsm.locs[first:last + 1]
	wsm.quats = wsm.quats[first:last + 1]
	wsm.frame_count = len(wsm.locs)
	wsm.duration = (wsm.frame_count - 1) / sample_rate if wsm.frame_count > 1 else 0.0


def heading_quat(dx: float, dz: float):
	"""Quaternion facing along (dx, dz), as a pure yaw about Y.

	Inverse of the forward vector the viewer draws: for q = (0, sy, 0, cw),
	forward = (2*cw*sy, 1 - 2*sy^2) = (sin yaw, cos yaw), so yaw = atan2(dx, dz).
	"""
	yaw = math.atan2(dx, dz)
	half = yaw / 2.0
	return [0.0, math.sin(half), 0.0, math.cos(half)]


def resample_path(wsm: Wsm, points, keep_headings: bool = False,
				  keep_height: bool = True):
	"""Replace the trajectory with `points`, resampled to the SAME frame count.

	`points` is a polyline of (x, z) in world metres - what a drawn stroke gives.
	Sampling is by ARC LENGTH so the animal moves at an even pace along the new
	shape rather than bunching wherever the mouse happened to slow down.

	frame_count and duration are deliberately NOT touched: redrawing where an
	animation goes must not change how long it takes, or it desynchronises from the
	clip that plays alongside it. Use `trim` for length.

	Headings are rebuilt to follow the tangent unless `keep_headings`, because a path
	redrawn without turning the animal leaves it walking sideways.
	"""
	if len(points) < 2:
		raise ValueError("need at least two points to draw a path")
	count = wsm.frame_count or len(wsm.locs)
	if count < 2:
		raise ValueError("this clip has fewer than two frames")

	# cumulative arc length along the drawn stroke
	spans = [0.0]
	for (x0, z0), (x1, z1) in zip(points, points[1:]):
		spans.append(spans[-1] + math.dist((x0, z0), (x1, z1)))
	total = spans[-1]
	if total <= 0:
		raise ValueError("the drawn path has zero length")

	heights = [p[1] for p in wsm.locs] if keep_height and wsm.locs else [0.0] * count
	locs, quats = [], []
	segment = 0
	for i in range(count):
		want = total * i / (count - 1)
		while segment < len(spans) - 2 and spans[segment + 1] < want:
			segment += 1
		run = spans[segment + 1] - spans[segment]
		t = 0.0 if run <= 0 else (want - spans[segment]) / run
		(x0, z0), (x1, z1) = points[segment], points[segment + 1]
		x, z = x0 + (x1 - x0) * t, z0 + (z1 - z0) * t
		y = heights[i] if i < len(heights) else 0.0
		locs.append([x, y, z])
		quats.append(None if keep_headings else [x1 - x0, z1 - z0])

	if keep_headings:
		quats = list(wsm.quats[:count]) or [[0.0, 0.0, 0.0, 1.0]] * count
	else:
		built = []
		for i, tangent in enumerate(quats):
			dx, dz = tangent
			if dx == 0.0 and dz == 0.0:
				# a degenerate segment: reuse the previous heading rather than snap
				built.append(built[-1] if built else [0.0, 0.0, 0.0, 1.0])
			else:
				built.append(heading_quat(dx, dz))
		quats = built
	wsm.locs, wsm.quats = locs, quats


def recentre(wsm: Wsm):
	"""Put the first frame at the origin, keeping the shape of the path."""
	if not wsm.locs:
		return
	x0, _y0, z0 = wsm.locs[0]
	translate(wsm, dx=-x0, dz=-z0)
