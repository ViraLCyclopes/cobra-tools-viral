"""JWE3 chunk culling cones and chunk caps.

Vanilla JWE3 chunks cap at exactly 64 verts / 64 tris and store backface-culling
cones that reproduce meshoptimizer's computeClusterBounds (Austroraptor juvenile
LOD0: 99.3% same cone/no-cone decision, the misses at its 0.1 threshold; the rest
is position quantization). Cobra used to write axis 0 / cutoff 1 (never cull)."""
import math
import os

import numpy as np
import pytest

from generated.formats.ms2.structs import ChunkedMesh as chunked_module
from plugin.modules_export.mesh_chunks import ChunkedMesh as Chunker, SOFT_MAX_TRIS, SOFT_MAX_VERTS

GAME = r"C:\Program Files (x86)\Steam\steamapps\common\Jurassic World Evolution 3\Win64\ovldata"
AUSTRO = os.path.join(GAME, r"ContentPDLC1\Dinosaurs\Land\Austroraptor\Juvenile\Austroraptor_Juvenile.ovl")


def quad(z=0.0):
    pos = np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]], dtype=np.float32)
    return np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32), pos


def test_flat_patch_gets_a_tight_cone_along_its_normal():
    tris, pos = quad()
    b = chunked_module.cluster_bounds(tris, pos)
    assert np.allclose(b["cone_axis"], [0, 0, 1], atol=1e-6)
    assert b["cone_cutoff"] == pytest.approx(0.0, abs=1e-6)


def test_closed_box_is_never_culled():
    pos = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float32)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = np.array([i for a, b, c, d in faces for i in (a, b, c, a, c, d)], dtype=np.uint32)
    b = chunked_module.cluster_bounds(tris, pos)
    assert not np.any(b["cone_axis"]) and b["cone_cutoff"] == 1.0


def test_zero_area_chunk_is_all_zero_like_vanilla_fins():
    pos = np.zeros((3, 3), dtype=np.float32)
    b = chunked_module.cluster_bounds(np.array([0, 1, 2], dtype=np.uint32), pos)
    assert not np.any(b["cone_axis"]) and b["cone_cutoff"] == 0.0


def test_missing_library_leaves_cones_alone(monkeypatch):
    monkeypatch.setattr(chunked_module, "_meshopt_bounds", False)
    tris, pos = quad()
    assert chunked_module.cluster_bounds(tris, pos) is None


def test_chunker_respects_vanilla_caps_and_loses_no_triangle():
    n = 60
    verts = [(x, y) for y in range(n) for x in range(n)]
    tris = []
    for y in range(n - 1):
        for x in range(n - 1):
            a = y * n + x
            tris += [(a, a + 1, a + n + 1), (a, a + n + 1, a + n)]
    out_chunks, out_verts = [], []
    cm = Chunker(tris, verts, out_chunks, out_verts, -1, False)
    cm.partition()
    assert sum(len(t) for _, t in out_chunks) == len(tris)
    for _, chunk_tris in out_chunks:
        assert len(chunk_tris) <= SOFT_MAX_TRIS == 64
        assert max(max(t) for t in chunk_tris) + 1 <= SOFT_MAX_VERTS == 64


def grid_mesh(n):
    verts = [((float(x), float(y), 0.0),) for y in range(n) for x in range(n)]
    tris = []
    for y in range(n - 1):
        for x in range(n - 1):
            a = y * n + x
            tris += [(a, a + 1, a + n + 1), (a, a + n + 1, a + n)]
    return tris, verts


def test_meshlet_chunks_keep_every_triangle_and_point_at_the_same_vertices():
    from plugin.modules_export.mesh_chunks import partition_meshlets
    tris, verts = grid_mesh(60)
    out_chunks, out_verts = [], []
    partition_meshlets(tris, verts, out_chunks, out_verts, 7)
    assert sum(len(t) for _, t in out_chunks) == len(tris)
    seen, offset = set(), 0
    for bone, local in out_chunks:
        assert bone == 7
        n = max(max(t) for t in local) + 1
        assert len(local) <= 64 and n <= 64
        for t in local:
            corners = tuple(sorted(out_verts[offset + i][0] for i in t))
            seen.add(corners)
        offset += n
    assert offset == len(out_verts)
    assert seen == {tuple(sorted(verts[i][0] for i in t)) for t in tris}


def test_meshlet_chunks_pack_small_separate_parts_together():
    from plugin.modules_export.mesh_chunks import partition_meshlets
    verts, tris = [], []
    for part in range(40):
        base = len(verts)
        verts += [((part * 3.0 + dx, dy, 0.0),) for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))]
        tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    greedy_chunks, greedy_verts = [], []
    Chunker(list(tris), verts, greedy_chunks, greedy_verts, -1, False).partition()
    meshlet_chunks, meshlet_verts = [], []
    partition_meshlets(tris, verts, meshlet_chunks, meshlet_verts, -1)
    assert len(greedy_chunks) == 40
    assert len(meshlet_chunks) <= 3


@pytest.mark.skipif(not os.path.isfile(AUSTRO), reason="JWE3 Austroraptor_Juvenile not installed")
def test_update_chunk_bounds_reproduces_vanilla(tmp_path):
    import subprocess, sys
    from generated.formats.ms2 import Ms2File
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "ovl_tool_cmd.py", "extract", AUSTRO, "-o", str(tmp_path),
                    "-g", "Jurassic World Evolution 3", "--type", ".ms2"], cwd=repo, check=True,
                   capture_output=True)
    ms2 = Ms2File()
    ms2.load(str(tmp_path / "models.ms2"), read_editable=True)

    def kind(axis, cut):
        return ("empty" if cut == 0 else "wide") if not np.any(axis) else "cone"

    total = same = 0
    errors = []
    for info in ms2.model_infos:
        for ob in info.model.lods[0].objects:
            m = ob.mesh
            raw = m.vertices.copy()
            # decoded positions -> the game frame the exporter writes in (fitted on 4,694 bounds)
            m.vertices[:, 0], m.vertices[:, 1], m.vertices[:, 2] = -raw[:, 0], raw[:, 2], -raw[:, 1]
            m.cull_cones = True
            for tc, vc in zip(m.tri_chunks, m.vert_chunks):
                if not len(tc.tri_indices):
                    continue
                s_axis = np.array([tc.cone_axis.x, tc.cone_axis.y, tc.cone_axis.z])
                s_cut = float(tc.cone_cutoff)
                tc.tri_indices = (np.asarray(tc.tri_indices, dtype=np.int64) - int(tc.value_min)).astype(np.uint8)
                # wipe Frontier's values to cobra's old default, so only a recomputation can pass
                tc.cone_apex.set((0.0, 0.0, 0.0))
                tc.cone_axis.set((0.0, 0.0, 0.0))
                tc.cone_cutoff = 1.0
                m.update_chunk_bounds(tc, vc)
                axis = np.array([tc.cone_axis.x, tc.cone_axis.y, tc.cone_axis.z])
                k, sk = kind(axis, float(tc.cone_cutoff)), kind(s_axis, s_cut)
                total += 1
                same += k == sk
                if k == sk == "cone":
                    errors.append(math.degrees(math.acos(min(1.0, float(axis @ s_axis)))))
    assert total > 4000, "read no chunks - the test would pass vacuously"
    assert same / total > 0.99
    assert np.median(errors) < 3.0
