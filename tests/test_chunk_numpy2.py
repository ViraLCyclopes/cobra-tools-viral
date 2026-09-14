"""Run with system Python and Blender's NumPy 2 Python environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generated.formats.ms2 import Ms2File, set_game
from generated.formats.ms2.structs.MeshDataWrap import MeshDataWrap
from generated.formats.ms2.enums.MeshFormat import MeshFormat
import numpy as np

m = Ms2File(); set_game(m.context, 'Jurassic World Evolution 3')
mesh = MeshDataWrap(m.context).mesh
mesh.mesh_format = MeshFormat.SEPARATE
mesh.shell_index = 0; mesh.shell_count = 0
mesh.material_effects = 0
mesh.pack_base = 512; mesh.precision = 512
# Many small legal chunks still need cumulative offsets wider than uint8.
mesh.tris = [(-1, [(0, 1, 59)]) for _ in range(8)]
offset = 0
for chunk in mesh.vert_chunks:
    assert type(chunk.vertex_count) is int
    assert chunk.vertex_count == 60
    offset += chunk.vertex_count
assert offset == 480
print('PASS: NumPy',np.__version__,'chunk counts and offsets stay Python ints; total',offset)
