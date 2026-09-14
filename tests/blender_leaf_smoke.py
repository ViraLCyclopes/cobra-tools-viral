"""Run in factory-startup background Blender, never in the user's scene."""
import sys
import logging
logging.disable(logging.CRITICAL)
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bpy
import numpy as np
from mathutils import Matrix, Quaternion
from plugin.leaf_bones import export_leaf, LEAF_CLASSES
from plugin.utils.transforms import Corrector
from plugin.utils.blender_util import bone_name_for_blender
from generated.formats.ms2 import Ms2File

P = ROOT.parents[1] / 'JWE 3 Luas/Base Game/Dinosaur Files/Motiongraph Research/New Bone Track Test'
source = P / 'source/models.ms2'
m = Ms2File(); m.load(str(source), read_editable=False)
rig = m.models_reader.bone_infos[1]
data = bpy.data.armatures.new('Leaf test')
obj = bpy.data.objects.new('Leaf test', data)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
worlds = []
for i, original in enumerate(rig.bones):
    local = Quaternion((original.rot.w, original.rot.x, original.rot.y, original.rot.z)).to_matrix().to_4x4()
    local.translation = (original.loc.x, original.loc.y, original.loc.z)
    parent = int(rig.parents[i])
    world = worlds[parent] @ local if parent < i else local
    worlds.append(world)
    bone = data.edit_bones.new(bone_name_for_blender(str(original.name)))
    bone.length = 0.1
    bone.matrix = Corrector(False).to_blender(world)
    if parent < i:
        bone.parent = data.edit_bones[parent]
for cls in LEAF_CLASSES:
    bpy.utils.register_class(cls)
data.edit_bones.active = data.edit_bones['def_c_head_joint']
assert bpy.ops.cobra.leaf_add() == {'FINISHED'}
leaf = data.edit_bones.active
leaf.name = 'VLVFX_muzzle_joint'
local = Matrix.Identity(4); local.translation = (-2, -.3, 0)
leaf.matrix = Corrector(False).to_blender(worlds[129] @ local)
bpy.ops.object.mode_set(mode='OBJECT')
out = P / 'blender_leaf_test/models.ms2'
export_leaf(obj, data.bones.active, source, out)
got = Ms2File(); got.load(str(out), read_editable=False)
g = got.models_reader.bone_infos[1]
assert int(g.bone_count) == int(g.joints.bone_count) == 185
assert list(g.joints.bone_to_joint) == list(rig.joints.bone_to_joint) + [-1]
assert list(g.parents) == list(rig.parents) + [129]
assert np.allclose([g.bones[-1].loc.x,g.bones[-1].loc.y,g.bones[-1].loc.z], [-2,-.3,0], atol=1e-5)
for i in range(184):
    assert np.array_equal(rig.inverse_bind_matrices[i].data, g.inverse_bind_matrices[i].data)
assert source.read_bytes()[m.buffer_2_offset:] == out.read_bytes()[got.buffer_2_offset:]
bpy.ops.object.mode_set(mode='EDIT')
data.edit_bones.active.head.x += .25
data.edit_bones.active.tail.x += .25
bpy.ops.object.mode_set(mode='OBJECT')
moved = P / 'blender_leaf_test/moved.ms2'
export_leaf(obj, data.bones.active, out, moved)
check = Ms2File(); check.load(str(moved), read_editable=False)
assert int(check.models_reader.bone_infos[1].bone_count) == 185
assert moved.read_bytes() != out.read_bytes()
try:
    export_leaf(obj, data.bones['Target4'], source, moved)
except ValueError:
    pass
else:
    raise AssertionError('Target export should fail')
# Reparent the existing leaf to the jaw while keeping armature-space placement.
bpy.ops.object.mode_set(mode='EDIT')
leaf = data.edit_bones['VLVFX_muzzle_joint']
before = leaf.matrix.copy()
jaw = next(b for b in data.edit_bones if 'jaw' in b.name.lower())
leaf.parent = jaw
leaf.use_connect = False
leaf.matrix = before
jaw_name = jaw.name
bpy.ops.object.mode_set(mode='OBJECT')
reparented = P / 'blender_leaf_test/jaw_parent.ms2'
export_leaf(obj, data.bones['VLVFX_muzzle_joint'], moved, reparented)
after = Ms2File(); after.load(str(reparented), read_editable=False)
ar = after.models_reader.bone_infos[1]
from plugin.utils.blender_util import bone_name_for_ovl
parent_index = [str(b.name) for b in ar.bones].index(bone_name_for_ovl(jaw_name))
assert int(ar.parents[-1]) == parent_index
leaf_bone = ar.bones[-1]
local_matrix = Quaternion((leaf_bone.rot.w,leaf_bone.rot.x,leaf_bone.rot.y,leaf_bone.rot.z)).to_matrix().to_4x4()
local_matrix.translation = (leaf_bone.loc.x,leaf_bone.loc.y,leaf_bone.loc.z)
assert np.allclose(np.array(Corrector(False).to_blender(worlds[parent_index] @ local_matrix)), np.array(before), atol=1e-5)
expected_bind = np.linalg.inv(np.array(ar.inverse_bind_matrices[parent_index].data).T) @ np.array(local_matrix)
assert np.allclose(np.array(ar.inverse_bind_matrices[-1].data).T @ expected_bind, np.eye(4), atol=1e-5)
assert list(ar.joints.bone_to_joint) == list(g.joints.bone_to_joint)
assert list(ar.parents[:-1]) == list(g.parents[:-1])
for i in range(184):
    assert np.array_equal(ar.inverse_bind_matrices[i].data, g.inverse_bind_matrices[i].data)
assert reparented.read_bytes()[after.buffer_2_offset:] == source.read_bytes()[m.buffer_2_offset:]
bpy.ops.object.mode_set(mode='EDIT')
leaf = data.edit_bones['VLVFX_muzzle_joint']
rotated_matrix = leaf.matrix @ Matrix.Rotation(0.6, 4, 'Y')
leaf.matrix = rotated_matrix
bpy.ops.object.mode_set(mode='OBJECT')
rotated = P / 'blender_leaf_test/jaw_rotated.ms2'
export_leaf(obj, data.bones['VLVFX_muzzle_joint'], reparented, rotated)
rot = Ms2File(); rot.load(str(rotated), read_editable=False)
rr = rot.models_reader.bone_infos[1]
rb = rr.bones[-1]
rl = Quaternion((rb.rot.w,rb.rot.x,rb.rot.y,rb.rot.z)).to_matrix().to_4x4()
rl.translation = (rb.loc.x,rb.loc.y,rb.loc.z)
assert np.allclose(np.array(Corrector(False).to_blender(worlds[parent_index] @ rl)), np.array(rotated_matrix), atol=1e-5)
assert list(rr.joints.bone_to_joint) == list(ar.joints.bone_to_joint)
assert rotated.read_bytes()[rot.buffer_2_offset:] == source.read_bytes()[m.buffer_2_offset:]
print('LEAF TEST PASS: add, move, head-to-jaw reparent, rotation, rest pose, inverse bind, physics maps and mesh bytes')
from plugin import export_ms2
# Exercise the regular export entry point in Edit Mode, including mode restore.
bpy.ops.object.mode_set(mode='EDIT')
data.edit_bones.active = data.edit_bones['VLVFX_muzzle_joint']
integrated = P / 'blender_leaf_test/integrated.ms2'
assert export_ms2.save(None, filepath=str(integrated), export_scope='LEAF', leaf_source=str(reparented)) == {'FINISHED'}
assert obj.mode == 'EDIT'
assert integrated.read_bytes() == rotated.read_bytes()
print('INTEGRATED MS2 EXPORT PASS: same output, Edit Mode restored')
