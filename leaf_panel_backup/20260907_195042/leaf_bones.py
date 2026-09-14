"""JWE3 rest-pose leaf editing without rebuilding meshes or animation."""
from io import BytesIO
from pathlib import Path
import os
import tempfile

import bpy
import numpy as np
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from mathutils import Matrix, Quaternion
from generated.formats.ms2 import Ms2File
from generated.formats.ms2.imports import name_type_map
from plugin.utils.transforms import Corrector
from plugin.utils.blender_util import bone_name_for_ovl


def rebuild(model, tail):
    model.info.mdl_2_count = len(model.model_infos)
    model.update_names()
    model.update_buffer_0_bytes()
    model.update_buffer_2_bytes()
    model.update_buffer_1_bytes()
    stream = BytesIO()
    model.write_fields(stream, model)
    return stream.getvalue() + tail


def export_leaf(armature, bone, source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('Choose a separate output MS2; keep the source intact')
    name = bone_name_for_ovl(bone.name)
    if not name.startswith('VLVFX_') or bone.parent is None or bone.children:
        raise ValueError('Select a childless, parented VLVFX_ bone')
    raw = source.read_bytes()
    model = Ms2File()
    model.load(str(source), read_editable=False)
    if model.context.version != 55:
        raise ValueError('This leaf exporter supports JWE3 MS2 version 55 only')
    tail = raw[model.buffer_2_offset:]
    if rebuild(model, tail) != raw:
        raise ValueError('Source failed the byte-identical round-trip check')
    model = Ms2File()
    model.load(str(source), read_editable=False)
    arm_names = {bone_name_for_ovl(b.name) for b in armature.data.bones}
    candidates = [b for b in model.models_reader.bone_infos
                  if b.bones and {str(x.name) for x in b.bones} <= arm_names]
    if candidates:
        largest = max(len(b.bones) for b in candidates)
        candidates = [b for b in candidates if len(b.bones) == largest]
    if len(candidates) != 1:
        raise ValueError('Source must contain exactly one rig matching this armature')
    rig = candidates[0]
    names = [str(b.name) for b in rig.bones]
    parent = names.index(bone_name_for_ovl(bone.parent.name))
    n = int(rig.bone_count)
    joints = rig.joints
    if int(rig.joint_count) and not (int(joints.bone_count) == len(joints.bone_to_joint) == n):
        raise ValueError('Source has an inconsistent physics bone map')
    corrector = Corrector(False)
    parent_world = corrector.from_blender(bone.parent.matrix_local)
    def rest_world(index):
        original = rig.bones[index]
        matrix = Quaternion((original.rot.w, original.rot.x, original.rot.y, original.rot.z)).to_matrix().to_4x4()
        matrix.translation = (original.loc.x, original.loc.y, original.loc.z)
        ancestor = int(rig.parents[index])
        return rest_world(ancestor) @ matrix if ancestor < len(rig.bones) else matrix
    stored_parent = np.array(rest_world(parent))
    if not np.allclose(np.array(parent_world), stored_parent, atol=0.002):
        raise ValueError(f'Parent rest pose differs from source; import the matching source rig (maximum difference {np.max(np.abs(np.array(parent_world)-stored_parent)):.5f})')
    world = corrector.from_blender(bone.matrix_local)
    local = parent_world.inverted() @ world
    if not np.allclose(list(local.to_scale()), [1, 1, 1], atol=0.001):
        raise ValueError('Leaf scale must be one')
    if name in names:
        index = names.index(name)
        if index in list(rig.parents) or int(rig.parents[index]) != parent:
            raise ValueError('Existing leaf must retain its source parent and have no children')
        if int(rig.joint_count) and int(joints.bone_to_joint[index]) != -1:
            raise ValueError('Cannot edit a physics joint with the leaf exporter')
    else:
        if n >= 254 or int(rig.zeros_count):
            raise ValueError('Unsupported rig size or padding')
        index = n
        arrays = {f: list(getattr(rig, f)) for f in
                  ('parents', 'name_indices', 'enumeration', 'jwe_3_nibbles')}
        rig.bone_count = rig.name_count = rig.bind_matrix_count = rig.parents_count = rig.enum_count = n + 1
        rig.bones.append(name_type_map['Bone'](rig.context, 0, None))
        rig.bones[-1].name = name
        rig.inverse_bind_matrices.append(name_type_map['Matrix44'](rig.context, 0, None))
        for field, values in arrays.items():
            rig.reset_field(field)
            getattr(rig, field)[:len(values)] = values
        rig.parents[index] = parent
        rig.enumeration[index] = index
        rig.jwe_3_nibbles[-1] = 0
        if int(rig.joint_count):
            old_map = list(joints.bone_to_joint)
            joints.bone_count = n + 1
            joints.reset_field('bone_to_joint')
            joints.bone_to_joint[:] = old_map + [-1]
    rig.bones[index].set_bone(local)
    source_parent_bind = Matrix(np.linalg.inv(np.array(rig.inverse_bind_matrices[parent].data).T))
    rig.inverse_bind_matrices[index].set_rows((source_parent_bind @ local).inverted())
    result = rebuild(model, tail)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix='.ms2', dir=output.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(result)
        check = Ms2File()
        check.load(temporary, read_editable=False)
        if result[check.buffer_2_offset:] != tail or rebuild(check, tail) != result:
            raise ValueError('Output verification failed')
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class COBRA_OT_leaf_source(bpy.types.Operator, ImportHelper):
    bl_idname = 'cobra.leaf_source'
    bl_label = 'Choose Source MS2'
    filename_ext = '.ms2'
    filter_glob: StringProperty(default='*.ms2', options={'HIDDEN'})

    def execute(self, context):
        context.object['cobra_leaf_source'] = self.filepath
        return {'FINISHED'}


class COBRA_OT_leaf_add(bpy.types.Operator):
    bl_idname = 'cobra.leaf_add'
    bl_label = 'Add VFX Leaf at Cursor'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        armature = context.object
        if armature.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bones = armature.data.edit_bones
        parent = bones.active
        if parent is None:
            self.report({'ERROR'}, 'Select a parent bone first')
            return {'CANCELLED'}
        bone = bones.new('VLVFX_attachment_joint')
        bone.length = max(parent.length * 0.2, 0.05)
        bone.matrix = parent.matrix.copy()
        bone.translate(armature.matrix_world.inverted() @ context.scene.cursor.location - bone.head)
        bone.parent = parent
        bone.use_connect = False
        bone.use_deform = False
        for other in bones:
            other.select = other.select_head = other.select_tail = False
        bone.select = bone.select_head = bone.select_tail = True
        bones.active = bone
        return {'FINISHED'}


class COBRA_OT_leaf_export(bpy.types.Operator, ExportHelper):
    bl_idname = 'cobra.leaf_export'
    bl_label = 'Export Active VFX Leaf'
    filename_ext = '.ms2'
    filter_glob: StringProperty(default='*.ms2', options={'HIDDEN'})

    def execute(self, context):
        armature = context.object
        mode = armature.mode
        try:
            if mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            bone = armature.data.bones.active
            if bone is None:
                raise ValueError('Select a VFX leaf first')
            export_leaf(armature, bone, armature.get('cobra_leaf_source', ''), self.filepath)
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        finally:
            if armature.mode != mode:
                bpy.ops.object.mode_set(mode=mode)
        self.report({'INFO'}, 'Leaf MS2 exported; mesh bytes preserved')
        return {'FINISHED'}


class COBRA_PT_leaf(bpy.types.Panel):
    bl_label = 'VFX Leaf Bones'
    bl_idname = 'COBRA_PT_leaf'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Cobra'

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def draw(self, context):
        layout = self.layout
        layout.label(text='Move leaves in Edit Mode')
        layout.operator('cobra.leaf_add')
        layout.operator('cobra.leaf_source')
        layout.label(text=Path(context.object.get('cobra_leaf_source', '')).name or 'No source selected')
        layout.operator('cobra.leaf_export')


LEAF_CLASSES = (COBRA_OT_leaf_source, COBRA_OT_leaf_add, COBRA_OT_leaf_export, COBRA_PT_leaf)
