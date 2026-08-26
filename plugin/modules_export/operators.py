import os

import bpy.utils.previews
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ExportHelper

from plugin import export_ms2, export_spl, export_manis, export_banis, export_fgm
from plugin.utils.operators import BaseOp


class ExportOp(BaseOp, ExportHelper):

    @property
    def kwargs(self) -> dict:
        return self.as_keywords(ignore=("axis_forward", "axis_up", "filter_glob", "check_existing"))


class ExportFgm(ExportOp):
    """Export to FGM file format (.fgm)"""
    bl_idname = "export_scene.cobra_fgm"
    bl_label = 'Export FGM'
    filename_ext = ".fgm"
    target = export_fgm.save
    filter_glob: StringProperty(default="*.fgm", options={'HIDDEN'})

    def invoke(self, context, _event):
        if not self.filepath:
            try:
                material = bpy.context.active_object.active_material
                self.filepath = material.name + self.filename_ext
            except:
                self.filepath = "None" + self.filename_ext
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class ExportMS2(ExportOp):
    """Export to MS2 file format (.MS2)"""
    bl_idname = "export_scene.cobra_ms2"
    bl_label = 'Export MS2'
    filename_ext = ".ms2"
    target = export_ms2.save
    filter_glob: StringProperty(default="*.ms2", options={'HIDDEN'})
    apply_transforms: BoolProperty(name="Apply Transforms",
                                   description="Automatically applies object transforms to meshes", default=False)
    update_rig: BoolProperty(name="Update Rigs", description="Updates rigs (bones, physics joints, hitchecks) from blender - may break skeletons",
                             default=False)
    use_stock_normals_tangents: BoolProperty(
        name="Use Original Normals & Tangents",
        description="Ignores the actual geometry and uses original normals and tangents stored as mesh attributes on import. Use case: if fur depends on custom normals",
        default=False)

    def invoke(self, context, _event):
        if not self.filepath:
            self.filepath = context.scene.name + self.filename_ext
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class ExportSPL(ExportOp):
    """Export to spline file format (.spl)"""
    bl_idname = "export_scene.cobra_spl"
    bl_label = 'Export SPL'
    filename_ext = ".spl"
    target = export_spl.save
    filter_glob: StringProperty(default="*.spl", options={'HIDDEN'})


class ExportManis(ExportOp):
    """Export to Cobra animations file format (.manis)"""
    bl_idname = "export_scene.cobra_manis"
    bl_label = 'Export Manis'
    filename_ext = ".manis"
    target = export_manis.save
    filter_glob: StringProperty(default="*.manis", options={'HIDDEN'})
    per_armature: BoolProperty(
        name="Per Armature",
        description="Exports a single manis for each armature, or lumps all armatures in one manis",
        default=False)
    export_mode: EnumProperty(
        name="Group By",
        description="How actions are grouped into manis files",
        items=(
            ('SOURCE', "Source Bundles",
             "One file per bundle the actions were imported from, which is how the game "
             "ships them. The chosen file name is only used for actions with no recorded "
             "source"),
            ('SINGLE', "One File",
             "Every action into the chosen file. Only sane when the scene holds a single "
             "bundle's actions"),
            ('ACTIVE', "Active Action Only",
             "Only each armature's active action, into the chosen file"),
        ),
        default='SOURCE')
    jwe3_scale_mode: EnumProperty(
        name="JWE3 Scale",
        description="How scale channels are written for JWE3 MANIS bundles",
        items=(
            ('OMIT', "Omit (Recommended)",
             "Remove scale channels from every exported clip. Current JWE3 dtype-0 "
             "scale blocks crash dinosaur bundles"),
            ('WRITE', "Write (Experimental)",
             "Write Blender scale channels for format research. Current output is "
             "known to crash JWE3 dinosaur bundles"),
        ),
        default='OMIT')

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "export_mode")
        layout.prop(self, "per_armature")
        if context.scene.cobra.game == "Jurassic World Evolution 3":
            layout.prop(self, "jwe3_scale_mode")
            scale_box = layout.box()
            if self.jwe3_scale_mode == 'OMIT':
                scale_box.label(text="Scale will be removed from every clip.", icon='CHECKMARK')
                scale_box.label(text="Rotation, translation, and floats are preserved.")
            else:
                scale_box.label(text="Experimental scale currently crashes JWE3.", icon='ERROR')
        # show exactly what will be written, using the same bucketing the export uses
        box = layout.box()
        try:
            manis_name = os.path.basename(self.filepath) or f"untitled{self.filename_ext}"
            datas, unstamped = export_manis.collect_export_map(
                context.scene, manis_name, self.per_armature, self.export_mode)
        except Exception:
            box.label(text="Could not preview export", icon='ERROR')
            return
        if not datas:
            box.label(text="No actions to export", icon='ERROR')
            return
        total = sum(len(a) for m in datas.values() for a in m.values())
        box.label(text=f"{total} clips into {len(datas)} file(s)", icon='ANIM')
        # the side region scrolls, so list every file rather than truncating
        folder = os.path.dirname(self.filepath)
        col = box.column(align=True)
        overwrite = 0
        for name in sorted(datas):
            count = sum(len(a) for a in datas[name].values())
            exists = bool(folder) and os.path.isfile(os.path.join(folder, name))
            overwrite += exists
            col.label(text=f"{count:>4}  {name}", icon='ERROR' if exists else 'FILE_BLANK')
        if overwrite:
            # SOURCE mode writes the vanilla file names, so exporting into the folder the
            # bundles were extracted to replaces the originals
            box.label(text=f"{overwrite} file(s) already exist here", icon='ERROR')
            box.label(text="and will be OVERWRITTEN.")
        if unstamped:
            box.separator()
            box.label(text=f"{len(unstamped)} action(s) have no source bundle,",
                      icon='INFO')
            box.label(text=f"so they go to '{manis_name}'.")
            box.label(text="Re-import them to record where they came from.")


class ExportBanis(ExportOp):
    """Export to Cobra baked animations file format (.banis)"""
    bl_idname = "export_scene.cobra_banis"
    bl_label = 'Export Banis'
    filename_ext = ".banis"
    target = export_banis.save
    filter_glob: StringProperty(default="*.banis", options={'HIDDEN'})
