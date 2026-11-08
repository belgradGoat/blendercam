"""Fabex 'brep/operators.py'

Thin Blender operators over the B-rep functions (the same code the API calls).
"""

from math import degrees, radians

import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from bpy.types import Operator

from ..exception import CamException
from . import bridge, curves

FACE_ID_ATTR = "brep_face_id"  # BRepKit's per-polygon face id on the display mesh


def selected_brep_faces(ob) -> list | None:
    """B-rep face ids of the selected polygons (None if none are selected)."""
    me = ob.data
    if ob.mode == "EDIT":  # attribute arrays are empty while editing: read the BMesh
        bm = bmesh.from_edit_mesh(me)
        layer = bm.faces.layers.int.get(FACE_ID_ATTR)
        if layer is None:
            return None
        ids = {f[layer] for f in bm.faces if f.select}
        return sorted(int(i) for i in ids) or None
    attr = me.attributes.get(FACE_ID_ATTR)
    if attr is None or attr.domain != "FACE" or not len(me.polygons):
        return None
    sel = np.zeros(len(me.polygons), dtype=bool)
    me.polygons.foreach_get("select", sel)
    if not sel.any():
        return None
    ids = np.zeros(len(me.polygons), dtype=np.int32)
    attr.data.foreach_get("value", ids)
    return sorted(int(i) for i in np.unique(ids[sel]))


class CamBrepExtractEdges(Operator):
    """Make curves from the exact edges, face loops or holes of a B-rep object - for
    Profile, Pocket and Drill operations"""

    bl_idname = "object.cam_brep_extract_edges"
    bl_label = "B-rep Edges to Curves"
    bl_options = {"REGISTER", "UNDO"}

    kind: EnumProperty(
        name="Edges",
        items=(
            ("CREASES", "Sharp Edges", "Edges where faces meet at an angle (no tangent seams)"),
            ("ALL", "All Edges", "Every edge of the B-rep"),
            ("NAKED", "Open Edges", "Edges used by one face only (sheet boundaries)"),
            ("OUTER_LOOPS", "Face Outlines", "The outer boundary of each face"),
            ("HOLES", "Holes", "Inner loops of faces: drilled holes, pockets, slots"),
        ),
        default="CREASES",
    )
    project: EnumProperty(
        name="Flatten",
        items=(
            ("NONE", "3D", "Keep the curves where they are"),
            ("TOP", "To Top", "Flatten each curve to its highest Z (holes: their top)"),
            (
                "BOTTOM",
                "To Bottom",
                "Flatten each curve to its lowest Z (holes: their bottom, so a Drill "
                "operation with Max Depth from Object drills each to its own depth)",
            ),
        ),
        default="NONE",
    )
    tolerance: FloatProperty(
        name="Tolerance",
        description="Max distance between the curves and the exact B-rep edges",
        default=0.00001,
        min=0.0000001,
        max=0.01,
        precision=5,
        unit="LENGTH",
    )
    crease_angle: FloatProperty(
        name="Sharp Angle",
        description="Faces meeting at a larger angle make a sharp edge",
        default=radians(30.0),
        min=radians(1.0),
        max=radians(89.0),
        subtype="ANGLE",
    )
    separate: BoolProperty(
        name="Separate Objects",
        description="One curve object per loop / chain instead of one object",
        default=False,
    )
    selected_faces: BoolProperty(
        name="Selected Faces Only",
        description="In Edit Mode, only the selected faces' edges, outlines or holes",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        return bridge.is_brep(context.active_object)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        props = self.properties
        layout.prop(props, "kind")
        if props.kind == "CREASES":
            layout.prop(props, "crease_angle")
        layout.prop(props, "project")
        layout.prop(props, "tolerance")
        layout.prop(props, "separate")
        ob = context.active_object
        if ob is not None and ob.mode == "EDIT":
            layout.prop(props, "selected_faces")

    def execute(self, context):
        ob = context.active_object
        faces = selected_brep_faces(ob) if self.selected_faces and ob.mode == "EDIT" else None
        try:
            objects, es = curves.extract_curves(
                ob,
                kind=self.kind.lower(),
                tolerance=self.tolerance,
                faces=faces,
                project=self.project.lower(),
                crease_angle=degrees(self.crease_angle),
                separate=self.separate,
            )
        except CamException as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        for w in es.warnings:
            self.report({"WARNING"}, w)
        if not objects:
            self.report({"WARNING"}, "No matching edges found")
            return {"CANCELLED"}
        if ob.mode == "OBJECT":
            for other in context.selected_objects:
                other.select_set(False)
            for new in objects:
                new.select_set(True)
            context.view_layer.objects.active = objects[0]
        names = ", ".join(o.name for o in objects[:3]) + ("..." if len(objects) > 3 else "")
        self.report({"INFO"}, f"{len(es.polylines)} curve(s) in {names}")
        return {"FINISHED"}
