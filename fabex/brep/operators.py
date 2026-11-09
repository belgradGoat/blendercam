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
from . import bridge, curves, fit, flow
from .bridge import FACE_ID_ATTR


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


def active_operation(context):
    """The scene's active CAM operation (None if there is none)."""
    s = context.scene
    ops = getattr(s, "cam_operations", None)
    if not ops or not 0 <= s.cam_active_operation < len(ops):
        return None
    return ops[s.cam_active_operation]


def _redraw(context) -> None:
    for window in context.window_manager.windows if context.window_manager else ():
        for area in window.screen.areas:
            area.tag_redraw()


class CamBrepFitCheck(Operator):
    """Check that the active operation's cutter fits its B-rep objects: the tightest concave
    curves, wall corners and sharp inside corners (or, without one, report the active
    B-rep object's limits)"""

    bl_idname = "object.cam_brep_fit_check"
    bl_label = "Check Cutter Fit"
    bl_options = {"REGISTER", "UNDO"}

    select: BoolProperty(
        name="Select Problem Faces",
        description="In Edit Mode, select the faces the cutter doesn't fit",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        o = active_operation(context)
        if o is not None and bridge.operation_brep_objects(o):
            return True
        return bridge.is_brep(context.active_object)

    def execute(self, context):
        o = active_operation(context)
        objects = bridge.operation_brep_objects(o) if o is not None else []
        if objects:
            reports = fit.check_operation(o, warn=False)
        else:
            ob = context.active_object
            try:
                reports = [fit.object_report(ob)]
            except CamException as e:
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
        problem_faces = {}
        for rep in reports:
            if rep.get("error"):
                self.report({"WARNING"}, f"{rep['object']}: {rep['error']['message']}")
                continue
            for line in rep["messages"]:
                self.report({"WARNING"}, line)
            if rep["fits"] is True and not rep["messages"]:
                self.report({"INFO"}, f"{rep['object']}: {fit.summary_text(rep)}")
            elif rep["fits"] is None:
                self.report({"INFO"}, f"{rep['object']}: {fit.summary_text(rep)}")
            problem_faces[rep["object"]] = [p["face"] for p in rep.get("problems", [])]
        ob = context.active_object
        if self.select and ob is not None and ob.mode == "EDIT" and ob.name in problem_faces:
            fit.select_faces(ob, problem_faces[ob.name])
        _redraw(context)
        return {"FINISHED"}


def _flow_object(context, o):
    """The B-rep object of operation `o` whose faces to use: the active one if it is a
    source, else the only source (None if ambiguous)."""
    objects = bridge.operation_brep_objects(o)
    active = context.active_object
    if active is not None and active in objects:
        return active
    return objects[0] if len(objects) == 1 else None


class CamBrepFlowFaces(Operator):
    """Machine only the faces selected in Edit Mode with the active Flowline operation
    (or, with Clear, every face again)"""

    bl_idname = "object.cam_brep_flow_faces"
    bl_label = "Use Selected Faces"
    bl_options = {"REGISTER", "UNDO"}

    clear: BoolProperty(name="Clear", description="Machine every face again", default=False)

    @classmethod
    def poll(cls, context):
        o = active_operation(context)
        return o is not None and bool(bridge.operation_brep_objects(o))

    def execute(self, context):
        o = active_operation(context)
        if self.clear:
            o.brep.flow_faces = ""
            self.report({"INFO"}, "Flowline: every face")
            _redraw(context)
            return {"FINISHED"}
        ob = _flow_object(context, o)
        if ob is None:
            self.report({"ERROR"}, "Make one of the operation's B-rep objects active")
            return {"CANCELLED"}
        faces = selected_brep_faces(ob)
        if not faces:
            self.report({"ERROR"}, f"Select faces of '{ob.name}' in Edit Mode first")
            return {"CANCELLED"}
        o.brep.flow_faces = ", ".join(map(str, faces))
        self.report({"INFO"}, f"Flowline: {len(faces)} face(s) of {ob.name}")
        _redraw(context)
        return {"FINISHED"}


class CamBrepFlowlinePreview(Operator):
    """Show the active operation's flowline passes as curves on its B-rep objects (before
    the cutter is dropped onto the model)"""

    bl_idname = "object.cam_brep_flowline_preview"
    bl_label = "Preview Flowline"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        o = active_operation(context)
        return o is not None and bool(bridge.operation_brep_objects(o))

    def execute(self, context):
        o = active_operation(context)
        try:
            results = flow.operation_preview(o)
        except CamException as e:  # FabexAPIError / BrepError too
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        lines = []
        for ob, preview, fl in results:
            s = fl.stats()
            lines.append(
                f"{ob.name}: {s['passes']} passes, {s['points']:,} points"
                + ("" if preview is not None else " (nothing to show)")
            )
        self.report({"INFO"}, "; ".join(lines))
        _redraw(context)
        return {"FINISHED"}
