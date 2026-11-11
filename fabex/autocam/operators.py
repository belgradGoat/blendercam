"""Thin Blender operators for the Auto CAM panel over the API's plan_cam / apply_plan /
verify_program (the same code an agent calls)."""

import bpy
from bpy.types import Operator

from ..exception import FabexAPIError


def _fx():
    from .. import api

    return api


def _part(context):
    """The panel's part, else the active object if it is a B-rep (read only: polls)."""
    ob = context.scene.cam_autocam.part
    if ob is None:
        ob = context.active_object
        if ob is None or ob.type != "MESH" or "brepkit_data" not in ob:
            return None
    return ob


def _fail(op, e) -> set:
    op.report({"ERROR"}, f"{getattr(e, 'code', 'ERROR')}: {e}")
    return {"CANCELLED"}


class CamAutocamPlan(Operator):
    """Plan machining the top of the B-rep part in one setup: stock, tools, operations,
    cutting data. Nothing in the scene changes until Apply"""

    bl_idname = "object.cam_autocam_plan"
    bl_label = "Plan"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return _part(context) is not None

    def execute(self, context):
        a = context.scene.cam_autocam
        if a.part is None:
            a.part = _part(context)
        options = {
            "allowance_top": a.allowance_top,
            "allowance_side": a.allowance_side,
            "grip": a.grip,
            "origin": a.origin,
        }
        if a.program_number:
            options["program_number"] = a.program_number
        try:
            plan = _fx().plan_cam(a.part.name, material=a.material, options=options)
        except FabexAPIError as e:
            return _fail(self, e)
        left = len(plan["coverage"]["uncovered"])
        self.report(
            {"INFO"},
            f"{len(plan['operations'])} operations, {len(plan['tools'])} tools"
            + (f", {left} face(s) left for later" if left else ""),
        )
        return {"FINISHED"}


class CamAutocamApply(Operator):
    """Make the plan: move the part to the plan's placement, create the operations and a
    chain, compute every path, write the FANUC program and verify it (blocks Blender while
    it computes)"""

    bl_idname = "object.cam_autocam_apply"
    bl_label = "Apply"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.get("fabex_autocam_plan"))

    def execute(self, context):
        try:
            r = _fx().apply_plan()
        except FabexAPIError as e:
            return _fail(self, e)
        if r["ok"]:
            p = r["program"] or {}
            self.report({"INFO"}, f"{p.get('file', '')}: {p.get('minutes', 0):.1f} min, verified")
        else:
            self.report({"WARNING"}, "Applied with problems: see the Auto CAM panel")
        return {"FINISHED"}


class CamAutocamVerify(Operator):
    """Simulate the applied plan's program on the stock and compare it with the part"""

    bl_idname = "object.cam_autocam_verify"
    bl_label = "Verify"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        from .apply import stored_plan

        plan = stored_plan(context.scene) or {}
        return bool(plan.get("applied", {}).get("program"))

    def execute(self, context):
        from .apply import stored_plan, store_plan

        try:
            rep = _fx().verify_program()
        except FabexAPIError as e:
            return _fail(self, e)
        plan = stored_plan(context.scene)
        plan["verify"] = {k: rep[k] for k in ("ok", "gouge", "leftover", "collisions", "plunges")}
        store_plan(context.scene, plan)
        bad = len(rep["leftover"].get("unexpected", []))
        self.report(
            {"INFO"} if rep["ok"] else {"WARNING"},
            f"gouge {rep['gouge']['max_mm']} mm, {bad} face(s) with unplanned leftover, "
            f"{len(rep['collisions'])} collision(s)",
        )
        return {"FINISHED"}


classes = (CamAutocamPlan, CamAutocamApply, CamAutocamVerify)
