"""Fabex 'autocam/api.py' - the machining-centre half of the scripting / agent API.

Machine profile, tool library, chains and FANUC programs. Functions register in the same
registry as fabex/api.py, so they show up in describe() and work through call_json().
"""

import os

import bpy

from ..exception import FabexAPIError
from ..utilities.api_utils import api_function
from ..utilities.simple_utils import safe_filename
from . import fanuc_export, state
from .camcore import check as cc_check
from .camcore import machine as cc_machine
from .camcore import tools as cc_tools
from .camcore.errors import CamcoreError


def _err(e: CamcoreError) -> FabexAPIError:
    return FabexAPIError(e.code, e.message, e.details)


def _fx():
    from .. import api  # late: fabex/api.py imports this module

    return api


@api_function("machine")
def machine_profile() -> dict:
    """The machine programs are written for (FANUC Robodrill defaults until set_machine):
    travel (m), spindle_min / spindle_max (rpm), feed_max / rapid (m/min), magazine (tool
    pockets), program_memory (kB), work_offset, the Fanuc blocks used (units_block,
    safe_start, home_z, end, aicc_on / aicc_off, coolant M-codes) and h_offset (H = T +
    h_offset). Also Fabex's own machine settings it keeps in step (cam_machine) and the
    presets set_machine() knows."""
    s = bpy.context.scene
    try:
        p = state.machine(s)
    except CamcoreError as e:
        raise _err(e) from e
    m = s.cam_machine
    return {
        "profile": p.to_dict(),
        "presets": sorted(cc_machine.PRESETS),
        "cam_machine": {
            "post_processor": m.post_processor,
            "unit_system": m.unit_system,
            "feedrate_max": m.feedrate_max,
            "spindle_max": m.spindle_max,
            "working_area": list(m.working_area),
        },
        "synced": m.post_processor == "FANUC"
        and abs(m.feedrate_max - p.feed_max) < 1e-6
        and abs(m.spindle_max - p.spindle_max) < 1e-3,
    }


@api_function("machine", mutates=True)
def set_machine(
    profile: dict | None = None, preset: str | None = None, reset: bool = False
) -> dict:
    """Change the machine profile: `profile` fields are laid over the current profile (or
    over `preset`, e.g. 'robodrill_d21mib5', or the defaults with reset=True). Unknown fields
    raise INVALID_PROPERTY with close matches. Also sets Fabex's machine settings to match
    (post processor FANUC, millimetres, travel, feed and spindle limits). Returns
    machine_profile()."""
    s = bpy.context.scene
    try:
        if preset:
            base = cc_machine.preset(preset)
        elif reset:
            base = cc_machine.MachineProfile()
        else:
            base = state.machine(s)
        p = cc_machine.MachineProfile.from_dict(profile or {}, base)
    except CamcoreError as e:
        raise _err(e) from e
    state.set_machine(s, p)
    return machine_profile()


@api_function("machine")
def tool_library(type: str | None = None) -> dict:
    """The tools plans may use: the user's library (set_tool_library) or the default
    catalogue built from Fabex's metric end / ball mill presets plus drills and spot drills.
    Each tool: id, type (END BALL BULL DRILL SPOT TAP), diameter, number (T, None = given
    by a plan), corner_radius, flutes, flute_length, reach, holder_diameter, tip_angle
    (deg), pitch, material, description, source. Lengths in metres. `type` filters."""
    try:
        tools, source = state.library(bpy.context.scene)
    except CamcoreError as e:
        raise _err(e) from e
    if type is not None:
        if type.upper() not in cc_tools.TOOL_TYPES:
            raise FabexAPIError(
                "INVALID_VALUE", f"type must be one of {cc_tools.TOOL_TYPES}", {"type": type}
            )
        tools = [t for t in tools if t.type == type.upper()]
    return {"source": source, "count": len(tools), "tools": [t.to_dict() for t in tools]}


@api_function("machine", mutates=True)
def set_tool_library(tools: list | None = None) -> dict:
    """Replace the tool library with your own tools - e.g. the turret map with fixed T
    numbers (`number`). Each tool needs id, type and diameter (m); missing flute length,
    reach, holder diameter and point angle get typical values. tools=None goes back to the
    default catalogue. Returns tool_library()."""
    s = bpy.context.scene
    try:
        parsed = None if tools is None else cc_tools.library_from_list(tools)
    except CamcoreError as e:
        raise _err(e) from e
    state.set_library(s, parsed)
    return tool_library()


def _ops(operations) -> list:
    fx = _fx()
    if not isinstance(operations, (list, tuple)) or not operations:
        raise FabexAPIError("INVALID_ARGUMENT", "operations must be a non-empty list", {})
    return [fx._resolve(op)[1] for op in operations]


@api_function("machine", mutates=True)
def create_chain(name: str, operations: list, filename: str | None = None) -> dict:
    """A chain: operations exported together into one program, in the given order (tool
    changes between them). An existing chain with that name gets these operations."""
    s = bpy.context.scene
    ops = _ops(operations)
    chain = next((c for c in s.cam_chains if c.name == name), None)
    if chain is None:
        chain = s.cam_chains.add()
        chain.name = name
        chain.index = len(s.cam_chains) - 1
    chain.operations.clear()
    for o in ops:
        chain.operations.add().name = o.name
    chain.filename = safe_filename(filename or name)
    s.cam_active_chain = list(s.cam_chains).index(chain)
    return {
        "chain": chain.name,
        "index": s.cam_active_chain,
        "operations": [o.name for o in ops],
        "filename": chain.filename,
    }


@api_function("machine", mutates=True)
def export_program(
    operations: list | None = None,
    chain: str | None = None,
    directory: str | None = None,
    program_number: int | None = None,
    title: str | None = None,
    work_offset: str | None = None,
) -> dict:
    """Write one FANUC program for computed operations (a list, in order) or a chain:
    tool changes with G43, coolant, AICC, canned drilling cycles. The file is
    <directory>/<title>.nc (directory: the export location / the .blend's folder).
    program_number defaults to the machine profile's. Returns the report: file, bytes,
    lines, minutes, tools, operations (moves, holes, time), envelope_mm, warnings and
    `check` - the program checked as text (check['ok'], check['errors']). Raises
    PATH_NOT_FOUND for operations without a path, TOOL_CONFLICT when one T number is used
    for different cutters, MACHINE_LIMIT when the magazine is too small."""
    fx = _fx()
    s = bpy.context.scene
    if (operations is None) == (chain is None):
        raise FabexAPIError("INVALID_ARGUMENT", "Give operations or chain (one of them)", {})
    if chain is not None:
        c = next((c for c in s.cam_chains if c.name == chain), None)
        if c is None:
            raise FabexAPIError(
                "INVALID_VALUE", f"No chain {chain!r}", {"chains": [c.name for c in s.cam_chains]}
            )
        operations = [o.name for o in c.operations]
        title = title or c.filename or c.name
    ops = _ops(operations)
    missing = [o.name for o in ops if bpy.data.objects.get(o.path_object_name or "") is None]
    if missing:
        raise FabexAPIError(
            "PATH_NOT_FOUND", f"No computed path for {missing}", {"operations": missing}
        )
    meshes = [bpy.data.objects[o.path_object_name].data for o in ops]
    directory = bpy.path.abspath(directory) if directory else fx._export_dir()
    if not directory:
        raise FabexAPIError(
            "INVALID_ARGUMENT",
            "The file is unsaved and has no export location: give a directory",
            {},
        )
    title = title or ops[0].name
    path = os.path.join(directory, safe_filename(title) + ".nc")
    return fanuc_export.write_program(
        path, ops, meshes, number=program_number, title=title, work_offset=work_offset or ""
    )


@api_function("machine")
def check_gcode(path: str | None = None, text: str | None = None) -> dict:
    """Check a FANUC program (a file or text) against the machine profile: framing,
    decimal points, tool / length offset / spindle before cutting, limits, G80 / AICC off
    before tool changes and the end, no incremental moves, no diagonal rapids down, the
    travel and the program memory. Returns ok, errors and warnings (rule, line, text,
    message), tools, envelope_mm, bytes, lines."""
    if (path is None) == (text is None):
        raise FabexAPIError("INVALID_ARGUMENT", "Give path or text (one of them)", {})
    if path is not None:
        try:
            with open(bpy.path.abspath(path), encoding="ascii", errors="replace") as f:
                text = f.read()
        except OSError as e:
            raise FabexAPIError("INVALID_VALUE", f"Can't read {path}: {e}", {}) from e
    try:
        return cc_check.check(text, state.machine(bpy.context.scene))
    except CamcoreError as e:
        raise _err(e) from e


# --- automatic CAM: cutting data, plan, apply, verify ----------------------------------------


def _library_tool(tool):
    tools, _ = state.library(bpy.context.scene)
    if isinstance(tool, dict):
        try:
            return cc_tools.Tool.from_dict(tool)
        except CamcoreError as e:
            raise _err(e) from e
    found = next((t for t in tools if t.id == tool), None)
    if found is None:
        import difflib

        raise FabexAPIError(
            "INVALID_VALUE",
            f"No tool {tool!r} in the tool library",
            {"close_matches": difflib.get_close_matches(str(tool), [t.id for t in tools], 5)},
        )
    return found


@api_function("plan")
def cutting_data(
    tool, material: str = "6061", engagement: str = "rough", ap: float | None = None,
    ae: float | None = None,
) -> dict:  # fmt: skip
    """Spindle speed, feeds and depth / width of cut for a tool (a library id like 'EM10',
    or a tool dict) in a material (6061, 6082, 7075, 2017; 'aluminium' = 6061) for a kind of
    cut: slot, pocket, rough, finish, floor, face (mills), drill, spot, tap. Carbide, flood
    coolant, capped by the machine profile's spindle speed, feed and power. Returns rpm,
    feed / plunge_feed (m/min), vc (m/min), fz / feed_per_rev / ap / ae / peck (m),
    ramp_angle (deg), power (kW), removal_rate (cm3/min), rpm_limited and notes."""
    from .camcore import cutting as cc_cutting

    t = _library_tool(tool)
    try:
        return cc_cutting.cutting_data(
            t, material, engagement, state.machine(bpy.context.scene), ap, ae
        ).to_dict()
    except CamcoreError as e:
        raise _err(e) from e


@api_function("plan", mutates=True)
def plan_cam(
    obj: str, material: str = "6061", options: dict | None = None, tolerance: float = 1e-5,
    geometry: bool = False,
) -> dict:  # fmt: skip
    """Plan machining the top of a B-rep part in one setup (OP1) on the machining centre:
    stock, placement (Z 0 at the part's top, X0 Y0 at the stock's centre), the tools from
    the tool library and the operations with cutting data, in order - face, rough (helical
    entry), spot, drill, floors, walls - each with why. Also: which faces each operation
    finishes and which faces stay uncovered (with the reason: the bottom is OP2, 3D faces,
    undercuts), corners left by the tool radius (rest) and warnings. Nothing in the scene
    changes; the plan is kept in the scene for apply_plan().

    options (lengths m): allowance_top (0.0005), allowance_side (0.002 per side), grip
    (0.006 below the part, held by the vise), through (0.0002: how far OP1 cuts below the
    part), jaw_clearance (0.002), stock ([x, y, z] size instead of the allowances), origin
    ('stock_center' | 'stock_corner'), skin (0.0002), spot (True), max_diameter (0.016),
    min_diameter (0.001), program_number. geometry=True includes the region curves and
    drill points (big)."""
    from ..brep.curves import brep_source_object
    from . import apply as ac_apply, partmodel
    from .camcore import planner

    s = bpy.context.scene
    ob = brep_source_object(obj)
    part, _ = partmodel.part_model(ob, float(tolerance))
    tools, source = state.library(s)
    opts = dict(options or {})
    opts["material"] = material
    opts.setdefault("tolerance", float(tolerance))
    from ..brep import bridge

    try:
        plan = planner.plan(part, tools, state.machine(s), opts, ob.name, bridge.body_hash(ob))
    except CamcoreError as e:
        raise _err(e) from e
    plan["placement"]["object_location"] = [float(x) for x in ob.matrix_world.translation]
    plan["tool_library"] = source
    ac_apply.store_plan(s, plan)
    return plan if geometry else planner.summary(plan)


@api_function("plan", mutates=True)
def apply_plan(
    plan: dict | None = None, compute: bool = True, export: bool = True, verify: bool = True,
    directory: str | None = None, program_number: int | None = None,
    resolution: float | None = None,
) -> dict:  # fmt: skip
    """Make a plan real (plan=None: the last one plan_cam made; an edited plan keeps its
    geometry by operation id): move the part to the plan's placement, replace what an
    earlier plan for the part made, create the source curves / drill points, the operations
    and a chain '<part> OP1', compute every path, write the FANUC program (export_program)
    and verify it against the part (verify_program). Returns placement, operations, chain,
    computed (per operation), program (file, minutes, tools, check) and verify, plus ok
    (everything computed, the program checks, no gouge or crash). Raises PLAN_INVALID,
    PLAN_STALE (the part changed or moved since planning)."""
    from . import apply as ac_apply

    s = bpy.context.scene
    if plan is None:
        plan = ac_apply.stored_plan(s)
        if plan is None:
            raise FabexAPIError("PLAN_INVALID", "No plan yet: call plan_cam first", {})
    return ac_apply.apply_plan(plan, compute, export, verify, directory, program_number, resolution)


@api_function("plan")
def verify_program(
    path: str | None = None, obj: str | None = None, stock: dict | None = None,
    resolution: float | None = None, tolerance: float = 0.02,
) -> dict:  # fmt: skip
    """Simulate a FANUC program on the stock and compare with a B-rep part: gouges (part
    material cut) and leftover (material left) per face - next to walls measured sideways,
    elsewhere vertically - crashes (rapids through material, the holder or a vise jaw),
    material above the flutes, mills plunging into material, and per operation the volume
    removed, air cutting and slotting. Defaults (path, obj, stock {min, max} in m, the
    jaws, tool lengths) come from the last applied plan. resolution: pixel size in mm
    (default 0.1, coarser for big stock). Lengths in the report are mm."""
    from . import apply as ac_apply

    s = bpy.context.scene
    plan = ac_apply.stored_plan(s) or {}
    applied = plan.get("applied", {})
    path = path or applied.get("program")
    if not path:
        raise FabexAPIError("INVALID_ARGUMENT", "Give path (no applied plan to take it from)", {})
    name = obj or plan.get("object")
    ob = bpy.data.objects.get(name or "")
    if ob is None:
        raise FabexAPIError("OBJECT_NOT_FOUND", f"No object {name!r}", {})
    stock = stock or plan.get("stock")
    if not stock:
        raise FabexAPIError("INVALID_ARGUMENT", "Give stock {min, max} (m)", {})
    same = plan and plan.get("object") == ob.name
    try:
        rep = ac_apply.verify_file(
            bpy.path.abspath(path),
            ob,
            stock,
            tools=ac_apply.tool_table(plan) if same else None,
            fixtures=plan.get("fixture", {}).get("jaws") if same else None,
            resolution=resolution,
            tolerance=tolerance,
            part_bottom=ac_apply.ob_bottom(plan) if same else None,
            plan=plan if same else None,
        )
    except OSError as e:
        raise FabexAPIError("INVALID_VALUE", f"Can't read {path}: {e}", {}) from e
    except CamcoreError as e:
        raise _err(e) from e
    return rep
