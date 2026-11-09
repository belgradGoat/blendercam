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
