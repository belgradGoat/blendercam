"""Fabex operations + their computed paths -> a FANUC program (post processor FANUC).

Moves are classified exactly as Fabex's own export does (camcore/moves.py), with these
deliberate differences for a machining centre:
- no move back to the path's synthetic origin vertex at the start of each operation;
- rapids never move across while going up or down;
- DRILL operations are written as canned cycles (autocam.drill_cycle) from the holes in
  their path, drilling at the operation's plunge feed, as Fabex does;
- F only on feed blocks; no G64; no splitting by vertex count;
- feed moves are thinned to a 1 um chord tolerance.
"""

import os

import numpy as np

from ..exception import FabexAPIError
from ..utilities.logging_utils import log
from .camcore import check as cc_check
from .camcore import fanuc
from .camcore.errors import CamcoreError
from .camcore.moves import classify, drill_holes
from .camcore.program import DrillCycle, OpBlock, ProgramSpec, ToolSpec
from . import state

last_report: dict | None = None  # the most recent export (for the API and the UI)

_TYPES = {
    "END": "END",
    "BALLNOSE": "BALL",
    "BULLNOSE": "BULL",
    "VCARVE": "SPOT",
    "BALLCONE": "BALL",
    "CYLCONE": "END",
    "CUSTOM": "END",
}


def _api_error(e: CamcoreError) -> FabexAPIError:
    return FabexAPIError(e.code, e.message, e.details)


def path_points(mesh) -> np.ndarray:
    """Path vertices in metres, as Fabex's export reads them (float32 storage)."""
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def tool_spec(o) -> ToolSpec:
    kind = _TYPES.get(o.cutter_type)
    if kind is None:
        raise FabexAPIError(
            "UNSUPPORTED",
            f"Operation {o.name!r}: cutter type {o.cutter_type} can't be written for a "
            "machining centre",
        )
    if o.strategy == "DRILL" and kind == "END":
        kind = "TAP" if o.autocam.drill_cycle == "G84" else "DRILL"
    d = o.cutter_diameter * 1000
    corner = {"BALL": d / 2, "BULL": o.bull_corner_radius * 1000}.get(kind, 0.0)
    return ToolSpec(
        number=int(o.cutter_id),
        type=kind,
        diameter=round(d, 6),
        corner_radius=round(corner, 6),
        flutes=int(o.cutter_flutes),
        pitch=round(o.autocam.tap_pitch * 1000, 6),
        description=(o.cutter_description or "").strip()[:30],
    )


def _cycle(o, tool: ToolSpec, holes_m: list, plunge_feed: float, clearance: float) -> DrillCycle:
    a = o.autocam
    holes = [(x * 1000, y * 1000, z * 1000) for x, y, z in holes_m]
    top = o.max_z * 1000
    depth = max(top - z for _, _, z in holes) if holes else 0.0
    cycle = a.drill_cycle
    if cycle == "AUTO":
        if tool.type == "SPOT" or depth <= tool.diameter:
            cycle = "G81"
        elif depth <= 4 * tool.diameter:
            cycle = "G73"
        else:
            cycle = "G83"
    peck = a.peck * 1000
    if not peck > 0:  # one diameter, or the stepdown when layers are on and it is smaller
        peck = min(o.stepdown * 1000, tool.diameter) if o.use_layers else tool.diameter
    r_plane = min(top + a.r_clearance * 1000, clearance)
    return DrillCycle(
        cycle=cycle,
        holes=holes,
        r_plane=r_plane,
        feed=plunge_feed,
        peck=peck if cycle in ("G73", "G83") else 0.0,
        dwell=a.dwell if cycle == "G82" else 0.0,
    )


def build_spec(
    operations,
    meshes,
    number: int | None = None,
    title: str = "",
    work_offset: str = "",
    machine=None,
    scene=None,
) -> tuple[ProgramSpec, list]:
    """A ProgramSpec from operations and their path meshes (same order), plus warnings."""
    import bpy

    scene = scene or bpy.context.scene
    profile = machine or state.machine(scene)
    cam_m = scene.cam_machine
    warnings = []
    blocks = []
    prev = None  # last point of the previous operation (m)
    for o, mesh in zip(operations, meshes):
        if o.machine_axes != "3":
            raise FabexAPIError("UNSUPPORTED", f"Operation {o.name!r}: only 3-axis is written")
        P = path_points(mesh)
        if len(P) < 2:
            warnings.append(f"{o.name}: empty path, left out")
            continue
        free = o.movement.free_height
        # the first vertex is the synthetic start point Fabex adds; the first operation's
        # moves are judged from it, later ones from where the previous one ended
        if prev is None:
            kinds = classify(P, free, o.plunge_angle)
            P, kinds = P[1:], kinds[1:]
        else:
            P = P[1:]
            kinds = classify(P, free, o.plunge_angle, prev)
        prev = P[-1]
        mill = min(o.feedrate, cam_m.feedrate_max)
        mill = max(mill, cam_m.feedrate_min) * 1000
        plunge = mill * o.plunge_feedrate / 100
        tool = tool_spec(o)
        clearance = free * 1000
        drill = None
        if o.strategy == "DRILL":
            holes = drill_holes(P, free)
            if holes is None:
                warnings.append(f"{o.name}: path is not plain drilling - written as moves")
            else:
                drill = _cycle(o, tool, holes, plunge, clearance)
        blocks.append(
            OpBlock(
                name=o.name,
                tool=tool,
                rpm=o.spindle_rpm,
                clockwise=o.movement.spindle_rotation == "CW",
                feed=mill,
                plunge_feed=plunge,
                clearance=clearance,
                points=P * 1000,
                kinds=kinds,
                drill=drill,
                coolant=o.autocam.coolant,
                aicc=o.autocam.aicc,
                header=[x for x in o.gcode_header.split(";") if x] if o.output_header else [],
                trailer=[x for x in o.gcode_trailer.split(";") if x] if o.output_trailer else [],
                strategy=o.strategy,
            )
        )
    if not blocks:
        raise FabexAPIError("PATH_NOT_FOUND", "No operation has a path to write")
    spec = ProgramSpec(
        number=int(number or profile.program_number),
        title=title,
        machine=profile,
        blocks=blocks,
        work_offset=work_offset,
    )
    return spec, warnings


def write_program(path: str, operations, meshes, number=None, title="", work_offset=""):
    """Write the program to `path`; returns the report (with the check)."""
    global last_report
    try:
        spec, warnings = build_spec(operations, meshes, number, title, work_offset)
        text, report = fanuc.write(spec)
        report["check"] = cc_check.check(text, spec.machine)
    except CamcoreError as e:
        raise _api_error(e) from e
    report["warnings"] = warnings + report["warnings"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="ascii", errors="replace", newline="\n") as f:
        f.write(text)
    report["file"] = os.path.abspath(path)
    last_report = report
    for w in report["warnings"]:
        log.warning(f"FANUC export: {w}")
    for e in report["check"]["errors"]:
        log.error(f"FANUC check: line {e['line']}: {e['message']}")
    return report


def export_fanuc(path: str, meshes, operations) -> dict:
    """export_gcode_path() for post processor FANUC (UI exports and chains)."""
    title = os.path.splitext(os.path.basename(path))[0]
    return write_program(path, operations, meshes, title=title)
