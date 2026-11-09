"""The machine profile and tool library of a .blend file.

Both are stored as JSON on the scene (`scene["fabex_autocam"]`), so they travel with the
file. Setting the profile also brings Fabex's own machine settings (`scene.cam_machine`) in
line: FANUC post, millimetres, travel, feed and spindle limits - Fabex refuses operations
faster than `cam_machine.feedrate_max`.
"""

import json
from pathlib import Path

from .camcore import tools as cc_tools
from .camcore.machine import MachineProfile

KEY = "fabex_autocam"
PRESET_DIR = Path(__file__).resolve().parents[1] / "presets" / "cam_cutters"


def _data(scene) -> dict:
    try:
        return json.loads(scene.get(KEY, "{}"))
    except (TypeError, ValueError):
        return {}


def _store(scene, data: dict) -> None:
    scene[KEY] = json.dumps(data)


def machine(scene) -> MachineProfile:
    saved = _data(scene).get("machine")
    return MachineProfile.from_dict(saved) if saved else MachineProfile()


def set_machine(scene, profile: MachineProfile, sync: bool = True) -> None:
    data = _data(scene)
    data["machine"] = profile.to_dict()
    _store(scene, data)
    if sync:
        sync_cam_machine(scene, profile)


def sync_cam_machine(scene, p: MachineProfile) -> None:
    m = scene.cam_machine
    m.post_processor = "FANUC"
    if m.unit_system != "MILLIMETERS":
        m.unit_system = "MILLIMETERS"
    m.working_area = tuple(float(t) for t in p.travel)
    m.feedrate_min = max(float(p.feed_min), 0.00001)
    m.feedrate_max = float(p.feed_max)
    m.feedrate_default = min(m.feedrate_default, m.feedrate_max)
    m.spindle_min = max(float(p.spindle_min), 0.00001)
    m.spindle_max = float(p.spindle_max)
    m.spindle_default = min(max(m.spindle_default, m.spindle_min), m.spindle_max)
    m.output_tool_change = True
    m.output_tool_definitions = True


def library(scene) -> tuple[list, str]:
    """(tools, source): the user's library if one was set, else the default catalogue."""
    saved = _data(scene).get("tools")
    if saved:
        return cc_tools.library_from_list(saved), "user"
    return cc_tools.default_library(PRESET_DIR), "default"


def set_library(scene, tools: list | None) -> None:
    data = _data(scene)
    if tools is None:
        data.pop("tools", None)
    else:
        data["tools"] = [t.to_dict() for t in tools]
    _store(scene, data)
