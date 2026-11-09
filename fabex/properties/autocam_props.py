"""Fabex 'autocam_props.py'

'CAM Machining Centre Properties' - per-operation settings used when writing a program for a
FANUC machining centre (post processor FANUC): coolant, AICC and drilling cycles.
"""

from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from ..constants import PRECISION


class CAM_AUTOCAM_Properties(PropertyGroup):
    coolant: EnumProperty(
        name="Coolant",
        items=(
            ("FLOOD", "Flood", "Flood coolant (M08)"),
            ("MIST", "Mist", "Mist coolant (M07)"),
            ("AIR", "Air", "Air blast (M-code from the machine profile)"),
            ("THROUGH", "Through Spindle", "Through-spindle coolant (M-code from the profile)"),
            ("OFF", "Off", "No coolant"),
        ),
        description="Coolant while this operation runs",
        default="FLOOD",
    )
    aicc: BoolProperty(
        name="AI Contour Control",
        description="Run the operation's moves with AI contour control (G05.1 Q1) when the "
        "machine profile has it",
        default=True,
    )
    drill_cycle: EnumProperty(
        name="Drill Cycle",
        items=(
            ("AUTO", "Auto", "G81 up to 1 x D deep, G73 up to 4 x D, G83 deeper"),
            ("G81", "G81 Drill", "Feed to depth, rapid out"),
            ("G82", "G82 Dwell", "Feed to depth, dwell, rapid out (spot drilling)"),
            ("G73", "G73 Chip Break", "Peck with a short retract to break chips"),
            ("G83", "G83 Peck", "Peck with a full retract to clear chips"),
            ("G84", "G84 Rigid Tap", "Rigid tapping (M29); feed = speed x pitch"),
        ),
        description="Canned cycle a DRILL operation is written with",
        default="AUTO",
    )
    peck: FloatProperty(
        name="Peck",
        description="Peck depth (Q) for G73 / G83; 0 = the drill diameter (or the stepdown "
        "when layers are on and it is smaller)",
        default=0.0,
        min=0.0,
        max=0.1,
        precision=PRECISION,
        unit="LENGTH",
    )
    dwell: FloatProperty(
        name="Dwell",
        description="Dwell at the bottom of the hole for G82 (seconds)",
        default=0.2,
        min=0.0,
        max=10.0,
    )
    r_clearance: FloatProperty(
        name="R Clearance",
        description="Height above the depth start (max Z) where drilling cycles start feeding",
        default=0.002,
        min=0.0,
        max=0.1,
        precision=PRECISION,
        unit="LENGTH",
    )
    tap_pitch: FloatProperty(
        name="Tap Pitch",
        description="Thread pitch for G84 rigid tapping",
        default=0.0,
        min=0.0,
        max=0.01,
        precision=PRECISION,
        unit="LENGTH",
    )
    plan_tag: StringProperty(
        name="Plan",
        description="The automatic CAM plan that made this operation (empty: made by hand)",
        default="",
    )
