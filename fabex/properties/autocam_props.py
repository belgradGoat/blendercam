"""Fabex 'autocam_props.py'

'CAM Machining Centre Properties' - per-operation settings used when writing a program for a
FANUC machining centre (post processor FANUC): coolant, AICC and drilling cycles. And the
Auto CAM panel's inputs (scene.cam_autocam).
"""

from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Object, PropertyGroup

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


def _is_brep(self, ob) -> bool:
    return ob.type == "MESH" and "brepkit_data" in ob


class CAM_AUTOCAM_SceneProperties(PropertyGroup):
    """Automatic CAM inputs (Auto CAM panel): the part, the material and the stock."""

    part: PointerProperty(
        name="Part",
        type=Object,
        poll=_is_brep,
        description="The B-rep part to plan (a BRepKit object)",
    )
    material: EnumProperty(
        name="Material",
        items=(
            ("6061", "6061-T6", "Aluminium 6061-T6"),
            ("6082", "6082-T6", "Aluminium EN AW-6082-T6"),
            ("7075", "7075-T6", "Aluminium 7075-T6"),
            ("2017", "2017A", "Aluminium 2017A (dural)"),
        ),
        default="6061",
    )
    allowance_top: FloatProperty(
        name="Top",
        description="Stock above the part's top (faced off)",
        default=0.0005,
        min=0.0,
        max=0.02,
        precision=PRECISION,
        unit="LENGTH",
    )
    allowance_side: FloatProperty(
        name="Sides",
        description="Stock around the part, each side",
        default=0.002,
        min=0.0,
        max=0.05,
        precision=PRECISION,
        unit="LENGTH",
    )
    grip: FloatProperty(
        name="Grip",
        description="Stock below the part, held by the vise",
        default=0.006,
        min=0.0025,
        max=0.05,
        precision=PRECISION,
        unit="LENGTH",
    )
    origin: EnumProperty(
        name="Origin",
        items=(
            ("stock_center", "Stock Centre", "X0 Y0 at the stock's centre, Z0 at the part's top"),
            ("stock_corner", "Stock Corner", "X0 Y0 at the stock's front-left corner"),
        ),
        default="stock_center",
    )
    program_number: IntProperty(
        name="Program",
        description="O-number of the program (0: the machine profile's)",
        default=0,
        min=0,
        max=9999,
    )
