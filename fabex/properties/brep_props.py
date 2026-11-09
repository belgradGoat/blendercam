"""Fabex 'brep_props.py'

'CAM B-rep Properties' - settings for operations whose sources are BRepKit B-rep objects.
"""

from math import radians

from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from ..constants import PRECISION


def _owner_operation(group):
    """The CAM operation a nested property group belongs to (None if detached)."""
    try:
        path = group.path_from_id()  # e.g. 'cam_operations[2].brep'
        return group.id_data.path_resolve(path.rsplit(".", 1)[0])
    except Exception:
        return None


def update_brep(self, context):
    o = _owner_operation(self)
    if o is not None:
        o.changed = True


class CAM_BREP_Properties(PropertyGroup):
    use_exact_mesh: BoolProperty(
        name="Exact B-rep Mesh",
        description="For BRepKit B-rep objects, give OpenCAMLib a mesh tessellated from the "
        "exact surfaces within the tolerance below, instead of the object's display mesh",
        default=True,
        update=update_brep,
    )

    mesh_tolerance: FloatProperty(
        name="B-rep Mesh Tolerance",
        description="Maximum distance between the B-rep mesh used for sampling and the exact "
        "surfaces. Smaller is more accurate but slower",
        default=0.00001,
        min=0.0000001,
        max=0.001,
        precision=PRECISION,
        unit="LENGTH",
        update=update_brep,
    )

    check_fit: BoolProperty(
        name="Check Cutter Fit",
        description="When computing a surface operation on BRepKit B-rep objects, check that "
        "the cutter fits the tightest concave curves and wall corners, and warn if not",
        default=True,
    )

    # --- Flowline strategy ------------------------------------------------------------------

    flow_faces: StringProperty(
        name="Faces",
        description="B-rep face ids to machine, e.g. '12, 15, 31' (empty: every face the "
        "cutter can reach from above). Use Selected Faces sets them from Edit Mode",
        default="",
        update=update_brep,
    )

    flow_direction: EnumProperty(
        name="Direction",
        description="Which way the passes run over each face",
        items=(
            (
                "AUTO",
                "Auto",
                "Around a pole (dome, corner) when the face has one, else along the face's "
                "longer direction",
            ),
            ("U", "U", "Along the face's U parameter lines"),
            ("V", "V", "Along the face's V parameter lines"),
        ),
        default="AUTO",
        update=update_brep,
    )

    flow_max_slope: FloatProperty(
        name="Max Slope",
        description="Leave out surface steeper than this (from level). Steep walls are better "
        "machined by Waterline",
        default=radians(80),
        min=radians(1),
        max=radians(90),
        subtype="ANGLE",
        update=update_brep,
    )

    flow_reverse: BoolProperty(
        name="Reverse",
        description="Run every pass the other way",
        default=False,
        update=update_brep,
    )
