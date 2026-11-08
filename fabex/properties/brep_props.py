"""Fabex 'brep_props.py'

'CAM B-rep Properties' - settings for operations whose sources are BRepKit B-rep objects.
"""

from bpy.props import (
    BoolProperty,
    FloatProperty,
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
