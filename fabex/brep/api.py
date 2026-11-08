"""Fabex 'brep/api.py' - the B-rep half of the scripting / agent API.

Functions here are registered in the same registry as fabex/api.py, so they show up in
describe() and can be called through call_json().
"""

import bpy

from ..exception import BrepError, FabexAPIError
from ..utilities.api_utils import api_function
from . import bridge, ocl_mesh


@api_function("brep")
def brep_kernel(refresh: bool = False) -> dict:
    """Where BRepKit was found (extension / module / path) and its version.
    Use refresh=True after enabling or installing BRepKit."""
    if refresh:
        bridge.find_kernel(refresh=True)
    return bridge.kernel_status()


@api_function("brep")
def list_brep_objects() -> list[dict]:
    """All objects carrying BRepKit B-rep data, with face/edge counts when BRepKit is
    available (an object can still be machined as a plain mesh without it)."""
    kernel = bridge.find_kernel()
    out = []
    for ob in bpy.data.objects:
        if not bridge.is_brep(ob):
            continue
        entry = {
            "name": ob.name,
            "body_hash": bridge.body_hash(ob),
            "faces": None,
            "trimmed_faces": None,
            "edges": None,
            "error": None,
        }
        if kernel is not None:
            try:
                body = bridge.read_body(ob)
                entry["faces"] = len(body.faces)
                entry["trimmed_faces"] = sum(1 for f in body.faces.values() if f.loops)
                entry["edges"] = len(body.edges)
            except BrepError as e:
                entry["error"] = e.to_dict()
        out.append(entry)
    return out


@api_function("brep")
def cam_tessellation(obj: str, tolerance: float | None = None, op=None) -> dict:
    """Tessellate a B-rep object from its exact surfaces - the mesh OpenCAMLib samples.

    tolerance: max distance from the exact surfaces, in metres. Default: the operation's
    'brep.mesh_tolerance' when `op` (name or index) is given, else 0.00001 (0.01 mm).
    Returns triangle count, the measured max deviation, timing, and whether the result
    came from the cache (an unchanged object and tolerance reuse the previous mesh)."""
    ob = bpy.data.objects.get(obj)
    if ob is None:
        raise FabexAPIError("OBJECT_NOT_FOUND", f"No object named '{obj}'", {"object": obj})
    if not bridge.is_brep(ob):
        raise BrepError("NOT_A_BREP", f"Object '{obj}' has no B-rep data", {"object": obj})
    if tolerance is None:
        if op is not None:
            from ..api import _resolve  # late import: fabex.api imports this module

            tolerance = _resolve(op)[1].brep.mesh_tolerance
        else:
            tolerance = 0.00001
    if not tolerance or tolerance <= 0:
        raise FabexAPIError("INVALID_VALUE", "tolerance must be positive", {"tolerance": tolerance})
    mesh, cached = ocl_mesh.cam_mesh_for(ob, float(tolerance))
    return {
        "object": ob.name,
        "body_hash": bridge.body_hash(ob),
        "cached": cached,
        **mesh.stats(),
    }
