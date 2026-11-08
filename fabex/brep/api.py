"""Fabex 'brep/api.py' - the B-rep half of the scripting / agent API.

Functions here are registered in the same registry as fabex/api.py, so they show up in
describe() and can be called through call_json().
"""

import bpy

from ..exception import BrepError
from ..utilities.api_utils import api_function
from . import bridge


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
