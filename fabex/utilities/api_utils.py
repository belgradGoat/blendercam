"""Fabex 'api_utils.py'

Registry and helpers shared by the scripting / agent API (fabex/api.py, fabex/brep/api.py).
"""

import functools
import inspect
import math

import bpy

from ..utilities.logging_utils import log

# name -> {"fn", "category", "mutates"}
FUNCTIONS: dict[str, dict] = {}

ERROR_CODES = {
    "INVALID_ARGUMENT": "A parameter is missing, unknown or has the wrong type",
    "UNKNOWN_FUNCTION": "call_json() was given a name that is not an API function",
    "OPERATION_NOT_FOUND": "No CAM operation with that name or index",
    "AMBIGUOUS_OPERATION": "Several operations share that name - use the index",
    "OBJECT_NOT_FOUND": "No object with that name",
    "COLLECTION_NOT_FOUND": "No collection with that name, or it is empty",
    "INVALID_PROPERTY": "The operation has no such property (details lists close matches)",
    "READ_ONLY_PROPERTY": "The property can't be set",
    "INVALID_VALUE": "The value was rejected (details has the reason / allowed values)",
    "SOURCE_INVALID": "The operation's geometry source is missing or empty",
    "COMPUTE_FAILED": "Path calculation failed (message has Fabex's error)",
    "CANCELLED": "Computation was cancelled or timed out",
    "BUSY": "Another computation is running",
    "PATH_NOT_FOUND": "The operation has no calculated path yet",
    "JOB_NOT_FOUND": "No compute job with that id",
    "UNSUPPORTED": "Not possible in this Blender session (e.g. jobs in background mode)",
    "BREPKIT_UNAVAILABLE": "BRepKit was not found (enable the extension or set FABEX_BREPKIT_PATH)",
    "NOT_A_BREP": "The object carries no BRepKit B-rep data",
    "BREP_PARSE_FAILED": "The B-rep data could not be read by the available BRepKit",
    "TESSELLATION_FAILED": "BRepKit could not tessellate the B-rep",
    "TOO_LARGE": "The result would be too big (details has the estimate and the limit)",
    "TOOL_CONFLICT": "One tool number (T) is used for different cutters",
    "MACHINE_LIMIT": "Beyond what the machine profile allows (magazine, coolant, travel)",
    "PLAN_INVALID": "The plan is not one plan_cam made (missing fields, geometry or version)",
    "PLAN_STALE": "The part changed or moved since it was planned: plan again",
    "CAM_ERROR": "A Fabex error without a specific code",
    "INTERNAL_ERROR": "Unexpected error - details has the exception",
}

_mutation_depth = 0


def api_function(category: str, mutates: bool = False):
    """Register a function as part of the public API.

    Mutating calls push one undo step (outermost call only, UI sessions only), so a user
    can undo what a script or agent did.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Results are always plain JSON types, whether called directly or via call_json.
            global _mutation_depth
            if not mutates:
                return jsonable(fn(*args, **kwargs))
            _mutation_depth += 1
            try:
                return jsonable(fn(*args, **kwargs))
            finally:
                _mutation_depth -= 1
                if _mutation_depth == 0:
                    undo_push(f"Fabex: {fn.__name__}")

        FUNCTIONS[fn.__name__] = {"fn": wrapper, "category": category, "mutates": mutates}
        return wrapper

    return decorator


def undo_push(message: str) -> None:
    if bpy.app.background:
        return
    try:
        bpy.ops.ed.undo_push(message=message)
    except Exception as e:  # no suitable context - not worth failing the call for
        log.debug(f"undo_push skipped: {e}")


def describe_functions() -> list[dict]:
    out = []
    for name in sorted(FUNCTIONS):
        entry = FUNCTIONS[name]
        fn = entry["fn"]
        out.append(
            {
                "name": name,
                "category": entry["category"],
                "mutates": entry["mutates"],
                "signature": f"{name}{inspect.signature(fn)}",
                "doc": inspect.getdoc(fn) or "",
            }
        )
    return out


def jsonable(value):
    """Convert API results to plain JSON types (numpy, mathutils, bpy arrays, sets...)."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        items = [jsonable(v) for v in value]
        try:
            return sorted(items)
        except TypeError:
            return items
    # numpy without importing it here
    if hasattr(value, "tolist") and hasattr(value, "dtype"):
        return jsonable(value.tolist())
    if isinstance(value, bpy.types.ID):
        return value.name
    try:
        return [jsonable(v) for v in value]  # mathutils / bpy_prop_array
    except TypeError:
        return str(value)
