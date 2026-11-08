"""Fabex 'api.py' - scripting / agent API.

Context-free functions for driving Fabex from Python: scripts, tests, or an AI agent
using the Blender MCP server's `execute_blender_code`. Every API function:

- addresses things by name or index - never by selection or active object,
- returns plain JSON-serialisable data (dicts, lists, numbers, strings),
- raises FabexAPIError(code, message, details) on failure (codes: describe()["error_codes"]),
- pushes one undo step if it changes the scene (UI sessions only).

Quick start (inside Blender):

    import importlib, bpy
    pkg = next(n for n in bpy.context.preferences.addons.keys() if n.endswith(".fabex"))
    fx = importlib.import_module(pkg + ".api")
    print(fx.call_json("describe"))          # everything below, as JSON
    print(fx.call_json("status"))

Units are metres (Fabex forces scene unit scale 1.0) and radians, unless a name says otherwise.
"""

import glob
import inspect
import json
import os
import tempfile
import time
import traceback
import uuid

import bpy
import numpy as np

from .exception import CamException, FabexAPIError
from .gcode.gcode_export import export_gcode_path
from .operators.async_op import AsyncCancelledException
from .operators.path_ops import _calc_path
from .utilities.addon_utils import fix_units
from .utilities.api_utils import (
    ERROR_CODES,
    FUNCTIONS,
    api_function,
    describe_functions,
    jsonable,
)
from .utilities.bounds_utils import get_bounds_worldspace
from .utilities.logging_utils import log
from .utilities.machine_utils import add_machine_area_object
from .utilities.operation_utils import source_valid
from .utilities.simple_utils import safe_filename
from .utilities.strategy_utils import get_strategy_list
from .version import __version__ as _fabex_version

# B-rep functions register themselves on import
from .brep import api as _brep_api  # noqa: F401
from .brep import bridge as _brep_bridge
from .brep import ocl_mesh as _brep_ocl_mesh

API_VERSION = "0.2"

# Properties applied first by set_operation_props, because others depend on them
_PRIORITY_PROPS = (
    "geometry_source",
    "object_name",
    "collection_name",
    "source_image_name",
    "machine_axes",
    "strategy",
    "cutter_type",
)

_busy: str | None = None  # "sync" or a job id
_jobs: dict[str, dict] = {}
_MAX_JOBS = 20


# --- helpers --------------------------------------------------------------------------------


def _scene():
    return bpy.context.scene


def _resolve(op):
    """(index, operation) for an operation name or index."""
    ops = _scene().cam_operations
    if isinstance(op, bool) or not isinstance(op, (int, str)):
        raise FabexAPIError("INVALID_ARGUMENT", "op must be an operation name or index", {})
    if isinstance(op, int):
        if 0 <= op < len(ops):
            return op, ops[op]
    else:
        matches = [i for i, o in enumerate(ops) if o.name == op]
        if len(matches) == 1:
            return matches[0], ops[matches[0]]
        if len(matches) > 1:
            raise FabexAPIError(
                "AMBIGUOUS_OPERATION",
                f"{len(matches)} operations are named '{op}' - use an index",
                {"indices": matches},
            )
    raise FabexAPIError(
        "OPERATION_NOT_FOUND",
        f"No operation {op!r}",
        {"operations": [o.name for o in ops]},
    )


def _activate(index: int) -> None:
    # Several Fabex property-update callbacks act on the *active* operation.
    _scene().cam_active_operation = index


def _override_kwargs() -> dict:
    """A window/3D-view context for Fabex code that calls bpy.ops (empty in background)."""
    wm = bpy.context.window_manager
    if wm is None:
        return {}
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                return {"window": window, "screen": window.screen, "area": area, "region": region}
    if wm.windows:
        window = wm.windows[0]
        return {"window": window, "screen": window.screen}
    return {}


def _props_dict(struct, prefix: str = "") -> dict:
    out = {}
    for prop in struct.bl_rna.properties:
        pid = prop.identifier
        if pid == "rna_type" or prop.is_hidden:
            continue
        if prefix and pid == "name":  # every nested PropertyGroup has an unused 'name'
            continue
        key = prefix + pid
        if prop.type == "COLLECTION":
            continue
        try:
            value = getattr(struct, pid)
        except Exception:
            continue
        if prop.type == "POINTER":
            if isinstance(value, bpy.types.PropertyGroup):
                out.update(_props_dict(value, key + "."))
            else:
                out[key] = value.name if isinstance(value, bpy.types.ID) else None
            continue
        out[key] = jsonable(value)
    return out


def _enum_identifiers(owner, name: str) -> list[str]:
    rna = owner.bl_rna.properties[name]
    items = [i.identifier for i in rna.enum_items]
    if items or name != "strategy":
        return items
    try:  # dynamic enum
        return [i[0] for i in get_strategy_list(owner, bpy.context)]
    except Exception:
        return []


def _set_prop(struct, path: str, value):
    owner = struct
    parts = path.split(".")
    for part in parts[:-1]:
        rna = owner.bl_rna.properties.get(part)
        if rna is None or rna.type != "POINTER":
            raise FabexAPIError(
                "INVALID_PROPERTY", f"'{part}' in '{path}' is not a property group", {}
            )
        owner = getattr(owner, part)

    name = parts[-1]
    rna = owner.bl_rna.properties.get(name)
    if rna is None or name == "rna_type" or rna.type in ("POINTER", "COLLECTION"):
        import difflib

        candidates = list(_props_dict(struct).keys())
        raise FabexAPIError(
            "INVALID_PROPERTY",
            f"Operation has no settable property '{path}'",
            {"close_matches": difflib.get_close_matches(path, candidates, n=5, cutoff=0.5)},
        )
    if rna.is_readonly:
        raise FabexAPIError("READ_ONLY_PROPERTY", f"'{path}' is read-only", {})

    if rna.type == "ENUM":
        allowed = _enum_identifiers(owner, name)
        if rna.is_enum_flag:
            value = set(value if isinstance(value, (list, tuple, set)) else [value])
            bad = [v for v in value if allowed and v not in allowed]
        else:
            bad = [value] if allowed and value not in allowed else []
        if bad:
            raise FabexAPIError(
                "INVALID_VALUE", f"{bad} not allowed for '{path}'", {"allowed": allowed}
            )
    try:
        setattr(owner, name, value)
    except (TypeError, ValueError, AttributeError) as e:
        raise FabexAPIError("INVALID_VALUE", f"Can't set '{path}' to {value!r}: {e}", {}) from e
    return jsonable(getattr(owner, name))


def _warnings(o) -> list[str]:
    return [w.strip() for w in o.info.warnings.splitlines() if w.strip()]


def _export_dir() -> str:
    s = _scene()
    if s.cam_names.default_export_location:
        return bpy.path.abspath(s.cam_names.default_export_location)
    if bpy.data.filepath:
        return os.path.dirname(bpy.data.filepath)
    return ""


def _export_name(o) -> str:
    return safe_filename(o.name if o.link_operation_file_names else o.filename)


def _find_exports(directory: str, name: str, since: float) -> list[str]:
    if not directory:
        return []
    files = glob.glob(os.path.join(glob.escape(directory), glob.escape(name) + "*"))
    return sorted(f for f in files if os.path.isfile(f) and os.path.getmtime(f) >= since - 1)


class _ExportSettings:
    """Temporarily point Fabex's auto-export at the requested destination."""

    def __init__(self, o, export):
        self.o = o
        self.names = _scene().cam_names
        self.saved = (o.auto_export, self.names.default_export_location)
        self.warnings = []
        if export is None:
            enabled, directory = o.auto_export, None
        elif export is False:
            enabled, directory = False, None
        elif export is True:
            enabled, directory = True, None
        elif isinstance(export, str):
            enabled, directory = True, bpy.path.abspath(export)
        else:
            raise FabexAPIError(
                "INVALID_ARGUMENT", "export must be None, a bool or a directory", {}
            )

        if enabled and directory is None and not _export_dir():
            # Unsaved file and no export location: Fabex would write relative to Blender's cwd
            directory = os.path.join(tempfile.gettempdir(), "fabex_gcode")
            self.warnings.append(f"File is unsaved; G-code written to {directory}")
        if directory:
            os.makedirs(directory, exist_ok=True)
            self.names.default_export_location = os.path.join(directory, "")
        o.auto_export = enabled
        self.enabled = enabled
        self.directory = _export_dir()

    def restore(self):
        self.o.auto_export, self.names.default_export_location = self.saved


def _path_stats(o) -> dict | None:
    ob = bpy.data.objects.get(o.path_object_name) if o.path_object_name else None
    if ob is None or ob.type != "MESH":
        return None
    mesh = ob.data
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mw = np.array(ob.matrix_world)
    co = co @ mw[:3, :3].T + mw[:3, 3]
    length = float(np.linalg.norm(np.diff(co, axis=0), axis=1).sum()) if n > 1 else 0.0
    return {
        "path_object": ob.name,
        "points": n,
        "length": length,
        "bbox_min": co.min(axis=0) if n else None,
        "bbox_max": co.max(axis=0) if n else None,
        # Fabex only estimates machining time while writing G-code, so this is set
        # by compute_operation (with export) and export_gcode, else None
        "estimated_minutes": None,
    }


def _begin(owner: str) -> None:
    global _busy
    if _busy is not None:
        raise FabexAPIError("BUSY", f"A computation is already running ({_busy})", {"busy": _busy})
    _busy = owner


def _end() -> None:
    global _busy
    _busy = None


def run_coroutine_sync(coro, deadline=None, on_progress=None, should_cancel=None):
    """Drive a Fabex coroutine (the progress_async protocol) to completion.

    Cancellation (deadline passed or should_cancel() true) is delivered the same way the
    ESC key does it in the UI, so Fabex cleans up normally.
    """
    to_send = None
    while True:
        try:
            msg = coro.send(to_send)
        except StopIteration as stop:
            return stop.value
        to_send = None
        if deadline is not None and time.monotonic() > deadline:
            to_send = AsyncCancelledException("Timed out")
        elif should_cancel is not None and should_cancel():
            to_send = AsyncCancelledException("Cancelled")
        if on_progress and isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "Progress:":
            on_progress(msg[1])


def _compute_result(index, o, status, started_wall, started, export, extra_warnings):
    state, ok, message = status
    result = {
        "op": o.name,
        "index": index,
        "ok": bool(ok),
        "status": state,
        "message": message,
        "duration_s": round(time.monotonic() - started, 3),
        "path": _path_stats(o) if ok else None,
        "warnings": extra_warnings + _warnings(o),
        "gcode_files": [],
        # for B-rep sources sampled by OpenCAMLib: exact or display mesh, deviation
        "brep_meshes": _brep_ocl_mesh.report(o),
    }
    if ok and export.enabled:
        result["gcode_files"] = _find_exports(export.directory, _export_name(o), started_wall)
        if result["gcode_files"] and result["path"]:
            result["path"]["estimated_minutes"] = o.info.duration
    return result


# --- general --------------------------------------------------------------------------------


@api_function("general")
def describe() -> dict:
    """Everything an agent needs: functions with signatures and docs, strategies, cutter
    types, post-processors, error codes and a quick-start sequence."""
    op_rna = bpy.types.Scene.bl_rna.properties["cam_operations"].fixed_type
    machine_rna = _scene().cam_machine.bl_rna
    try:
        strategies = [
            {"id": s[0], "name": s[1], "description": s[2]}
            for s in get_strategy_list(None, bpy.context)
        ]
    except Exception:
        strategies = []
    return {
        "api_version": API_VERSION,
        "fabex_version": ".".join(map(str, _fabex_version)),
        "import": "importlib.import_module(<fabex package> + '.api'); "
        "package = next(n for n in bpy.context.preferences.addons.keys() if n.endswith('.fabex'))",
        "conventions": [
            "Operations are addressed by name or index (op=...)",
            "Lengths in metres, angles in radians",
            "Errors raise FabexAPIError; call_json() returns {'ok': False, 'error': {...}} instead",
            "Property paths for nested settings use dots, e.g. 'movement.type'",
        ],
        "functions": describe_functions(),
        "strategies": strategies,
        "cutter_types": [
            {"id": i.identifier, "name": i.name}
            for i in op_rna.properties["cutter_type"].enum_items
        ],
        "post_processors": [
            i.identifier for i in machine_rna.properties["post_processor"].enum_items
        ],
        "error_codes": ERROR_CODES,
        "quickstart": [
            "status()",
            "create_operation('MyObject', strategy='PARALLEL', props={'cutter_type': 'BALLNOSE', "
            "'cutter_diameter': 0.006})",
            "compute_operation('<op>', export='/tmp/gcode')   # or start_compute() + job_status()",
            "path_info('<op>')",
        ],
    }


@api_function("general")
def status() -> dict:
    """Session overview: versions, file, units, operations, busy state, OpenCAMLib and BRepKit."""
    s = _scene()
    try:
        import opencamlib  # noqa: F401

        ocl = True
    except ImportError:
        ocl = False
    return {
        "api_version": API_VERSION,
        "fabex_version": ".".join(map(str, _fabex_version)),
        "blender": bpy.app.version_string,
        "background": bpy.app.background,
        "file": bpy.data.filepath or None,
        "scene": s.name,
        "render_engine": s.render.engine,
        "units": {"system": s.unit_settings.system, "scale_length": s.unit_settings.scale_length},
        "post_processor": s.cam_machine.post_processor,
        "export_dir": _export_dir() or None,
        "operations": [o.name for o in s.cam_operations],
        "active_operation": s.cam_active_operation if len(s.cam_operations) else None,
        "busy": _busy,
        "opencamlib": ocl,
        "brepkit": _brep_bridge.kernel_status(),
    }


@api_function("general")
def call_json(name: str, **kwargs) -> str:
    """Call API function `name` and return a JSON string: {"ok": true, "result": ...} or
    {"ok": false, "error": {"code", "message", "details"}}. Never raises."""
    try:
        entry = FUNCTIONS.get(name)
        if entry is None or name == "call_json":
            raise FabexAPIError(
                "UNKNOWN_FUNCTION", f"No API function '{name}'", {"functions": sorted(FUNCTIONS)}
            )
        try:
            inspect.signature(entry["fn"]).bind(**kwargs)
        except TypeError as e:
            raise FabexAPIError("INVALID_ARGUMENT", str(e), {}) from e
        payload = {"ok": True, "result": entry["fn"](**kwargs)}
    except FabexAPIError as e:
        payload = {"ok": False, "error": e.to_dict()}
    except CamException as e:
        payload = {"ok": False, "error": {"code": "CAM_ERROR", "message": str(e), "details": {}}}
    except Exception as e:
        payload = {
            "ok": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": f"{type(e).__name__}: {e}",
                "details": {"traceback": traceback.format_exc(limit=6)},
            },
        }
    return json.dumps(jsonable(payload), sort_keys=True)


# --- operations -----------------------------------------------------------------------------


@api_function("operations")
def list_operations() -> list[dict]:
    """All CAM operations with their key settings and whether a path exists."""
    out = []
    for i, o in enumerate(_scene().cam_operations):
        out.append(
            {
                "index": i,
                "name": o.name,
                "strategy": o.strategy,
                "geometry_source": o.geometry_source,
                "object_name": o.object_name,
                "collection_name": o.collection_name,
                "cutter_type": o.cutter_type,
                "cutter_diameter": o.cutter_diameter,
                "has_path": bool(o.path_object_name and o.path_object_name in bpy.data.objects),
            }
        )
    return out


@api_function("operations")
def get_operation(op) -> dict:
    """All settings of one operation as a flat dict ('movement.type', 'optimisation.use_exact'...)."""
    index, o = _resolve(op)
    return {"index": index, "props": _props_dict(o), "warnings": _warnings(o)}


@api_function("operations", mutates=True)
def set_operation_props(op, props: dict) -> dict:
    """Set operation settings from a flat dict of property paths to values.

    Source, axes, strategy and cutter type are applied first; the rest in the given order.
    Unknown names fail with INVALID_PROPERTY (details.close_matches), bad values with
    INVALID_VALUE (details.allowed for enums). Returns the values as Fabex stored them."""
    index, o = _resolve(op)
    if not isinstance(props, dict):
        raise FabexAPIError("INVALID_ARGUMENT", "props must be a dict", {})
    _activate(index)
    keys = [k for k in _PRIORITY_PROPS if k in props] + [
        k for k in props if k not in _PRIORITY_PROPS
    ]
    changed = {key: _set_prop(o, key, props[key]) for key in keys}
    return {
        "op": o.name,
        "index": index,
        "changed": changed,
        "valid": o.valid,
        "warnings": _warnings(o),
    }


@api_function("operations", mutates=True)
def create_operation(
    obj_name: str | None = None,
    strategy: str = "PARALLEL",
    name: str | None = None,
    collection_name: str | None = None,
    props: dict | None = None,
) -> dict:
    """Create an operation on an object (or a collection) - the scripted 'Add Operation'.

    Like the UI, it sets metric units and adds the CAM_Machine area object if missing.
    `props` is passed to set_operation_props()."""
    s = _scene()
    if collection_name:
        coll = bpy.data.collections.get(collection_name)
        if coll is None or len(coll.objects) == 0:
            raise FabexAPIError(
                "COLLECTION_NOT_FOUND", f"No non-empty collection '{collection_name}'", {}
            )
        objects = list(coll.objects)
    elif obj_name:
        ob = bpy.data.objects.get(obj_name)
        if ob is None:
            raise FabexAPIError("OBJECT_NOT_FOUND", f"No object '{obj_name}'", {})
        objects = [ob]
    else:
        raise FabexAPIError("INVALID_ARGUMENT", "Give obj_name or collection_name", {})

    fix_units()
    minz = get_bounds_worldspace(objects)[2]
    s.cam_operations.add()
    index = len(s.cam_operations) - 1
    o = s.cam_operations[index]
    _activate(index)
    if collection_name:
        o.geometry_source = "COLLECTION"
        o.collection_name = collection_name
    else:
        o.object_name = objects[0].name
    o.min_z = minz

    warnings = []
    try:
        _set_prop(o, "strategy", strategy)
    except FabexAPIError:
        s.cam_operations.remove(index)
        if len(s.cam_operations):
            s.cam_active_operation = len(s.cam_operations) - 1
        raise
    o.name = name or s.cam_names.operation_name_full
    o.filename = s.cam_names.file_name_full
    if sum(1 for other in s.cam_operations if other.name == o.name) > 1:
        warnings.append(f"Another operation is also named '{o.name}' - address it by index")

    if s.objects.get("CAM_Machine") is None:
        try:
            with bpy.context.temp_override(**_override_kwargs()):
                add_machine_area_object()
        except Exception as e:
            warnings.append(f"Could not add the CAM_Machine area object: {e}")

    if props:
        set_operation_props(index, props)
    return {"op": o.name, "index": index, "valid": o.valid, "warnings": warnings + _warnings(o)}


@api_function("operations", mutates=True)
def remove_operation(op, delete_path: bool = True) -> dict:
    """Delete an operation (and by default its path object)."""
    index, o = _resolve(op)
    name = o.name
    path_name = o.path_object_name
    s = _scene()
    s.cam_operations.remove(index)
    if len(s.cam_operations):  # the active-index update callback reads the active operation
        s.cam_active_operation = min(index, len(s.cam_operations) - 1)
    removed_path = None
    if delete_path and path_name and path_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[path_name])
        removed_path = path_name
    return {"removed": name, "removed_path": removed_path}


# --- compute & export -----------------------------------------------------------------------


def _start(index, o, export):
    """Common setup for compute_operation / start_compute."""
    _activate(index)
    _brep_ocl_mesh.begin_report(o)
    if not source_valid(o, bpy.context):
        raise FabexAPIError(
            "SOURCE_INVALID",
            f"Operation '{o.name}' has no valid geometry source",
            {
                "geometry_source": o.geometry_source,
                "object_name": o.object_name,
                "collection_name": o.collection_name,
            },
        )
    return _ExportSettings(o, export)


@api_function("compute", mutates=True)
def compute_operation(op, export=None, timeout_s: float | None = None) -> dict:
    """Calculate an operation's toolpath and wait for it (blocks Blender meanwhile).

    export: None = follow the operation's 'auto_export' setting, False = don't export,
    True = export next to the .blend (or the export location), or a directory path.
    Returns ok/status/message, path stats, warnings and the written G-code files.
    For long computations in the UI use start_compute() + job_status()."""
    index, o = _resolve(op)
    _begin("sync")
    try:
        export_settings = _start(index, o, export)
        started_wall, started = time.time(), time.monotonic()
        deadline = started + timeout_s if timeout_s else None
        try:
            with bpy.context.temp_override(**_override_kwargs()):
                status_ = run_coroutine_sync(_calc_path(None, bpy.context), deadline=deadline)
            result = _compute_result(
                index, o, status_, started_wall, started, export_settings, export_settings.warnings
            )
        finally:
            export_settings.restore()
    finally:
        _end()
    return result


@api_function("compute", mutates=True)
def start_compute(op, export=None, timeout_s: float | None = None) -> dict:
    """Start calculating a toolpath in the background of the UI (bpy.app.timers) and return
    a job id at once. Poll job_status(job_id); cancel with cancel_job(job_id).
    Same `export` options as compute_operation(). Not available in background mode."""
    if bpy.app.background:
        raise FabexAPIError(
            "UNSUPPORTED", "Timers don't run in background mode - use compute_operation()", {}
        )
    index, o = _resolve(op)
    job_id = uuid.uuid4().hex[:8]
    _begin(job_id)
    try:
        export_settings = _start(index, o, export)
    except Exception:
        _end()
        raise

    started = time.monotonic()
    job = {
        "job_id": job_id,
        "op": o.name,
        "index": index,
        "state": "running",
        "progress": None,
        "started_wall": time.time(),
        "started": started,
        "deadline": started + timeout_s if timeout_s else None,
        "cancel": False,
        "result": None,
        "error": None,
    }
    coro = _calc_path(None, bpy.context)
    pending = [None]  # value to send into the coroutine on the next step

    def finish(state, status_=None, error=None):
        export_settings.restore()
        job["state"] = state
        if status_ is not None:
            job["result"] = _compute_result(
                index,
                o,
                status_,
                job["started_wall"],
                job["started"],
                export_settings,
                export_settings.warnings,
            )
        job["error"] = error
        _end()

    def step():
        try:
            with bpy.context.temp_override(**_override_kwargs()):
                budget = time.monotonic() + 0.05
                while time.monotonic() < budget:
                    try:
                        msg = coro.send(pending[0])
                    except StopIteration as stop:
                        state = (
                            "done"
                            if stop.value[1]
                            else ("cancelled" if stop.value[0] == "CANCELLED" else "error")
                        )
                        finish(state, stop.value)
                        return None
                    pending[0] = None
                    if job["deadline"] is not None and time.monotonic() > job["deadline"]:
                        pending[0] = AsyncCancelledException("Timed out")
                    elif job["cancel"]:
                        pending[0] = AsyncCancelledException("Cancelled")
                    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "Progress:":
                        job["progress"] = msg[1]
            return 0.01
        except Exception as e:
            log.error(f"Fabex API job {job_id} failed: {e}")
            finish("error", error={"code": "INTERNAL_ERROR", "message": f"{type(e).__name__}: {e}"})
            return None

    _jobs[job_id] = job
    while len(_jobs) > _MAX_JOBS:
        oldest = next(iter(_jobs))
        if _jobs[oldest]["state"] == "running":
            break
        del _jobs[oldest]
    bpy.app.timers.register(step, first_interval=0.0)
    return {"job_id": job_id, "op": o.name, "index": index}


@api_function("compute")
def job_status(job_id: str) -> dict:
    """State of a compute job: running | done | error | cancelled, with progress and - once
    finished - the same result dict as compute_operation()."""
    job = _jobs.get(job_id)
    if job is None:
        raise FabexAPIError("JOB_NOT_FOUND", f"No job '{job_id}'", {"jobs": list(_jobs)})
    return {
        "job_id": job_id,
        "op": job["op"],
        "state": job["state"],
        "progress": job["progress"],
        "elapsed_s": round(time.monotonic() - job["started"], 3),
        "result": job["result"],
        "error": job["error"],
    }


@api_function("compute", mutates=True)
def cancel_job(job_id: str) -> dict:
    """Ask a running compute job to stop (takes effect at its next progress step)."""
    job = _jobs.get(job_id)
    if job is None:
        raise FabexAPIError("JOB_NOT_FOUND", f"No job '{job_id}'", {"jobs": list(_jobs)})
    if job["state"] == "running":
        job["cancel"] = True
    return {"job_id": job_id, "state": job["state"], "cancel_requested": job["cancel"]}


@api_function("compute")
def path_info(op) -> dict:
    """Stats of an operation's existing toolpath: point count, length, bounding box.
    (Machining time is only estimated while writing G-code: see compute_operation /
    export_gcode.) Raises PATH_NOT_FOUND if it hasn't been calculated."""
    index, o = _resolve(op)
    stats = _path_stats(o)
    if stats is None:
        raise FabexAPIError("PATH_NOT_FOUND", f"Operation '{o.name}' has no path yet", {})
    return {"op": o.name, "index": index, **stats, "warnings": _warnings(o)}


@api_function("compute", mutates=True)
def export_gcode(op, directory: str | None = None) -> dict:
    """Write G-code for an already calculated operation. `directory` defaults to the export
    location / the .blend's folder. Returns the written files."""
    index, o = _resolve(op)
    ob = bpy.data.objects.get(o.path_object_name) if o.path_object_name else None
    if ob is None:
        raise FabexAPIError("PATH_NOT_FOUND", f"Operation '{o.name}' has no path yet", {})
    _activate(index)
    settings = _ExportSettings(o, directory if directory else True)
    started = time.time()
    try:
        export_gcode_path(_export_name(o), [ob.data], [o])
    finally:
        settings.restore()
    files = _find_exports(settings.directory, _export_name(o), started)
    return {
        "op": o.name,
        "files": files,
        "estimated_minutes": o.info.duration if files else None,
        "warnings": settings.warnings,
    }


# Expose every registered API function (including the B-rep ones from fabex/brep/api.py)
# as an attribute of this module, so `fx.<name>(...)` and call_json("<name>") match.
for _name, _entry in FUNCTIONS.items():
    globals().setdefault(_name, _entry["fn"])
__all__ = sorted(FUNCTIONS) + ["run_coroutine_sync", "API_VERSION"]
