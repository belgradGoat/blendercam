# Scripting and Agent API

`fabex/api.py` (with its B-rep half, `fabex/brep/api.py`) drives Fabex from Python: from
scripts, tests, or an AI agent running code in Blender through a Blender MCP server's
`execute_blender_code`. Every function:

- addresses things by **name or index**, never by the selection or the active object;
- returns plain **JSON-able** data (dicts, lists, numbers, strings);
- raises `FabexAPIError(code, message, details)` on failure, with stable error codes;
- pushes one undo step when it changes the scene (UI sessions only).

`describe()` returns all of it as data: every function with its signature and docstring, the
strategies, cutter types, post-processors, error codes and a quick start. An agent needs
nothing else to begin.

## Getting the module

```python
import importlib, bpy
pkg = next(n for n in bpy.context.preferences.addons.keys() if n.endswith(".fabex"))
fx = importlib.import_module(pkg + ".api")

print(fx.call_json("describe"))      # everything, as JSON
print(fx.call_json("status"))        # versions, operations, OpenCAMLib, BRepKit
```

`call_json(function, **kwargs)` never raises. It returns `{"ok": true, "result": ...}` or
`{"ok": false, "error": {"code", "message", "details"}}` as a JSON string, ready to print
from MCP. The function's name goes first, positionally, so arguments called `name` reach
the function: `call_json("create_operation", obj_name="Part", name="Rough")`. The functions
are also attributes of the module (`fx.brep_info("Part")`).

## Conventions

- **Units.** Lengths are metres; Fabex sets the scene unit scale to 1. Operation properties
  that are angles are radians, as Blender stores them. B-rep arguments ending in `_angle`,
  and `max_slope`, are degrees.
- **Property paths.** Nested operation settings use dots: `"movement.type"`,
  `"optimisation.use_opencamlib"`, `"brep.mesh_tolerance"`. `describe_operation_props(search)`
  lists them with their type, unit, range, default and allowed values.
  `set_operation_props()` checks every name and value against Blender's property
  definitions and lists close matches for typos.
- **The stock top is Z 0.** Fabex only machines below each operation's depth start (`max_z`,
  default 0). A part above Z 0 gives an empty path with no error: move it down (as
  `brep_info()` suggests) or raise `max_z`.
- **Blocking.** `compute_operation()` blocks Blender until the path is done. In a UI session,
  `start_compute()` + `job_status()` runs in the background instead (Blender timers; not
  available in background mode).

## Functions

| Category | Function | What it does |
|---|---|---|
| general | `describe()` | Functions, strategies, cutter types, error codes, quick start |
| | `status()` | Versions, file, units, operations, busy state, OpenCAMLib, BRepKit |
| | `call_json(name, **kw)` | Call any function, get JSON back, never raise |
| operations | `list_operations()`, `get_operation(op)` | Operations and their settings |
| | `create_operation(obj_name, strategy, name, collection_name, props)` | The scripted *Add Operation* |
| | `set_operation_props(op, props)` | Validated property changes |
| | `describe_operation_props(search)` | Every setting: type, unit, range, default, allowed values |
| | `remove_operation(op)` | Delete an operation (and its path) |
| compute | `compute_operation(op, export, timeout_s)` | Calculate and wait; path stats, warnings, G-code files |
| | `start_compute(...)`, `job_status(id)`, `cancel_job(id)` | The same, as a background job |
| | `path_info(op)` | Stats of an existing path |
| | `export_gcode(op, directory)` | Write G-code for a computed path |
| brep | `list_brep_objects()` | BRepKit objects in the file |
| | `brep_info(obj, ...)` | What a B-rep is made of, and suggested next steps |
| | `brep_holes(obj, ...)` | Holes and bosses, numbered |
| | `extract_edges(obj, kind, ...)` | Curves from exact edges, outlines and holes |
| | `fit_check(obj, ...)` | Will a cutter fit? |
| | `flowline_preview(obj, ...)` | Flowline passes as a curve, or a dry run |
| | `create_flowline_operation(obj, ...)` | A Flowline finishing operation |
| | `cam_tessellation(obj, tolerance)` | The exact mesh OpenCAMLib samples |
| | `brep_kernel(refresh)` | Where BRepKit was found |
| machine | `machine_profile()`, `set_machine(profile, preset, reset)` | The FANUC machine programs are written for |
| | `tool_library(type)`, `set_tool_library(tools)` | Tools for planning: catalogue or your turret map |
| | `create_chain(name, operations)` | Operations exported together, in this order |
| | `export_program(operations or chain, directory, ...)` | One FANUC program, with its check |
| | `check_gcode(path or text)` | Check a FANUC program as the control reads it |
| plan | `cutting_data(tool, material, engagement)` | rpm, feeds, depth and width of cut for a tool in aluminium |
| | `plan_cam(obj, material, options)` | A machining plan for the top of a B-rep part; changes nothing |
| | `apply_plan(plan, compute, export, verify)` | Make the plan: operations, chain, paths, program, verification |
| | `verify_program(path, obj, stock)` | Simulate a FANUC program on the stock and compare with the part |

The docstrings (in `describe()`) document every argument and result field.

## From a B-rep object to G-code

The B-rep functions are designed to be chained, each answering the question the next one
needs:

```python
fx.list_brep_objects()              # which objects carry an exact B-rep
info = fx.brep_info("Part")         # faces, holes, limits, placement, suggestions
```

`brep_info()` describes every face from the exact surfaces:

- **surface**: plane, cylinder, cone, sphere, torus or freeform, with its axis, centre and
  radii;
- **role**: `top`, `floor`, `slope`, `wall`, `hole_wall`, `fillet` (concave blend), `round`
  (convex blend), `curved`, `bottom` or `underside`;
- **curvature**, slope range, the area out of reach from above, its neighbours and the kind
  of edge between them (tangent, concave, convex).

It also lists the **holes** and other openings. Each one is numbered, with its depth to the
deepest face inside, where its walls end (`walls_bottom`), and `prismatic` when it has
vertical walls and a level floor (2.5D). It also gives the **limits** (largest
ball and end mill that fit), the **placement** (is the part below Z 0?), and
**suggestions**: a machining sequence as ready-to-run calls.

| step | strategy | made with |
|---|---|---|
| `place` | | Python that moves the part's top to Z 0 |
| `rough` | PARALLEL | flat end mill, layers, 0.5 mm skin |
| `drill` | DRILL | one per drill size: `extract_edges(kind='holes', holes=[...], project='bottom')` |
| `pocket` | POCKET | each prismatic opening, flat end mill that fits its corners |
| `walls` | CUTOUT | other openings' vertical walls, profiled from inside down to where they end |
| `finish` | FLOWLINE | curved faces up to 80 degrees steep, the largest common ball that fits, stepover for a 0.01 mm scallop |
| `cut_out` | CUTOUT | the outline, down to the part's bottom |

Running them is plain Python:

```python
for step in info["suggestions"]:
    if "python" in step:
        exec(step["python"], {"bpy": bpy})
    for call in step.get("calls", []):
        eval("fx." + call)

for op in fx.list_operations():
    r = fx.compute_operation(op["name"], export="/tmp/gcode")
    print(op["name"], r["ok"], r["error"], r["warnings"], r["gcode_files"])
```

Z values in the calls assume the `place` step ran. The suggestions are a starting point,
not a verified plan. Look at the warnings,
`fit_check()` and the paths, and adjust: other cutters, a finer stepover, fewer faces. The
pieces work on their own too:

```python
fx.fit_check("Part", cutter_type="BALLNOSE", diameter=0.004)            # fits? max_tip_radius
fx.flowline_preview("Part", faces=[7, 8], stepover=0.0004, create=False) # dry run: passes, points
fx.create_flowline_operation("Part", faces=[7, 8], diameter=0.003, stepover=0.0003)
fx.extract_edges("Part", kind="holes", holes=[0, 1], project="bottom", name="Drill6")
fx.create_operation("Drill6", strategy="DRILL",
                    props={"cutter_diameter": 0.006, "min_z_from": "OBJECT"})
```

Hole numbers are the same in `brep_info()`, `brep_holes()` and `extract_edges(holes=...)`.
Face ids are BRepKit's face ids, also stored per polygon in the display mesh's
`brep_face_id` attribute.

## FANUC machining centres

`set_machine()` describes the machine, a FANUC Robodrill by default, and switches Fabex to
the FANUC post. Then `export_program()` writes one program for computed operations, with:
- tool changes;
- `G43`;
- coolant;
- AICC;
- canned drilling cycles.

It returns a report that includes `check`: the program read back and checked as G-code.
Give each different cutter its own `cutter_id` (T number); one number used for two cutters
raises `TOOL_CONFLICT`. Per-operation settings live under `autocam.*`: `coolant`, `aicc`,
`drill_cycle`, `peck`, `dwell`, `r_clearance`, `tap_pitch`. See
[FANUC Machining Centres](fanuc.md).

```python
fx.set_machine(preset="robodrill_d21mib5", profile={"spindle_max": 24000})
fx.set_operation_props("Drill 5", {"cutter_id": 2, "autocam.drill_cycle": "G83"})
r = fx.export_program(["Pocket", "Drill 5", "Cutout"], directory="/tmp/nc", program_number=1001)
r["check"]["ok"], r["check"]["errors"], r["minutes"], r["tools"]
```

## Automatic CAM

`plan_cam()` plans one setup from the top for 2.5D features, and nothing in the scene
changes. The plan holds:
- the stock and the placement (Z 0 at the part's top, X0 Y0 at the stock's centre);
- the tools, numbered;
- the operations in order, each with its Fabex properties, cutting data and `why`;
- `coverage`: which faces each operation finishes, and the faces left uncovered with the
  reason;
- `rest`: sharp corners left with the tool's radius;
- `holes`: drilled or milled, and why;
- `warnings`.

`apply_plan()` makes it real and returns:
- `computed` per operation;
- `program`: the file, the minutes and the check;
- `verify`: gouge, leftover (expected / unexpected), collisions, plunges, and per
  operation the removal, air cutting and slotting.

`ok` means everything computed, the program checks, and there is no gouge or crash.
Applying again replaces only what the last plan for that part made. See
[Automatic CAM](autocam.md).

```python
plan = fx.plan_cam("Bracket", material="6061", options={"allowance_side": 0.003})
result = fx.apply_plan()
result["ok"], result["program"]["file"], result["verify"]["leftover"]["unexpected"]
```

## Results worth knowing

`compute_operation()` returns `ok`, `status`, `message`, `error`, path stats (`points`,
`length`, bounding box, `estimated_minutes` only when G-code was written), `warnings`, the
written `gcode_files`, and for B-rep sources:

- `brep_meshes`: which mesh OpenCAMLib sampled, with the deviation reached;
- `brep_fit`: the cutter fit check;
- `brep_flowline`: the Flowline pattern per object.

A calculation that fails does **not** raise: `ok` is false and `error` is
`{"code": "COMPUTE_FAILED", "message": ...}` (or `CANCELLED`), so the warnings and reports
survive. Check `result["ok"]`, not just call_json's own `ok`.

The automatic cutter fit check of a Flowline operation looks only at the faces it
machines (when it names them).

`flowline_preview()` refuses patterns over 4 million points with `TOO_LARGE`. Its details
give the estimate and `scale_steps_by`, the factor to multiply stepover and step along by.

## Errors

The codes are in `describe()["error_codes"]`. The common ones:

| Code | Meaning |
|---|---|
| `OBJECT_NOT_FOUND`, `OPERATION_NOT_FOUND` | Wrong name: the details list what exists |
| `INVALID_PROPERTY`, `INVALID_VALUE` | Rejected setting: the details give close matches or allowed values |
| `NOT_A_BREP`, `BREPKIT_UNAVAILABLE` | Not a BRepKit object, or BRepKit not found |
| `TOO_LARGE` | A pattern would be too big: the details say by how much |
| `TOOL_CONFLICT` | One T number used for different cutters (or twice in a tool library) |
| `MACHINE_LIMIT` | Beyond the machine profile: magazine, tool number, coolant M-code |
| `PLAN_INVALID`, `PLAN_STALE` | Not a plan from `plan_cam`, or the part changed or moved since planning |
| `BUSY` | Another computation is running |
| `UNSUPPORTED` | e.g. `start_compute()` in background mode |
| `COMPUTE_FAILED`, `CANCELLED` | In compute results (`result["error"]`), not raised |

## Testing

- **Pure tests** (no Blender) of the B-rep geometry in `fabex/brep/brepcam/`:
  `FABEX_BREPKIT_PATH=~/path/to/BRepKit python -m pytest tests/brep`. Use a Python with numpy
  and pytest. `python -m pytest tests/autocam` tests `fabex/autocam/camcore/`: the program
  writer (golden programs), the checker, moves, the machine profile and tools, cutting
  data, the planner (on synthetic parts) and the verifier. It needs shapely too, for
  example with the versions Blender 4.5 has:
  `uv run --no-project --python 3.11 --with pytest --with numpy==1.26.4 --with
  fabex/wheels/shapely-2.0.5-cp311-cp311-macosx_11_0_arm64.whl python -m pytest tests/autocam`.
- **Headless checks**, run with
  `FABEX_BREPKIT_PATH=... blender -b --factory-startup -P tests/<check>.py`, and again with
  `FABEX_TEST_BREPKIT_EXTENSION=1` to use the BRepKit extension:

  | Check | Covers |
  |---|---|
  | `brep_blender_check.py` | Kernel discovery |
  | `brep_ocl_check.py` | Exact meshes |
  | `brep_edges_check.py` | Edges to curves |
  | `brep_fit_check.py` | Fit check |
  | `brep_flowline_check.py` | Preview |
  | `brep_flowline_op_check.py` | The strategy: gouge check and G-code reference |
  | `brep_info_check.py` | Classification, and every suggestion run and computed |
  | `autocam_fanuc_check.py` | FANUC programs: profile, tools, cycles at the B-rep holes, chains |
  | `autocam_plan_check.py` | Automatic CAM on two parts: plan, apply, program check, verify; re-applying, an edited plan, a stale plan |
- `tests/test_api_parity.py` computes the reference operations through the UI operator and
  through the API, and compares the G-code.

## Not in the API yet

- Automatic CAM covers one setup from the top and 2.5D features. Still to come:
  - 3D finishing of fillets and curved faces;
  - rest machining;
  - the second setup (the bottom);
  - tapping, chamfering and adaptive clearing.

- Tabs (bridges) on profiles: `bpy.ops.scene.cam_bridges_add()` on the active operation.
- Chains for the other post processors are written by Fabex's own chain operator; the
  API's `export_program()` writes FANUC programs only.
- Which settings each strategy uses (Fabex's panels decide; `describe_operation_props()` lists
  them all).
- The PARALLEL pattern sits on a fixed grid, so next to a wall it can leave up to a stepover.
