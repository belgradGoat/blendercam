# B-rep Objects (BRepKit)

BRepKit is a NURBS / B-rep modelling extension for Blender. A BRepKit
object looks like an ordinary mesh, but it also carries the exact model: every face is a
NURBS surface with its trim curves, stored on the object (`obj["brepkit_data"]`). The mesh
is only a display copy.

Fabex machines a BRepKit object like any other mesh. When it can find BRepKit, it also
uses the exact model for five things a mesh can't give:

| Feature | What it does | Where |
|---|---|---|
| **Exact sampling** | The cutter is dropped onto the exact surfaces, and waterlines are traced around them, by BRepKit itself: no tessellation error | Operation > Optimisation > Exact |
| **Exact meshes for OpenCAMLib** | Where exact sampling doesn't apply, OpenCAMLib samples a mesh made from the exact surfaces, to a tolerance you set, instead of the coarse display mesh | Operation > Optimisation > Exact |
| **Edges to curves** | Curves from the exact edges, face outlines and holes: sources for Profile, Pocket and Drill | Curve Tools panel, Fabex menu |
| **Cutter fit check** | Does the cutter reach every fillet, hole and inside corner? | Cutter panel, Fabex menu |
| **Flowline strategy** | Finishing passes that follow each face's surface (around a dome, along a fillet) | Strategy: *Flowline (B-rep)* |

Everything is also available to scripts and AI agents: see [Scripting and Agent API](agent_api.md).

## Finding BRepKit

Fabex looks for BRepKit in this order:

1. the enabled BRepKit extension;
2. an importable `brepkit` Python package;
3. an installed but disabled BRepKit extension;
4. a BRepKit checkout named by the `FABEX_BREPKIT_PATH` environment variable.

If none is found, B-rep objects are machined as plain meshes and the B-rep features say
"BRepKit not found".

## Placing the part

Fabex only machines below each operation's **Depth Start** (`max_z`, 0 by default): the
top of the stock is at Z 0. A part modelled standing on the ground (Z 0) is above the
stock, and its toolpath is empty. Move the part down so that its top is at Z 0, or raise
Depth Start. B-rep operations warn when the part reaches above the depth start.

## Exact sampling

With **Optimisation > Exact** and OpenCAMLib on, operations on B-rep objects are sampled
on the exact surfaces by BRepKit's drop-cutter (`brepkit.cam`) instead of OpenCAMLib:

- **Exact B-rep Sampling**: on by default. For end mills, ball and bull nose cutters (other
  cutters use the exact mesh below).
- Every height is a real contact of the cutter with the surfaces, an edge or a corner: the
  toolpath carries no tessellation error (on a 15 mm sphere a ball-nose finish lands within
  a nanometre of the analytic surface, where the 0.01 mm mesh gives 11 micrometres).
- **Skin** (stock to leave) is exact: the cutter grown by the skin all round touches the
  surface where the cutter itself would touch the surface offset by the skin.
- **Waterline** traces BRepKit's waterlines (constant-Z loops on the exact surfaces) when
  every source of the operation is a B-rep object.
- Other objects in the same operation (plain meshes, curves) are still sampled by
  OpenCAMLib, and each point takes the higher of the two heights.

The sampled body is cached per object, placement and cutter. Objects with modifiers, and
any case BRepKit can't handle, fall back to the exact mesh with a warning.

## Exact meshes for OpenCAMLib

Where exact sampling is off or doesn't apply, operations on B-rep objects give OpenCAMLib a
mesh made from the exact surfaces:

- **Exact B-rep Mesh**: on by default.
- **Tolerance**: the largest distance between that mesh and the exact surfaces (0.01 mm by
  default). The panel shows the deviation actually reached.

The mesh is cached per object and tolerance. The display mesh is used instead when the
object has modifiers. Without OpenCAMLib (the Bullet / depth image route), the display mesh
is always used.

## B-rep Edges to Curves

**B-rep Edges to Curves** (Curve Tools panel, or the Fabex menu) makes curve objects from
the active B-rep object:

- **Creases**: sharp edges. Tangent seams and fillet edges are left out.
- **All**, **Naked** (the open edges of a sheet) or **Shared** edges.
- **Outer Loops**: each face's outline.
- **Holes**: the inner loops of faces (drilled holes, pockets, slots). A through hole is
  listed once, from its top.

**Project** flattens the curves: *Top* / *Bottom* puts each curve at its highest / lowest
point (a hole at its top or its bottom), which suits Drill and Pocket with *Min Z From:
Object*. In Edit Mode only the selected faces are used. Extracting again replaces the
curves made before, so operations keep pointing at them.

## Cutter fit check

**Check Cutter Fit** (the magnifier next to *Check Fit on B-rep* in the Cutter panel, or
the Fabex menu) compares the active operation's cutter with the exact model:

- the tightest **concave radius** a ball, a bull nose's corner or a ball cone's tip must fit
  into (fillets, holes, dishes);
- the tightest **inside radius of steep walls**, which limits flat-bottomed cutters (pocket
  corners, holes);
- **sharp inside corners** that no round cutter reaches, with the material left in them.

The result shows under the cutter settings. In Edit Mode the faces that are too tight get
selected. With *Check Fit on B-rep* on, surface strategies run the check before computing
and add warnings.

## Flowline (B-rep)

A finishing strategy for B-rep objects. Passes follow each face's own parameter lines:
rings around a dome's pole, along a fillet, across a blend. Adjacent passes are never
further apart than the stepover, measured on the surface, and passes end exactly on each
face's trim (holes, outline). Heights come from Fabex's usual tool-drop sampling
(OpenCAMLib when available), so the cutter never goes below the model.

### Options

- **Faces**: B-rep face ids to machine, e.g. `12, 15, 31`. Empty means every face the cutter
  reaches from above. **Use Selected Faces** (in Edit Mode) fills it from the selection.
- **Direction**: *Auto* (rings around a pole, otherwise along the longer direction), *U* or
  *V*.
- **Max Slope**: leaves out surface steeper than this angle from level. Steep walls are
  better cut by Waterline or Profile.
- **Reverse**: runs every pass the other way. One-way passes otherwise follow the
  climb / conventional setting.
- **Skin**: material to leave.
- **Toolpath**: **Stepover** (the largest distance between passes) and **Detail** (the
  distance between points along a pass).
- **Preview Flowline**: draws the passes as a curve object, and lists the passes and the
  largest gap per object.

Layers are switched off when you pick Flowline, because it is a single finishing pass over
the surface. Movement *Meander* gives zigzag passes; other movement types cut one way.

The *Fin_Ball_3,0_Flowline* preset sets up a 3 mm ball-nose Flowline operation.

## Limitations

- Each face is patterned on its own: passes don't line up across face boundaries.
- Fabex machines from above (3 axis): undercut faces get no passes.
- On very large parts the exact mesh may not reach a very fine tolerance; the panel and the
  compute warnings show the deviation reached.
- Exact sampling with a flat end mill is several times slower than with ball and bull nose
  cutters (its rim's contact is searched around the rim); waterlines take a few seconds a
  level on a filleted part.
