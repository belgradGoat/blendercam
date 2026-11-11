# Automatic CAM

Fabex can plan the machining of a B-rep part by itself, for a FANUC machining centre
(a Robodrill by default, see [FANUC Machining Centres](fanuc.md)) in aluminium. It then makes
the plan, computes it, writes the program and checks it against the part:

1. **Plan** (`plan_cam`) chooses the stock, the placement, the tools and the operations,
   with cutting data and a reason for each. It also lists which faces each operation
   finishes and which it doesn't. Nothing in the scene changes.
2. **Apply** (`apply_plan`) moves the part into place and creates the operations and a
   chain. It then computes every path, writes the program and verifies it.
3. **Verify** (`verify_program`) simulates the program on the stock and compares the
   result with the part. It reports gouges, leftover material and collisions.

This version plans **one setup, from the top**, for **2.5D** features: tops, floors,
vertical walls, steps, pockets, slots, windows, and round holes. Other faces are listed as
not machined, with the reason:
- the bottom, which is machined from the other side (a second setup, planned later);
- fillets, chamfers and other 3D faces, which need 3D finishing;
- undercuts, which can't be reached from above.

## In Blender

The **[ Auto CAM ]** panel (Properties > Render, next to the Machine panel) holds the inputs:
- the part;
- the material (6061, 6082, 7075, 2017A);
- the stock allowances: top, sides, and grip below the part;
- where X0 Y0 goes;
- the program number.

**Plan** shows the operations with their tools, what is left for later, corners the tools
can't reach, and warnings. **Apply** makes everything. While Apply computes, Blender
doesn't respond. **Verify** runs the check again, for example after you edit operations
and write the program again.

## From a script or an agent

```python
plan = fx.plan_cam("Bracket", material="6061")
for op in plan["operations"]:
    print(op["name"], op["tool"], op["cutting"]["rpm"], op["cutting"]["feed"], op["why"])
plan["coverage"]["uncovered"]   # faces left for later, with the reason
plan["rest"]                    # sharp inside corners the end mills leave round
plan["warnings"]                # e.g. a hole at a tap drill size

result = fx.apply_plan()        # or fx.apply_plan(edited_plan)
result["ok"]                    # computed, the program checks, no gouge or crash
result["program"]["file"], result["program"]["minutes"], result["program"]["check"]
result["verify"]["gouge"], result["verify"]["leftover"]["unexpected"]
```

`plan_cam` returns the plan without its geometry: the region outlines and the drill
points. The full plan is kept in the scene. An agent can edit the plan it got back and pass
it to `apply_plan`, for example:
- change cutting data or other operation properties;
- drop an operation;
- change the T numbers.

The geometry is found again by the operation ids.

## How the plan is made

**Stock and placement.** The stock is the part's bounding box plus the allowances:
- 0.5 mm on top, which is faced off;
- 2 mm on each side;
- 6 mm of grip below the part, held in the vise.

Or give `stock=[x, y, z]` for the size you have. The part is placed so that **Z 0 is its top**
and **X0 Y0 is the stock's centre** (or its front-left corner with
`origin='stock_corner'`), on the profile's work offset (G54). Apply moves the part object
there, once.

**Level bands.** The heights of the part's level faces cut the stock into bands, together
with the tops of other faces and the cut-through depth (0.2 mm below the part's bottom).
Within a band the part's cross-section doesn't change. What has to go in a band is the
stock minus that cross-section, split into regions:
- the outside;
- pockets;
- steps and slots;
- windows;
- holes that aren't drilled.

The same region in consecutive bands is one region.

**Tools.** Each region gets one end mill: the largest in the tool library that reaches
everything in it, except sharp inside corners. A closed region also needs room for a helical
entry. Tools are capped at 16 mm, and at a quarter of the stock's width. A tool already in
use is preferred when it is nearly as big. Corners that keep the tool's radius are listed in
`rest`.

**Operations per region:**

| Operation | Strategy | What it does |
|---|---|---|
| rough | Pocket | Helical entry (ramp if it doesn't fit); leaves 0.2 mm on walls and floors |
| floor | Pocket | One pass over a real floor, still 0.2 mm off the walls |
| walls | Profile inside the region | Full height in one pass (layers only past the flute length); the outside is finished last |

The band above the part's top is the **facing** pass.

**Holes.** A round vertical hole is drilled when the library has that drill and its bottom
is through or a drill point (cone):
- It is spot drilled first, to the hole's diameter. If the spot drill would hit a wall
  next to the hole, a smaller one is used.
- It is then drilled with G81, G73 (chip breaking) or G83 by depth.
- Through holes go past the bottom by the drill's point plus 0.5 mm.

Other holes, including flat-bottomed ones, are milled as regions. Holes at a tap drill size
are flagged, because tapping is not planned yet.

**Order:** face, rough (band by band from the top), spot, drill, floors, walls. Tools are
numbered by first use, unless your turret map (`set_tool_library`) fixes the numbers.

**Cutting data** (`cutting_data(tool, material, engagement)`) assumes carbide tools and flood
coolant. These are conservative starting values:
- The spindle's maximum caps the speed. The chip load stays the same, so the feed drops
  with it.
- The depth of cut is 0.5 × D for pocketing, which tolerates the full-width first ring of
  each layer.
- The feed is reduced if a cut would need more than 80% of the spindle's power.
- Drills feed by D per revolution with a peck of one diameter.

## Verification

The program is read as G-code, so what is verified is what the machine gets. The stock is
a height map (0.1 mm pixels by default), and each move lowers it by the cutter's swept
shape. Before each move is applied, it is checked for:
- **a rapid through material** (a crash);
- **the holder** or **a vise jaw** hitting something;
- **material above the flutes**;
- **a mill feeding straight down** into material;
- **slotting**: the width of cut along each stretch.

Afterwards the stock is compared with the part's exact mesh, face by face:
- **gouges**: part material cut away;
- **leftover**: material left where the part should end.

Next to vertical walls both are measured sideways from the wall, elsewhere vertically.
Leftover the plan expected (faces left for later, the radius in sharp corners) is listed
under `expected`. Anything else is listed under `unexpected`. `ok` means no gouge and no crash.

The resolution limits what the height map sees. A sideways gouge smaller than a pixel only
shows where it covers a pixel centre.

## Before the first part

Air-cut the program first, with the work offset raised. Check the profile against the
machine (see [FANUC Machining Centres](fanuc.md)). Also:
- **Workholding.** The plan assumes a vise holding the grip below the part, with the jaws
  2 mm below the deepest cut.
- **Cutting data.** It is a starting point: compare it with your tool maker's data.
- **The bottom face** is still to be machined in a second setup.
