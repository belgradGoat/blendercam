"""Automatic 2.5D machining plan for one setup (OP1: from the top), in aluminium.

    plan(part, tools, machine, options) -> dict   (JSON-able; apply it with apply_plan)

The part model (camcore/part.py) is placed so that Z 0 is the part's top and X0 Y0 the
stock's centre (or front-left corner), the stock is the part's box plus allowances, held
in a vise by `grip` of extra material below the part.

How the work is found - level bands:
the heights of the level faces (top, floors), the tops of non-level faces and the cut-through
depth (part bottom - `through`) cut the stock into bands. Within a band the part's
cross-section (section()) doesn't change, so the material to remove there is
(stock + margin) - section, split into connected regions: the outside, pockets, steps, slots,
windows, holes that aren't drilled. The same region in consecutive bands is one region.
Every region gets one end mill - the largest that reaches every wall (only sharp inside
corners may be left, with the tool's radius) and, for closed regions, room for a helical
entry - and up to three operations:
  rough   POCKET, helical entry, leaving `skin` on walls and floors
  floor   POCKET, one pass over the region's floor (when the band ends on a real floor)
  walls   CUTOUT inside the region, full height (layers only past the flute length)
The band above the part's top is the facing pass.

Round vertical holes with a matching drill (and a through or cone bottom) are spot drilled
and drilled instead; the rest is milled as regions.

Order: face, rough (band by band, top down), spot, drill, floors, walls (outside last).
Tools are numbered in the order they are first used, unless the library fixes the number.
Every face is listed as covered (by which operations) or uncovered (with the reason).

Lengths in metres, feeds m/min (API units).
"""

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from math import ceil, radians, tan

import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box
from shapely.geometry.polygon import orient

from . import cutting
from .errors import CamcoreError, check_keys
from .machine import MachineProfile
from .part import Z_TOL, PartModel, as_rings, polygons
from .tools import Tool, to_fabex_props

PLAN_VERSION = 1
ORIGINS = ("stock_center", "stock_corner")
QS = 16  # quad segments for the planner's round offsets
SHARP_TURN = 30.0  # degrees: a corner turning more than this is sharp (fillets are sampled finer)
CORNER_REACH = 1.5  # residue within this many tool radii of a sharp corner is corner residue
HELIX_FRACTION = 0.5  # helix radius / tool radius (Fabex: helix_diameter %, of the radius)
TAP_DRILLS = {2.5: "M3", 3.3: "M4", 4.2: "M5", 5.0: "M6", 6.8: "M8", 8.5: "M10", 10.2: "M12"}
DRILL_MATCH = 1e-5  # m: a drill this close to the hole's diameter drills it
CLEARANCE = 0.005  # m above the stock: rapids / the free height
DRILL_LEVEL_GAP = 0.003  # holes whose tops differ more are drilled from separate R planes


@dataclass
class PlanOptions:
    material: str = "6061"
    allowance_top: float = 0.0005  # stock above the part's top (faced off)
    allowance_side: float = 0.002  # stock around the part, each side
    grip: float = 0.006  # stock below the part, held by the vise
    through: float = 0.0002  # how far OP1 cuts below the part's bottom
    jaw_clearance: float = 0.002  # the deepest cut stays this far above the vise jaws
    stock: list | None = None  # [x, y, z] stock size; None = the part + allowances
    origin: str = "stock_center"
    skin: float = 0.0002  # left by roughing on walls and floors
    spot: bool = True  # spot drill before drilling
    max_diameter: float = 0.016  # largest end mill to use
    min_diameter: float = 0.001  # smallest end mill to use
    tolerance: float = 1e-5
    program_number: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "PlanOptions":
        data = dict(data or {})
        check_keys(cls, data, "plan option")
        o = cls(**data)
        o.validate()
        return o

    def validate(self) -> None:
        bad = []
        for name in ("allowance_top", "allowance_side", "grip", "through", "jaw_clearance"):
            if getattr(self, name) < 0:
                bad.append(f"{name} must not be negative")
        for name in ("skin", "max_diameter", "min_diameter", "tolerance"):
            if not getattr(self, name) > 0:
                bad.append(f"{name} must be positive")
        if self.origin not in ORIGINS:
            bad.append(f"origin must be one of {ORIGINS}")
        if self.min_diameter > self.max_diameter:
            bad.append("min_diameter is larger than max_diameter")
        if self.stock is not None and (
            len(self.stock) != 3 or not all(float(s) > 0 for s in self.stock)
        ):
            bad.append("stock is [x, y, z] (m), all positive")
        if self.grip < self.through + self.jaw_clearance:
            bad.append("grip must be at least through + jaw_clearance")
        if bad:
            raise CamcoreError("INVALID_VALUE", "; ".join(bad), {"options": self.to_dict()})


# --------------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------------


def _mitre(g, d):
    return g.buffer(d, join_style="mitre", mitre_limit=2.0)


def _close(g, eps: float = 1e-6):
    """Fill hairline cracks between faces (a morphological closing, sharp corners kept)."""
    if g.is_empty:
        return g
    return _mitre(_mitre(g, eps), -eps)


def _thick(g, tol: float) -> bool:
    """Is any part of g thicker than `tol`?"""
    return not g.is_empty and not g.buffer(-tol / 2, quad_segs=4).is_empty


def _sharp_corners(g, turn_deg: float = SHARP_TURN) -> list:
    """Corners of a region where its boundary turns inwards (round the region's inside) by
    more than turn_deg: where a round tool can't reach."""
    out = []
    lim = np.radians(turn_deg)
    for p in polygons(g):
        p = orient(p, 1.0)  # exterior CCW, holes CW: the inside is always on the left
        for ring in [p.exterior] + list(p.interiors):
            P = np.asarray(ring.coords)[:-1, :2]
            if len(P) < 3:
                continue
            a = P - np.roll(P, 1, axis=0)
            b = np.roll(P, -1, axis=0) - P
            cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            dot = (a * b).sum(1)
            turn = np.arctan2(cross, dot)  # > 0: a left turn = a corner of the inside
            out.extend(P[turn > lim].tolist())
    return out


def _fit(C, stock, r: float, skin: float, closed: bool, tol: float) -> tuple[bool, list]:
    """Can an end mill of radius r machine region C? (ok, residue pieces [(x, y, size)]).
    Everything of C inside the stock must be reachable, except near sharp corners; a closed
    region also needs room for a helical entry."""
    if closed and C.buffer(-(r + skin) * (1 + HELIX_FRACTION), quad_segs=QS).is_empty:
        return False, []
    opened = C.buffer(-r, quad_segs=QS).buffer(r, quad_segs=QS)
    residue = C.intersection(stock).difference(opened)
    if not _thick(residue, tol):
        return True, []
    corners = _sharp_corners(C)
    zone = shapely.union_all([Point(x, y).buffer(CORNER_REACH * r, quad_segs=8) for x, y in corners]) if corners else Polygon()  # fmt: skip
    pieces = []
    for piece in polygons(residue):
        if not _thick(piece, tol):
            continue
        if _thick(piece.difference(zone), tol):
            return False, []
        c = piece.representative_point()
        pieces.append((c.x, c.y, float(np.sqrt(piece.area))))
    return True, pieces


def _margin(r: float, skin: float) -> float:
    """How far a region reaches past the stock edge for a tool of radius r: always room for a
    ring of the tool between the stock edge and the part, and the stock edge within reach."""
    return 2 * r + 2 * skin + 0.0005


def _trim(C, stock, r: float, skin: float):
    return C.intersection(_mitre(stock, _margin(r, skin)))


# --------------------------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------------------------


@dataclass
class Region:
    band_top: float
    band_bottom: float
    polygon: object
    open: bool  # reaches outside the stock (entered from outside)
    floor: bool  # the band ends on a floor of the part inside this region
    floor_faces: list
    facing: bool = False
    tool: Tool | None = None
    residue: list | None = None


def _bands(part: PartModel, stock_top: float, z_end: float, drilled: list) -> list:
    zs = {stock_top, z_end}
    zs.update(z for z in part.level_heights() if z_end < z < stock_top)
    zs.update(o.z_max for o in part.obstacles if z_end < o.z_max < stock_top)
    zs.update(h.z_top for h in drilled if z_end < h.z_top < stock_top)
    ordered = []
    for z in sorted(zs, reverse=True):
        if not ordered or ordered[-1] - z > Z_TOL:
            ordered.append(z)
    return list(zip(ordered[:-1], ordered[1:]))


def _regions(part, stock, stock_top, z_end, drilled, margin, tol) -> list:
    outer = _mitre(stock, margin)
    ring = outer.difference(stock)
    out = []
    for z_hi, z_lo in _bands(part, stock_top, z_end, drilled):
        S = _close(part.section((z_hi + z_lo) / 2, drilled))
        R = outer.difference(S)
        for C in polygons(R):
            if not _thick(C, 2 * tol) or C.area < (10 * tol) ** 2:
                continue
            floor_faces = [
                lv.face
                for lv in part.levels
                if abs(lv.z - z_lo) <= Z_TOL and lv.polygon.intersection(C).area > (10 * tol) ** 2
            ]
            out.append(
                Region(
                    band_top=z_hi,
                    band_bottom=z_lo,
                    polygon=C,
                    open=bool(C.intersects(ring)),
                    floor=bool(floor_faces),
                    floor_faces=floor_faces,
                    facing=z_lo >= part.top - Z_TOL,
                )
            )
    return _merge(out, tol)


def _same(a, b, tol) -> bool:
    return a.symmetric_difference(b).area <= (10 * tol) ** 2 * 4


def _merge(regions: list, tol) -> list:
    """The same region in consecutive bands (no floor between them) becomes one."""
    merged = []
    for r in regions:
        prev = next(
            (
                m
                for m in merged
                if abs(m.band_bottom - r.band_top) <= Z_TOL
                and not m.floor
                and not m.facing
                and not r.facing
                and m.open == r.open
                and _same(m.polygon, r.polygon, tol)
            ),
            None,
        )
        if prev is None:
            merged.append(r)
        else:
            prev.band_bottom = r.band_bottom
            prev.floor, prev.floor_faces = r.floor, r.floor_faces
    return merged


# --------------------------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------------------------


def _mm(x: float) -> str:
    return f"{round(x * 1000, 3) + 0.0:g} mm"


class _Planner:
    def __init__(self, part, tools, machine, opts, object_name, body_hash):
        self.part0 = part
        self.tools = list(tools)
        self.machine = machine
        self.opts = opts
        self.name = object_name or part.name
        self.body_hash = body_hash
        self.mat_key, self.mat = cutting.material(opts.material)
        self.warnings = list(part.warnings)
        self.ops = []
        self.geometry = {}
        self.used = []  # tools in order of first use
        self.cutting_cache = {}

    # --- stock and placement -----------------------------------------------------------------

    def place(self):
        o, p = self.opts, self.part0
        lo, hi = np.asarray(p.bbox_min, float), np.asarray(p.bbox_max, float)
        size = hi - lo
        if o.stock is not None:
            stock_size = np.asarray(o.stock, float)
            if np.any(stock_size[:2] < size[:2] - 1e-9) or stock_size[2] < size[2] + o.through:
                raise CamcoreError(
                    "INVALID_VALUE",
                    f"The stock {np.round(stock_size * 1000, 3).tolist()} mm is smaller than "
                    f"the part {np.round(size * 1000, 3).tolist()} mm",
                )
            side = (stock_size[:2] - size[:2]) / 2
            top = min(o.allowance_top, stock_size[2] - size[2] - o.through - o.jaw_clearance)
            top = max(top, 0.0)
            grip = stock_size[2] - size[2] - top
        else:
            side = np.array([o.allowance_side, o.allowance_side])
            top, grip = o.allowance_top, o.grip
            stock_size = np.array(
                [size[0] + 2 * side[0], size[1] + 2 * side[1], size[2] + top + grip]
            )
        s_lo_xy = lo[:2] - side
        if o.origin == "stock_center":
            shift_xy = -(s_lo_xy + stock_size[:2] / 2)
        else:
            shift_xy = -s_lo_xy
        self.T = np.array([shift_xy[0], shift_xy[1], -hi[2]])
        self.part = p.translated(*self.T)
        s_lo = np.array([s_lo_xy[0] + shift_xy[0], s_lo_xy[1] + shift_xy[1], -size[2] - grip])
        self.stock_min = s_lo
        self.stock_max = s_lo + stock_size
        self.stock_top = float(self.stock_max[2])
        self.grip = float(grip)
        self.allowance_top = float(top)
        self.stock_poly = box(s_lo[0], s_lo[1], self.stock_max[0], self.stock_max[1])
        self.z_end = self.part.bottom - o.through
        self.clear = self.stock_top + CLEARANCE
        jaw_top = self.z_end - o.jaw_clearance
        if jaw_top <= self.stock_min[2] + 1e-9:
            self.warnings.append(
                "No stock is left below the deepest cut for the vise to grip: increase grip"
            )
        jaw = 0.02  # jaw thickness (only for the collision check)
        width = max(0.15, stock_size[0] + 0.05)
        cx = (self.stock_min[0] + self.stock_max[0]) / 2
        self.jaws = [
            {"min": [cx - width / 2, self.stock_min[1] - jaw, self.stock_min[2] - 0.05],
             "max": [cx + width / 2, self.stock_min[1], jaw_top]},
            {"min": [cx - width / 2, self.stock_max[1], self.stock_min[2] - 0.05],
             "max": [cx + width / 2, self.stock_max[1] + jaw, jaw_top]},
        ]  # fmt: skip
        self.jaw_top = jaw_top

    # --- tools ------------------------------------------------------------------------------

    def _cap(self) -> float:
        sx, sy = (self.stock_max - self.stock_min)[:2]
        return min(self.opts.max_diameter, max(0.006, 0.25 * min(sx, sy)))

    def _mills(self, cap: float) -> list:
        o = self.opts
        return sorted(
            (
                t
                for t in self.tools
                if t.type == "END" and o.min_diameter - 1e-9 <= t.diameter <= cap + 1e-9
            ),
            key=lambda t: -t.diameter,
        )

    def _use(self, tool: Tool) -> Tool:
        if all(t.id != tool.id for t in self.used):
            self.used.append(tool)
        return tool

    def _choose(self, reg: Region, cap: float):
        """(tool, residue, notes) for a region: the largest end mill that fits, or a tool
        already in use if it is nearly as big."""
        depth = self.stock_top - reg.band_bottom
        skin = 0.0 if reg.facing else self.opts.skin
        tol = max(self.opts.tolerance, 1e-6)

        def ok(t):
            if t.reach < depth + 0.001:
                return None
            C = (
                _trim(reg.polygon, self.stock_poly, t.diameter / 2, skin)
                if reg.open
                else reg.polygon
            )
            good, residue = _fit(C, self.stock_poly, t.diameter / 2, skin, not reg.open, tol)
            return residue if good else None

        best, best_res = None, None
        for t in self._mills(cap):
            res = ok(t)
            if res is not None:
                best, best_res = t, res
                break
        if best is None:
            return None, None
        for t in sorted(self.used, key=lambda t: -t.diameter):
            if t.type != "END" or t.diameter > best.diameter + 1e-12 or t is best:
                continue
            if t.diameter < 0.6 * best.diameter:
                break
            res = ok(t)
            if res is not None:
                return t, res
        return best, best_res

    def _cutting(self, tool: Tool, engagement: str) -> cutting.CuttingData:
        key = (tool.id, engagement)
        if key not in self.cutting_cache:
            self.cutting_cache[key] = cutting.cutting_data(
                tool, self.mat_key, engagement, self.machine
            )
        return self.cutting_cache[key]

    # --- operations -------------------------------------------------------------------------

    def _add(self, phase, strategy, tool, cd, props, geometry, faces, why, **extra):
        op_id = f"op{len(self.ops) + 1}"
        self.ops.append(
            {
                "id": op_id,
                "phase": phase,
                "strategy": strategy,
                "tool": tool.id,
                "cutting": cd.to_dict(),
                "props": props,
                "faces": sorted(set(faces)),
                "why": why,
                **extra,
            }
        )
        self.geometry[op_id] = geometry
        return op_id

    def _common(self, tool, cd, plunge_pct=None) -> dict:
        props = to_fabex_props(tool)
        props.pop("cutter_id", None)  # numbered at the end
        props.update(
            {
                "feedrate": round(cd.feed, 5),
                "plunge_feedrate": (
                    plunge_pct if plunge_pct is not None
                    else round(100 * cd.plunge_feed / cd.feed, 2)
                ),  # fmt: skip
                "spindle_rpm": round(cd.rpm, 1),
                "movement.free_height": round(self.clear, 6),
                "movement.type": "CLIMB",
                "movement.spindle_rotation": "CW",
                "autocam.coolant": "FLOOD",
                "autocam.aicc": True,
            }
        )
        return props

    def _region_geometry(self, reg, tool):
        C = reg.polygon
        if reg.open:
            C = _trim(C, self.stock_poly, tool.diameter / 2, 0.0 if reg.facing else self.opts.skin)
        return C

    def _pocket_props(self, tool, cd, z_top, z_bottom, skin, layers, stepover) -> dict:
        p = self._common(tool, cd)
        p.update(
            {
                "min_z_from": "CUSTOM",
                "max_z": round(z_top, 7),
                "min_z": round(z_bottom, 7),
                "use_layers": bool(layers),
                "stepdown": round(min(cd.ap, max(z_top - z_bottom, 1e-5)), 7),
                "skin": round(skin, 7),
                "distance_between_paths": round(stepover, 7),
                "pocket_option": "INSIDE",
                "pocket_type": "PERIMETER",
                "straight": True,
                "first_down": False,
                "movement.helix_enter": True,
                "movement.helix_diameter": 100 * HELIX_FRACTION,
                "movement.ramp_in_angle": round(radians(max(cd.ramp_angle, 1.0)), 6),
                "movement.stay_low": True,
            }
        )
        return p

    def regions_ops(self, regions, drilled):
        cap = self._cap()
        faced = [r for r in regions if r.facing]
        rest = [r for r in regions if not r.facing]
        # big regions choose first, so later ones can share their tools
        for reg in sorted(rest, key=lambda r: -r.polygon.area):
            reg.tool, reg.residue = self._choose(reg, cap)
            if reg.tool is not None:
                self._use(reg.tool)
        for reg in faced:  # the largest end mill already chosen faces the top
            chosen = [r.tool for r in rest if r.tool is not None]
            if chosen:
                reg.tool, reg.residue = max(chosen, key=lambda t: t.diameter), []
            else:
                reg.tool, reg.residue = self._choose(reg, cap)
            if reg.tool is not None:
                self._use(reg.tool)
        self.unreached = [r for r in regions if r.tool is None]
        for r in self.unreached:
            c = r.polygon.representative_point()
            self.warnings.append(
                f"A region at X{c.x * 1000:.1f} Y{c.y * 1000:.1f} "
                f"(Z {_mm(r.band_top)} to {_mm(r.band_bottom)}) fits no end mill in the "
                "library: it is left uncut"
            )
        regions = [r for r in regions if r.tool is not None]
        self.regions = regions

        # facing
        for reg in faced:
            t = self._use(reg.tool)
            cd = self._cutting(t, "face")
            depth = reg.band_top - reg.band_bottom
            ap = min(0.001, 0.9 * t.flute_length)
            props = self._pocket_props(t, cd, reg.band_top, reg.band_bottom, 0.0, depth > ap, cd.ae)
            props["stepdown"] = round(min(ap, depth), 7)
            tops = [lv.face for lv in self.part.levels if abs(lv.z - self.part.top) <= Z_TOL]
            self._add(
                "face",
                "POCKET",
                t,
                cd,
                props,
                {"kind": "region", "z": reg.band_bottom, "polygons": as_rings(self._region_geometry(reg, t))},
                tops,
                f"Face the top: {_mm(depth)} off the stock with a {_mm(t.diameter)} end mill, "
                f"{_mm(cd.ae)} stepover.",
                z_top=reg.band_top,
                z_bottom=reg.band_bottom,
            )  # fmt: skip

        # roughing, band by band from the top
        self.rough_ids = {}
        for reg in sorted(regions, key=lambda r: (-r.band_top, r.band_bottom)):
            if reg.facing:
                continue
            t = self._use(reg.tool)
            cd = self._cutting(t, "pocket")
            # Fabex's skin is sideways only: stop `skin` above a real floor
            bottom = reg.band_bottom + (self.opts.skin if reg.floor else 0.0)
            props = self._pocket_props(t, cd, reg.band_top, bottom, self.opts.skin, True, cd.ae)
            where = "outside" if reg.open else "pocket"
            self.rough_ids[id(reg)] = self._add(
                "rough",
                "POCKET",
                t,
                cd,
                props,
                {"kind": "region", "z": reg.band_bottom, "polygons": as_rings(self._region_geometry(reg, t))},
                [],
                f"Rough the {where} region from Z {_mm(reg.band_top)} to {_mm(reg.band_bottom)}: "
                f"{_mm(t.diameter)} end mill, {_mm(cd.ap)} deep x {_mm(cd.ae)} wide, helical "
                f"entry at {cd.ramp_angle:g} deg, leaving {_mm(self.opts.skin)}.",
                z_top=reg.band_top,
                z_bottom=reg.band_bottom,
                region={"open": reg.open, "area": round(reg.polygon.area, 10)},
            )  # fmt: skip

    def hole_ops(self, drilled):
        o = self.opts
        spots = sorted((t for t in self.tools if t.type == "SPOT"), key=lambda t: t.diameter)
        if o.spot and drilled and not spots:
            self.warnings.append("The library has no spot drill: holes are drilled unspotted")
        by_tool = {}
        for h, tool in drilled:
            by_tool.setdefault(tool.id, (tool, []))[1].append(h)
        # spot: to the hole's diameter with a spot drill at least that big, if it clears the
        # walls above the hole; else a smaller one to 80% of its diameter; else none
        if o.spot and spots:
            groups, dia = {}, {}
            for h, _ in drilled:
                d = h.diameter
                above = self.part.section(h.z_top + 10 * Z_TOL)  # not the hole's own face

                def clear(s):
                    disc = Point(h.center[0], h.center[1]).buffer(
                        s.diameter / 2 + 0.0002, quad_segs=16
                    )
                    return disc.intersection(above).area <= 1e-10

                spot = next((s for s in spots if s.diameter >= d - 1e-9 and clear(s)), None)
                dia[h.index] = d
                if spot is None:
                    spot = next((s for s in reversed(spots) if s.diameter < d and clear(s)), None)
                    if spot is None:
                        self.warnings.append(
                            f"Hole {h.index}: no spot drill clears the walls next to it - not spotted"
                        )
                        continue
                    dia[h.index] = 0.8 * spot.diameter
                groups.setdefault(spot.id, (spot, []))[1].append(h)
            for spot, holes in groups.values():
                for level in self._levels(holes):
                    t = self._use(spot)
                    cd = self._cutting(t, "spot")
                    pts = []
                    for h in level:
                        depth = (dia[h.index] / 2) / tan(radians(t.tip_angle / 2))
                        pts.append([h.center[0], h.center[1], h.z_top - depth])
                    full = all(dia[h.index] >= h.diameter - 1e-9 for h in level)
                    self._drill_op("spot", t, cd, level, pts, "G81", 0.0,
                                   f"Spot drill {len(level)} hole(s) with the {_mm(t.diameter)} "
                                   f"{t.tip_angle:g} deg spot drill, "
                                   + ("to the hole diameter." if full else
                                      "smaller than the hole (a bigger one would hit a wall)."))  # fmt: skip
        # drill
        for tool, holes in sorted(by_tool.values(), key=lambda x: x[0].diameter):
            for level in self._levels(holes):
                t = self._use(tool)
                cd = self._cutting(t, "drill")
                tip = (t.diameter / 2) / tan(radians(t.tip_angle / 2))
                pts, faces, notes = [], [], []
                for h in level:
                    if h.bottom == "through":
                        z = h.z_bottom - tip - 0.0005
                    else:
                        z = h.z_bottom
                        if h.tip_angle and abs(h.tip_angle - t.tip_angle) > 5:
                            notes.append(
                                f"hole {h.index}: the model's {h.tip_angle:g} deg cone bottom "
                                f"will be {t.tip_angle:g} deg"
                            )
                    pts.append([h.center[0], h.center[1], z])
                    faces += h.faces
                deepest = max(lv.z_top - p[2] for lv, p in zip(level, pts))
                kind = (
                    "G81"
                    if deepest <= t.diameter
                    else "G73" if deepest <= 4 * t.diameter else "G83"
                )
                why = (f"Drill {len(level)} x {_mm(t.diameter)} hole(s), {_mm(deepest)} deep "
                       f"({kind}" + (f", {_mm(cd.peck)} pecks" if kind != "G81" else "") + ")"
                       + ("; through holes go " + _mm(tip + 0.0005) + " past the bottom"
                          if any(h.bottom == "through" for h in level) else "") + ".")  # fmt: skip
                self._drill_op("drill", t, cd, level, pts, "AUTO", cd.peck, why, faces, notes)

    def _levels(self, holes) -> list:
        """Holes grouped by the height of their tops (one R plane per group)."""
        groups = []
        for h in sorted(holes, key=lambda h: -h.z_top):
            if groups and groups[-1][0].z_top - h.z_top <= DRILL_LEVEL_GAP:
                groups[-1].append(h)
            else:
                groups.append([h])
        return groups

    def _drill_op(self, phase, t, cd, holes, pts, cycle, peck, why, faces=(), notes=()):
        props = self._common(t, cd, plunge_pct=100.0)
        props.update(
            {
                "min_z_from": "OBJECT",
                "max_z": round(max(h.z_top for h in holes), 7),
                "use_layers": False,
                "drill_type": "ALL_POINTS",
                "autocam.drill_cycle": cycle,
                "autocam.peck": round(peck, 7),
                "autocam.r_clearance": 0.002,
                "autocam.aicc": False,
            }
        )
        self._add(
            phase,
            "DRILL",
            t,
            cd,
            props,
            {"kind": "points", "points": np.round(np.asarray(pts), 7).tolist()},
            list(faces),
            why,
            holes=[h.index for h in holes],
            notes=list(notes),
            z_top=max(h.z_top for h in holes),
            z_bottom=min(p[2] for p in pts),
        )

    def finish_ops(self):
        floors, walls = [], []
        for reg in self.regions:
            if reg.facing:
                continue
            t = reg.tool
            C = self._region_geometry(reg, t)
            if reg.floor:
                # the floor to size, the walls still `skin` away (the walls pass takes it)
                cd = self._cutting(t, "floor")
                z_top = reg.band_bottom + self.opts.skin + 0.0001
                props = self._pocket_props(t, cd, z_top, reg.band_bottom, 0.0, False, cd.ae)
                floors.append((t, reg, cd, props, _mitre(C, -self.opts.skin)))
            cd = self._cutting(t, "finish")
            height = reg.band_top - reg.band_bottom
            cap = 0.9 * t.flute_length
            p = self._common(t, cd)
            p.update(
                {
                    "min_z_from": "CUSTOM",
                    "max_z": round(reg.band_top, 7),
                    "min_z": round(reg.band_bottom, 7),
                    "use_layers": bool(height > cap),
                    "stepdown": round(min(cap, max(height, 1e-5)), 7),
                    "skin": 0.0,
                    "cut_type": "INSIDE",
                    "outlines_count": 1,
                    "straight": True,
                    "first_down": False,
                    "movement.stay_low": True,
                }
            )
            walls.append((t, reg, cd, p, C))
        order = {t.id: i for i, t in enumerate(self.used)}
        for t, reg, cd, props, C in sorted(
            floors, key=lambda x: (order[x[0].id], -x[1].band_bottom)
        ):
            self._add(
                "floor",
                "POCKET",
                t,
                cd,
                props,
                {"kind": "region", "z": reg.band_bottom, "polygons": as_rings(C)},
                reg.floor_faces,
                f"Finish the floor at Z {_mm(reg.band_bottom)}: one pass, {_mm(cd.ae)} stepover.",
                z_top=props["max_z"],
                z_bottom=reg.band_bottom,
            )  # fmt: skip
        for t, reg, cd, props, C in sorted(
            walls, key=lambda x: (x[1].open, order[x[0].id], -x[1].band_top)
        ):
            faces = self._walls_of(reg, C)
            what = "the outside" if reg.open else "the walls"
            self._add(
                "walls",
                "CUTOUT",
                t,
                cd,
                props,
                {"kind": "region", "z": reg.band_bottom, "polygons": as_rings(C)},
                faces,
                f"Finish {what} from Z {_mm(reg.band_top)} to {_mm(reg.band_bottom)} in "
                + (f"{ceil((reg.band_top - reg.band_bottom) / props['stepdown'] - 1e-9)} passes"
                   if props["use_layers"] else "one pass") + ", climb milling.",
                z_top=reg.band_top,
                z_bottom=reg.band_bottom,
            )  # fmt: skip

    def _walls_of(self, reg, C) -> list:
        boundary = C.boundary
        near = max(10 * self.opts.tolerance, 2e-5)
        out = []
        for w in self.part.walls:
            if w.z_max <= reg.band_bottom + Z_TOL or w.z_min >= reg.band_top - Z_TOL:
                continue
            if not len(w.points):
                continue
            d = shapely.distance(shapely.points(w.points), boundary)
            if float(np.median(d)) <= near:
                out.append(w.face)
        return out

    # --- holes ------------------------------------------------------------------------------

    def choose_holes(self):
        """[(hole, drill)] to drill; the rest stay as regions to mill."""
        drills = [t for t in self.tools if t.type == "DRILL"]
        drilled, self.hole_rows = [], []
        for h in self.part.holes:
            row = {"hole": h.index, "diameter": h.diameter, "kind": h.kind, "bottom": h.bottom}
            if h.diameter is not None:
                tap = TAP_DRILLS.get(round(h.diameter * 1000, 1))
                if tap:
                    self.warnings.append(
                        f"Hole {h.index} (Ø{h.diameter * 1000:g} mm) is the {tap} tap drill "
                        "size: if it is threaded, tapping is not planned yet"
                    )
                    row["tap_drill_for"] = tap
            reason = None
            if not (h.circular and h.vertical and h.diameter):
                reason = "not a round vertical hole"
            elif h.kind not in ("through", "blind"):
                reason = f"{h.kind} opening"
            elif h.bottom not in ("through", "cone"):
                reason = f"{h.bottom} bottom: a drill leaves a cone"
            tool = None
            if reason is None:
                tool = next(
                    (t for t in drills if abs(t.diameter - h.diameter) <= DRILL_MATCH), None
                )
                if tool is None:
                    reason = f"no Ø{h.diameter * 1000:g} mm drill in the library"
            if tool is not None:
                tip = (tool.diameter / 2) / tan(radians(tool.tip_angle / 2))
                depth = h.z_top - h.z_bottom + (tip + 0.0005 if h.bottom == "through" else 0.0)
                if depth > tool.flute_length:
                    reason = f"{_mm(depth)} deep: past the drill's {_mm(tool.flute_length)} flutes"
                    tool = None
                elif self.stock_top - (h.z_bottom - tip) > tool.reach:
                    reason = "deeper than the drill reaches"
                    tool = None
            if tool is not None:
                drilled.append((h, tool))
                row.update({"action": "drill", "tool": tool.id})
            else:
                row.update({"action": "mill", "why": reason})
            self.hole_rows.append(row)
        return drilled

    # --- coverage and numbering -------------------------------------------------------------

    def coverage(self) -> dict:
        covered = {}
        for op in self.ops:
            for f in op["faces"]:
                covered.setdefault(int(f), []).append(op["id"])
        uncovered = []
        reasons = {
            "bottom": "the bottom: machined from the other side (OP2, a second setup - not planned yet)",
            "fillet": "a fillet: 3D finishing (not in 2.5D plans yet)",
            "round": "a rounded edge: 3D finishing (not in 2.5D plans yet)",
            "curved": "a curved face: 3D finishing (not in 2.5D plans yet)",
            "slope": "a sloped face: 3D finishing (not in 2.5D plans yet)",
            "underside": "faces down: not reachable from above",
        }  # fmt: skip
        for f in self.part.faces:
            fid = int(f["face"])
            if fid in covered:
                continue
            role = f.get("role", "")
            reason = reasons.get(role, "not reached: no tool in the library fits there")
            uncovered.append({"face": fid, "role": role, "reason": reason})
        return {"faces": {str(k): v for k, v in sorted(covered.items())}, "uncovered": uncovered}

    def number_tools(self) -> list:
        fixed = {t.number for t in self.used if t.number is not None}
        nxt, numbers = 1, {}
        for t in self.used:
            if t.number is not None:
                numbers[t.id] = int(t.number)
                continue
            while nxt in fixed:
                nxt += 1
            numbers[t.id] = nxt
            fixed.add(nxt)
        for op in self.ops:
            op["number"] = numbers[op["tool"]]
            op["props"]["cutter_id"] = numbers[op["tool"]]
        rows = []
        for t in self.used:
            d = t.to_dict()
            d["number"] = numbers[t.id]
            d["operations"] = [op["id"] for op in self.ops if op["tool"] == t.id]
            rows.append(d)
        if len(rows) > self.machine.magazine:
            self.warnings.append(f"{len(rows)} tools, the magazine holds {self.machine.magazine}")
        return rows

    # --- the whole plan ---------------------------------------------------------------------

    def run(self) -> dict:
        self.place()
        drilled = self.choose_holes()
        holes = [h for h, _ in drilled]
        cap = self._cap()
        margin = _margin(cap / 2, self.opts.skin) + 0.001
        regions = _regions(
            self.part,
            self.stock_poly,
            self.stock_top,
            self.z_end,
            holes,
            margin,
            max(self.opts.tolerance, 1e-6),
        )
        self.regions_ops(regions, holes)
        self.hole_ops(drilled)
        self.finish_ops()
        # facing first, then the order operations were added per phase
        rank = {"face": 0, "rough": 1, "spot": 2, "drill": 3, "floor": 4, "walls": 5}
        self.ops.sort(key=lambda op: (rank[op["phase"]], int(op["id"][2:])))
        # tools by first use in the final order
        first = {}
        for op in self.ops:
            first.setdefault(op["tool"], len(first))
        self.used.sort(key=lambda t: first.get(t.id, 1e9))
        tools = self.number_tools()
        for i, op in enumerate(self.ops):
            op["name"] = f"{self.name} {i + 1:02d} {op['phase']}"
        rest = []
        for reg in self.regions:
            for x, y, size in reg.residue or []:
                rest.append(
                    {
                        "at": [round(x, 7), round(y, 7)],
                        "z": [round(reg.band_bottom, 7), round(reg.band_top, 7)],
                        "size": round(size, 7),
                        "radius_left": round(reg.tool.diameter / 2, 7),
                        "why": "a sharp inside corner: the end mill leaves its radius",
                    }
                )
        options = self.opts.to_dict()
        digest = hashlib.sha1(
            json.dumps([self.name, self.body_hash, options, self.machine.to_dict(),
                        [t.to_dict() for t in self.tools]], sort_keys=True, default=str).encode()
        ).hexdigest()[:10]  # fmt: skip
        return {
            "version": PLAN_VERSION,
            "id": f"{self.name}-{digest}",
            "tag": f"autocam:{self.name}",
            "object": self.name,
            "body_hash": self.body_hash,
            "material": {"key": self.mat_key, "name": self.mat["name"]},
            "machine": self.machine.name,
            "options": options,
            "placement": {
                "origin": self.opts.origin,
                "translation": [round(float(x), 9) for x in self.T],
                "work_offset": self.machine.work_offset,
                "z0": "the part's top",
            },
            "stock": {
                "min": [round(float(x), 7) for x in self.stock_min],
                "max": [round(float(x), 7) for x in self.stock_max],
                "size": [round(float(x), 7) for x in self.stock_max - self.stock_min],
                "allowance_top": round(self.allowance_top, 7),
                "grip": round(self.grip, 7),
            },
            "fixture": {"kind": "vise", "jaw_top": round(self.jaw_top, 7), "jaws": self.jaws},
            "clearance": round(self.clear, 7),
            "z_end": round(self.z_end, 7),
            "setups": [
                {
                    "name": "OP1",
                    "side": "top",
                    "work_offset": self.machine.work_offset,
                    "operations": [op["id"] for op in self.ops],
                }
            ],
            "tools": tools,
            "operations": self.ops,
            "holes": self.hole_rows,
            "coverage": self.coverage(),
            "rest": rest,
            "warnings": self.warnings,
            "geometry": self.geometry,
        }


def _plain(x):
    """numpy scalars / arrays and tuples -> plain JSON types."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def plan(
    part: PartModel,
    tools: list,
    machine: MachineProfile,
    options: dict | None = None,
    object_name: str | None = None,
    body_hash: str | None = None,
) -> dict:
    """A machining plan for the top side of `part` (see the module doc)."""
    opts = PlanOptions.from_dict(options)
    if not part.levels:
        raise CamcoreError("INVALID_VALUE", "The part has no level face facing up to start from")
    return _plain(_Planner(part, tools, machine, opts, object_name, body_hash).run())


def summary(p: dict) -> dict:
    """The plan without its geometry (what an agent reads)."""
    return {k: v for k, v in p.items() if k != "geometry"}
