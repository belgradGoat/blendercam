"""Verify a FANUC program against the part: simulate the stock and compare.

    verify(program_text, triangles, face_ids, stock_min, stock_max, ...) -> dict

The program is read as G-code (like check.py), so what is verified is what the machine
gets: G0 / G1 / G2 / G3 in G17, canned drilling cycles expanded, tool changes (M06) with
the tools' shapes from the program's tool comments `(T1 D10. END 3FL)` or a `tools` table.

The stock is a height map seen from above (one Z per pixel). Every move lowers it by the
cutter's shape swept along the move (flat, ball, bull nose, drill point). Before each move
is applied it is checked:
- a rapid (G0) that would remove material: a crash;
- material reaching above the flutes (the shank rubs) or the holder (a crash);
- a mill plunging straight down with its centre into material (should ramp or helix);
- the width of cut along each stretch of feed moves (slotting: the full diameter);
- moves into fixtures (vise jaws given as boxes).
At the end the stock is compared with the part (its height map from the exact mesh):
material left (leftover) and material cut that belongs to the part (gouge), per face.
Next to vertical walls both are measured sideways from the wall, elsewhere vertically.

The part's bottom is the lowest Z the part needs: outside the part and in through holes
only material above it counts as leftover (below is grip stock for the other side).

Units: millimetres. Resolution: the pixel size (default from the stock size).
"""

import math
import re
import time
from dataclasses import dataclass

import numpy as np

from .errors import CamcoreError

RAPID, FEED = 0, 1
CUT_EPS = 0.001  # mm of material that counts as cut
RAPID_EPS = 0.01  # mm a rapid may graze
PLUNGE_LIMIT = 0.5  # mm a mill may feed straight down into material with its centre
SLOT_RATIO = 0.9  # width of cut / diameter counted as slotting
MAX_PIXELS = 4_000_000
HOME_Z = 1e5  # after G28 Z: far above everything
WALL_NZ = 0.3  # triangles with |nz| below this are walls
WALL_REACH = 1.0  # mm: leftover / gouge this close to a wall is measured from the wall
ARC_CHORD = 0.005  # mm: arcs are split into chords this close

_COMMENT = re.compile(r"\(([^)]*)\)")
_WORD = re.compile(r"([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")
_TOOL = re.compile(r"T(\d+) D([0-9.]+) (\w+)")
_CYCLES = {"73", "81", "82", "83", "84", "85", "86", "89"}


# --------------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------------


@dataclass
class Cutter:
    number: int
    type: str  # END BALL BULL DRILL SPOT TAP
    diameter: float  # mm
    corner_radius: float = 0.0
    tip_angle: float = 0.0  # degrees (DRILL / SPOT)
    flute_length: float = 0.0  # mm (0 = not checked)
    reach: float = 0.0  # mm below the holder (0 = not checked)
    holder_diameter: float = 0.0

    @property
    def r(self) -> float:
        return self.diameter / 2

    @property
    def mill(self) -> bool:
        return self.type in ("END", "BALL", "BULL")

    def profile(self, d: np.ndarray) -> np.ndarray:
        """Height of the cutting edge above the tip at distance d from the axis (inf
        outside)."""
        r = self.r
        h = np.full(d.shape, np.inf)
        inside = d <= r + 1e-9
        dd = np.minimum(d[inside], r)
        if self.type == "BALL":
            h[inside] = r - np.sqrt(np.maximum(r * r - dd * dd, 0.0))
        elif self.type == "BULL" and self.corner_radius > 0:
            rc = min(self.corner_radius, r)
            flat = r - rc
            x = np.maximum(dd - flat, 0.0)
            h[inside] = rc - np.sqrt(np.maximum(rc * rc - x * x, 0.0))
        elif self.type in ("DRILL", "SPOT"):
            angle = self.tip_angle or (140.0 if self.type == "DRILL" else 90.0)
            h[inside] = dd / math.tan(math.radians(angle / 2))
        else:
            h[inside] = 0.0
        return h


def _cutters(text: str, tools: dict | None) -> dict:
    out = {}
    for n, d, kind in _TOOL.findall(text):
        kind = kind.upper()
        out[int(n)] = Cutter(int(n), kind if kind in ("END", "BALL", "BULL", "DRILL", "SPOT", "TAP") else "END", float(d))  # fmt: skip
        if kind == "BALL":
            out[int(n)].corner_radius = float(d) / 2
    for n, t in (tools or {}).items():
        base = out.get(int(n))
        fields = {k: v for k, v in dict(t).items() if k in Cutter.__dataclass_fields__}
        fields.pop("number", None)
        if base is None:
            if "diameter" not in fields:
                raise CamcoreError("INVALID_VALUE", f"Tool T{n}: needs a diameter")
            out[int(n)] = Cutter(int(n), **{"type": "END", **fields})
        else:
            for k, v in fields.items():
                setattr(base, k, v)
    return out


# --------------------------------------------------------------------------------------------
# G-code -> moves
# --------------------------------------------------------------------------------------------


@dataclass
class Move:
    a: tuple
    b: tuple
    kind: int  # RAPID / FEED
    tool: int | None
    op: int  # index into ops
    line: int
    drill: bool = False  # part of a canned cycle


def _arc(a, b, center, cw: bool) -> list:
    """Points (excluding a) along an XY arc from a to b around center (z linear)."""
    ax, ay, az = a
    bx, by, bz = b
    cx, cy = center
    r = math.hypot(ax - cx, ay - cy)
    t0 = math.atan2(ay - cy, ax - cx)
    t1 = math.atan2(by - cy, bx - cx)
    sweep = t1 - t0
    if cw:
        while sweep >= -1e-12:
            sweep -= 2 * math.pi
    else:
        while sweep <= 1e-12:
            sweep += 2 * math.pi
    if abs(ax - bx) < 1e-9 and abs(ay - by) < 1e-9:
        sweep = -2 * math.pi if cw else 2 * math.pi
    step = 2 * math.acos(max(1 - ARC_CHORD / max(r, 1e-9), -1.0)) if r > ARC_CHORD else math.pi
    n = max(2, int(math.ceil(abs(sweep) / max(step, 1e-3))))
    pts = []
    for i in range(1, n + 1):
        t = t0 + sweep * i / n
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t), az + (bz - az) * i / n))
    pts[-1] = (bx, by, bz)
    return pts


def moves(text: str) -> tuple[list, list]:
    """(moves, operations) of a program: absolute positions in mm; operations are the
    comment lines that stand alone (the writer puts one before each operation)."""
    out, ops = [], ["(start)"]
    pos = [None, None, HOME_Z]
    motion = None
    cycle = None
    cycle_r = cycle_z = None
    initial = None
    retract98 = True
    absolute = True
    tool = pending = None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip().upper()
        if not line or line == "%":
            continue
        comments = _COMMENT.findall(line)
        body = _COMMENT.sub("", line).strip()
        if not body:
            if comments and not _TOOL.match(comments[0]) and n > 2:
                ops.append(comments[0].strip())
            continue
        words = _WORD.findall(body)
        g = {w[1].lstrip("0") or "0" for w in words if w[0] == "G"}
        g = {x[:-2] if x.endswith(".0") else x for x in g}
        mcodes = {int(float(w[1])) for w in words if w[0] == "M"}
        vals = {w[0]: float(w[1]) for w in words if w[0] not in "GM"}
        if "T" in vals:
            pending = int(vals["T"])
        if 6 in mcodes and pending is not None:
            tool = pending
        if "90" in g:
            absolute = True
        if "91" in g:
            absolute = False
        if "98" in g:
            retract98 = True
        if "99" in g:
            retract98 = False
        if "28" in g or "30" in g:
            if "Z" in vals:
                pos[2] = HOME_Z
            if "X" in vals or "Y" in vals:
                pos[0] = pos[1] = None
            continue
        if "53" in g:
            continue
        if "80" in g:
            cycle = None
        for code in ("0", "1", "2", "3"):
            if code in g:
                motion, cycle = int(code), None
        new_cycle = g & _CYCLES
        if new_cycle:
            cycle = sorted(new_cycle)[0]
            initial = pos[2]
            cycle_r = cycle_z = None
        if not absolute or not any(k in vals for k in "XYZ") and not new_cycle:
            continue
        target = [vals.get(k, pos[i]) for i, k in enumerate("XYZ")]
        op = len(ops) - 1
        if cycle is not None:
            cycle_r = vals.get("R", cycle_r)
            cycle_z = vals.get("Z", cycle_z)
            if cycle_r is None or cycle_z is None or None in target[:2]:
                continue
            x, y = target[0], target[1]
            top = initial if initial is not None and initial < HOME_Z else cycle_r
            if pos[0] is not None and (pos[0], pos[1]) != (x, y):
                out.append(Move((pos[0], pos[1], top), (x, y, top), RAPID, tool, op, n, True))
            out.append(Move((x, y, top), (x, y, cycle_r), RAPID, tool, op, n, True))
            out.append(Move((x, y, cycle_r), (x, y, cycle_z), FEED, tool, op, n, True))
            back = top if retract98 else cycle_r
            out.append(Move((x, y, cycle_z), (x, y, back), RAPID, tool, op, n, True))
            pos = [x, y, back]
            continue
        if None in pos or None in target:  # position not known yet (start, after homing)
            pos = [t if t is not None else p for t, p in zip(target, pos)]
            continue
        a = tuple(pos)
        if motion in (2, 3) and ("I" in vals or "J" in vals or "R" in vals):
            if "R" in vals:
                rr = vals["R"]
                mx, my = (a[0] + target[0]) / 2, (a[1] + target[1]) / 2
                dx, dy = target[0] - a[0], target[1] - a[1]
                q = math.hypot(dx, dy)
                h = math.sqrt(max(rr * rr - q * q / 4, 0.0))
                s = (1 if (motion == 3) == (rr > 0) else -1) / max(q, 1e-12)
                center = (mx - s * h * dy, my + s * h * dx)
            else:
                center = (a[0] + vals.get("I", 0.0), a[1] + vals.get("J", 0.0))
            prev = a
            for p in _arc(a, tuple(target), center, motion == 2):
                out.append(Move(prev, p, FEED, tool, op, n))
                prev = p
        elif motion in (0, 1):
            out.append(Move(a, tuple(target), RAPID if motion == 0 else FEED, tool, op, n))
        pos = target
    return out, ops


# --------------------------------------------------------------------------------------------
# Rasters
# --------------------------------------------------------------------------------------------


@dataclass
class Grid:
    x0: float  # mm, the grid's lower-left corner
    y0: float
    px: float
    nx: int
    ny: int

    def window(self, x_lo, x_hi, y_lo, y_hi):
        i0 = max(int(math.floor((x_lo - self.x0) / self.px - 0.5)), 0)
        i1 = min(int(math.ceil((x_hi - self.x0) / self.px - 0.5)), self.nx - 1)
        j0 = max(int(math.floor((y_lo - self.y0) / self.px - 0.5)), 0)
        j1 = min(int(math.ceil((y_hi - self.y0) / self.px - 0.5)), self.ny - 1)
        return i0, i1, j0, j1

    def centers(self, i0, i1, j0, j1):
        xs = self.x0 + (np.arange(i0, i1 + 1) + 0.5) * self.px
        ys = self.y0 + (np.arange(j0, j1 + 1) + 0.5) * self.px
        return np.meshgrid(xs, ys)


def raster_part(tris: np.ndarray, face_ids: np.ndarray, grid: Grid, chunk: int = 4_000_000):
    """(height (ny, nx), face (ny, nx)) of the highest triangle above each pixel centre
    (-inf / -1 where there is none)."""
    H = np.full((grid.ny, grid.nx), -np.inf)
    F = np.full((grid.ny, grid.nx), -1, dtype=np.int64)
    T = np.asarray(tris, float).reshape(-1, 3, 3)
    fid = np.asarray(face_ids, dtype=np.int64)
    if not len(T):
        return H, F
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    keep = np.abs(area2) > 1e-12
    T, fid, area2 = T[keep], fid[keep], area2[keep]
    lo, hi = T[:, :, :2].min(1), T[:, :, :2].max(1)
    i0 = np.clip(np.ceil((lo[:, 0] - grid.x0) / grid.px - 0.5), 0, grid.nx).astype(np.int64)
    i1 = np.clip(np.floor((hi[:, 0] - grid.x0) / grid.px - 0.5), -1, grid.nx - 1).astype(np.int64)
    j0 = np.clip(np.ceil((lo[:, 1] - grid.y0) / grid.px - 0.5), 0, grid.ny).astype(np.int64)
    j1 = np.clip(np.floor((hi[:, 1] - grid.y0) / grid.px - 0.5), -1, grid.ny - 1).astype(np.int64)
    w = np.maximum(i1 - i0 + 1, 0)
    h = np.maximum(j1 - j0 + 1, 0)
    n = w * h
    Hf, Ff = H.ravel(), F.ravel()
    start = 0
    while start < len(T):
        cum = np.cumsum(n[start:])
        stop = start + max(int(np.searchsorted(cum, chunk)), 1)
        sl = slice(start, stop)
        cnt = n[sl]
        total = int(cnt.sum())
        start = stop
        if total == 0:
            continue
        tri = np.repeat(np.arange(sl.start, sl.stop), cnt)
        offs = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ww = w[tri]
        ix = i0[tri] + offs % ww
        iy = j0[tri] + offs // ww
        px = grid.x0 + (ix + 0.5) * grid.px
        py = grid.y0 + (iy + 0.5) * grid.px
        A, B, C = T[tri, 0], T[tri, 1], T[tri, 2]
        d = area2[tri]
        l1 = ((B[:, 0] - px) * (C[:, 1] - py) - (C[:, 0] - px) * (B[:, 1] - py)) / d
        l2 = ((C[:, 0] - px) * (A[:, 1] - py) - (A[:, 0] - px) * (C[:, 1] - py)) / d
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        z = l1 * A[:, 2] + l2 * B[:, 2] + l3 * C[:, 2]
        idx = (iy * grid.nx + ix)[inside]
        z, f = z[inside], fid[tri][inside]
        order = np.lexsort((z, idx))
        idx, z, f = idx[order], z[order], f[order]
        last = np.r_[idx[1:] != idx[:-1], True]
        idx, z, f = idx[last], z[last], f[last]
        better = z > Hf[idx]
        Hf[idx[better]] = z[better]
        Ff[idx[better]] = f[better]
    return H, F


def _wall_segments(tris, face_ids) -> dict:
    """{face: (n, 4) XY segments} of the steep triangles' projected edges."""
    T = np.asarray(tris, float).reshape(-1, 3, 3)
    if not len(T):
        return {}
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-12
    nz = np.zeros(len(T))
    nz[ok] = np.abs(nrm[ok, 2]) / ln[ok]
    steep = ok & (nz < WALL_NZ)
    out = {}
    fid = np.asarray(face_ids)
    for f in np.unique(fid[steep]):
        sel = T[steep & (fid == f)]
        segs = np.concatenate(
            [np.hstack([sel[:, i, :2], sel[:, (i + 1) % 3, :2]]) for i in range(3)]
        )
        long = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) > 1e-9
        out[int(f)] = segs[long]
    return out


def _seg_distance(px, py, segs) -> np.ndarray:
    """Distance from points to the nearest of the segments (chunked)."""
    best = np.full(len(px), np.inf)
    if not len(segs):
        return best
    ax, ay, bx, by = segs.T
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    step = max(1, 2_000_000 // max(len(segs), 1))
    for s in range(0, len(px), step):
        X, Y = px[s : s + step, None], py[s : s + step, None]
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L, 0, 1)
        d = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
        best[s : s + step] = d.min(axis=1)
    return best


def _wall_labels(grid: Grid, walls: dict, reach_px: int):
    """(face (ny, nx), steps (ny, nx)): the wall face within reach_px pixels of each pixel,
    by growing the walls' pixels outwards one ring at a time (-1 / inf where none)."""
    L = np.full((grid.ny, grid.nx), -1, dtype=np.int64)
    D = np.full((grid.ny, grid.nx), np.inf)
    for f, segs in walls.items():
        for ax, ay, bx, by in segs:
            n = max(2, int(math.hypot(bx - ax, by - ay) / (grid.px / 2)) + 2)
            t = np.linspace(0, 1, n)
            ix = np.floor((ax + t * (bx - ax) - grid.x0) / grid.px).astype(np.int64)
            iy = np.floor((ay + t * (by - ay) - grid.y0) / grid.px).astype(np.int64)
            ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
            L[iy[ok], ix[ok]] = f
            D[iy[ok], ix[ok]] = 0
    for step in range(1, reach_px + 1):
        frontier = D == step - 1
        if not frontier.any():
            break
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            src = frontier[max(-dy, 0) : grid.ny - max(dy, 0), max(-dx, 0) : grid.nx - max(dx, 0)]
            dst = (
                slice(max(dy, 0), grid.ny - max(-dy, 0)),
                slice(max(dx, 0), grid.nx - max(-dx, 0)),
            )
            Lsrc = L[max(-dy, 0) : grid.ny - max(dy, 0), max(-dx, 0) : grid.nx - max(dx, 0)]
            take = src & ~np.isfinite(D[dst])
            D[dst][take] = step
            L[dst][take] = Lsrc[take]
    return L, D


# --------------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------------


def _seg_dist_grid(X, Y, p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = dx * dx + dy * dy
    if L < 1e-18:
        return np.hypot(X - p0[0], Y - p0[1])
    t = np.clip(((X - p0[0]) * dx + (Y - p0[1]) * dy) / L, 0.0, 1.0)
    return np.hypot(X - (p0[0] + t * dx), Y - (p0[1] + t * dy))


def _box_distance_xy(a, b, lo, hi) -> float:
    """XY distance between segment a-b and an axis-aligned rectangle."""
    pts = np.linspace(np.asarray(a[:2]), np.asarray(b[:2]), 8)
    cx = np.clip(pts[:, 0], lo[0], hi[0])
    cy = np.clip(pts[:, 1], lo[1], hi[1])
    d = float(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).min())
    if d == 0.0:
        return 0.0
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
    ax, ay = a[0], a[1]
    dx, dy = b[0] - ax, b[1] - ay
    L = dx * dx + dy * dy
    if L > 0:
        t = np.clip(((corners[:, 0] - ax) * dx + (corners[:, 1] - ay) * dy) / L, 0, 1)
        d = min(d, float(np.hypot(corners[:, 0] - ax - t * dx, corners[:, 1] - ay - t * dy).min()))
    return d


class _Sim:
    def __init__(self, grid, stock_top, cutters, fixtures, ops, tol):
        self.g = grid
        self.S = np.full((grid.ny, grid.nx), float(stock_top))
        self.top0 = float(stock_top)
        self.cutters = cutters
        self.fixtures = fixtures or []
        self.ops = ops
        self.tol = tol
        self.collisions, self.plunges = [], []
        self.stats = {}
        self.run = None  # (op, tool, length, area, max ap) of the current feed stretch
        self.missing_tools = set()

    def stat(self, op, tool):
        key = op
        if key not in self.stats:
            self.stats[key] = {
                "name": self.ops[op] if op < len(self.ops) else str(op),
                "tool": tool,
                "removed_mm3": 0.0,
                "feed_mm": 0.0,
                "air_mm": 0.0,
                "max_width_ratio": 0.0,
                "slot_mm": 0.0,
                "slot_max_depth_mm": 0.0,
            }
        return self.stats[key]

    def _end_run(self):
        if self.run is None:
            return
        op, c, length, area, ap = self.run
        self.run = None
        if length < max(c.diameter, 3 * self.g.px):
            return
        ratio = min(area / length / c.diameter, 1.0)  # the first bite of a stretch counts once
        st = self.stat(op, c.number)
        st["max_width_ratio"] = max(st["max_width_ratio"], ratio)
        if ratio >= SLOT_RATIO:
            st["slot_mm"] += length
            st["slot_max_depth_mm"] = max(st["slot_max_depth_mm"], ap)

    def _record(self, lst, kind, mv, c, at, amount, note=""):
        if len(lst) >= 50:
            return
        lst.append(
            {
                "kind": kind,
                "operation": self.ops[mv.op] if mv.op < len(self.ops) else str(mv.op),
                "line": mv.line,
                "tool": c.number if c else None,
                "at": [round(float(at[0]), 3), round(float(at[1]), 3), round(float(at[2]), 3)],
                "mm": round(float(amount), 4),
                **({"note": note} if note else {}),
            }
        )

    def fixtures_check(self, mv, c):
        z_low = min(mv.a[2], mv.b[2])
        if z_low >= HOME_Z / 2:
            return
        for fx in self.fixtures:
            lo, hi = fx["min"], fx["max"]
            if z_low >= hi[2]:
                continue
            d = _box_distance_xy(mv.a, mv.b, lo, hi)
            reach_r = c.r
            if c.reach and c.holder_diameter and z_low + c.reach < hi[2]:
                reach_r = max(reach_r, c.holder_diameter / 2)
            if d < reach_r - 1e-6:
                self._record(self.collisions, "fixture", mv, c, mv.b, reach_r - d,
                             fx.get("name", "fixture"))  # fmt: skip

    def apply(self, mv: Move):
        c = self.cutters.get(mv.tool)
        if c is None:
            self.missing_tools.add(mv.tool)
            return
        a, b = np.asarray(mv.a, float), np.asarray(mv.b, float)
        if min(a[2], b[2]) >= HOME_Z / 2:
            return
        self.fixtures_check(mv, c)
        st = self.stat(mv.op, c.number)
        if min(a[2], b[2]) >= self.top0 - 1e-9:  # above all the stock
            if mv.kind == FEED:
                length = float(np.linalg.norm(b - a))
                st["feed_mm"] += length
                st["air_mm"] += length
            self._end_run()
            return
        g = self.g
        r = c.r
        rr = r
        holder = bool(
            c.reach and c.holder_diameter > c.diameter and min(a[2], b[2]) + c.reach < self.top0
        )
        if holder:
            rr = c.holder_diameter / 2
        i0, i1, j0, j1 = g.window(
            min(a[0], b[0]) - rr, max(a[0], b[0]) + rr, min(a[1], b[1]) - rr, max(a[1], b[1]) + rr
        )
        length = float(np.linalg.norm(b - a))
        dxy = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if i0 > i1 or j0 > j1:
            if mv.kind == FEED:
                st["feed_mm"] += length
                st["air_mm"] += length
            self._end_run()
            return
        X, Y = g.centers(i0, i1, j0, j1)
        cur = self.S[j0 : j1 + 1, i0 : i1 + 1]
        dz = b[2] - a[2]
        # the swept cutter: pieces short enough in Z that each is cut at its lowest Z
        if dxy < g.px / 2:
            low = a if a[2] <= b[2] else b
            pieces = [(low, low, low[2])]
        else:
            n = int(min(max(math.ceil(abs(dz) / max(self.tol / 2, 1e-4)), 1), 256))
            ts = np.linspace(0, 1, n + 1)
            P = a + (b - a) * ts[:, None]
            pieces = [(P[k], P[k + 1], min(P[k][2], P[k + 1][2])) for k in range(n)]
        env = np.full(cur.shape, np.inf)
        dmin = np.full(cur.shape, np.inf)
        z_low = min(a[2], b[2])
        for p0, p1, zl in pieces:
            d = _seg_dist_grid(X, Y, p0, p1)
            np.minimum(env, zl + c.profile(d), out=env)
            np.minimum(dmin, d, out=dmin)
        if c.type == "TAP":
            env[:] = np.inf  # a tap cuts the thread, not the stock
        removed = np.where(np.isfinite(env), cur - env, 0.0)
        cut = removed > CUT_EPS
        depth = float(removed.max()) if removed.size else 0.0
        # checks before cutting
        if mv.kind == RAPID and depth > RAPID_EPS:
            k = np.unravel_index(np.argmax(removed), removed.shape)
            self._record(self.collisions, "rapid", mv, c, (X[k], Y[k], env[k]), depth,
                         "a rapid move through material")  # fmt: skip
        if c.flute_length > 0 and cut.any():
            above = np.where(dmin <= r, cur - (z_low + c.flute_length), -np.inf)
            over = float(above.max())
            if over > self.tol:
                k = np.unravel_index(np.argmax(above), above.shape)
                self._record(self.collisions, "shank", mv, c, (X[k], Y[k], cur[k]), over,
                             "material above the flutes")  # fmt: skip
        if holder:
            above = np.where(dmin <= rr, cur - (z_low + c.reach), -np.inf)
            over = float(above.max())
            if over > self.tol:
                k = np.unravel_index(np.argmax(above), above.shape)
                self._record(self.collisions, "holder", mv, c, (X[k], Y[k], cur[k]), over,
                             "the holder hits material")  # fmt: skip
        if mv.kind == FEED and c.mill and dz < 0 and dxy < 0.1 * abs(dz):
            centre = dmin <= max(0.3 * r, g.px)
            if centre.any():
                into = float((cur[centre] - z_low).max())
                if into > PLUNGE_LIMIT:
                    self._record(self.plunges, "plunge", mv, c, b, into,
                                 "the mill feeds straight down into material")  # fmt: skip
        # cut
        np.minimum(cur, env, out=cur)
        vol = float(removed[cut].sum()) * g.px * g.px
        st["removed_mm3"] += vol
        if mv.kind == FEED:
            st["feed_mm"] += length
            if not cut.any():
                st["air_mm"] += length
        # width of cut along stretches of sideways feed moves
        if mv.kind == FEED and c.mill and dxy >= g.px and abs(dz) < 0.2 * dxy and not mv.drill:
            bite = removed > 0.01
            area = float(bite.sum()) * g.px * g.px
            engaged = 0.0  # length along the move over which it cut (moves may start in air)
            if bite.any():
                ux, uy = (b[0] - a[0]) / dxy, (b[1] - a[1]) / dxy
                proj = (X[bite] - a[0]) * ux + (Y[bite] - a[1]) * uy
                engaged = min(float(proj.max() - proj.min()), dxy)
            if self.run is None or self.run[0] != mv.op or self.run[1] is not c:
                self._end_run()
                self.run = (mv.op, c, 0.0, 0.0, 0.0)
            op, cc, ln, ar, ap = self.run
            self.run = (op, cc, ln + engaged, ar + area, max(ap, depth))
            if self.run[2] >= max(c.diameter, 3 * g.px):
                self._end_run()
        else:
            self._end_run()


def _report_faces(values, faces, kinds, at_xy, px) -> list:
    rows = {}
    for v, f, kind, (x, y) in zip(values, faces, kinds, at_xy):
        row = rows.setdefault(int(f), {"face": int(f), "max_mm": 0.0, "area_mm2": 0.0,
                                       "measured": kind, "at": None})  # fmt: skip
        row["area_mm2"] += px * px
        if v > row["max_mm"]:
            row["max_mm"] = float(v)
            row["at"] = [round(float(x), 3), round(float(y), 3)]
            row["measured"] = kind
    out = sorted(rows.values(), key=lambda r: -r["max_mm"])
    for r in out:
        r["max_mm"] = round(r["max_mm"], 4)
        r["area_mm2"] = round(r["area_mm2"], 3)
    return out


def verify(
    program: str,
    triangles,
    face_ids,
    stock_min,
    stock_max,
    part_bottom: float | None = None,
    tools: dict | None = None,
    fixtures: list | None = None,
    resolution: float | None = None,
    tolerance: float = 0.02,
    expected_faces: dict | None = None,
    expected_zones: list | None = None,
) -> dict:
    """Simulate `program` on the stock box and compare with the part (see the module doc).

    triangles (n, 3, 3) and face_ids (n,): the part, mm, in the program's coordinates.
    tools: {T: {type, diameter, corner_radius, tip_angle, flute_length, reach,
    holder_diameter}} (mm) over the program's tool comments. fixtures: [{min, max, name}]
    boxes (mm). resolution: pixel size (mm). Leftover that was planned is listed apart:
    expected_faces {face: reason} (faces left for later) and expected_zones [{at: [x, y],
    radius, reason}] (mm: e.g. the radius a tool leaves in a sharp corner); leftover has
    "expected" and "unexpected" rows. Returns ok, gouge, leftover, collisions, plunges,
    operations (removed volume, feed / air length, width of cut, slotting)."""
    started = time.perf_counter()
    lo, hi = np.asarray(stock_min, float), np.asarray(stock_max, float)
    if np.any(hi <= lo):
        raise CamcoreError("INVALID_VALUE", "The stock box is empty")
    size = hi - lo
    px = float(resolution) if resolution else max(0.1, math.sqrt(size[0] * size[1] / MAX_PIXELS))
    px = max(px, math.sqrt(size[0] * size[1] / (4 * MAX_PIXELS)))
    nx, ny = max(int(math.ceil(size[0] / px)), 1), max(int(math.ceil(size[1] / px)), 1)
    grid = Grid(lo[0], lo[1], px, nx, ny)
    T = np.asarray(triangles, float).reshape(-1, 3, 3)
    fids = np.asarray(face_ids, dtype=np.int64)
    if part_bottom is None:
        part_bottom = float(T[:, :, 2].min()) if len(T) else lo[2]
    cutters = _cutters(program, tools)
    mvs, ops = moves(program)
    sim = _Sim(grid, hi[2], cutters, fixtures, ops, tolerance)
    for mv in mvs:
        sim.apply(mv)
    sim._end_run()

    part_z, part_f = raster_part(T, fids, grid)
    target = np.maximum(part_z, part_bottom)
    diff = sim.S - target
    covered = np.isfinite(part_z)
    walls = _wall_segments(T, fids)
    reach_px = max(1, int(math.ceil(WALL_REACH / px)))
    wall_face, wall_steps = _wall_labels(grid, walls, reach_px)
    X, Y = grid.centers(0, nx - 1, 0, ny - 1)

    def classify(mask, amount):
        """Per pixel: (face, amount, how measured) - from the nearest wall when the material
        stands against one, else the face below (vertically)."""
        jj, ii = np.nonzero(mask)
        amt = amount[jj, ii]
        wf, ws = wall_face[jj, ii], wall_steps[jj, ii]
        own = part_f[jj, ii]
        near = (wf >= 0) & (~covered[jj, ii] | (amt > 2 * (ws + 1) * px))
        face = np.where(near, wf, own)
        value = amt.copy()
        kind = np.where(near, "sideways", "vertical").astype(object)
        xs, ys = X[jj, ii], Y[jj, ii]
        for f in np.unique(face[near]):
            sel = near & (face == f)
            value[sel] = _seg_distance(xs[sel], ys[sel], walls.get(int(f), np.zeros((0, 4))))
        far = ~near & (own < 0)
        kind[far] = "vertical (no face: outside the part)"
        real = value > tolerance  # pixels on a wall's line, within the tolerance of it: none
        return face[real], value[real], kind[real], np.column_stack([xs, ys])[real]

    left = diff > tolerance
    gouge = covered & (diff < -tolerance)
    lf, lv, lk, lxy = classify(left, diff)
    gf, gv, gk, gxy = classify(gouge, -diff)
    leftover_faces = _report_faces(lv, lf, lk, lxy, px)
    gouge_faces = _report_faces(gv, gf, gk, gxy, px)
    # planned leftover: in the expected zones, or on faces left for later
    zone = np.zeros(left.shape, dtype=bool)
    zone_reason = []
    for z in expected_zones or []:
        zx, zy = z["at"]
        zone |= (X - zx) ** 2 + (Y - zy) ** 2 <= float(z["radius"]) ** 2
        zone_reason.append(z.get("reason", "expected"))
    later = {int(k): v for k, v in (expected_faces or {}).items()}
    expected_rows, unexpected_rows = [], []
    if (left & zone).any():
        zf, zv, zk, zxy = classify(left & zone, diff)
        for row in _report_faces(zv, zf, zk, zxy, px):
            reason = later.get(row["face"]) or (sorted(set(zone_reason))[0] if zone_reason else "")
            expected_rows.append({**row, "reason": reason})
    rf, rv, rk, rxy = classify(left & ~zone, diff)
    for row in _report_faces(rv, rf, rk, rxy, px):
        if row["face"] in later:
            expected_rows.append({**row, "reason": later[row["face"]]})
        else:
            unexpected_rows.append(row)
    ops_out = []
    for k in sorted(sim.stats):
        st = sim.stats[k]
        if not st["feed_mm"] and not st["removed_mm3"]:
            continue  # only positioning (tool changes, rapids)
        ops_out.append(
            {
                "name": st["name"],
                "tool": st["tool"],
                "removed_cm3": round(st["removed_mm3"] / 1000, 4),
                "feed_mm": round(st["feed_mm"], 1),
                "air_fraction": round(st["air_mm"] / st["feed_mm"], 3) if st["feed_mm"] else 0.0,
                "max_width_ratio": round(st["max_width_ratio"], 3),
                "slot_mm": round(st["slot_mm"], 1),
                "slot_max_depth_mm": round(st["slot_max_depth_mm"], 3),
            }
        )
    warnings = []
    if sim.missing_tools:
        warnings.append(f"No shape for tools {sorted(t for t in sim.missing_tools if t is not None)}: their moves were not simulated")  # fmt: skip
    if not len(T):
        warnings.append("No part triangles: everything left counts as leftover")
    crash = [c for c in sim.collisions if c["kind"] in ("rapid", "holder", "fixture")]
    return {
        "ok": not gouge_faces and not crash,
        "resolution_mm": round(px, 5),
        "tolerance_mm": tolerance,
        "grid": [nx, ny],
        "gouge": {
            "max_mm": max((r["max_mm"] for r in gouge_faces), default=0.0),
            "area_mm2": round(float(gouge.sum()) * px * px, 3),
            "faces": gouge_faces,
        },
        "leftover": {
            "max_mm": max((r["max_mm"] for r in leftover_faces), default=0.0),
            "area_mm2": round(float(left.sum()) * px * px, 3),
            "faces": leftover_faces,
            "expected": expected_rows,
            "unexpected": unexpected_rows,
        },
        "collisions": sim.collisions,
        "plunges": sim.plunges,
        "operations": ops_out,
        "moves": len(mvs),
        "warnings": warnings,
        "seconds": round(time.perf_counter() - started, 3),
    }
