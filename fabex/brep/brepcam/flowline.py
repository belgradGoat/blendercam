"""Fabex 'brepcam/flowline.py'

Flowline finishing patterns from a B-rep's exact surfaces. No bpy.

    flowline(kernel, body, params, matrix, faces) -> Flowline   passes + per-face stats

Passes follow each face's parameter lines (isocurves), so they follow the shape of the
face - around a dome, along a fillet, across a blend. For every face:

  1. direction: passes run along u or v. AUTO: around a pole (rings) when the face has
     one, else along the longer direction (fewer, longer passes).
  2. stations across the passes are placed by the largest 3D distance between
     neighbouring parameter lines, so adjacent passes are never further apart than the
     stepover anywhere on the face (measured: max_gap).
  3. each pass is cut to the face's trim (outline, holes) - within `tolerance` of the
     exact trim curves - and sampled every step_along (3D arc length). Points steeper
     than max_slope (or facing down) are left out; the cut ends are put on the slope
     limit. Passes around a closed face (cylinder, pole) stay closed.
  4. zigzag or one-way order.
  5. tool-centre XY = contact point + tip_radius * normal.xy: a cutter dropped there
     touches the surface at the contact point with its round tip (ball, bull nose
     corner, ball cone tip). Flat-bottomed cutters get the contact XY (they touch with
     their rim, so their spacing is approximate). Points where the tip is too big for
     the curve along the pass (the centre moves backwards) are left out.

Z isn't decided here: Fabex drops the cutter at the centre XY onto the whole model, so
the result can't gouge. Normals point out of the material (BRepKit solids); bodies with
open edges (sheets) have their faces turned to face up first.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .edges import loop_pcurves, sample_curve
from .fitcheck import UNDERCUT_Z, _face_sign_world

DIRECTIONS = ("AUTO", "U", "V")
ORDERS = ("ZIGZAG", "ONEWAY")
MAX_POINTS = 4_000_000  # whole pattern, before slope / misfit filtering
MAX_STATIONS = 20_000  # per face
CHUNK = 65536  # points per surface evaluation
POLE_REL = 1e-7  # a side shorter than this times the face size is a pole
EDGE_CHUNK = 256  # lines per trim crossing batch


class PatternTooLarge(ValueError):
    """The pattern would have more points than allowed (raise step_along / stepover)."""

    def __init__(self, message: str, estimate: int, limit: int):
        super().__init__(message)
        self.estimate, self.limit = int(estimate), int(limit)


@dataclass
class FlowlineParams:
    """stepover: max 3D distance between adjacent passes. step_along: max 3D distance
    between points along a pass. direction: "AUTO" | "U" | "V" (the parameter the passes
    follow). order: "ZIGZAG" | "ONEWAY". max_slope: degrees from level; steeper points
    are left out (90: everything but faces facing down). tip_radius: round tip of the
    cutter, for the tool-centre XY. tolerance: trim accuracy. reverse: flip every pass."""

    stepover: float
    step_along: float
    direction: str = "AUTO"
    order: str = "ZIGZAG"
    max_slope: float = 80.0
    tip_radius: float = 0.0
    tolerance: float = 1e-5
    reverse: bool = False
    max_points: int = MAX_POINTS

    def validated(self) -> "FlowlineParams":
        direction, order = str(self.direction).upper(), str(self.order).upper()
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}")
        if order not in ORDERS:
            raise ValueError(f"order must be one of {', '.join(ORDERS)}")
        for name in ("stepover", "step_along", "tolerance"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive")
        if not 0.0 < float(self.max_slope) <= 90.0:
            raise ValueError("max_slope must be in (0, 90] degrees")
        if not float(self.tip_radius) >= 0.0:
            raise ValueError("tip_radius can't be negative")
        if int(self.max_points) < 1:
            raise ValueError("max_points must be positive")
        return FlowlineParams(
            float(self.stepover),
            float(self.step_along),
            direction,
            order,
            float(self.max_slope),
            float(self.tip_radius),
            float(self.tolerance),
            bool(self.reverse),
            int(self.max_points),
        )

    @property
    def nz_min(self) -> float:
        """Smallest normal.z a contact point may have."""
        if self.max_slope >= 90.0:
            return UNDERCUT_Z
        return float(np.cos(np.radians(self.max_slope)))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Pass:
    """One pass over one face, in cutting order. World space."""

    face: int
    station: int  # index across the face
    points: np.ndarray  # (n, 3) contact points
    normals: np.ndarray  # (n, 3) unit normals, towards the tool
    uv: np.ndarray  # (n, 2) surface parameters of the contact points
    centers: np.ndarray  # (n, 2) tool-centre XY to drop the cutter at
    closed: bool = False  # closed loop (the first point isn't repeated)

    @property
    def length(self) -> float:
        P = np.vstack([self.points, self.points[:1]]) if self.closed else self.points
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    def tool_tips(self, tip: float) -> np.ndarray:
        """Lowest points of a round tip of radius `tip` touching the contact points."""
        T = self.points + tip * self.normals
        T[:, 2] -= tip
        return T

    def reversed(self) -> "Pass":
        return Pass(
            self.face,
            self.station,
            self.points[::-1].copy(),
            self.normals[::-1].copy(),
            self.uv[::-1].copy(),
            self.centers[::-1].copy(),
            self.closed,
        )


@dataclass
class FaceFlow:
    """What happened on one face."""

    face: int
    name: str
    direction: str | None = None  # "U" / "V": the parameter the passes follow
    stations: int = 0
    passes: int = 0
    points: int = 0
    length: float = 0.0  # contact length of the passes
    cross_length: float = 0.0  # widest distance across the passes
    max_gap: float = 0.0  # largest 3D distance between adjacent passes (measured)
    closed: bool = False  # passes go around (a closed face or a pole)
    poles: list = field(default_factory=list)  # sides that collapse to a point
    steep_points: int = 0  # samples left out: steeper than max_slope / facing down
    misfit_points: int = 0  # samples left out: the tip doesn't fit the curve
    note: str | None = None  # why there are no passes
    seconds: float = 0.0

    def summary(self) -> dict:
        out = asdict(self)
        for k in ("length", "cross_length", "max_gap", "seconds"):
            out[k] = round(float(out[k]), 9 if k != "seconds" else 4)
        return out


@dataclass
class Flowline:
    params: FlowlineParams
    passes: list  # [Pass], in order
    faces: list  # [FaceFlow]
    seconds: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def points(self) -> int:
        return int(sum(len(p.points) for p in self.passes))

    @property
    def length(self) -> float:
        return float(sum(p.length for p in self.passes))

    def stats(self) -> dict:
        return {
            "passes": len(self.passes),
            "closed_passes": sum(p.closed for p in self.passes),
            "points": self.points,
            "length": round(self.length, 9),
            "max_gap": round(max((f.max_gap for f in self.faces if f.passes), default=0.0), 9),
            "faces": [f.summary() for f in self.faces],
            "params": self.params.to_dict(),
            "seconds": round(self.seconds, 4),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------------------------
# Surface evaluation
# --------------------------------------------------------------------------------------------


def _grid(basis, S, us, vs) -> np.ndarray:
    """Points on the tensor grid us x vs, shape (len(us), len(vs), 3). With the kernel's
    dense basis matrices and BLAS (much faster than NurbsSurface.grid for big grids)."""
    us = np.clip(np.asarray(us, dtype=float), *S.domain_u)
    vs = np.clip(np.asarray(vs, dtype=float), *S.domain_v)
    if basis is None:
        return S.grid(us, vs)[0, 0]
    Nu = basis(S.knots_u, S.degree_u, S.shape[0], us)[0]
    Nv = basis(S.knots_v, S.degree_v, S.shape[1], vs)[0]
    if len(us) * S.shape[1] <= len(vs) * S.shape[0]:
        T = np.tensordot(Nu, S.cpw, axes=(1, 0))  # (nu, n_v ctrl, 4)
        W = np.tensordot(T, Nv, axes=(1, 1)).transpose(0, 2, 1)
    else:
        T = np.tensordot(Nv, S.cpw, axes=(1, 1))  # (nv, n_u ctrl, 4)
        W = np.tensordot(Nu, T, axes=(1, 1))
    return W[..., :3] / W[..., 3:]


def _frame(S, sign: float, u, v):
    """Points and unit normals (towards the tool; NaN where degenerate) at (u, v)."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    P = np.empty((len(u), 3))
    n = np.empty((len(u), 3))
    for s in range(0, len(u), CHUNK):
        D = S.derivatives(u[s : s + CHUNK], v[s : s + CHUNK], 1)
        N = np.cross(D[1, 0], D[0, 1]) * sign
        with np.errstate(invalid="ignore", divide="ignore"):
            N = N / np.linalg.norm(N, axis=1)[:, None]
        P[s : s + CHUNK], n[s : s + CHUNK] = D[0, 0], N
    return P, n


def _fill_normals(n: np.ndarray) -> np.ndarray:
    """Degenerate normals (poles) from their neighbours along the pass."""
    bad = ~np.all(np.isfinite(n), axis=1)
    if not bad.any():
        return n
    good = np.flatnonzero(~bad)
    if not len(good):
        n[:] = (0.0, 0.0, 1.0)
        return n
    idx = np.flatnonzero(bad)
    for k in range(3):
        n[idx, k] = np.interp(idx, good, n[good, k])
    n[idx] /= np.maximum(np.linalg.norm(n[idx], axis=1), 1e-300)[:, None]
    return n


# --------------------------------------------------------------------------------------------
# Trim regions
# --------------------------------------------------------------------------------------------


def _trim_polygons(face, S, tol: float):
    """uv polygons of the face's loops, sampled within `tol` (world) of the exact trims.
    None for an untrimmed face."""
    if face.loops is None:
        return None
    polys = []
    for li in range(len(face.loops)):
        pts = []
        for pc in loop_pcurves(face, li):
            t, _, _ = sample_curve(S, pc, tol)
            pts.append(pc.evaluate(t)[:-1])
        polys.append(np.concatenate(pts))
    return polys


def _crossings(edges, lines) -> list:
    """Sorted x1 values where the polygon edges cross each line x2 = c (one array per
    line). edges: (A, B) segment ends, (m, 2) arrays in (x1, x2) coordinates."""
    A, B = edges
    ax, ay, bx, by = A[:, 0], A[:, 1], B[:, 0], B[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (bx - ax) / (by - ay)
    out = []
    lines = np.asarray(lines, dtype=float)
    for s in range(0, len(lines), EDGE_CHUNK):
        c = lines[s : s + EDGE_CHUNK][:, None]
        cross = (ay[None] > c) != (by[None] > c)
        with np.errstate(invalid="ignore"):  # level edges never cross: their x is unused
            x = ax[None] + (c - ay[None]) * slope[None]
        out.extend(np.sort(xr[row]) for row, xr in zip(cross, x))
    return out


def _swap(edges):
    return None if edges is None else (edges[0][:, ::-1], edges[1][:, ::-1])


def _mask(edges, xs, ys) -> np.ndarray:
    """Inside the trim region on the grid xs x ys (even-odd), shape (len(xs), len(ys)).
    edges in (x, y) coordinates; None: everything is inside."""
    if edges is None:
        return np.ones((len(xs), len(ys)), dtype=bool)
    ys = np.asarray(ys, dtype=float)
    rows = _crossings(_swap(edges), xs)  # y crossings of the lines x = xs[i]
    return np.array([np.searchsorted(r, ys, side="right") % 2 == 1 for r in rows])


def _intervals(edges, lines, lo: float, hi: float) -> list:
    """Inside intervals [x0, x1] along each line y = c, clipped to [lo, hi]."""
    if edges is None:
        return [np.array([[lo, hi]])] * len(lines)
    out = []
    for xs in _crossings(edges, lines):
        if len(xs) % 2:
            xs = xs[:-1]  # a line through a vertex, numerically: drop the odd crossing
        pairs = np.clip(xs.reshape(-1, 2), lo, hi)
        out.append(pairs[pairs[:, 1] > pairs[:, 0]])
    return out


# --------------------------------------------------------------------------------------------
# Layout: direction and stations
# --------------------------------------------------------------------------------------------


@dataclass
class _Layout:
    flow: FaceFlow
    S: object  # world-space surface
    sign: float
    along: int = 0  # 0: passes follow u, 1: follow v
    a_range: tuple = (0.0, 1.0)  # region range along the passes
    a_domain: tuple = (0.0, 1.0)
    edges: tuple | None = None  # trim polygon edges in (along, cross) coordinates
    stations: np.ndarray = field(default_factory=lambda: np.zeros(0))
    full_lengths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    estimate: float = 0.0  # points, before filtering
    size: float = 0.0

    def uv(self, a, c):
        a, c = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(c, dtype=float))
        return (a, c) if self.along == 0 else (c, a)


def _poles(G: np.ndarray, lo, hi, dom, size: float) -> list:
    """Sides of the domain the region reaches that collapse to a point."""
    out = []
    for side, row, k, end in (
        ("u0", G[0, :], 0, 0),
        ("u1", G[-1, :], 0, 1),
        ("v0", G[:, 0], 1, 0),
        ("v1", G[:, -1], 1, 1),
    ):
        span = dom[k][1] - dom[k][0]
        reached = abs((lo, hi)[end][k] - dom[k][end]) <= 1e-9 * span
        length = float(np.linalg.norm(np.diff(row, axis=0), axis=1).sum())
        if reached and length <= POLE_REL * max(size, 1e-300):
            out.append(side)
    return out


def _layout(basis, face, M, sign: float, p: FlowlineParams) -> _Layout:
    S = face.surface.transformed(M)
    flow = FaceFlow(face.id, face.name)
    lay = _Layout(flow, S, sign)
    dom = (S.domain_u, S.domain_v)
    polys = _trim_polygons(face, S, p.tolerance)
    lo = np.array([dom[0][0], dom[1][0]])
    hi = np.array([dom[0][1], dom[1][1]])
    edges = None
    if polys is not None:
        allp = np.concatenate(polys)
        lo, hi = np.maximum(allp.min(0), lo), np.minimum(allp.max(0), hi)
        edges = (allp, np.concatenate([np.roll(q, -1, axis=0) for q in polys]))
    if not np.all(hi > lo):
        flow.note = "no area"
        return lay

    # coarse look: size, poles, reachable part, direction
    us, vs = np.linspace(lo[0], hi[0], 33), np.linspace(lo[1], hi[1], 33)
    UU, VV = np.meshgrid(us, vs, indexing="ij")
    P, n = _frame(S, sign, UU.ravel(), VV.ravel())
    G, nz = P.reshape(33, 33, 3), n[:, 2].reshape(33, 33)
    inside = _mask(edges, us, vs)
    pts = G[inside] if inside.any() else G.reshape(-1, 3)
    lay.size = size = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    flow.poles = _poles(G, lo, hi, dom, size)
    if size <= p.tolerance * 1e-3:
        flow.note = "no area"
        return lay
    reach = inside & ~(np.isfinite(nz) & (nz < p.nz_min))
    if not reach.any():
        flow.note = "too steep or facing down"
        return lay
    reach_fraction = reach.sum() / max(inside.sum(), 1)

    if p.direction != "AUTO":
        along = 0 if p.direction == "U" else 1
    else:
        v_pole = any(s[0] == "v" for s in flow.poles)
        u_pole = any(s[0] == "u" for s in flow.poles)
        if v_pole != u_pole:
            along = 0 if v_pole else 1  # rings around the pole
        else:
            lengths = []
            for axis in (0, 1):
                seg = np.linalg.norm(np.diff(G, axis=axis), axis=2)
                both = inside[1:] & inside[:-1] if axis == 0 else inside[:, 1:] & inside[:, :-1]
                lengths.append(float(np.where(both, seg, 0.0).sum(axis).max()))
            along = 0 if lengths[0] >= lengths[1] else 1
    lay.along = along
    flow.direction = "UV"[along]
    ia, ic = along, 1 - along
    a_lo, a_hi, c_lo, c_hi = lo[ia], hi[ia], lo[ic], hi[ic]
    lay.a_range, lay.a_domain = (float(a_lo), float(a_hi)), dom[ia]
    lay.edges = edges if along == 0 else _swap(edges)

    def grid_ac(a_vals, c_vals):
        if along == 0:
            return _grid(basis, S, a_vals, c_vals)
        return _grid(basis, S, c_vals, a_vals).transpose(1, 0, 2)

    def mask_ac(a_vals, c_vals):
        return _mask(lay.edges, a_vals, c_vals)

    degrees = (S.degree_u, S.degree_v)
    spans_a, spans_c = S.shape[ia] - degrees[ia], S.shape[ic] - degrees[ic]
    a_s = np.linspace(a_lo, a_hi, int(np.clip(8 * spans_a + 1, 65, 257)))

    def chords(c_s):
        """Largest distance between neighbouring cross samples, over the passes."""
        Gc, m = grid_ac(a_s, c_s), mask_ac(a_s, c_s)
        seg = np.linalg.norm(np.diff(Gc, axis=1), axis=2)
        return np.where(m[:, 1:] | m[:, :-1], seg, 0.0).max(0)

    c_s = np.linspace(c_lo, c_hi, int(np.clip(8 * spans_c + 1, 65, 513)))
    ch = chords(c_s)
    k = int(np.clip(np.ceil(8.0 * ch.sum() / p.stepover) + 1, len(c_s), 4097))
    if k > len(c_s):
        c_s = np.linspace(c_lo, c_hi, k)
        ch = chords(c_s)
    total = float(ch.sum())
    flow.cross_length = total
    n = max(1, int(np.ceil(total / p.stepover * (1.0 - 1e-12))))
    if n + 1 > MAX_STATIONS:
        raise PatternTooLarge(
            f"Face {face.id} ({face.name}) needs {n + 1} passes at this stepover",
            (n + 1) * 2,
            p.max_points,
        )
    cum = np.concatenate([[0.0], np.cumsum(ch)])
    cum = cum + np.linspace(0.0, max(total, 1e-300) * 1e-12, len(cum))  # strictly rising
    stations = np.interp(np.linspace(0.0, cum[-1], n + 1), cum, c_s)
    eps = 1e-9 * (c_hi - c_lo)
    lay.stations = np.clip(stations, c_lo + eps, c_hi - eps)
    flow.stations = len(lay.stations)

    # measured: gaps between adjacent passes, pass lengths (for the point estimate)
    G2, m2 = grid_ac(a_s, lay.stations), mask_ac(a_s, lay.stations)
    if len(lay.stations) > 1:
        gap = np.linalg.norm(np.diff(G2, axis=1), axis=2)
        both = m2[:, 1:] & m2[:, :-1]
        flow.max_gap = float(np.where(both, gap, 0.0).max()) if both.any() else 0.0
    seg_a = np.linalg.norm(np.diff(G2, axis=0), axis=2)
    lay.full_lengths = seg_a.sum(0)
    inside_len = np.where(m2[1:] & m2[:-1], seg_a, 0.0).sum(0)
    lay.estimate = float((inside_len.sum() / p.step_along + 2 * len(lay.stations)) * reach_fraction)

    # passes around a closed face
    span = dom[ia][1] - dom[ia][0]
    if abs(a_lo - dom[ia][0]) <= 1e-9 * span and abs(a_hi - dom[ia][1]) <= 1e-9 * span:
        ends = grid_ac(np.array(dom[ia]), np.linspace(c_lo, c_hi, 9))
        flow.closed = bool(np.linalg.norm(ends[0] - ends[1], axis=1).max() <= p.tolerance)
    return lay


# --------------------------------------------------------------------------------------------
# Sampling the passes
# --------------------------------------------------------------------------------------------


@dataclass
class _Piece:
    """A kept stretch of one sampled interval."""

    a: np.ndarray
    P: np.ndarray
    n: np.ndarray
    at_lo: bool  # starts at the first sample of the station's first interval
    at_hi: bool  # ends at the last sample of the station's last interval
    whole: bool  # the station's only interval, nothing left out
    bisect_start: float | None = None  # steep neighbour before / after: bisect to it
    bisect_end: float | None = None


def _bisect_slope(lay: _Layout, a_ok, a_bad, c, nz_min: float, iters: int = 40):
    """Parameters where normal.z crosses nz_min, between kept and steep samples."""
    a_ok, a_bad = a_ok.copy(), a_bad.copy()
    for _ in range(iters):
        mid = 0.5 * (a_ok + a_bad)
        _, n = _frame(lay.S, lay.sign, *lay.uv(mid, c))
        ok = np.isfinite(n[:, 2]) & (n[:, 2] >= nz_min)
        a_ok = np.where(ok, mid, a_ok)
        a_bad = np.where(ok, a_bad, mid)
    return a_ok


def _sample(lay: _Layout, p: FlowlineParams) -> list:
    """The face's passes, in cutting order."""
    flow, S = lay.flow, lay.S
    if not len(lay.stations):
        return []
    a_lo, a_hi = lay.a_range
    dom_lo, dom_hi = lay.a_domain
    tiny = 1e-9 * (dom_hi - dom_lo)
    min_len = p.tolerance * 1e-3
    iso_dir = "v" if lay.along == 0 else "u"  # iso_curve("v", t) runs along u
    intervals = _intervals(lay.edges, lay.stations, a_lo, a_hi)

    # parameters of every sample, interval by interval
    runs = []  # (station index, interval index, interval count, a values)
    for j, (c, ivs) in enumerate(zip(lay.stations, intervals)):
        if not len(ivs):
            continue
        nd = int(np.clip(np.ceil(2.0 * lay.full_lengths[j] / p.step_along) + 1, 33, 65537))
        a_dense = np.linspace(a_lo, a_hi, nd)
        Q = S.iso_curve(iso_dir, float(c)).evaluate(a_dense)
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
        cum = cum + np.linspace(0.0, max(cum[-1], 1e-300) * 1e-12, nd)
        for k, (a0, a1) in enumerate(ivs):
            s0, s1 = np.interp([a0, a1], a_dense, cum)
            if s1 - s0 <= min_len:
                continue  # a pole, or a sliver of the trim
            m = max(1, int(np.ceil((s1 - s0) / p.step_along * (1.0 - 1e-12))))
            a = np.interp(np.linspace(s0, s1, m + 1), cum, a_dense)
            a[0], a[-1] = a0, a1
            runs.append((j, k, len(ivs), a))
    if not runs:
        flow.note = "no area"
        return []

    a_all = np.concatenate([r[3] for r in runs])
    c_all = np.concatenate([np.full(len(r[3]), lay.stations[r[0]]) for r in runs])
    P_all, n_all = _frame(S, lay.sign, *lay.uv(a_all, c_all))
    nz_min, tip = p.nz_min, p.tip_radius

    pieces: dict[int, list] = {}
    tasks = []  # (piece, "start" / "end", a_ok, a_bad, c)
    start = 0
    for j, k, count, a in runs:
        end = start + len(a)
        P, n = P_all[start:end], _fill_normals(n_all[start:end].copy())
        start = end
        steep = n[:, 2] < nz_min
        misfit = np.zeros(len(a), dtype=bool)
        if tip > 0 and len(a) > 1:
            C = P[:, :2] + tip * n[:, :2]
            dP, dC = np.diff(P[:, :2], axis=0), np.diff(C, axis=0)
            back = (dP * dC).sum(1) < -1e-12 * p.step_along**2
            misfit[:-1] |= back
            misfit[1:] |= back
            misfit &= ~steep
        flow.steep_points += int(steep.sum())
        flow.misfit_points += int(misfit.sum())
        keep = ~steep & ~misfit
        if not keep.any():
            continue
        # stretches of kept samples
        edges_ = np.flatnonzero(np.diff(np.concatenate([[0], keep.view(np.int8), [0]])))
        c = float(lay.stations[j])
        for s0, s1 in edges_.reshape(-1, 2):
            pc = _Piece(
                a[s0:s1],
                P[s0:s1],
                n[s0:s1],
                at_lo=(k == 0 and s0 == 0 and abs(a[0] - dom_lo) <= tiny),
                at_hi=(k == count - 1 and s1 == len(a) and abs(a[-1] - dom_hi) <= tiny),
                whole=(count == 1 and s0 == 0 and s1 == len(a)),
            )
            if s0 > 0 and steep[s0 - 1]:
                tasks.append((pc, "start", a[s0], a[s0 - 1], c))
            if s1 < len(a) and steep[s1]:
                tasks.append((pc, "end", a[s1 - 1], a[s1], c))
            pieces.setdefault(j, []).append(pc)

    # cut ends onto the slope limit
    if tasks:
        a_ok = np.array([t[2] for t in tasks])
        a_bad = np.array([t[3] for t in tasks])
        cs = np.array([t[4] for t in tasks])
        a_b = _bisect_slope(lay, a_ok, a_bad, cs, nz_min)
        P_b, n_b = _frame(S, lay.sign, *lay.uv(a_b, cs))
        n_b = _fill_normals(n_b)
        for (pc, where, *_), ab, pb, nb in zip(tasks, a_b, P_b, n_b):
            if where == "start" and np.linalg.norm(pb - pc.P[0]) > min_len:
                pc.a, pc.P, pc.n = np.r_[ab, pc.a], np.vstack([pb, pc.P]), np.vstack([nb, pc.n])
            elif where == "end" and np.linalg.norm(pb - pc.P[-1]) > min_len:
                pc.a, pc.P, pc.n = np.r_[pc.a, ab], np.vstack([pc.P, pb]), np.vstack([pc.n, nb])

    passes, emitted = [], 0
    for j in sorted(pieces):
        group = pieces[j]
        closed = False
        if flow.closed:
            if len(group) == 1 and group[0].whole and group[0].at_lo and group[0].at_hi:
                pc = group[0]  # a whole ring: drop the repeated seam point
                pc.a, pc.P, pc.n = pc.a[:-1], pc.P[:-1], pc.n[:-1]
                closed = True
            elif len(group) > 1 and group[0].at_lo and group[-1].at_hi:
                first, last = group[0], group.pop()  # joined across the seam
                first.a = np.r_[last.a, first.a[1:] + (dom_hi - dom_lo)]
                first.P = np.vstack([last.P, first.P[1:]])
                first.n = np.vstack([last.n, first.n[1:]])
        out = []
        for pc in group:
            if len(pc.P) < 2:
                continue
            if not closed and np.linalg.norm(np.diff(pc.P, axis=0), axis=1).sum() <= min_len:
                continue
            a = np.where(pc.a > dom_hi, pc.a - (dom_hi - dom_lo), pc.a)
            c = np.full(len(a), lay.stations[j])
            uv = np.column_stack(lay.uv(a, c))
            centers = pc.P[:, :2] + tip * pc.n[:, :2]
            out.append(Pass(flow.face, int(j), pc.P, pc.n, uv, centers, closed))
        if not out:
            continue
        if p.order == "ZIGZAG" and emitted % 2 == 1:
            out = [q.reversed() for q in reversed(out)]
        emitted += 1
        passes.extend(out)
    if p.reverse:
        passes = [q.reversed() for q in passes]
    flow.passes = len(passes)
    flow.points = int(sum(len(q.points) for q in passes))
    flow.length = float(sum(q.length for q in passes))
    if not passes and flow.note is None:
        if flow.misfit_points > flow.steep_points:
            flow.note = "the cutter's tip is too big for the surface's curve"
        else:
            flow.note = "too steep or facing down" if flow.steep_points else "nothing left"
    return passes


# --------------------------------------------------------------------------------------------
# Pattern
# --------------------------------------------------------------------------------------------


def flowline(kernel, body, params: FlowlineParams, matrix=None, faces=None) -> Flowline:
    """Flowline passes over a body's faces (all, or the face ids in `faces`), in world
    space (`matrix`: the object's 4x4 world matrix). See the module doc.

    Raises ValueError for bad params or a singular matrix, KeyError for unknown faces and
    PatternTooLarge when the pattern would exceed params.max_points."""
    started = time.perf_counter()
    p = params.validated()
    if faces is not None:
        faces = [int(f) for f in faces]
        unknown = sorted(set(faces) - set(body.faces))
        if unknown:
            raise KeyError(f"No faces {unknown}")
    M = np.eye(4) if matrix is None else np.asarray(matrix, dtype=float)
    A, t = M[:3, :3], M[:3, 3]
    det = float(np.linalg.det(A))
    if abs(det) < 1e-300:
        raise ValueError("the matrix is singular")
    base = 1.0 if det > 0 else -1.0  # a mirroring matrix flips the cross product
    axis = np.array([0.0, 0.0, 1.0])
    warnings = []
    is_open = any(not e.is_shared for e in body.edges.values())
    if is_open:
        warnings.append("Open body (sheet): faces were turned to face up (+Z)")
    basis = None
    if kernel is not None and kernel.has("nurbs.basis", "basis_matrix"):
        basis = kernel.module("nurbs.basis").basis_matrix

    layouts = []
    for fid, face in body.faces.items():
        if faces is not None and fid not in faces:
            continue
        sign = base
        if is_open:
            sign = base * _face_sign_world(face.surface, A, t, axis, base)
        t0 = time.perf_counter()
        try:
            lay = _layout(basis, face, M, sign, p)
        except PatternTooLarge:
            raise
        except Exception as e:  # a broken face shouldn't stop the others
            warnings.append(f"Face {fid} ({face.name}) was skipped: {e}")
            continue
        lay.flow.seconds = time.perf_counter() - t0
        layouts.append(lay)

    estimate = int(sum(lay.estimate for lay in layouts))
    if estimate > p.max_points:
        raise PatternTooLarge(
            f"The flowline pattern would have about {estimate} points (limit "
            f"{p.max_points}): use a larger step_along or stepover, or fewer faces",
            estimate,
            p.max_points,
        )

    passes = []
    for lay in layouts:
        t0 = time.perf_counter()
        try:
            passes.extend(_sample(lay, p))
        except Exception as e:
            warnings.append(f"Face {lay.flow.face} ({lay.flow.name}) was skipped: {e}")
            lay.flow.note = f"failed: {e}"
        lay.flow.seconds += time.perf_counter() - t0
    return Flowline(
        p, passes, [lay.flow for lay in layouts], time.perf_counter() - started, warnings
    )
