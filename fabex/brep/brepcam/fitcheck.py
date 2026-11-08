"""Fabex 'brepcam/fitcheck.py'

Will a cutter fit the B-rep? Curvature and inside corners from the exact surfaces. No bpy.

    analyze(body, matrix, faces, ...) -> FitAnalysis   cutter independent (cache it)
    FitAnalysis.summary()                              radii, sharp corners, per-face stats
    FitAnalysis.check(cutter, skin)                    fits?, problems, corner rest, advice

Normals point out of the material (BRepKit solids), towards the tool. A surface is concave
where it curves towards its normal, i.e. where its largest principal curvature along the
normal is positive. Three things limit a cutter coming from above (+Z):

  concave radius  1 / largest concave curvature, over every point the tool can reach
                  (normal.z >= 0). A ball - or a bull nose's corner, a ball cone's tip -
                  of radius r fits where this is >= r + skin.
  wall radius     the inside radius of the contour lines (horizontal sections) of walls
                  steeper than wall_angle: a pocket's corner radius, a hole's radius. A
                  cutter of radius R fits where this is >= R + skin.
  sharp corners   concave edges where faces meet at an angle (no fillet). No round cutter
                  reaches into them; reported with the material a cutter leaves there.

Curvature is sampled on a grid of cell centres inside each face's trim, then the tightest
samples are refined by zooming in, so analytic radii come out to ~1e-9 relative.
Bodies with open edges (sheets) have no inside / outside: their faces are turned to face
up (+Z) first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .edges import coedge_curve, loop_pcurves

INF = float("inf")
UNDERCUT_Z = -1e-3  # normal.z below this: the tool can't reach the point from above
FIT_TOL = 1e-6  # relative: radii this close count as fitting (float32 settings, fits)
FLOOR_COS = np.cos(np.radians(1.0))  # floors: within 1 degree of level
MIN_CELLS, MAX_CELLS = 8, 96  # grid cells per face direction
CHUNK = 2048

CHECKED = ("END", "BALLNOSE", "BULLNOSE", "BALLCONE")  # other types: not checked


@dataclass
class Cutter:
    """type: Fabex cutter type. corner_radius: BULLNOSE corner / BALLCONE ball radius."""

    type: str
    diameter: float
    corner_radius: float | None = None

    @property
    def radius(self) -> float:
        return 0.5 * float(self.diameter)

    @property
    def checked(self) -> bool:
        return self.type in CHECKED

    @property
    def tip_radius(self) -> float | None:
        """Radius of the round tip that has to fit concave curvature (END: a sharp edge)."""
        if self.type == "END":
            return 0.0
        if self.type == "BALLNOSE":
            return self.radius
        if self.type in ("BULLNOSE", "BALLCONE"):
            return min(float(self.corner_radius or 0.0), self.radius)
        return None

    @property
    def side_radius(self) -> float | None:
        """Radius that has to fit the walls' inside corners: flat-bottomed tools only (a
        ball's side never limits more than its tip; tapered sides aren't checked)."""
        return self.radius if self.type in ("END", "BULLNOSE") else None

    def wall_need(self, sin_slope, skin: float = 0.0):
        """Contour radius a wall needs where the tool touches it at this slope: the tool's
        radius at the contact point (a bull nose touches a less than vertical wall on its
        corner, inside its full radius), plus the skin."""
        tip = self.tip_radius or 0.0
        return self.radius - tip * (1.0 - sin_slope) + skin * sin_slope

    def corner_rest(self, kind: str, angle: float) -> float | None:
        """Material left in a sharp inside corner, along its bisector (None: unknown)."""
        f = _rest_factor(angle)
        if not self.checked:
            return None
        if kind == "vertical":  # round in plan: the full radius
            return None if self.type == "BALLCONE" else self.radius * f
        if kind == "floor" and self.type == "END":
            return 0.0
        if self.type == "END":
            return None  # a flat end on a sloped crease: depends on the slopes
        return self.tip_radius * f

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "diameter": float(self.diameter),
            "corner_radius": None if self.corner_radius is None else float(self.corner_radius),
            "tip_radius": self.tip_radius,
            "side_radius": self.side_radius,
        }


def _rest_factor(angle_deg: float) -> float:
    """Gap between a corner and a round tool of radius 1 touching both of its faces."""
    half = np.radians(min(float(angle_deg), 179.0)) / 2
    return float(1.0 / np.cos(half) - 1.0)


def _num(x):
    """JSON-safe float: inf (no limit) -> None."""
    return None if x is None or not np.isfinite(x) else float(x)


# --------------------------------------------------------------------------------------------
# Differential geometry in world space
# --------------------------------------------------------------------------------------------


class _Frame:
    """World-space surface data on a (u, v) grid, normals out of the material."""

    def __init__(self, surface, us, vs, A, t, sign, axis):
        D = surface.grid(np.asarray(us, float), np.asarray(vs, float), nd=2)
        D = D @ A.T
        P = D[0, 0] + t
        Su, Sv, Suu, Suv, Svv = D[1, 0], D[0, 1], D[2, 0], D[1, 1], D[0, 2]
        N = np.cross(Su, Sv) * sign
        ln = np.linalg.norm(N, axis=-1)
        lu, lv = np.linalg.norm(Su, axis=-1), np.linalg.norm(Sv, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            n = N / ln[..., None]
            E, F, G = (Su * Su).sum(-1), (Su * Sv).sum(-1), (Sv * Sv).sum(-1)
            L, M, Nn = (Suu * n).sum(-1), (Suv * n).sum(-1), (Svv * n).sum(-1)
            den = E * G - F * F
            K = (L * Nn - M * M) / den
            H = (E * Nn - 2 * F * M + G * L) / (2 * den)
            k1 = H + np.sqrt(np.maximum(H * H - K, 0.0))  # most concave principal curvature
            # contour lines: horizontal tangent t = axis x n; Meusnier for the plane section
            th = np.cross(axis, n)
            sin_slope = np.linalg.norm(th, axis=-1)
            th = th / sin_slope[..., None]
            a_, b_ = _solve2(E, F, G, (th * Su).sum(-1), (th * Sv).sum(-1))
            I = a_ * a_ * E + 2 * a_ * b_ * F + b_ * b_ * G
            II = a_ * a_ * L + 2 * a_ * b_ * M + b_ * b_ * Nn
            kc = II / I / sin_slope
        self.P, self.n, self.k, self.kc, self.sin = P, n, k1, kc, sin_slope
        self.nz = (n * axis).sum(-1)
        self.density = ln  # area per unit (u, v)
        self.lu, self.lv = lu, lv

    def valid(self, ref: float) -> np.ndarray:
        """Not degenerate (poles, collapsed sides) and finite."""
        ok = self.density > 1e-7 * ref
        ok &= np.minimum(self.lu, self.lv) > 1e-12 * np.maximum(self.lu, self.lv)
        return ok & np.isfinite(self.k) & np.all(np.isfinite(self.n), axis=-1)


def _solve2(E, F, G, a, b):
    """Solve [[E, F], [F, G]] x = [a, b] elementwise."""
    det = E * G - F * F
    return (G * a - F * b) / det, (E * b - F * a) / det


# --------------------------------------------------------------------------------------------
# Trim regions
# --------------------------------------------------------------------------------------------


def _loop_polygons(face) -> list:
    """uv polygons of a trimmed face's loops (None for an untrimmed face)."""
    if face.loops is None:
        return None
    polys = []
    for li in range(len(face.loops)):
        pts = []
        for pc in loop_pcurves(face, li):
            n = max(33, 8 * len(getattr(pc, "cpw", ())) + 1)
            pts.append(pc.evaluate(np.linspace(0.0, 1.0, n))[:-1])
        polys.append(np.concatenate(pts))
    return polys


def _inside(uv: np.ndarray, polys) -> np.ndarray:
    """Even-odd point-in-region test (orientation independent), in chunks."""
    if polys is None:
        return np.ones(len(uv), dtype=bool)
    out = np.zeros(len(uv), dtype=bool)
    for s in range(0, len(uv), CHUNK):
        P = uv[s : s + CHUNK]
        inside = np.zeros(len(P), dtype=bool)
        for poly in polys:
            a, b = poly, np.roll(poly, -1, axis=0)
            py, px = P[:, 1][:, None], P[:, 0][:, None]
            ay, by = a[:, 1][None], b[:, 1][None]
            crosses = (ay > py) != (by > py)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_at = a[:, 0][None] + (py - ay) * (b[:, 0] - a[:, 0])[None] / (by - ay)
            inside ^= (np.sum(crosses & (px < x_at), axis=1) % 2).astype(bool)
        out[s : s + CHUNK] = inside
    return out


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# --------------------------------------------------------------------------------------------
# Faces
# --------------------------------------------------------------------------------------------


@dataclass
class FaceFit:
    """One face's curvature samples (world space) and its refined tightest spots."""

    face: int
    name: str
    area: float
    size: float  # bounding box diagonal of the sampled points
    undercut_fraction: float
    k: np.ndarray  # concave curvature at reachable samples (<= 0: flat or convex)
    w: np.ndarray  # their area weights
    kc: np.ndarray  # contour concave curvature at wall samples
    wc: np.ndarray  # their area weights
    sc: np.ndarray  # their sin(slope)
    Pc: np.ndarray  # their points
    concave_radius: float = INF
    concave_at: np.ndarray | None = None
    wall_radius: float = INF
    wall_at: np.ndarray | None = None
    wall_sin: float = 1.0

    def concave_fraction(self) -> float:
        """Area fraction curving towards the tool (radius under 1000x the face size)."""
        if self.area <= 0:
            return 0.0
        return float(self.w[self.k * self.size > 1e-3].sum() / self.area)

    def tight_fraction(self, need: float) -> float:
        """Area fraction where a tip of radius `need` doesn't fit."""
        if self.area <= 0 or need <= 0:
            return 0.0
        return float(self.w[self.k > (1.0 + FIT_TOL) / need].sum() / self.area)

    def wall_misfit(self, cutter, skin: float):
        """(contour radius, need, point, area fraction) where the walls are too tight for
        the cutter's side, worst first - or None. Checks the wall samples and the refined
        tightest point."""
        rho = np.append(
            np.where(self.kc > 0, 1.0 / np.maximum(self.kc, 1e-300), INF), self.wall_radius
        )
        sin = np.append(self.sc, self.wall_sin)
        need = cutter.wall_need(sin, skin)
        short = rho < need * (1 - FIT_TOL)
        if not short.any():
            return None
        i = int(np.argmin(np.where(short, rho - need, INF)))
        at = self.Pc[i] if i < len(self.Pc) else self.wall_at
        frac = float(self.wc[short[:-1]].sum() / self.area) if self.area > 0 else 0.0
        return float(rho[i]), float(need[i]), at, frac

    def summary(self) -> dict:
        return {
            "face": self.face,
            "name": self.name,
            "area": float(self.area),
            "concave_radius": _num(self.concave_radius),
            "concave_at": None if self.concave_at is None else self.concave_at.tolist(),
            "wall_radius": _num(self.wall_radius),
            "wall_at": None if self.wall_at is None else self.wall_at.tolist(),
            "concave_fraction": round(self.concave_fraction(), 6),
            "undercut_fraction": round(float(self.undercut_fraction), 6),
        }


def _cells(surface, direction: int) -> int:
    n_ctrl = surface.shape[direction]
    degree = surface.degree_u if direction == 0 else surface.degree_v
    return int(np.clip(4 * (n_ctrl - degree) + 4, MIN_CELLS, MAX_CELLS))


def _refine(frame_at, inside, start, h, domain, pick, iters: int = 7, n: int = 5):
    """Zoom in on the largest `pick(frame)` around uv `start` (cell size h).

    Returns (value, point, sin slope) of the best valid sample seen."""
    (ua, ub), (va, vb) = domain
    u, v = start
    hu, hv = h
    best, best_P, best_sin = -INF, None, 1.0
    for _ in range(iters):
        us = np.clip(np.linspace(u - hu, u + hu, n), ua, ub)
        vs = np.clip(np.linspace(v - hv, v + hv, n), va, vb)
        fr = frame_at(us, vs)
        val = pick(fr)
        UU, VV = np.meshgrid(us, vs, indexing="ij")
        ok = np.isfinite(val) & inside(np.column_stack([UU.ravel(), VV.ravel()])).reshape(val.shape)
        if ok.any():
            i = np.unravel_index(np.argmax(np.where(ok, val, -INF)), val.shape)
            if val[i] > best:
                best, best_P, best_sin = float(val[i]), fr.P[i].copy(), float(fr.sin[i])
                u, v = float(us[i[0]]), float(vs[i[1]])
        hu, hv = hu / 2.5, hv / 2.5
    return best, best_P, best_sin


def analyze_face(face, A, t, sign, axis, wall_cos, candidates: int = 3) -> FaceFit:
    surface = face.surface
    (ua, ub), (va, vb) = surface.domain_u, surface.domain_v
    nu, nv = _cells(surface, 0), _cells(surface, 1)
    hu, hv = (ub - ua) / nu, (vb - va) / nv
    us = ua + (np.arange(nu) + 0.5) * hu
    vs = va + (np.arange(nv) + 0.5) * hv
    polys = _loop_polygons(face)
    fr = _Frame(surface, us, vs, A, t, sign, axis)
    ref = float(np.nanmax(fr.density)) if np.isfinite(fr.density).any() else 1.0
    UU, VV = np.meshgrid(us, vs, indexing="ij")
    uv = np.column_stack([UU.ravel(), VV.ravel()])
    ok = (_inside(uv, polys).reshape(UU.shape)) & fr.valid(ref)
    weight = np.where(ok, fr.density * hu * hv, 0.0)
    area = float(weight.sum())
    reach = ok & (fr.nz >= UNDERCUT_Z)
    wall = reach & (np.abs(fr.nz) <= wall_cos) & np.isfinite(fr.kc)
    P = fr.P[ok]
    size = float(np.linalg.norm(P.max(0) - P.min(0))) if len(P) else 0.0
    under = float(weight[ok & ~reach].sum() / area) if area > 0 else 0.0
    ff = FaceFit(
        face.id,
        face.name,
        area,
        size,
        under,
        fr.k[reach],
        weight[reach],
        fr.kc[wall],
        weight[wall],
        fr.sin[wall],
        fr.P[wall],
    )

    def frame_at(us_, vs_):
        return _Frame(surface, us_, vs_, A, t, sign, axis)

    def inside(uv_):
        return _inside(uv_, polys)

    def refine(mask, values, pick):
        """Best refined (curvature, point, sin slope) from the top candidates in `mask`."""
        vals = np.where(mask, values, -INF)
        flat = np.argsort(vals, axis=None)[::-1][:candidates]
        if mask.sum() >= 4 and np.array_equal(mask, reach):
            # constant over the whole face (cylinder, sphere): nothing to refine
            top, low = float(values[mask].max()), float(values[mask].min())
            if top > 0 and top - low <= 1e-9 * top:
                i, j = np.unravel_index(flat[0], vals.shape)
                return top, fr.P[i, j].copy(), float(fr.sin[i, j])
        best = (-INF, None, 1.0)
        for f in flat:
            i, j = np.unravel_index(f, vals.shape)
            if not np.isfinite(vals[i, j]) or vals[i, j] <= 0:
                break
            found = _refine(frame_at, inside, (us[i], vs[j]), (hu, hv), domain, pick)
            if found[0] > best[0]:
                best = found
        return best

    domain = (surface.domain_u, surface.domain_v)
    k, at, _ = refine(
        reach, fr.k, lambda f: np.where(f.valid(ref) & (f.nz >= UNDERCUT_Z), f.k, np.nan)
    )
    if k > 0 and k * max(size, 1e-300) > 1e-6:
        ff.concave_radius, ff.concave_at = 1.0 / k, at
    kc, at, sin = refine(
        wall,
        fr.kc,
        lambda f: np.where(
            f.valid(ref) & (f.nz >= UNDERCUT_Z) & (np.abs(f.nz) <= wall_cos), f.kc, np.nan
        ),
    )
    if kc > 0 and kc * max(size, 1e-300) > 1e-6:
        ff.wall_radius, ff.wall_at, ff.wall_sin = 1.0 / kc, at, sin
    return ff


# --------------------------------------------------------------------------------------------
# Sharp inside corners
# --------------------------------------------------------------------------------------------


def _left_is_inside(face, c) -> bool:
    """Is the face's interior to the left of the coedge's own parameter direction (in uv)?"""
    if not c.is_trim:
        return c.side in ("v0", "u1")  # side_line runs in ascending side parameter
    poly = np.concatenate(
        [pc.evaluate(np.linspace(0, 1, 17))[:-1] for pc in loop_pcurves(face, c.loop)]
    )
    ccw = _signed_area(poly) > 0
    return ccw == (c.loop == 0)


def _coedge_frame(body, c, ts, A, sign):
    """World points, outward normals and inward (into the face) directions at ts."""
    face = body.faces[c.face]
    surface, pc = coedge_curve(body, c)
    uv = pc.evaluate(ts)
    h = 1e-6
    duv = pc.evaluate(np.clip(ts + h, 0, 1)) - pc.evaluate(np.clip(ts - h, 0, 1))
    left = np.column_stack([-duv[:, 1], duv[:, 0]])
    if not _left_is_inside(face, c):
        left = -left
    D = surface.derivatives(uv[:, 0], uv[:, 1], 1) @ A.T
    Su, Sv = D[1, 0], D[0, 1]
    n = np.cross(Su, Sv) * sign
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.linalg.norm(n, axis=1)[:, None]
    d = left[:, :1] * Su + left[:, 1:] * Sv
    return D[0, 0], n, d


def find_corners(body, A, t, signs, axis, wall_cos, corner_angle: float, faces=None) -> list:
    """Concave edges where two faces meet at >= corner_angle degrees (no fillet), unless
    one of the faces faces away from the tool."""
    out = []
    cos_min = np.cos(np.radians(corner_angle))
    ts = np.linspace(0.0, 1.0, 11)[1:-1]
    tb = np.linspace(0.0, 1.0, 257)
    for e in body.edges.values():
        if not e.is_shared:
            continue
        ca, cb = e.coedges
        if ca.face == cb.face:
            continue  # a seam of one closed face
        if faces is not None and ca.face not in faces and cb.face not in faces:
            continue
        sa, sb = signs.get(ca.face, 1.0), signs.get(cb.face, 1.0)
        PA, nA, dA = _coedge_frame(body, ca, ts, A, sa)
        PB_all, nB_all, dB_all = _coedge_frame(body, cb, tb, A, sb)
        j = np.argmin(np.linalg.norm(PA[:, None] - PB_all[None], axis=2), axis=1)
        nB, dB = nB_all[j], dB_all[j]
        T = np.gradient(PA, axis=0)
        T /= np.maximum(np.linalg.norm(T, axis=1), 1e-300)[:, None]

        def across(d):
            d = d - (d * T).sum(1)[:, None] * T
            return d / np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]

        cos = np.clip((nA * nB).sum(1), -1.0, 1.0)
        score = 0.5 * ((across(dB) * nA).sum(1) + (across(dA) * nB).sum(1))
        good = np.isfinite(cos) & np.isfinite(score)
        if not good.any():
            continue
        sharp = good & (cos <= cos_min)
        if not sharp.any() or np.mean(score[sharp]) <= 0:
            continue
        zA, zB = (nA * axis).sum(1), (nB * axis).sum(1)
        if (zA[sharp] < UNDERCUT_Z).all() or (zB[sharp] < UNDERCUT_Z).all():
            continue  # one face faces down (an overhang): out of reach from above
        wall = (np.abs(zA) <= wall_cos) & (np.abs(zB) <= wall_cos)
        floor = ((zA >= FLOOR_COS) & (np.abs(zB) <= wall_cos)) | (
            (zB >= FLOOR_COS) & (np.abs(zA) <= wall_cos)
        )
        kind = "vertical" if wall[sharp].all() else "floor" if floor[sharp].all() else "other"
        surface, pc = coedge_curve(body, ca)
        uv = pc.evaluate(np.linspace(0.0, 1.0, 129))
        Q = surface.evaluate(uv[:, 0], uv[:, 1]) @ A.T + t
        angle = float(np.degrees(np.arccos(cos[sharp].min())))
        out.append(
            {
                "edge": int(e.id),
                "faces": [int(ca.face), int(cb.face)],
                "face_names": [body.faces[ca.face].name, body.faces[cb.face].name],
                "kind": kind,
                "angle": round(angle, 4),
                "length": float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()),
                "midpoint": Q[len(Q) // 2].tolist(),
                "z_min": float(Q[:, 2].min()),
                "z_max": float(Q[:, 2].max()),
            }
        )
    return out


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------


@dataclass
class FitAnalysis:
    faces: list  # [FaceFit]
    corners: list  # [dict]
    wall_angle: float
    corner_angle: float
    seconds: float = 0.0
    warnings: list = field(default_factory=list)

    def _tightest(self, attr: str) -> dict:
        best = min(self.faces, key=lambda f: getattr(f, attr + "_radius"), default=None)
        if best is None or not np.isfinite(getattr(best, attr + "_radius")):
            return {"radius": None, "at": None, "face": None, "face_name": None}
        return {
            "radius": float(getattr(best, attr + "_radius")),
            "at": getattr(best, attr + "_at").tolist(),
            "face": best.face,
            "face_name": best.name,
        }

    def min_concave_radius(self) -> dict:
        """Tightest concave curvature the tool can reach (radius None: none at all)."""
        return self._tightest("concave")

    def min_wall_radius(self) -> dict:
        """Tightest inside radius of the walls' contour lines (radius None: none)."""
        return self._tightest("wall")

    def summary(self) -> dict:
        return {
            "min_concave_radius": self.min_concave_radius(),
            "min_wall_radius": self.min_wall_radius(),
            "sharp_corners": self.corners,
            "faces": [f.summary() for f in self.faces],
            "wall_angle": self.wall_angle,
            "corner_angle": self.corner_angle,
            "seconds": round(self.seconds, 4),
            "warnings": list(self.warnings),
        }

    def check(self, cutter: Cutter, skin: float = 0.0) -> dict:
        """Does `cutter` (leaving `skin`) reach every concave surface and wall corner?

        fits is False when some concave radius < tip radius + skin or some wall radius <
        cutter radius + skin; sharp corners can't be reached by any round cutter and are
        reported separately (with the material left in them). None: not checked for
        this cutter type."""
        skin = float(skin or 0.0)
        tip, side = cutter.tip_radius, cutter.side_radius
        problems = []
        for f in self.faces if cutter.checked else ():
            if tip is not None and tip + skin > 0:
                need = tip + skin
                if f.concave_radius < need * (1 - FIT_TOL):
                    problems.append(
                        {
                            "check": "concave_radius",
                            "face": f.face,
                            "face_name": f.name,
                            "radius": float(f.concave_radius),
                            "need": need,
                            "at": f.concave_at.tolist(),
                            "tight_fraction": round(f.tight_fraction(need), 6),
                        }
                    )
            if side is not None:
                miss = f.wall_misfit(cutter, skin)
                if miss is not None:
                    problems.append(
                        {
                            "check": "wall_radius",
                            "face": f.face,
                            "face_name": f.name,
                            "radius": miss[0],
                            "need": miss[1],
                            "at": np.asarray(miss[2]).tolist(),
                            "tight_fraction": round(miss[3], 6),
                        }
                    )
        problems.sort(key=lambda p: p["radius"])
        corners = []
        for c in self.corners:
            rest = cutter.corner_rest(c["kind"], c["angle"]) if cutter.checked else None
            corners.append({**c, "rest": rest})
        rests = [c["rest"] for c in corners if c["rest"] is not None]
        conc, wall = self.min_concave_radius()["radius"], self.min_wall_radius()["radius"]
        return {
            "cutter": cutter.to_dict(),
            "skin": skin,
            "checked": cutter.checked,
            "fits": (not problems) if cutter.checked else None,
            "problems": problems,
            "sharp_corners": corners,
            "corner_rest_max": max(rests) if rests else None,
            "max_tip_radius": None if conc is None else max(conc - skin, 0.0),
            "max_cutter_radius": None if wall is None else max(wall - skin, 0.0),
        }


def analyze(
    body,
    matrix=None,
    faces=None,
    wall_angle: float = 80.0,
    corner_angle: float = 10.0,
    axis=(0.0, 0.0, 1.0),
) -> FitAnalysis:
    """Curvature limits and sharp inside corners of a body, in world space (see module doc).

    faces: only these face ids (corners: edges touching them). wall_angle: faces steeper
    than this (degrees from level) are walls. corner_angle: smallest angle between two
    faces' normals that counts as a sharp corner."""
    started = time.perf_counter()
    if not 0.0 < float(wall_angle) <= 90.0:
        raise ValueError("wall_angle must be in (0, 90] degrees")
    if not 0.0 < float(corner_angle) < 180.0:
        raise ValueError("corner_angle must be in (0, 180) degrees")
    if faces is not None:
        faces = {int(f) for f in faces}
        unknown = sorted(faces - set(body.faces))
        if unknown:
            raise KeyError(f"No faces {unknown}")
    M = np.eye(4) if matrix is None else np.asarray(matrix, dtype=float)
    A, t = M[:3, :3], M[:3, 3]
    det = float(np.linalg.det(A))
    if abs(det) < 1e-300:
        raise ValueError("the matrix is singular")
    base = 1.0 if det > 0 else -1.0  # a mirroring matrix flips the cross product
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    wall_cos = float(np.cos(np.radians(wall_angle)))
    warnings = []
    is_open = any(not e.is_shared for e in body.edges.values())
    signs = {fid: base for fid in body.faces}
    if is_open:
        for fid, face in body.faces.items():
            signs[fid] = base * _face_sign_world(face.surface, A, t, axis, base)
        warnings.append("Open body (sheet): faces were turned to face up (+Z)")
    out = []
    for fid, face in body.faces.items():
        if faces is not None and fid not in faces:
            continue
        try:
            out.append(analyze_face(face, A, t, signs[fid], axis, wall_cos))
        except Exception as e:  # a broken face shouldn't stop the others
            warnings.append(f"Face {fid} ({face.name}) was skipped: {e}")
    corners = find_corners(body, A, t, signs, axis, wall_cos, corner_angle, faces)
    return FitAnalysis(
        out,
        corners,
        float(wall_angle),
        float(corner_angle),
        time.perf_counter() - started,
        warnings,
    )


def _face_sign_world(surface, A, t, axis, base) -> float:
    """+1 / -1 relative to `base`: which way the face's world normals face the tool."""
    us = np.linspace(*surface.domain_u, 9)
    vs = np.linspace(*surface.domain_v, 9)
    f = _Frame(surface, us, vs, A, t, base, axis)
    ok = np.isfinite(f.nz) & np.isfinite(f.density)
    return -1.0 if float((f.nz[ok] * f.density[ok]).sum()) < 0 else 1.0
