"""Fabex 'brepcam/classify.py'

What each face and edge of a B-rep is, for planning CAM. No bpy.

    classify(body, matrix, tolerance, ...) -> Classification   (.summary(): JSON-able)

Everything is measured from the exact surfaces, in world space, with normals pointing out
of the material (BRepKit solids), for a tool coming from above (+Z).

Surface: the simplest analytic surface within `tolerance` of every sample of the face
(inside its trim and along its boundary), with its parameters:

    plane     normal, point
    cylinder  axis, center (a point on the axis), radius
    cone      axis, apex, half_angle (degrees)
    sphere    center, radius
    torus     axis, center, radius (of the tube's centre circle), minor_radius
    freeform  none of these

Curvature: "flat" | "concave" (curving towards the normal - towards the tool, like a
fillet or a hole) | "convex" | "saddle" | "mixed", and the smallest radius on the face.

Edges between faces: "tangent" (normals within tangent_angle along the whole edge),
"concave" (an inside corner), "convex" (an outside edge), "naked" (used by one face: an
open sheet's boundary) or "seam" (a closed face meeting itself).

Role - what the face is for machining from above:

    top        level, facing up: the highest such faces (face them)
    floor      level, facing up, lower down (pocket and step floors)
    slope      flat, facing up, between level and wall_angle
    wall       steeper than wall_angle everywhere (sides, pocket walls)
    hole_wall  the wall of a round hole (see brepcam.edges.find_holes); the walls of other
               openings (pockets, slots) are walls. Both get the opening's index ("hole")
    fillet     a concave blend: a curved face tangent to two other faces that turns from
               one to the other (fillets in pockets, along floors and around bosses,
               spherical corners)
    round      a convex blend (rounded outside edges)
    curved     any other face facing up (domes, freeform surfaces); may be partly
               undercut (see undercut_fraction)
    bottom     level, facing down: the lowest such faces (the side it stands on)
    underside  any other face facing down (overhangs: out of reach from above)
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import fitcheck as FC
from .edges import indexed_holes, loop_pcurves, public_hole

SURFACES = ("plane", "cylinder", "cone", "sphere", "torus", "freeform")
ROLES = (
    "top",
    "floor",
    "slope",
    "wall",
    "hole_wall",
    "fillet",
    "round",
    "curved",
    "bottom",
    "underside",
)
EDGE_KINDS = ("tangent", "concave", "convex", "naked", "seam")

LEVEL_COS = FC.FLOOR_COS  # level: normals within 1 degree of +Z (or -Z)
UNDERCUT_Z = FC.UNDERCUT_Z
FLAT_REL = 1e-3  # curvature radius over 1000x the face size counts as flat
BLEND_TURN = 15.0  # degrees a blend turns between the faces it joins
MIN_INSIDE = 64  # samples inside the trim, refining the grid for narrow faces
MAX_SAMPLES = 200_000
CHUNK = 16384


def _num(x, digits: int | None = None):
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, digits) if digits is not None else x


def _vec(v) -> list:
    return [float(x) for x in v]


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# --------------------------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------------------------


@dataclass
class _Samples:
    P: np.ndarray  # (m, 3) world points
    n: np.ndarray  # (m, 3) unit normals, out of the material
    k1: np.ndarray  # most concave principal curvature (> 0: curving towards the normal)
    k2: np.ndarray  # most convex principal curvature
    w: np.ndarray  # area weights (0 for boundary samples)


def _frame(surface, uv, A, t, sign) -> tuple[_Samples, np.ndarray]:
    """World points, normals and principal curvatures at uv points (pointwise, chunked), and
    the area density |Su x Sv| there (weights still unset)."""
    out = {k: [] for k in ("P", "n", "k1", "k2", "dens")}
    for s in range(0, len(uv), CHUNK):
        c = uv[s : s + CHUNK]
        D = surface.derivatives(c[:, 0], c[:, 1], 2) @ A.T
        Su, Sv, Suu, Suv, Svv = D[1, 0], D[0, 1], D[2, 0], D[1, 1], D[0, 2]
        N = np.cross(Su, Sv) * sign
        ln = np.linalg.norm(N, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            n = N / ln[:, None]
            E, F, G = (Su * Su).sum(-1), (Su * Sv).sum(-1), (Sv * Sv).sum(-1)
            L, M, Nn = (Suu * n).sum(-1), (Suv * n).sum(-1), (Svv * n).sum(-1)
            den = E * G - F * F
            K = (L * Nn - M * M) / den
            H = (E * Nn - 2 * F * M + G * L) / (2 * den)
            root = np.sqrt(np.maximum(H * H - K, 0.0))
            out["k1"].append(H + root)
            out["k2"].append(H - root)
        out["P"].append(D[0, 0] + t)
        out["n"].append(n)
        out["dens"].append(ln)
    cat = {
        k: np.concatenate(v) if v else np.zeros((0, 3) if k in "Pn" else 0) for k, v in out.items()
    }
    return _Samples(cat["P"], cat["n"], cat["k1"], cat["k2"], np.zeros(len(uv))), cat["dens"]


def _natural_boundary(surface, n: int = 33) -> np.ndarray:
    (ua, ub), (va, vb) = surface.domain_u, surface.domain_v
    s = np.linspace(0.0, 1.0, n)[:-1]
    return np.concatenate(
        [
            np.column_stack([ua + s * (ub - ua), np.full_like(s, va)]),
            np.column_stack([np.full_like(s, ub), va + s * (vb - va)]),
            np.column_stack([ub - s * (ub - ua), np.full_like(s, vb)]),
            np.column_stack([np.full_like(s, ua), vb - s * (vb - va)]),
        ]
    )


def face_samples(face, A, t, sign):
    """Samples of one face - cell centres inside its trim (area weighted) and points along
    its boundary - without degenerate ones (poles); and its world bounding box (lo, hi)."""
    surface = face.surface
    (ua, ub), (va, vb) = surface.domain_u, surface.domain_v
    nu, nv = FC._cells(surface, 0), FC._cells(surface, 1)
    polys = FC._loop_polygons(face)
    while True:
        hu, hv = (ub - ua) / nu, (vb - va) / nv
        us = ua + (np.arange(nu) + 0.5) * hu
        vs = va + (np.arange(nv) + 0.5) * hv
        UU, VV = np.meshgrid(us, vs, indexing="ij")
        uv = np.column_stack([UU.ravel(), VV.ravel()])
        inside = FC._inside(uv, polys)
        if inside.sum() >= MIN_INSIDE or nu * nv * 9 > MAX_SAMPLES:
            break
        nu, nv = nu * 3, nv * 3
    uv_in = uv[inside]
    uv_b = np.concatenate(polys) if polys is not None else _natural_boundary(surface)
    s, density = _frame(surface, np.concatenate([uv_in, uv_b]), A, t, sign)
    s.w[: len(uv_in)] = density[: len(uv_in)] * hu * hv
    ok = np.all(np.isfinite(s.n), axis=1) & np.isfinite(s.k1) & np.isfinite(s.k2)
    ref = float(density[ok].max()) if ok.any() else 0.0
    ok &= density > 1e-7 * ref
    samples = _Samples(s.P[ok], s.n[ok], s.k1[ok], s.k2[ok], s.w[ok])
    return samples, _extent(face, A, t, s.P[: len(uv_in)], uv_in, (hu, hv), polys)


def _zoom(evaluate, x0, h, lo, hi, accept, best, sgn, k, iters: int = 10, n: int = 7):
    """Zoom in on the largest sgn * coordinate k of evaluate(x) around x0 (1 or 2 params)."""
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    h = np.atleast_1d(np.asarray(h, dtype=float))
    for _ in range(iters):
        axes = [
            np.clip(np.linspace(x[i] - h[i], x[i] + h[i], n), lo[i], hi[i]) for i in range(len(x))
        ]
        X = np.column_stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")])
        val = sgn * evaluate(X)[:, k]
        val = np.where(accept(X), val, -np.inf)
        i = int(np.argmax(val))
        if val[i] > best:
            best, x = float(val[i]), X[i]
        h = h / 3.0
    return best


def _extent(face, A, t, P_in, uv_in, h, polys):
    """World bounding box of a face: its extremes inside (zoomed in from the best cell
    centre) and along every boundary curve (zoomed in along the curve) - to ~1e-9 of the
    face size, so a part's top can be put exactly at Z 0."""
    surface = face.surface
    (ua, ub), (va, vb) = surface.domain_u, surface.domain_v

    def at_uv(X):
        return surface.evaluate(X[:, 0], X[:, 1]) @ A.T + t

    def inside(X):
        return FC._inside(X, polys)

    curves = []
    for li in range(len(face.loops) if face.loops else 1):
        curves += loop_pcurves(face, li)
    ts = np.linspace(0.0, 1.0, 129)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for pc in curves:

        def on_curve(X, pc=pc):
            return at_uv(np.atleast_2d(pc.evaluate(X[:, 0])))

        P = on_curve(ts[:, None])
        for k in range(3):
            for sgn in (1.0, -1.0):
                i = int(np.argmax(sgn * P[:, k]))
                if i in (0, len(ts) - 1) or np.ptp(P[:, k]) <= 1e-12 * (1.0 + abs(P[i, k])):
                    best = sgn * P[i, k]  # at a vertex, or level: exact already
                else:
                    best = _zoom(
                        on_curve,
                        [ts[i]],
                        [ts[1]],
                        [0.0],
                        [1.0],
                        lambda X: np.ones(len(X), bool),
                        sgn * P[i, k],
                        sgn,
                        k,
                    )
                if sgn > 0:
                    hi[k] = max(hi[k], best)
                else:
                    lo[k] = min(lo[k], -best)
    if len(P_in):
        span = np.ptp(P_in, axis=0)
        for k in range(3):
            for sgn in (1.0, -1.0):
                i = int(np.argmax(sgn * P_in[:, k]))
                edge = hi[k] if sgn > 0 else -lo[k]
                if sgn * P_in[i, k] < edge - 0.05 * span[k]:
                    continue  # well inside the boundary's extreme (planes always are)
                best = _zoom(
                    at_uv, uv_in[i], h, [ua, va], [ub, vb], inside, sgn * P_in[i, k], sgn, k
                )
                if sgn > 0:
                    hi[k] = max(hi[k], best)
                else:
                    lo[k] = min(lo[k], -best)
    return lo, hi


# --------------------------------------------------------------------------------------------
# Analytic surface fits
# --------------------------------------------------------------------------------------------


def _fit_plane(s: _Samples):
    c = s.P.mean(axis=0)
    _, _, Vt = np.linalg.svd(s.P - c, full_matrices=False)
    nrm = Vt[2]
    if float((s.n @ nrm).mean()) < 0:
        nrm = -nrm
    dev = float(np.abs((s.P - c) @ nrm).max())
    return {"type": "plane", "normal": _vec(nrm), "point": _vec(c)}, dev


def _fit_sphere(s: _Samples):
    c0 = s.P.mean(axis=0)
    Q = s.P - c0
    A = np.column_stack([2 * Q, np.ones(len(Q))])
    sol, *_ = np.linalg.lstsq(A, (Q * Q).sum(1), rcond=None)
    c = sol[:3]
    r2 = sol[3] + c @ c
    if not np.isfinite(r2) or r2 <= 0:
        return None, np.inf
    r = float(np.sqrt(r2))
    dev = float(np.abs(np.linalg.norm(Q - c, axis=1) - r).max())
    return {"type": "sphere", "center": _vec(c + c0), "radius": r}, dev


def _perp_basis(a):
    e1 = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def _fit_revolved(s: _Samples):
    """Cylinder and cone: the normals make a constant angle with the axis, and every normal
    line meets it. Returns (cylinder, dev), (cone, dev)."""
    none = (None, np.inf), (None, np.inf)
    nc = s.n - s.n.mean(axis=0)
    _, V = np.linalg.eigh(nc.T @ nc)
    a = V[:, 0]
    if a[2] < 0 or (a[2] == 0 and a[0] < 0):
        a = -a
    e1, e2 = _perp_basis(a)
    P2 = np.column_stack([s.P @ e1, s.P @ e2])
    n2 = np.column_stack([s.n @ e1, s.n @ e2])
    ln = np.linalg.norm(n2, axis=1)
    good = ln > 1e-6
    if good.sum() < 3:
        return none
    m = np.column_stack([-n2[good, 1], n2[good, 0]]) / ln[good, None]
    rhs = (m * P2[good]).sum(1)
    c2, _, rank, _ = np.linalg.lstsq(m, rhs, rcond=None)
    if rank < 2:
        return none
    rho = np.linalg.norm(P2 - c2, axis=1)
    sa = s.P @ a
    base = c2[0] * e1 + c2[1] * e2  # the axis point at s = 0
    inward = (base + sa[:, None] * a) - s.P  # towards the axis
    concave = bool(float((inward * s.n).sum(1).mean()) > 0)
    r = 0.5 * float(rho.max() + rho.min())
    cyl = {"type": "cylinder", "axis": _vec(a), "center": _vec(base), "radius": r}
    cyl["_concave"] = concave
    dev_cyl = float(np.abs(rho - r).max())
    out_cyl = (cyl, dev_cyl)
    X = np.column_stack([np.ones_like(sa), sa])
    (alpha, beta), *_ = np.linalg.lstsq(X, rho, rcond=None)
    if abs(beta) < 1e-9 or abs(beta) > 1e6:
        return out_cyl, (None, np.inf)
    half = float(np.arctan(abs(beta)))
    dev_cone = float(np.abs(rho - alpha - beta * sa).max() * np.cos(half))
    apex = base + (-alpha / beta) * a
    cone = {
        "type": "cone",
        "axis": _vec(a if beta > 0 else -a),  # from the apex towards the wide end
        "apex": _vec(apex),
        "half_angle": float(np.degrees(half)),
        "_concave": concave,
    }
    return out_cyl, (cone, dev_cone)


def _fit_torus(s: _Samples, size: float):
    """A tube of constant radius around a circle: one principal curvature is constant and
    its centres of curvature lie on a circle."""
    best = (None, np.inf)
    for k in (s.k1, s.k2):
        km = float(np.median(k))
        if abs(km) * size < 1e-3:
            continue
        if np.abs(k - km).max() > 1e-3 * abs(km) + 1e-9:
            continue
        C = s.P + s.n / km
        c0 = C.mean(axis=0)
        _, _, Vt = np.linalg.svd(C - c0, full_matrices=False)
        ax = Vt[2]
        if ax[2] < 0:
            ax = -ax
        x, y = (C - c0) @ Vt[0], (C - c0) @ Vt[1]
        A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        R2 = sol[2] + sol[0] ** 2 + sol[1] ** 2
        if not np.isfinite(R2) or R2 <= 0:
            continue
        R = float(np.sqrt(R2))
        center = c0 + sol[0] * Vt[0] + sol[1] * Vt[1]
        q = s.P - center
        h = q @ ax
        radial = q - h[:, None] * ax
        rl = np.linalg.norm(radial, axis=1)
        closest = center + R * radial / np.maximum(rl, 1e-300)[:, None]
        dev = float(np.abs(np.linalg.norm(s.P - closest, axis=1) - 1.0 / abs(km)).max())
        if dev < best[1]:
            best = (
                {
                    "type": "torus",
                    "axis": _vec(ax),
                    "center": _vec(center),
                    "radius": R,
                    "minor_radius": 1.0 / abs(km),
                    "_concave": km > 0,
                },
                dev,
            )
    return best


def fit_surface(s: _Samples, tol: float, size: float) -> tuple[dict, float]:
    """The simplest analytic surface within tol of every sample (or freeform)."""
    if len(s.P) < 4:
        return {"type": "freeform"}, None
    flat = 1.0 / max(size, 1e-300) * FLAT_REL
    curved = (np.abs(s.k1) > flat) | (np.abs(s.k2) > flat)
    fit, dev = _fit_plane(s)
    if dev <= tol:
        return fit, dev
    if curved.any():
        sph, dev = _fit_sphere(s)
        if sph is not None and dev <= tol and sph["radius"] < 1e3 * size:
            return sph, dev
        (cyl, dev_cyl), (cone, dev_cone) = _fit_revolved(s)
        if cyl is not None and dev_cyl <= tol and cyl["radius"] < 1e3 * size:
            return cyl, dev_cyl
        if cone is not None and dev_cone <= tol and cone["half_angle"] < 89.0:
            return cone, dev_cone
        tor, dev = _fit_torus(s, size)
        if tor is not None and dev <= tol:
            return tor, dev
    return {"type": "freeform"}, None


# --------------------------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------------------------


def edge_kinds(body, A, signs, tangent_angle: float) -> dict:
    """{edge id: {"kind", "angle" (max between the normals, degrees), "faces"}}."""
    out = {}
    ts = np.linspace(0.0, 1.0, 11)[1:-1]
    tb = np.linspace(0.0, 1.0, 257)
    for e in body.edges.values():
        if not e.is_shared:
            out[e.id] = {"kind": "naked", "angle": None, "faces": [int(e.coedges[0].face)]}
            continue
        ca, cb = e.coedges
        if ca.face == cb.face:
            out[e.id] = {"kind": "seam", "angle": None, "faces": [int(ca.face)]}
            continue
        PA, nA, dA = FC._coedge_frame(body, ca, ts, A, signs[ca.face])
        PB, nB, dB = FC._coedge_frame(body, cb, tb, A, signs[cb.face])
        j = np.argmin(np.linalg.norm(PA[:, None] - PB[None], axis=2), axis=1)
        nB, dB = nB[j], dB[j]
        T = np.gradient(PA, axis=0)
        T /= np.maximum(np.linalg.norm(T, axis=1), 1e-300)[:, None]

        def across(d):
            d = d - (d * T).sum(1)[:, None] * T
            return d / np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]

        cos = np.clip((nA * nB).sum(1), -1.0, 1.0)
        score = 0.5 * ((across(dB) * nA).sum(1) + (across(dA) * nB).sum(1))
        good = np.isfinite(cos) & np.isfinite(score)
        if not good.any():
            out[e.id] = {"kind": "tangent", "angle": None, "faces": [int(ca.face), int(cb.face)]}
            continue
        angle = float(np.degrees(np.arccos(cos[good].min())))
        if angle <= tangent_angle:
            kind = "tangent"
        else:
            sharp = good & (np.degrees(np.arccos(cos)) > tangent_angle)
            kind = "concave" if float(score[sharp].mean()) > 0 else "convex"
        out[e.id] = {"kind": kind, "angle": round(angle, 4), "faces": [int(ca.face), int(cb.face)]}
    return out


# --------------------------------------------------------------------------------------------
# Faces
# --------------------------------------------------------------------------------------------


@dataclass
class FaceInfo:
    face: int
    name: str
    surface: dict
    deviation: float | None
    curvature: str
    min_radius: float | None
    area: float
    bbox_min: list
    bbox_max: list
    nz_min: float
    nz_max: float
    undercut_fraction: float
    neighbors: list = field(default_factory=list)
    role: str = "curved"
    hole: int | None = None

    @property
    def z_min(self) -> float:
        return self.bbox_min[2]

    @property
    def z_max(self) -> float:
        return self.bbox_max[2]

    def slope_range(self) -> list:
        """Degrees from level: 0 facing straight up, 90 vertical, 180 facing down."""
        return [
            round(float(np.degrees(np.arccos(np.clip(self.nz_max, -1, 1)))), 4),
            round(float(np.degrees(np.arccos(np.clip(self.nz_min, -1, 1)))), 4),
        ]

    def summary(self) -> dict:
        surface = {k: v for k, v in self.surface.items() if not k.startswith("_")}
        return {
            "face": self.face,
            "name": self.name,
            "role": self.role,
            "surface": surface,
            "deviation": _num(self.deviation),
            "curvature": self.curvature,
            "min_radius": _num(self.min_radius),
            "area": float(self.area),
            "z_min": float(self.z_min),
            "z_max": float(self.z_max),
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "slope_range": self.slope_range(),
            "undercut_fraction": round(float(self.undercut_fraction), 6),
            "hole": self.hole,
            "neighbors": self.neighbors,
        }


def _curvature(s: _Samples, size: float) -> tuple[str, float | None]:
    flat = FLAT_REL / max(size, 1e-300)
    cc = s.k1 > flat
    cv = s.k2 < -flat
    kmax = np.maximum(np.abs(s.k1), np.abs(s.k2))
    rmin = float(1.0 / kmax.max()) if len(kmax) and kmax.max() > flat else None
    if not cc.any() and not cv.any():
        return "flat", None
    if cc.any() and not cv.any():
        return "concave", rmin
    if cv.any() and not cc.any():
        return "convex", rmin
    if (cc & cv).mean() > 0.5 * (cc | cv).mean():
        return "saddle", rmin
    return "mixed", rmin


def analyze_face(face, A, t, sign, axis, tol: float):
    s, (lo, hi) = face_samples(face, A, t, sign)
    size = float(np.linalg.norm(hi - lo))
    surface, dev = fit_surface(s, tol, size)
    curvature, rmin = _curvature(s, size)
    if "_concave" in surface and curvature not in ("concave", "convex"):
        curvature = "concave" if surface["_concave"] else "convex"
    if surface["type"] == "plane":
        curvature, rmin = "flat", None
    elif surface["type"] in ("sphere", "cylinder"):
        rmin = surface["radius"]
    elif surface["type"] == "torus":
        rmin = min(rmin or np.inf, surface["minor_radius"])
    nz = s.n @ axis
    area = float(s.w.sum())
    under = float(s.w[nz < UNDERCUT_Z].sum() / area) if area > 0 else 0.0
    return FaceInfo(
        face=int(face.id),
        name=face.name,
        surface=surface,
        deviation=dev,
        curvature=curvature,
        min_radius=rmin,
        area=area,
        bbox_min=_vec(lo),
        bbox_max=_vec(hi),
        nz_min=float(nz.min()) if len(nz) else 0.0,
        nz_max=float(nz.max()) if len(nz) else 0.0,
        undercut_fraction=under,
    )


def blend_turn(body, fid: int, neighbors: list, A, sign) -> tuple[float, bool | None]:
    """How far a face turns between its tangent neighbours: (largest angle in degrees between
    its normals at the edges it shares with two different tangent faces, at nearest points;
    concave across?). A fillet turns from one face to the other (90 degrees between a floor
    and a wall); a can's side between its top and bottom blends doesn't turn at all."""
    by_face = {}
    for nb in neighbors:
        if nb["kind"] == "tangent":
            by_face.setdefault(nb["face"], []).append(nb["edge"])
    if len(by_face) < 2:
        return 0.0, None
    ts = np.linspace(0.0, 1.0, 17)
    frames = []
    for edge_ids in by_face.values():
        Ps, ns = [], []
        for eid in edge_ids:
            c = next(c for c in body.edges[eid].coedges if c.face == fid)
            P, n, _ = FC._coedge_frame(body, c, ts, A, sign)
            good = np.all(np.isfinite(n), axis=1)
            Ps.append(P[good])
            ns.append(n[good])
        frames.append((np.concatenate(Ps), np.concatenate(ns)))
    best, concave = 0.0, None
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            (P1, n1), (P2, n2) = frames[i], frames[j]
            if not len(P1) or not len(P2):
                continue
            k = np.argmin(np.linalg.norm(P1[:, None] - P2[None], axis=2), axis=1)
            angle = np.degrees(np.arccos(np.clip((n1 * n2[k]).sum(1), -1.0, 1.0)))
            if angle.max() > best:
                best = float(angle.max())
                wide = angle >= 0.5 * best
                concave = bool(float(((P2[k] - P1) * n1).sum(1)[wide].mean()) > 0)
    return best, concave


def _is_blend(f: FaceInfo, turn: float) -> bool:
    """A fillet or rounded edge: a curved face turning at least BLEND_TURN degrees between two
    faces it is tangent to."""
    return f.surface["type"] != "plane" and turn >= BLEND_TURN


def _level(f: FaceInfo, up: bool) -> bool:
    return f.nz_min >= LEVEL_COS if up else f.nz_max <= -LEVEL_COS


def _role(f: FaceInfo, turn, concave, hole_wall: bool, z_top, z_bottom, tol, wall_cos):
    if _is_blend(f, turn):
        return "fillet" if concave else "round"
    if hole_wall:
        return "hole_wall"
    if _level(f, True):
        return "top" if f.z_max >= z_top - tol else "floor"
    if _level(f, False):
        return "bottom" if f.z_min <= z_bottom + tol else "underside"
    if max(abs(f.nz_min), abs(f.nz_max)) <= wall_cos:
        return "wall"
    if f.nz_max < UNDERCUT_Z:
        return "underside"
    if f.surface["type"] == "plane":
        return "slope" if f.nz_min > 0 else "underside"
    return "curved"


def _prismatic(hole: dict, infos: dict, wall_cos: float) -> bool:
    """Only vertical walls and level floors facing up: a 2.5D pocket (POCKET, DRILL)."""
    for fid in hole["faces"]:
        f = infos.get(fid)
        if f is None:
            return False
        vertical = max(abs(f.nz_min), abs(f.nz_max)) <= 1e-6
        if not (vertical or _level(f, True)):
            return False
    return bool(hole["walls"])


# --------------------------------------------------------------------------------------------
# Body
# --------------------------------------------------------------------------------------------


@dataclass
class Classification:
    faces: list  # [FaceInfo] (only the requested ones)
    edges: dict  # edge id -> kind dict
    holes: list  # public hole dicts (through / blind / open)
    bosses: list
    closed: bool
    bbox_min: list
    bbox_max: list
    tolerance: float
    seconds: float = 0.0
    warnings: list = field(default_factory=list)

    def face(self, fid: int) -> FaceInfo:
        return next(f for f in self.faces if f.face == fid)

    def by_role(self, *roles) -> list:
        return [f.face for f in self.faces if f.role in roles]

    def summary(self) -> dict:
        kinds = Counter(e["kind"] for e in self.edges.values())
        return {
            "closed": self.closed,
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "size": [float(b - a) for a, b in zip(self.bbox_min, self.bbox_max)],
            "counts": {
                "faces": len(self.faces),
                "surface": dict(Counter(f.surface["type"] for f in self.faces)),
                "role": dict(Counter(f.role for f in self.faces)),
                "edges": {k: kinds.get(k, 0) for k in EDGE_KINDS},
                "holes": len(self.holes),
                "bosses": len(self.bosses),
            },
            "faces": [f.summary() for f in self.faces],
            "edges": [{"edge": int(eid), **e} for eid, e in sorted(self.edges.items())],
            "holes": self.holes,
            "bosses": self.bosses,
            "tolerance": self.tolerance,
            "seconds": round(self.seconds, 4),
            "warnings": list(self.warnings),
        }


def classify(
    body,
    matrix=None,
    tolerance: float = 1e-5,
    faces=None,
    wall_angle: float = 80.0,
    tangent_angle: float = 1.0,
    axis=(0.0, 0.0, 1.0),
) -> Classification:
    """Surface type, curvature, orientation and role of every face, the kind of every edge,
    and the body's holes (see the module doc). World space via `matrix`.

    tolerance: how close (world units) a face must be to a plane / cylinder / ... to count as
    one. faces: report only these face ids (the body's extent and the roles still use every
    face). wall_angle: faces steeper than this (degrees from level) everywhere are walls.
    tangent_angle: edges whose faces' normals stay within this (degrees) are tangent."""
    started = time.perf_counter()
    if not tolerance or tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if not 0.0 < float(wall_angle) <= 90.0:
        raise ValueError("wall_angle must be in (0, 90] degrees")
    if not 0.0 <= float(tangent_angle) < 90.0:
        raise ValueError("tangent_angle must be in [0, 90) degrees")
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
    base = 1.0 if det > 0 else -1.0
    axis = _unit(axis)
    wall_cos = float(np.cos(np.radians(wall_angle)))
    warnings = []
    closed = all(e.is_shared for e in body.edges.values())
    signs = {fid: base for fid in body.faces}
    if not closed:
        for fid, face in body.faces.items():
            signs[fid] = base * FC._face_sign_world(face.surface, A, t, axis, base)
        warnings.append("Open body (sheet): faces were turned to face up (+Z)")

    infos = {}
    for fid, face in body.faces.items():
        try:
            infos[fid] = analyze_face(face, A, t, signs[fid], axis, float(tolerance))
        except Exception as e:  # a broken face shouldn't stop the others
            warnings.append(f"Face {fid} ({face.name}) was skipped: {e}")
    if not infos:
        raise ValueError("no face could be analysed")
    lo = np.min([f.bbox_min for f in infos.values()], axis=0)
    hi = np.max([f.bbox_max for f in infos.values()], axis=0)

    edges = edge_kinds(body, A, signs, float(tangent_angle))
    for eid, e in edges.items():
        if len(e["faces"]) != 2:
            continue
        a, b = e["faces"]
        for me, other in ((a, b), (b, a)):
            if me in infos:
                infos[me].neighbors.append(
                    {"face": other, "edge": int(eid), "kind": e["kind"], "angle": e["angle"]}
                )

    found, bosses, hole_warnings = indexed_holes(body, float(tolerance), M)
    warnings += hole_warnings
    holes = [public_hole(h) for h in found]
    bosses = [public_hole(h) for h in bosses]
    wall_of, round_walls = {}, set()
    for h in holes:
        h["prismatic"] = _prismatic(h, infos, wall_cos)
        for w in h["faces"]:
            wall_of.setdefault(w, h["index"])
        if h["circular"] and h["kind"] in ("through", "blind"):
            round_walls.update(h["walls"])

    # top: the highest level faces facing up (the rest are floors); bottom likewise
    tol = float(tolerance)
    ups = [f.z_max for f in infos.values() if _level(f, True)]
    downs = [f.z_min for f in infos.values() if _level(f, False)]
    z_top = max(ups) if ups else np.inf
    z_bottom = min(downs) if downs else -np.inf
    for fid, f in infos.items():
        turn, concave = 0.0, None
        if f.surface["type"] != "plane":
            try:
                turn, concave = blend_turn(body, fid, f.neighbors, A, signs[fid])
            except Exception as e:
                warnings.append(f"Face {fid} ({f.name}): blend check failed: {e}")
        f.hole = wall_of.get(fid)
        f.role = _role(f, turn, concave, fid in round_walls, z_top, z_bottom, tol, wall_cos)

    shown = [infos[fid] for fid in sorted(infos) if faces is None or fid in faces]
    if faces is not None:
        edges = {eid: e for eid, e in edges.items() if set(e["faces"]) & faces}
        holes = [h for h in holes if (set(h["faces"]) | {h["face"]}) & faces]
        bosses = [h for h in bosses if (set(h["walls"]) | {h["face"]}) & faces]
    return Classification(
        faces=shown,
        edges=edges,
        holes=holes,
        bosses=bosses,
        closed=closed,
        bbox_min=_vec(lo),
        bbox_max=_vec(hi),
        tolerance=tol,
        seconds=time.perf_counter() - started,
        warnings=warnings,
    )
