"""Fabex 'brepcam/edges.py'

B-rep edges and face loops as polylines within an absolute tolerance. No bpy.

Everything is sampled from the exact geometry: an edge from its first coedge (a trim
pcurve, or a natural side of its face, mapped through the face's surface), a face loop
from its trims in order. Points are added until every chord is within `tolerance` of the
curve, so curves made from them can be machined directly (Profile, Pocket, Drill).

    extract(body, kind, ...) -> EdgeSet   polylines with provenance (+ holes for "holes")
    find_holes(body, ...)    -> [Hole]    inner face loops grouped by their walls

Kinds:
    all          every edge
    naked        edges used by one face only (open boundaries of sheets)
    shared       edges between two faces (or a seam of one face)
    creases      shared edges whose faces meet at an angle >= crease_angle (sharp edges;
                 tangent seams and fillet edges are left out)
    outer_loops  the outer boundary loop of each face
    holes        inner loops of faces (drilled holes, pockets, slots), one per opening:
                 the two ends of a through hole count once, entered from the higher end

Edge kinds are chained into the longest open or closed polylines (they break where
more than two edges meet). With `project` = "top" / "bottom" every polyline is flattened
to its highest / lowest Z (for holes: the hole's top / bottom) - or a number, to that Z -
polylines that collapse (vertical edges, walls) are dropped and coinciding ones are kept
once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tessellate import linear_scale

KINDS = ("all", "naked", "shared", "creases", "outer_loops", "holes")
EDGE_KINDS = ("all", "naked", "shared", "creases")
PROJECT = ("none", "top", "bottom")

MAX_POINTS = 8192  # per sampled trim / edge
MAX_ANGLE = np.radians(15.0)  # max turning between consecutive chords
CHUNK = 1024


# --------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------


class UVLine:
    """A straight pcurve p0 -> p1 in parameter space (a natural side of a face)."""

    n_ctrl = 2

    def __init__(self, p0, p1):
        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)

    def evaluate(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.p0[None] + t[:, None] * (self.p1 - self.p0)[None]


def side_line(surface, side: str) -> UVLine:
    """A natural side as a pcurve, in ascending along-side parameter."""
    (ua, ub), (va, vb) = surface.domain_u, surface.domain_v
    return {
        "u0": UVLine((ua, va), (ua, vb)),
        "u1": UVLine((ub, va), (ub, vb)),
        "v0": UVLine((ua, va), (ub, va)),
        "v1": UVLine((ua, vb), (ub, vb)),
    }[side]


def natural_loop(surface) -> list[UVLine]:
    """The domain rectangle, counter-clockwise (same order as BRepKit's natural_loop)."""
    (ua, ub), (va, vb) = surface.domain_u, surface.domain_v
    return [
        UVLine((ua, va), (ub, va)),
        UVLine((ub, va), (ub, vb)),
        UVLine((ub, vb), (ua, vb)),
        UVLine((ua, vb), (ua, va)),
    ]


def loop_pcurves(face, loop: int = 0) -> list:
    """The pcurves of a face loop, in loop order (untrimmed faces: the natural loop)."""
    if face.loops is None:
        if loop != 0:
            raise IndexError(f"face {face.id} has no loop {loop}")
        return natural_loop(face.surface)
    return [t.pcurve for t in face.loops[loop]]


def coedge_curve(body, c):
    """(surface, pcurve) of a coedge, in the coedge's own parameter direction."""
    face = body.faces[c.face]
    if c.is_trim:
        return face.surface, face.loops[c.loop][c.index].pcurve
    return face.surface, side_line(face.surface, c.side)


def sample_curve(
    surface, pcurve, tol: float, max_angle: float = MAX_ANGLE, max_points: int = MAX_POINTS
):
    """(t, points, capped): samples t in [0, 1] of surface(pcurve(t)) whose chords stay
    within `tol` of the curve (checked at chord midpoints) and turn by <= max_angle."""
    ns = max(getattr(surface, "shape", (2, 2)))
    t = np.linspace(0.0, 1.0, max(17, 4 * getattr(pcurve, "n_ctrl", 2) + 1, 2 * ns + 1))
    cos_max = np.cos(max_angle)
    capped = False
    while True:
        uv = pcurve.evaluate(t)
        P = surface.evaluate(uv[:, 0], uv[:, 1])
        tm = 0.5 * (t[:-1] + t[1:])
        uvm = pcurve.evaluate(tm)
        M = surface.evaluate(uvm[:, 0], uvm[:, 1])
        bad = np.linalg.norm(M - 0.5 * (P[:-1] + P[1:]), axis=1) > tol
        seg = np.diff(P, axis=0)
        ln = np.linalg.norm(seg, axis=1)
        real = ln > tol * 1e-3
        seg = seg / np.maximum(ln, 1e-300)[:, None]
        turn = (np.sum(seg[:-1] * seg[1:], axis=1) < cos_max) & real[:-1] & real[1:]
        bad[:-1] |= turn
        bad[1:] |= turn
        if not bad.any():
            break
        if len(t) + int(bad.sum()) > max_points:
            capped = True
            break
        t = np.sort(np.concatenate([t, tm[bad]]))
    return t, P, capped


def unit_normals(surface, uv) -> np.ndarray:
    """Unit normals at uv points; NaN rows where the surface is degenerate (poles)."""
    D = surface.derivatives(uv[:, 0], uv[:, 1], 1)
    n = np.cross(D[1, 0], D[0, 1])
    ln = np.linalg.norm(n, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / ln[:, None]
    n[ln <= 1e-300] = np.nan
    return n


def edge_angle(body, edge, samples: int = 9) -> float:
    """Max angle (degrees, 0..90) between the two faces' normals along a shared edge.

    Orientation-independent (|cos|), so faces with flipped normals still read as smooth.
    0 for naked edges."""
    if not edge.is_shared:
        return 0.0
    ca, cb = edge.coedges
    sa, pa = coedge_curve(body, ca)
    sb, pb = coedge_curve(body, cb)
    t = np.linspace(0.0, 1.0, samples + 2)[1:-1]
    uva = pa.evaluate(t)
    Pa = sa.evaluate(uva[:, 0], uva[:, 1])
    tb = np.linspace(0.0, 1.0, 257)
    uvb = pb.evaluate(tb)
    Pb = sb.evaluate(uvb[:, 0], uvb[:, 1])
    j = np.argmin(np.linalg.norm(Pa[:, None] - Pb[None], axis=2), axis=1)
    na, nb = unit_normals(sa, uva), unit_normals(sb, uvb[j])
    cos = np.abs(np.sum(na * nb, axis=1))
    cos = cos[np.isfinite(cos)]
    if len(cos) == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(cos.min(), 0.0, 1.0))))


# --------------------------------------------------------------------------------------------
# Polylines
# --------------------------------------------------------------------------------------------


@dataclass
class Polyline:
    points: np.ndarray  # (n, 3), world space; closed polylines don't repeat the first point
    closed: bool
    edges: list = field(default_factory=list)  # B-rep edge ids it was made from
    faces: list = field(default_factory=list)  # B-rep face ids it bounds
    loop: tuple | None = None  # (face id, loop index) for face loops and holes
    hole: dict | None = None

    @property
    def length(self) -> float:
        P = np.vstack([self.points, self.points[:1]]) if self.closed else self.points
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    def summary(self) -> dict:
        out = {
            "closed": self.closed,
            "points": int(len(self.points)),
            "length": self.length,
            "z_min": float(self.points[:, 2].min()),
            "z_max": float(self.points[:, 2].max()),
            "edges": sorted(int(e) for e in self.edges),
            "faces": sorted(int(f) for f in self.faces),
        }
        if self.loop is not None:
            out["loop"] = [int(self.loop[0]), int(self.loop[1])]
        return out


@dataclass
class EdgeSet:
    kind: str
    tolerance: float
    project: object  # "none" / "top" / "bottom" or a Z value
    polylines: list
    holes: list = field(default_factory=list)  # hole dicts, in polyline order for "holes"
    warnings: list = field(default_factory=list)

    def stats(self) -> dict:
        return {
            "kind": self.kind,
            "tolerance": self.tolerance,
            "project": self.project,
            "splines": len(self.polylines),
            "closed": sum(p.closed for p in self.polylines),
            "points": int(sum(len(p.points) for p in self.polylines)),
            "warnings": list(self.warnings),
        }


def _to_world(P: np.ndarray, matrix) -> np.ndarray:
    if matrix is None:
        return P
    M = np.asarray(matrix, dtype=float)
    return P @ M[:3, :3].T + M[:3, 3]


def _normal_to_world(n: np.ndarray, matrix) -> np.ndarray:
    if matrix is not None:
        n = np.linalg.inv(np.asarray(matrix, dtype=float)[:3, :3]).T @ n
    ln = np.linalg.norm(n)
    return n / ln if ln > 0 else n


def _dedupe_consecutive(P: np.ndarray, eps: float, closed: bool) -> np.ndarray:
    if len(P) < 2:
        return P
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) > eps
    P = P[keep]
    if closed and len(P) > 1 and np.linalg.norm(P[-1] - P[0]) <= eps:
        P = P[:-1]
    return P


def _area_xy(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _degenerate(pl: Polyline, tol: float, projected: bool) -> bool:
    if len(pl.points) < 2:
        return True
    if pl.closed and projected:
        # a wall seen from above: no area left (width below ~2 tol)
        return abs(_area_xy(pl.points)) < tol * max(pl.length, 1e-300) or len(pl.points) < 3
    return pl.length < tol


def _point_segment_distance(Q: np.ndarray, P: np.ndarray, closed: bool) -> np.ndarray:
    """Distance from each point of Q to polyline P (2D or 3D), chunked over Q."""
    B = np.vstack([P, P[:1]]) if closed else P
    a, ab = B[:-1], np.diff(B, axis=0)
    L2 = np.maximum(np.sum(ab * ab, axis=1), 1e-300)
    out = np.empty(len(Q))
    for s in range(0, len(Q), CHUNK):
        q = Q[s : s + CHUNK]
        aq = q[:, None] - a[None]
        t = np.clip(np.sum(aq * ab[None], axis=2) / L2[None], 0.0, 1.0)
        out[s : s + CHUNK] = np.linalg.norm(aq - t[..., None] * ab[None], axis=2).min(axis=1)
    return out


def _same_xy(A: Polyline, B: Polyline, tol: float) -> bool:
    if A.closed != B.closed:
        return False
    a, b = A.points[:, :2], B.points[:, :2]
    if np.abs(a.min(0) - b.min(0)).max() > 2 * tol or np.abs(a.max(0) - b.max(0)).max() > 2 * tol:
        return False
    return bool(
        _point_segment_distance(a, b, B.closed).max() <= 2 * tol
        and _point_segment_distance(b, a, A.closed).max() <= 2 * tol
    )


def _merge_into(keep: Polyline, other: Polyline) -> None:
    keep.edges = sorted(set(keep.edges) | set(other.edges))
    keep.faces = sorted(set(keep.faces) | set(other.faces))


def _is_level(project) -> bool:
    return isinstance(project, (int, float)) and not isinstance(project, bool)


def flatten(pl: Polyline, project, z: float | None = None) -> Polyline:
    """A copy at constant Z: `z`, a numeric `project`, else the polyline's highest
    ("top") / lowest ("bottom") Z."""
    if _is_level(project):
        z = float(project)
    if z is None:
        z = pl.points[:, 2].max() if project == "top" else pl.points[:, 2].min()
    P = pl.points.copy()
    P[:, 2] = z
    return Polyline(P, pl.closed, list(pl.edges), list(pl.faces), pl.loop, pl.hole)


def project_polylines(polylines: list, project, tol: float) -> list:
    """Flatten to the top / bottom Z, drop collapsed polylines, keep coinciding ones once."""
    if project == "none":
        return [p for p in polylines if not _degenerate(p, tol, False)]
    flat = []
    for pl in polylines:
        z = None
        if pl.hole is not None and not _is_level(project):
            z = pl.hole["z_top"] if project == "top" else pl.hole["z_bottom"]
        q = flatten(pl, project, z)
        q.points = _dedupe_consecutive(q.points, tol * 1e-3, q.closed)
        if not _degenerate(q, tol, True):
            flat.append(q)
    out: list[Polyline] = []
    for pl in flat:
        for i, kept in enumerate(out):
            if _same_xy(pl, kept, tol):
                higher = pl.points[0, 2] > kept.points[0, 2]
                if not _is_level(project) and higher == (project == "top"):
                    _merge_into(pl, kept)
                    out[i] = pl
                else:
                    _merge_into(kept, pl)
                break
        else:
            out.append(pl)
    return out


def chain(polylines: list, merge_tol: float, xy: bool = False) -> list:
    """Join open polylines end to end where exactly two ends meet (within merge_tol;
    xy=True compares ends in XY only, for flattened polylines at different Z)."""
    closed = [p for p in polylines if p.closed]
    opened = [p for p in polylines if not p.closed]
    if not opened:
        return closed
    ends = np.array([[p.points[0], p.points[-1]] for p in opened])  # (n, 2, 3)
    # cluster end points on a grid of merge_tol cells (neighbouring cells checked)
    flat = ends.reshape(-1, 3).copy()
    if xy:
        flat[:, 2] = 0.0
    cells = np.floor(flat / merge_tol).astype(np.int64)
    grid: dict = {}
    node = np.full(len(flat), -1)
    n_nodes = 0
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for idx, (p, c) in enumerate(zip(flat, cells)):
        found = -1
        for o in offsets:
            for other in grid.get((c[0] + o[0], c[1] + o[1], c[2] + o[2]), ()):
                if np.linalg.norm(flat[other] - p) <= merge_tol:
                    found = node[other]
                    break
            if found >= 0:
                break
        if found < 0:
            found = n_nodes
            n_nodes += 1
        node[idx] = found
        grid.setdefault(tuple(c), []).append(idx)
    node = node.reshape(-1, 2)
    at: dict = {}
    for i in range(len(opened)):
        for end in (0, 1):
            at.setdefault(int(node[i, end]), []).append((i, end))

    used = np.zeros(len(opened), dtype=bool)

    def follow(origin: int, end: int):
        """Walk from polyline origin's `end`; returns [(j, forward)] and whether it came back
        to the origin (a closed chain)."""
        seq, i = [], origin
        n = int(node[i, end])
        while len(at[n]) == 2:
            (j, e), (k, f) = at[n]
            nxt = (k, f) if (j, e) == (i, end) else (j, e)
            j, e = nxt
            if j == origin:
                return seq, True
            if used[j]:
                break
            used[j] = True
            forward = e == 0
            seq.append((j, forward))
            i, end = j, 1 - e
            n = int(node[i, end])
        return seq, False

    out = list(closed)
    for start in range(len(opened)):
        if used[start]:
            continue
        used[start] = True
        fwd, loop = follow(start, 1)
        back = []
        if not loop:
            back, _ = follow(start, 0)
        seq = [(j, not f) for j, f in reversed(back)] + [(start, True)] + fwd
        pts, edges, faces = [], [], []
        for j, forward in seq:
            P = opened[j].points if forward else opened[j].points[::-1]
            pts.append(P if not pts else P[1:])
            edges += opened[j].edges
            faces += opened[j].faces
        P = np.concatenate(pts)
        is_closed = loop or (
            len(seq) == 1 and node[start, 0] == node[start, 1] and len(at[int(node[start, 0])]) == 2
        )
        gap = P[-1] - P[0]
        if xy:
            gap = gap[:2]
        if is_closed and len(P) > 2 and np.linalg.norm(gap) <= merge_tol:
            P = P[:-1]
        out.append(Polyline(P, bool(is_closed), sorted(set(edges)), sorted(set(faces))))
    return out


# --------------------------------------------------------------------------------------------
# Face loops and holes
# --------------------------------------------------------------------------------------------


def sample_loop(face, loop: int, tol: float, matrix=None):
    """(points world (n, 3), uv (n, 2), capped) of a closed face loop."""
    pts, uvs, capped = [], [], False
    for pc in loop_pcurves(face, loop):
        t, P, c = sample_curve(face.surface, pc, tol)
        pts.append(P[:-1])
        uvs.append(pc.evaluate(t)[:-1])
        capped |= c
    P, uv = np.concatenate(pts), np.concatenate(uvs)
    # degenerate sides (poles) repeat one point
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) > tol * 1e-3
    return _to_world(P[keep], matrix), uv[keep], capped


def fit_circle(P: np.ndarray) -> dict:
    """Best plane + circle through 3D points: center, radius, axis, max radial residual."""
    c0 = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c0, full_matrices=False)
    e1, e2, axis = Vt[0], Vt[1], Vt[2]
    if axis[2] < 0:
        axis = -axis
    x, y = (P - c0) @ e1, (P - c0) @ e2
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
    center = c0 + cx * e1 + cy * e2
    planar = float(np.abs((P - c0) @ axis).max())
    radial = float(np.abs(np.linalg.norm(P - center, axis=1) - r).max())
    return {"center": center, "radius": r, "axis": axis, "residual": max(planar, radial)}


def _edge_index(body):
    """{(face, loop, index) or (face, side): edge} for every coedge."""
    idx = {}
    for e in body.edges.values():
        for c in e.coedges:
            idx[(c.face, c.loop, c.index) if c.is_trim else (c.face, c.side)] = e
    return idx


def _face_edges(body, fid) -> list:
    return [e for e in body.edges.values() if any(c.face == fid for c in e.coedges)]


def _adjacency(body) -> dict:
    """{face: [(neighbour face, this face's coedge, the neighbour's coedge)]}."""
    adj = {}
    for e in body.edges.values():
        if not e.is_shared or e.coedges[0].face == e.coedges[1].face:
            continue
        a, b = e.coedges
        adj.setdefault(a.face, []).append((b.face, a, b))
        adj.setdefault(b.face, []).append((a.face, b, a))
    return adj


def _inner(c) -> bool:
    return c.is_trim and c.loop >= 1


def _region(adj, walls, entry) -> tuple[set, set]:
    """(every face inside an opening, the far ends) - from its walls on, through faces' outer
    boundaries only: it stops at the entry face, at another face's inner loop (the far end
    of a through hole) and at its own faces' inner loops (a hole inside the opening, e.g.
    drilled into a pocket floor, is an opening of its own)."""
    seen, far = set(walls), set()
    todo = list(walls)
    while todo:
        f = todo.pop()
        for g, cf, cg in adj.get(f, ()):
            if g == entry or g in seen or _inner(cf):
                continue
            if _inner(cg):
                far.add((g, cg.loop))
                continue
            seen.add(g)
            todo.append(g)
    return seen, far


def find_holes(body, tolerance: float, matrix=None, faces=None) -> tuple[list, list]:
    """Inner face loops grouped into openings. Returns (holes, warnings); each hole is a dict
    with the entry loop's world points under "_points" (dropped from JSON output).

    A hole's walls are the faces across its loop's edges; its faces are the walls and
    everything beyond them (floor fillets, the floor) up to another face's inner loop,
    which makes it a through hole (otherwise it is blind). Depth and z_bottom reach the
    deepest of its faces, walls_bottom where the walls end. Several loops sharing the same
    walls are one hole, entered from the highest one. Loops whose walls stand out of the
    face (a boss, e.g. a dome on a plate) get kind "boss"."""
    scale = linear_scale(matrix)
    tol = tolerance / scale
    index = _edge_index(body)
    adj = _adjacency(body)
    warnings = []
    found = []
    for fid, face in body.faces.items():
        if face.loops is None or (faces is not None and fid not in faces):
            continue
        for li in range(1, len(face.loops)):
            P, uv, capped = sample_loop(face, li, tol, matrix)
            if capped:
                warnings.append(f"hole loop {li} of face {fid} hit the {MAX_POINTS}-point cap")
            walls = set()
            for k in range(len(face.loops[li])):
                e = index.get((fid, li, k))
                if e is not None:
                    walls.update(c.face for c in e.coedges if c.face != fid)
            ends, through = set(), False
            region, far = _region(adj, walls, fid) if walls else (set(), set())
            through = bool(far)
            W, walls_bottom = [P], None
            for w in sorted(region):
                wf = body.faces[w]
                for lj in range(len(wf.loops) if wf.loops else 1):
                    Q = sample_loop(wf, lj, tol * 10, matrix)[0]
                    W.append(Q)
                    if w in walls:
                        low = float(Q[:, 2].min())
                        walls_bottom = low if walls_bottom is None else min(walls_bottom, low)
            for w in walls:
                for e in _face_edges(body, w):
                    for c in e.coedges:
                        if c.face in walls or c.face == fid:
                            continue
                        ends.add(c.face)
                        through |= c.is_trim and c.loop >= 1
            W = np.concatenate(W)
            # the entry face's normal at the opening (the surface continues across it)
            m = uv.mean(axis=0)
            n = unit_normals(face.surface, m[None])[0]
            axis = _normal_to_world(n, matrix) if np.all(np.isfinite(n)) else np.array([0, 0, 1.0])
            centroid = P.mean(axis=0)
            # BRepKit solids have outward normals: a hole's walls go into the material
            # (against the normal), a boss's walls stand out of the face
            s = (W - centroid) @ axis
            d_in, d_out = float(max(-s.min(), 0.0)), float(max(s.max(), 0.0))
            boss = bool(walls) and d_out > d_in
            depth = (d_out if boss else d_in) if walls else 0.0
            circle = fit_circle(P)
            circular = circle["residual"] <= max(2 * tolerance, 1e-9 * np.ptp(P, axis=0).max())
            center = circle["center"] if circular else centroid
            found.append(
                {
                    "face": int(fid),
                    "face_name": getattr(face, "name", ""),
                    "loop": li,
                    "kind": (
                        "boss"
                        if boss
                        else ("through" if through else "blind") if walls and ends else "open"
                    ),
                    "circular": bool(circular),
                    "center": [float(x) for x in center],
                    "radius": float(circle["radius"]) if circular else None,
                    "diameter": 2 * float(circle["radius"]) if circular else None,
                    "circle_residual": circle["residual"],
                    "axis": [float(x) for x in axis],
                    "vertical": bool(abs(axis[2]) >= 0.999),
                    "depth": depth,
                    "z_top": float(max(P[:, 2].max(), W[:, 2].max())),
                    "z_bottom": float(W[:, 2].min()),
                    "walls_bottom": walls_bottom,
                    "size_xy": [float(x) for x in np.ptp(P[:, :2], axis=0)],
                    "walls": sorted(int(w) for w in walls),
                    "faces": sorted(int(f) for f in region),
                    "other_ends": [],
                    "_points": P,
                }
            )
    # one hole per set of walls, entered from the highest end
    holes, by_walls = [], {}
    for h in found:
        key = frozenset(h["walls"])
        if not key:
            holes.append(h)
            continue
        kept = by_walls.get(key)
        if kept is None:
            by_walls[key] = h
            holes.append(h)
        elif h["center"][2] > kept["center"][2]:
            h["other_ends"] = kept["other_ends"] + [[kept["face"], kept["loop"]]]
            holes[holes.index(kept)] = h
            by_walls[key] = h
        else:
            kept["other_ends"].append([h["face"], h["loop"]])
    return holes, warnings


def public_hole(h: dict) -> dict:
    return {k: v for k, v in h.items() if not k.startswith("_")}


def _number(h: dict, numbered: list):
    """The whole-body number of a hole found among some faces' loops only."""
    walls = frozenset(h["walls"])
    for g in numbered:
        if walls and frozenset(g["walls"]) == walls:
            return g["index"]
        if (g["face"], g["loop"]) == (h["face"], h["loop"]) or [h["face"], h["loop"]] in g[
            "other_ends"
        ]:
            return g["index"]
    return None


def indexed_holes(body, tolerance: float, matrix=None, faces=None, select=None):
    """(holes, bosses, warnings), each numbered ("index") over the whole body - the numbers
    brep_info / brep_holes report and extract_edges(holes=[...]) selects by.

    faces: only the inner loops of these faces (as find_holes; a through hole selected by its
    lower face is entered from there). select: only these hole numbers (ValueError for
    unknown ones)."""
    found, warnings = find_holes(body, tolerance, matrix)
    holes = [h for h in found if h["kind"] != "boss"]
    bosses = [h for h in found if h["kind"] == "boss"]
    for group in (holes, bosses):
        for i, h in enumerate(group):
            h["index"] = i
    count = len(holes)
    if faces is not None:
        local, warnings = find_holes(body, tolerance, matrix, faces)
        numbered_holes, numbered_bosses = holes, bosses
        holes = [h for h in local if h["kind"] != "boss"]
        bosses = [h for h in local if h["kind"] == "boss"]
        for h in holes:
            h["index"] = _number(h, numbered_holes)
        for h in bosses:
            h["index"] = _number(h, numbered_bosses)
    if select is not None:
        select = {int(i) for i in select}
        unknown = sorted(select - set(range(count)))
        if unknown:
            raise ValueError(f"no holes {unknown} (the body has {count}, numbered from 0)")
        holes = [h for h in holes if h["index"] in select]
    return holes, bosses, warnings


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def extract(
    body,
    kind: str = "all",
    tolerance: float = 1e-5,
    matrix=None,
    faces=None,
    project="none",
    crease_angle: float = 30.0,
    holes=None,
) -> EdgeSet:
    """Polylines of a body's edges / loops / holes, in world space (see module docstring).

    tolerance: max distance between the polylines and the exact curves (world units).
    faces: optional face ids - only edges touching them, or only their loops / holes.
    holes: only these hole numbers (kind "holes"; see indexed_holes)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    select, holes = holes, []
    if select is not None and kind != "holes":
        raise ValueError("holes=[...] selects holes: use it with kind 'holes'")
    if not (_is_level(project) and np.isfinite(project)) and project not in PROJECT:
        raise ValueError(f"project must be one of {PROJECT} or a Z value, not {project!r}")
    if not tolerance or tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if faces is not None:
        faces = {int(f) for f in faces}
        unknown = sorted(faces - set(body.faces))
        if unknown:
            raise KeyError(f"no faces {unknown} (faces: {sorted(body.faces)})")
    tol = tolerance / linear_scale(matrix)
    warnings: list[str] = []
    polylines: list[Polyline] = []

    if kind in EDGE_KINDS:
        for eid, e in body.edges.items():
            if faces is not None and not any(c.face in faces for c in e.coedges):
                continue
            if kind == "naked" and e.is_shared:
                continue
            if kind in ("shared", "creases") and not e.is_shared:
                continue
            if kind == "creases" and edge_angle(body, e) < crease_angle:
                continue
            _, P, capped = sample_curve(*coedge_curve(body, e.coedges[0]), tol)
            if capped:
                warnings.append(f"edge {eid} hit the {MAX_POINTS}-point cap")
            polylines.append(
                Polyline(_to_world(P, matrix), False, [eid], sorted({c.face for c in e.coedges}))
            )
    elif kind == "outer_loops":
        for fid, face in body.faces.items():
            if faces is not None and fid not in faces:
                continue
            P, _, capped = sample_loop(face, 0, tol, matrix)
            if capped:
                warnings.append(f"outer loop of face {fid} hit the {MAX_POINTS}-point cap")
            polylines.append(Polyline(P, True, [], [fid], (fid, 0)))
    else:
        holes, _, warnings = indexed_holes(body, tolerance, matrix, faces, select)
        for h in holes:
            polylines.append(
                Polyline(
                    h["_points"], True, [], [h["face"]] + h["walls"], (h["face"], h["loop"]), h
                )
            )

    if kind in EDGE_KINDS:
        gaps = [e.gap for e in body.edges.values()]
        merge_tol = max(2 * tolerance, 4 * max(gaps, default=0.0) * linear_scale(matrix))
        polylines = chain(polylines, merge_tol)
    polylines = project_polylines(polylines, project, tolerance)
    if kind in EDGE_KINDS and project != "none":
        # pieces left after flattening (e.g. a box's top edges, its sides gone) join up
        polylines = [flatten(p, project) for p in chain(polylines, merge_tol, xy=True)]
    if kind == "holes":
        holes = [p.hole for p in polylines]
    return EdgeSet(kind, tolerance, project, polylines, [public_hole(h) for h in holes], warnings)
