"""Triangles of a B-rep body within an absolute tolerance, for drop-cutter sampling.

No bpy. The kernel (brepcam.kernel.Kernel) is passed in.

BRepKit's tessellator takes a *relative* chord tolerance (a fraction of each face's
control-net diagonal) and checks it along grid lines only. A CAM tolerance is an
absolute distance between the mesh and the exact surface, so cam_mesh():

1. converts the absolute tolerance to a chord tolerance, with a safety factor for the
   errors the chord test can't see (quad centres, triangle diagonals, trim regions);
2. tessellates, then splits polygons into triangles (quads along the shorter diagonal);
3. measures the real deviation: points on the triangles (centroid and edge midpoints)
   are projected onto the exact surfaces by Newton iteration;
4. if the measured deviation exceeds the tolerance, tightens and repeats.

Cost guards (BRepKit's trimmed-face meshing gets slow, and memory hungry, at very fine
relative tolerances - a 2 m part at 0.01 mm took minutes and ~100 GB):

- the relative chord tolerance never goes below MIN_CHORD_TOL; when the requested
  tolerance needs finer, the mesh is made at the limit and a warning says so (the
  measured deviation is still reported, so within_tolerance tells the truth);
- BRepKit's point-in-trim-region and distance-to-trim tests are run in chunks of points
  while meshing, so their (points x boundary samples) temporaries stay small;
- refinement passes stop at the limit, or when a pass doesn't improve the deviation.

All lengths are in the units of the result's world space (Blender: metres).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

SAFETY = 0.5  # chord tolerance as a fraction of the requested absolute tolerance
MIN_CHORD_TOL = 2e-5  # relative to the face size; 10x finer than BRepKit's display default
MAX_SEGMENTS = 512  # per face direction
SAMPLE_TRIANGLES = 20000  # triangles measured per pass (all of them below this)
CHUNK_POINTS = 1024  # points per chunk for BRepKit's trim-region tests


@dataclass
class CamMesh:
    triangles: np.ndarray  # (n, 3, 3) world-space triangle corners
    face_ids: np.ndarray  # (n,) B-rep face id per triangle
    tolerance: float  # requested max deviation (world units)
    max_deviation: float  # measured max deviation (world units); nan if not measured
    measured_points: int  # how many points were projected for max_deviation
    chord_tol: float  # relative chord tolerance passed to the kernel (last pass)
    passes: int  # tessellation passes (1 + refinements)
    seconds: float
    unmerged_edges: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def within_tolerance(self) -> bool:
        return bool(np.isfinite(self.max_deviation) and self.max_deviation <= self.tolerance)

    def stats(self) -> dict:
        lo = self.triangles.reshape(-1, 3).min(axis=0) if len(self.triangles) else None
        hi = self.triangles.reshape(-1, 3).max(axis=0) if len(self.triangles) else None
        return {
            "triangles": int(len(self.triangles)),
            "faces": int(len(np.unique(self.face_ids))),
            "tolerance": float(self.tolerance),
            "max_deviation": (
                float(self.max_deviation) if np.isfinite(self.max_deviation) else None
            ),
            "within_tolerance": self.within_tolerance,
            "measured_points": int(self.measured_points),
            "chord_tol": float(self.chord_tol),
            "passes": int(self.passes),
            "seconds": round(float(self.seconds), 3),
            "bbox_min": None if lo is None else lo.tolist(),
            "bbox_max": None if hi is None else hi.tolist(),
            "unmerged_edges": list(self.unmerged_edges),
            "warnings": list(self.warnings),
        }


# --- helpers ---------------------------------------------------------------------------------


def linear_scale(matrix) -> float:
    """Largest stretch factor of a 4x4 (or 3x3) transform."""
    if matrix is None:
        return 1.0
    m = np.asarray(matrix, dtype=float)[:3, :3]
    return float(np.linalg.svd(m, compute_uv=False).max()) or 1.0


def _face_diag(surface) -> float:
    ctrl = surface.ctrl.reshape(-1, 3)
    return float(np.linalg.norm(ctrl.max(0) - ctrl.min(0))) or 1.0


def chord_tolerance(body, local_tol: float) -> float:
    """Relative chord tolerance so that every face meets `local_tol` (body units)."""
    diag = max((_face_diag(f.surface) for f in body.faces.values()), default=1.0)
    return SAFETY * local_tol / diag


def _chunked(fn, chunk: int):
    """fn(points, polylines) evaluated over chunks of points (fn is per-point)."""

    def wrapper(uv, polylines):
        P = np.atleast_2d(np.asarray(uv, dtype=float))
        if len(P) <= chunk:
            return fn(P, polylines)
        return np.concatenate([fn(P[i : i + chunk], polylines) for i in range(0, len(P), chunk)])

    wrapper._fabex_chunked = True
    return wrapper


@contextmanager
def bounded_trim_tests(kernel, chunk: int = CHUNK_POINTS):
    """Run BRepKit's per-point trim tests in chunks while meshing (restored afterwards).

    BRepKit's points_in_region / distance_to_polylines broadcast every point against
    every boundary sample at once; for a big trimmed face that is tens of GB.
    """
    patched = []
    try:
        module = kernel.module("tessellate.trimmed")
    except Exception:
        module = None
    if module is not None:
        for name in ("points_in_region", "distance_to_polylines"):
            fn = getattr(module, name, None)
            if callable(fn) and not getattr(fn, "_fabex_chunked", False):
                setattr(module, name, _chunked(fn, chunk))
                patched.append((name, fn))
    try:
        yield
    finally:
        for name, fn in patched:
            setattr(module, name, fn)


def triangulate(tess) -> tuple[np.ndarray, np.ndarray]:
    """Corner indices (n, 3) into tess.corner_verts, and the face id per triangle.

    Quads are split along their shorter diagonal; larger polygons are fanned.
    """
    sizes = np.asarray(tess.poly_sizes, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    fids = np.asarray(tess.face_ids)
    P = np.asarray(tess.points)[np.asarray(tess.corner_verts)]
    tris, tri_fids = [], []

    s3 = starts[sizes == 3]
    tris.append(s3[:, None] + np.arange(3)[None, :])
    tri_fids.append(fids[sizes == 3])

    s4 = starts[sizes == 4]
    if len(s4):
        a, b, c, d = (s4 + k for k in range(4))
        ac = np.linalg.norm(P[a] - P[c], axis=1)
        bd = np.linalg.norm(P[b] - P[d], axis=1)
        use_ac = ac <= bd
        first = np.where(use_ac[:, None], np.stack([a, b, c], 1), np.stack([a, b, d], 1))
        second = np.where(use_ac[:, None], np.stack([a, c, d], 1), np.stack([b, c, d], 1))
        tris += [first, second]
        tri_fids += [fids[sizes == 4]] * 2

    for k in np.unique(sizes[sizes > 4]):
        sk = starts[sizes == k]
        for j in range(1, k - 1):
            tris.append(np.stack([sk, sk + j, sk + j + 1], 1))
            tri_fids.append(fids[sizes == k])

    if not tris:
        return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(tris).astype(np.int64), np.concatenate(tri_fids).astype(np.int64)


def _raw_uv(surface, uv, points) -> np.ndarray:
    """Corner uvs as surface parameters.

    BRepKit reports grid-face uvs normalized to [0, 1] and trimmed-face uvs as raw
    parameters; rather than rely on that, pick whichever reading reproduces the points.
    """
    (ua, ub), (va, vb) = surface.domain_u, surface.domain_v
    mapped = np.column_stack([ua + uv[:, 0] * (ub - ua), va + uv[:, 1] * (vb - va)])
    k = min(len(uv), 32)
    idx = np.linspace(0, len(uv) - 1, k).astype(int)
    err_raw = np.linalg.norm(surface.evaluate(uv[idx, 0], uv[idx, 1]) - points[idx], axis=1)
    err_map = np.linalg.norm(surface.evaluate(mapped[idx, 0], mapped[idx, 1]) - points[idx], axis=1)
    return uv if err_raw.max() <= err_map.max() else mapped


def foot_points(surface, Q, u, v, iters: int = 8) -> np.ndarray:
    """Distance from points Q to the surface, by Gauss-Newton from the seeds (u, v).

    The points are close to the surface (within a tessellation tolerance), so the
    second-derivative terms of full Newton are negligible and first derivatives suffice.
    Any surface point bounds the distance from above, so where the iteration can't
    improve (degenerate Jacobian at a pole, clamped at the boundary) the result stays a
    safe over-estimate.
    """
    (ua, ub), (va, vb) = surface.domain_u, surface.domain_v
    u = np.clip(u, ua, ub)
    v = np.clip(v, va, vb)
    best = np.full(len(Q), np.inf)
    step_tol = 1e-12 * max(ub - ua, vb - va)
    for _ in range(iters):
        D = surface.derivatives(u, v, 1)
        S, Su, Sv = D[0, 0], D[1, 0], D[0, 1]
        r = S - Q
        best = np.minimum(best, np.linalg.norm(r, axis=1))
        f1, f2 = np.sum(Su * r, 1), np.sum(Sv * r, 1)
        j11, j12, j22 = np.sum(Su * Su, 1), np.sum(Su * Sv, 1), np.sum(Sv * Sv, 1)
        det = j11 * j22 - j12 * j12
        safe = np.abs(det) > 1e-300
        du = np.where(safe, -(j22 * f1 - j12 * f2) / np.where(safe, det, 1), 0.0)
        dv = np.where(safe, -(j11 * f2 - j12 * f1) / np.where(safe, det, 1), 0.0)
        u = np.clip(u + du, ua, ub)
        v = np.clip(v + dv, va, vb)
        if max(np.abs(du).max(), np.abs(dv).max()) < step_tol:
            break
    S = surface.evaluate(u, v)
    return np.minimum(best, np.linalg.norm(S - Q, axis=1))


# barycentric weights of the measured points: centroid and the three edge midpoints
_BARY = np.array(
    [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]], dtype=float
)


def measure_deviation(body, tess, corners, face_ids, sample=SAMPLE_TRIANGLES, seed=0):
    """(max deviation in body units, points measured) of the triangles from the surfaces."""
    if len(corners) == 0:
        return 0.0, 0
    pick = np.arange(len(corners))
    if sample and len(pick) > sample:
        pick = np.sort(np.random.default_rng(seed).choice(len(pick), sample, replace=False))
    points = np.asarray(tess.points)[np.asarray(tess.corner_verts)]
    uvs = np.asarray(tess.corner_uv, dtype=float)
    worst, count = 0.0, 0
    for fid in np.unique(face_ids[pick]):
        surface = body.faces[int(fid)].surface
        c = corners[pick[face_ids[pick] == fid]]  # (m, 3) corner indices
        flat = c.ravel()
        raw = _raw_uv(surface, uvs[flat], points[flat]).reshape(-1, 3, 2)
        P = points[c]  # (m, 3, 3)
        Q = np.einsum("kj,mjc->mkc", _BARY, P).reshape(-1, 3)
        UV = np.einsum("kj,mjc->mkc", _BARY, raw).reshape(-1, 2)
        d = foot_points(surface, Q, UV[:, 0], UV[:, 1])
        worst = max(worst, float(d.max()))
        count += len(Q)
    return worst, count


# --- main entry ------------------------------------------------------------------------------


def cam_mesh(
    kernel,
    body,
    matrix=None,
    tolerance: float = 1e-5,
    verify: bool = True,
    max_refinements: int = 3,
    max_segments: int = MAX_SEGMENTS,
    sample: int = SAMPLE_TRIANGLES,
    min_chord_tol: float = MIN_CHORD_TOL,
) -> CamMesh:
    """Triangulate `body` so that it deviates at most `tolerance` from the exact surfaces.

    `matrix` (4x4) maps the body's local space to the result space; `tolerance` is in
    result units. With verify=False the deviation isn't measured (max_deviation = nan).
    min_chord_tol bounds the cost (see the module docstring); lower it to allow finer
    meshes of large parts at the price of time.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    t0 = time.perf_counter()
    TessOptions = kernel.module("tessellate.grid").TessOptions
    tessellate_body = kernel.module("tessellate.body").tessellate_body

    scale = linear_scale(matrix)
    local_tol = tolerance / scale
    wanted = chord_tolerance(body, local_tol)
    chord = max(wanted, min_chord_tol)
    warnings = []
    if wanted < min_chord_tol:
        warnings.append(
            f"the tolerance {tolerance:.3g} is very fine for a part this size - meshed at "
            f"the relative limit {min_chord_tol:g} instead (see max_deviation)"
        )
    passes = 0
    previous = float("inf")
    with bounded_trim_tests(kernel):
        while True:
            passes += 1
            opts = TessOptions(chord_tol=chord, max_segments=max_segments)
            tess = tessellate_body(body, opts)
            corners, face_ids = triangulate(tess)
            if not verify:
                deviation, measured = float("nan"), 0
                break
            dev_local, measured = measure_deviation(body, tess, corners, face_ids, sample)
            deviation = dev_local * scale
            if (
                deviation <= tolerance
                or passes > max_refinements
                or chord <= min_chord_tol
                or deviation > 0.9 * previous  # tightening no longer helps (segment cap)
            ):
                break
            previous = deviation
            # tighten in proportion to the overshoot (deviation ~ chord_tol)
            chord = max(min_chord_tol, chord * max(0.2, 0.8 * tolerance / deviation))

    if verify and deviation > tolerance:
        warnings.append(
            f"measured deviation {deviation:.3g} exceeds the tolerance {tolerance:.3g} "
            f"after {passes} pass(es)"
        )
    unmerged = [int(e) for e in getattr(tess, "unmerged", [])]
    if unmerged:
        warnings.append(f"{len(unmerged)} shared edge(s) left open in the mesh")

    points = np.asarray(tess.points, dtype=float)[np.asarray(tess.corner_verts)]
    tris = points[corners] if len(corners) else np.zeros((0, 3, 3))
    if matrix is not None:
        M = np.asarray(matrix, dtype=float)
        tris = tris @ M[:3, :3].T + M[:3, 3]
    return CamMesh(
        triangles=tris,
        face_ids=face_ids,
        tolerance=float(tolerance),
        max_deviation=float(deviation),
        measured_points=int(measured),
        chord_tol=float(chord),
        passes=passes,
        seconds=time.perf_counter() - t0,
        unmerged_edges=unmerged,
        warnings=warnings,
    )
