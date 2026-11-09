"""Moves of a computed Fabex path: what kind each one is, drill holes, simplification.

`classify` follows Fabex's G-code export (gcode/gcode_export.py) so a program written from
the same path cuts the same way:

- a move is a plunge when it points downward within the plunge angle of straight down
  (Fabex: `downvector.angle(v - last) < pi/2 - plunge_angle`), checked first;
- otherwise it is a rapid when it ends at or above the free (clearance) height;
- otherwise it is a feed move.
"""

import numpy as np

from .errors import CamcoreError

RAPID, FEED, PLUNGE = 0, 1, 2
KIND_NAMES = {RAPID: "rapid", FEED: "feed", PLUNGE: "plunge"}


def classify(points, free_height: float, plunge_angle: float, prev=None) -> np.ndarray:
    """Kind of the move into each point. `prev` is where the machine is before points[0]
    (None: points[0] is a rapid, like the first vertex Fabex writes)."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    kinds = np.full(len(P), FEED, dtype=np.int8)
    if not len(P):
        return kinds
    start = P[:1] if prev is None else np.asarray(prev, dtype=np.float64).reshape(1, 3)
    v = P - np.vstack([start, P[:-1]])
    length = np.linalg.norm(v, axis=1)
    plunge = (length > 0) & (-v[:, 2] > length * np.sin(plunge_angle))
    rapid = ~plunge & (P[:, 2] >= free_height)
    if prev is None:
        plunge[0], rapid[0] = False, True
    kinds[rapid] = RAPID
    kinds[plunge] = PLUNGE
    return kinds


def drill_holes(points, free_height: float, tol: float = 1e-6) -> list[tuple] | None:
    """Holes of a DRILL path: (x, y, deepest z) per run of points below the free height at
    one XY position, in path order. None if the path moves sideways below the free height
    (not a plain drilling path)."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    holes = []
    run = None
    for x, y, z in P:
        if z >= free_height:
            if run is not None:
                holes.append(tuple(run))
                run = None
            continue
        if run is None:
            run = [x, y, z]
        elif abs(x - run[0]) > tol or abs(y - run[1]) > tol:
            return None
        else:
            run[2] = min(run[2], z)
    if run is not None:
        holes.append(tuple(run))
    return holes


def _dp_keep(P: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker: points of the polyline P to keep so every dropped point lies within
    tol of the kept polyline."""
    n = len(P)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, seg = P[i], P[j] - P[i]
        q = P[i + 1 : j] - a
        l2 = float(seg @ seg)
        if l2 == 0.0:
            d = np.linalg.norm(q, axis=1)
        else:
            t = np.clip(q @ seg / l2, 0.0, 1.0)
            d = np.linalg.norm(q - t[:, None] * seg, axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return keep


def simplify(points, kinds, tol: float) -> np.ndarray:
    """Mask of points to keep: feed moves are thinned to within `tol` (chord deviation);
    rapids, plunges and the ends of every feed run are kept."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    kinds = np.asarray(kinds)
    keep = np.ones(len(P), dtype=bool)
    if tol <= 0 or len(P) < 3:
        return keep
    feed = kinds == FEED
    # runs of consecutive feed moves: points s-1 .. e (the run starts where the last
    # non-feed move ended)
    edges = np.diff(np.concatenate([[0], feed.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    for s, e in zip(starts, ends):
        lo = max(s - 1, 0)
        if e - lo < 2:
            continue
        keep[lo : e + 1] = _dp_keep(P[lo : e + 1], tol)
    return keep


def split_rapid(a, b) -> list:
    """Rapid from a to b without a diagonal: up first, across, then down."""
    ax, ay, az = a
    bx, by, bz = b
    same_xy = ax == bx and ay == by
    if same_xy or az == bz:
        return [(bx, by, bz)]
    if bz > az:
        return [(ax, ay, bz), (bx, by, bz)]
    return [(bx, by, az), (bx, by, bz)]


def check_points(points) -> np.ndarray:
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not np.all(np.isfinite(P)):
        raise CamcoreError("INVALID_VALUE", "Path has non-finite coordinates")
    return P
