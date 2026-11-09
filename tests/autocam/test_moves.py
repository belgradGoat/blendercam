"""Move classification (Fabex parity), drill holes, simplification, rapid splitting."""

import math

import numpy as np
import pytest

from camcore import moves


def fabex_kinds(points, free_height, plunge_angle, prev=None):
    """Fabex's export loop (gcode_export.py), one vertex at a time."""
    out = []
    last = None if prev is None else np.asarray(prev, float)
    for vi, v in enumerate(np.asarray(points, float)):
        if last is None:
            out.append(moves.RAPID)  # vertex 0 of the first operation
            last = v
            continue
        vect = v - last
        length = float(np.linalg.norm(vect))
        plunge = False
        if length > 0:
            angle = math.acos(max(-1.0, min(1.0, -vect[2] / length)))
            plunge = angle < math.pi / 2 - plunge_angle
        if plunge:
            out.append(moves.PLUNGE)
        elif v[2] >= free_height:
            out.append(moves.RAPID)
        else:
            out.append(moves.FEED)
        last = v
    return np.array(out, dtype=np.int8)


def random_path(rng, n=400):
    pts = [np.zeros(3) + [0, 0, 0.005]]
    for _ in range(n):
        step = rng.normal(0, 0.002, 3)
        r = rng.random()
        if r < 0.15:
            step[:2] = 0
            step[2] = -abs(step[2]) * 3  # straight down
        elif r < 0.25:
            step[2] = 0.005 - pts[-1][2]  # up to the free height
        elif r < 0.35:
            step = np.zeros(3)  # repeated point
        nxt = pts[-1] + step
        nxt[2] = min(nxt[2], 0.008)
        pts.append(nxt)
    return np.array(pts)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("plunge_angle", [0.0, math.pi / 6, math.pi / 3])
def test_classify_matches_fabex(seed, plunge_angle):
    pts = random_path(np.random.default_rng(seed))
    np.testing.assert_array_equal(
        moves.classify(pts, 0.005, plunge_angle), fabex_kinds(pts, 0.005, plunge_angle)
    )
    prev = [0.01, 0.01, 0.005]
    np.testing.assert_array_equal(
        moves.classify(pts, 0.005, plunge_angle, prev),
        fabex_kinds(pts, 0.005, plunge_angle, prev),
    )


def test_classify_simple():
    pts = [[0, 0, 5], [10, 0, 5], [10, 0, 0], [20, 0, -1], [30, 0, -1], [30, 0, 5]]
    k = moves.classify(np.array(pts) / 1000, 0.005, math.pi / 6)
    assert list(k) == [moves.RAPID, moves.RAPID, moves.PLUNGE, moves.FEED, moves.FEED, moves.RAPID]


def test_drill_holes_with_pecks():
    f = 5.0
    pts = [
        [0, 0, 5], [10, 10, 5], [10, 10, 0], [10, 10, -4], [10, 10, 0], [10, 10, -8], [10, 10, 5],
        [20, 10, 5], [20, 10, 0], [20, 10, -3], [20, 10, 5],
    ]  # fmt: skip
    assert moves.drill_holes(pts, f) == [(10, 10, -8), (20, 10, -3)]


def test_drill_holes_rejects_sideways_moves():
    pts = [[0, 0, 5], [10, 10, 0], [11, 10, -2], [11, 10, 5]]
    assert moves.drill_holes(pts, 5.0) is None


def _max_deviation(original, kept):
    """Largest distance from an original point to the kept polyline."""
    worst = 0.0
    for p in original:
        best = np.inf
        for a, b in zip(kept[:-1], kept[1:]):
            ab = b - a
            t = 0.0 if not ab.any() else np.clip((p - a) @ ab / (ab @ ab), 0, 1)
            best = min(best, np.linalg.norm(p - (a + t * ab)))
        worst = max(worst, best)
    return worst


def test_simplify_within_tolerance():
    t = np.linspace(0, math.pi, 400)
    arc = np.stack([20 * np.cos(t), 20 * np.sin(t), np.full_like(t, -1.0)], axis=1)
    line = np.stack([np.linspace(-20, -60, 200), np.zeros(200), np.full(200, -1.0)], axis=1)
    pts = np.vstack([[[20, 0, 5]], arc, line, [[-60, 0, 5]]])
    kinds = moves.classify(pts, 5.0, math.pi / 6)
    keep = moves.simplify(pts, kinds, 0.01)
    assert keep[0] and keep[-1] and keep.sum() < len(pts) / 3
    # the straight part collapses to its ends
    assert keep[401:601].sum() <= 2
    assert _max_deviation(pts[1:-1], pts[keep][1:-1]) <= 0.01 + 1e-12
    # rapids and plunges are never removed
    assert keep[kinds != moves.FEED].all()


def test_simplify_off_keeps_everything():
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    kinds = np.array([moves.RAPID, moves.FEED, moves.FEED, moves.FEED])
    assert moves.simplify(pts, kinds, 0).all()
    assert list(moves.simplify(pts, kinds, 0.001)) == [True, False, False, True]


def test_split_rapid():
    assert moves.split_rapid((0, 0, 5), (10, 10, 5)) == [(10, 10, 5)]
    assert moves.split_rapid((0, 0, 5), (0, 0, 1)) == [(0, 0, 1)]
    assert moves.split_rapid((0, 0, 1), (10, 10, 5)) == [(0, 0, 5), (10, 10, 5)]  # up first
    assert moves.split_rapid((0, 0, 5), (10, 10, 1)) == [(10, 10, 5), (10, 10, 1)]  # down last
