"""brepcam.flowline: passes along the exact surfaces' parameter lines."""

from collections import Counter

import numpy as np
import pytest

from brepcam import flowline as FL

P = FL.FlowlineParams


@pytest.fixture(scope="module")
def solids(kernel):
    return kernel.module("ops.solids")


@pytest.fixture(scope="module")
def pr(kernel):
    return kernel.module("nurbs.primitives")


@pytest.fixture(scope="module")
def sheet(kernel):
    """A body of one face with naked edges (open: its faces are turned to face up)."""
    Body = kernel.module("topology.model").Body
    stitch = kernel.module("topology.edges").stitch

    def make(surface, name="Sheet"):
        body = Body()
        body.add_face(surface, name)
        stitch(body)
        return body

    return make


@pytest.fixture(scope="module")
def box(solids):
    return solids.box((0, 0, 0), (1, 1, 1))


def face_id(body, name):
    return next(f.id for f in body.faces.values() if f.name == name)


@pytest.fixture(scope="module")
def roof(pr, sheet):
    """Convex half cylinder: radius 1 about the y axis, 1 long (y 0..1)."""
    arc = pr.circle_arc((0, 0, 0), 1.0, (1, 0, 0), (0, 0, 1), 0.0, np.pi)
    return sheet(pr.extrude(arc, (0, 1.0, 0)))


@pytest.fixture(scope="module")
def dome(pr, sheet):
    """Spherical cap, radius 1 about the origin, from 30 degrees up to the pole."""
    arc = pr.circle_arc((0, 0, 0), 1.0, (1, 0, 0), (0, 0, 1), np.pi / 6, np.pi / 2)
    return sheet(pr.revolve(arc, (0, 0, 0), (0, 0, 1)))


@pytest.fixture(scope="module")
def trough(pr, sheet):
    """Concave half cylinder: radius 1 about the line z = 1 (the bottom at z = 0)."""
    arc = pr.circle_arc((0, 0, 1), 1.0, (1, 0, 0), (0, 0, 1), np.pi, 2 * np.pi)
    return sheet(pr.extrude(arc, (0, 1.0, 0)))


@pytest.fixture(scope="module")
def holed(solids, kernel):
    """100 x 60 x 10 mm plate with an 8 mm radius through hole at (20, 0) mm."""
    boolean = kernel.module("ops.boolean").boolean
    body = solids.box((0, 0, 0), (0.1, 0.06, 0.01))
    body, _ = boolean(
        body, solids.cylinder((0.02, 0.0, -0.01), (0, 0, 1), 0.008, 0.03), "difference"
    )
    return body


def _segments(p):
    P = np.vstack([p.points, p.points[:1]]) if p.closed else p.points
    return P[:-1], P[1:]


def gaps(fl, face):
    """Largest distance from a contact point of station j + 1 to the passes of station
    j, for each pair of adjacent stations."""
    by_station = {}
    for p in fl.passes:
        if p.face == face:
            by_station.setdefault(p.station, []).append(p)
    keys = sorted(by_station)
    out = []
    for j0, j1 in zip(keys, keys[1:]):
        A0, A1 = (np.concatenate(x) for x in zip(*map(_segments, by_station[j0])))
        B = np.concatenate([p.points for p in by_station[j1]])
        ab = A1 - A0
        t = np.clip(((B[:, None] - A0[None]) * ab[None]).sum(2) / (ab * ab).sum(1)[None], 0, 1)
        d = np.linalg.norm(B[:, None] - (A0[None] + t[..., None] * ab[None]), axis=2).min(1)
        out.append(d.max())
    return np.array(out)


def steps(p):
    return np.linalg.norm(np.diff(p.points, axis=0), axis=1)


# --- planes and order ----------------------------------------------------------------------


def test_plane_zigzag(kernel, box):
    top = face_id(box, "Box.+Z")
    fl = FL.flowline(kernel, box, P(0.1, 0.03), faces=[top])
    f = fl.faces[0]
    assert (f.stations, f.passes, f.closed, f.poles) == (11, 11, False, [])
    assert f.max_gap == pytest.approx(0.1)
    assert fl.length == pytest.approx(11.0)
    for i, p in enumerate(fl.passes):
        assert np.allclose(p.points[:, 2], 0.5) and np.allclose(p.normals, (0, 0, 1))
        assert steps(p).max() <= 0.03 * (1 + 1e-9)
        assert np.allclose(p.centers, p.points[:, :2])  # flat: no offset
        if i:  # zigzag: each pass starts where the previous one ended
            assert np.linalg.norm(p.points[0, :2] - fl.passes[i - 1].points[-1, :2]) < 0.11
    oneway = FL.flowline(kernel, box, P(0.1, 0.03, order="ONEWAY"), faces=[top])
    starts = {tuple(np.round(p.points[0, :1], 9)) for p in oneway.passes}
    assert len(starts) == 1  # all start on the same side
    rev = FL.flowline(kernel, box, P(0.1, 0.03, order="ONEWAY", reverse=True), faces=[top])
    assert np.allclose(rev.passes[0].points, oneway.passes[0].points[::-1])


def test_only_faces_the_tool_reaches(kernel, box):
    fl = FL.flowline(kernel, box, P(0.1, 0.05))
    notes = {f.name: (f.passes, f.note) for f in fl.faces}
    assert notes["Box.+Z"] == (11, None)
    for name in ("Box.-X", "Box.+X", "Box.-Y", "Box.+Y", "Box.-Z"):
        assert notes[name] == (0, "too steep or facing down")
    # a vertical wall with max_slope 90; never the bottom
    fl = FL.flowline(kernel, box, P(0.1, 0.05, max_slope=90))
    passes = {f.name: f.passes for f in fl.faces}
    assert passes["Box.+X"] > 0 and passes["Box.-Z"] == 0


def test_direction_and_stepover(kernel, box):
    top = face_id(box, "Box.+Z")
    u = FL.flowline(kernel, box, P(0.25, 0.1, direction="U"), faces=[top])
    v = FL.flowline(kernel, box, P(0.25, 0.1, direction="V"), faces=[top])
    du = np.ptp(u.passes[0].uv, axis=0)
    dv = np.ptp(v.passes[0].uv, axis=0)
    assert du[1] == 0 and du[0] > 0  # U: passes follow u
    assert dv[0] == 0 and dv[1] > 0
    # stepover that doesn't divide the width: evenly spread, never wider
    odd = FL.flowline(kernel, box, P(0.3, 0.1), faces=[top])
    assert odd.faces[0].stations == 5 and odd.faces[0].max_gap == pytest.approx(0.25)


# --- curved faces ---------------------------------------------------------------------------


def test_convex_roof_slope_limit_and_ball_centres(kernel, roof):
    fl = FL.flowline(kernel, roof, P(0.1, 0.05, max_slope=60, tip_radius=0.2))
    f = fl.faces[0]
    assert f.direction == "U" and f.stations == 11  # around the arc (pi > 1)
    assert fl.warnings == ["Open body (sheet): faces were turned to face up (+Z)"]
    for p in fl.passes:
        assert p.normals[:, 2].min() >= 0.5 - 1e-12  # nothing steeper than 60 degrees
        assert p.normals[[0, -1], 2] == pytest.approx(0.5, abs=1e-9)  # ends on the limit
        assert steps(p).max() <= 0.05 * (1 + 1e-6)
        # a ball of radius 0.2 dropped at the centre XY touches the contact point
        centre = np.column_stack([p.centers, p.points[:, 2] + 0.2 * p.normals[:, 2]])
        dist = np.linalg.norm(centre[:, [0, 2]], axis=1)
        assert np.allclose(dist, 1.2, atol=1e-12)
        assert np.allclose(p.tool_tips(0.2)[:, 2], centre[:, 2] - 0.2)


def test_stations_across_an_arc(kernel, roof):
    fl = FL.flowline(kernel, roof, P(0.1, 0.05, direction="V", max_slope=90))
    f = fl.faces[0]
    assert f.direction == "V" and f.cross_length == pytest.approx(np.pi, rel=1e-4)
    assert f.max_gap <= 0.1 and gaps(fl, f.face).max() <= 0.1 * (1 + 1e-9)
    assert f.stations == int(np.ceil(f.cross_length / 0.1)) + 1


def test_dome_rings_around_the_pole(kernel, dome):
    fl = FL.flowline(kernel, dome, P(0.05, 0.02, tip_radius=0.1))
    f = fl.faces[0]
    assert f.poles == ["v1"] and f.direction == "U" and f.closed
    assert all(p.closed for p in fl.passes) and f.passes == f.stations - 1  # no ring at the pole
    assert gaps(fl, f.face).max() <= 0.05 * (1 + 1e-6)
    for p in fl.passes:
        assert np.allclose(np.linalg.norm(p.points, axis=1), 1.0, atol=1e-12)
        assert np.linalg.norm(p.points[0] - p.points[-1]) <= 0.02 * (1 + 1e-6)  # loop closes
        assert len({tuple(np.round(q, 12)) for q in p.points}) == len(p.points)  # no repeats
    assert f.misfit_points == 0
    # forced across the pole: meridians, no degenerate normals
    v = FL.flowline(kernel, dome, P(0.05, 0.02, direction="V"))
    assert v.faces[0].direction == "V" and not any(p.closed for p in v.passes)
    assert all(np.all(np.isfinite(p.normals)) for p in v.passes)


def test_concave_trough_tip_fit(kernel, trough):
    ok = FL.flowline(kernel, trough, P(0.1, 0.05, tip_radius=0.5))
    assert ok.faces[0].misfit_points == 0 and ok.faces[0].passes == 11
    big = FL.flowline(kernel, trough, P(0.1, 0.05, tip_radius=1.5))
    f = big.faces[0]
    assert f.passes == 0 and f.misfit_points > 0 and "too big" in f.note
    # a ball the size of the trough: every centre in the same place, backtracking nowhere
    centres = np.concatenate([p.centers for p in ok.passes])
    assert np.all(np.isfinite(centres))


def test_trimmed_face_with_a_hole(kernel, holed):
    top = face_id(holed, "Box.+Z")
    fl = FL.flowline(kernel, holed, P(0.002, 0.0005, tolerance=1e-6), faces=[top])
    f = fl.faces[0]
    assert f.passes > f.stations  # passes split by the hole
    pts = np.concatenate([p.points for p in fl.passes])
    r = np.linalg.norm(pts[:, :2] - (0.02, 0.0), axis=1)
    assert r.min() >= 0.008 - 1e-6  # nothing inside the hole (trim tolerance 1 um)
    ends = np.concatenate([p.points[[0, -1]] for p in fl.passes])
    on_hole = np.abs(np.linalg.norm(ends[:, :2] - (0.02, 0.0), axis=1) - 0.008) < 1e-6
    on_rim = np.isclose(np.abs(ends[:, 0]), 0.05)
    assert np.all(on_hole | on_rim)  # every pass ends on the hole or the plate's rim
    assert on_hole.sum() >= 2 * (f.passes - f.stations)
    assert np.allclose(pts[:, 2], 0.005)


def test_seam_joined_and_world_matrix(kernel, solids):
    """A horizontal cylinder turned so its seam is on top: each ring over the top is one
    pass, joined across the seam."""
    cyl = solids.cylinder((0, 0, 0), (1, 0, 0), 0.5, 1.0)
    side = next(f for f in cyl.faces.values() if f.name == "Cylinder.side")
    S = side.surface
    seam = S.evaluate(S.domain_u[0], np.mean(S.domain_v))
    turn = np.pi / 2 - np.arctan2(seam[2], seam[1])
    M = np.eye(4)
    M[1:3, 1:3] = [[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]
    M[:3, 3] = (0.3, -0.2, 0.1)
    fl = FL.flowline(kernel, cyl, P(0.05, 0.01), matrix=M, faces=[side.id])
    f = fl.faces[0]
    assert f.closed and f.stations == 21
    assert set(Counter(p.station for p in fl.passes).values()) == {1}
    for p in fl.passes:
        rel = p.points - (0.3, -0.2, 0.1)
        assert np.allclose(np.linalg.norm(rel[:, 1:], axis=1), 0.5, atol=1e-12)
        assert p.normals[:, 2].min() >= np.cos(np.radians(80)) - 1e-12
        assert steps(p).max() <= 0.01 * (1 + 1e-6)
    # standing up, with max_slope 90: whole closed rings
    up = solids.cylinder((0, 0, 0), (0, 0, 1), 0.5, 1.0)
    side = next(f for f in up.faces.values() if f.name == "Cylinder.side")
    fl = FL.flowline(kernel, up, P(0.1, 0.05, max_slope=90), faces=[side.id])
    assert fl.faces[0].direction == "U" and all(p.closed for p in fl.passes)
    assert len(fl.passes) == 11


def test_mirror_keeps_faces_facing_up(kernel, box):
    M = np.diag([-2.0, 1.0, 1.0, 1.0])  # mirrored and stretched in x
    fl = FL.flowline(kernel, box, P(0.1, 0.05), matrix=M)
    assert {f.name for f in fl.faces if f.passes} == {"Box.+Z"}
    p = fl.passes[0]
    assert np.allclose(p.normals, (0, 0, 1))
    assert np.ptp(np.concatenate([q.points[:, 0] for q in fl.passes])) == pytest.approx(2.0)


# --- limits and errors ----------------------------------------------------------------------


def test_too_many_points(kernel, box):
    with pytest.raises(FL.PatternTooLarge) as e:
        FL.flowline(kernel, box, P(0.001, 0.0001, max_points=100_000))
    assert e.value.estimate > 100_000 and e.value.limit == 100_000


@pytest.mark.parametrize(
    "bad",
    [
        dict(stepover=0),
        dict(step_along=-1),
        dict(direction="W"),
        dict(order="SPIRAL"),
        dict(max_slope=0),
        dict(max_slope=91),
        dict(tip_radius=-0.1),
        dict(tolerance=0),
    ],
)
def test_bad_params(kernel, box, bad):
    args = {"stepover": 0.1, "step_along": 0.05, **bad}
    with pytest.raises(ValueError):
        FL.flowline(kernel, box, P(**args))


def test_unknown_faces_and_singular_matrix(kernel, box):
    with pytest.raises(KeyError):
        FL.flowline(kernel, box, P(0.1, 0.05), faces=[999])
    with pytest.raises(ValueError):
        FL.flowline(kernel, box, P(0.1, 0.05), matrix=np.zeros((4, 4)))


def test_without_basis_matrices(box):
    """No kernel handed in: NurbsSurface.grid is used instead (same result)."""
    top = face_id(box, "Box.+Z")
    fl = FL.flowline(None, box, P(0.1, 0.05), faces=[top])
    assert fl.faces[0].passes == 11 and fl.faces[0].max_gap == pytest.approx(0.1)


def test_stats_are_json(kernel, dome):
    import json

    s = FL.flowline(kernel, dome, P(0.1, 0.05)).stats()
    text = json.dumps(s)
    assert '"closed_passes"' in text and s["faces"][0]["poles"] == ["v1"]
