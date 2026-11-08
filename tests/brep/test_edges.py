"""brepcam.edges: B-rep edges, face loops and holes as polylines within a tolerance."""

import numpy as np
import pytest

from brepcam import edges as E


@pytest.fixture(scope="module")
def solids(kernel):
    return kernel.module("ops.solids")


@pytest.fixture(scope="module")
def plate(kernel, solids):
    """100 x 60 x 10 mm plate (z -5..5 mm): two through holes and one 5 mm deep blind hole."""
    boolean = kernel.module("ops.boolean").boolean
    body = solids.box((0, 0, 0), (0.1, 0.06, 0.01))
    for c, r, z0, h in (
        ((0.02, 0.0), 0.004, -0.01, 0.03),
        ((-0.03, -0.01), 0.003, -0.01, 0.03),
        ((-0.02, 0.015), 0.0025, 0.0, 0.02),
    ):
        body, _ = boolean(body, solids.cylinder((*c, z0), (0, 0, 1), r, h), "difference")
    return body


def face_named(body, name):
    return next(fid for fid, f in body.faces.items() if f.name == name)


def chord_points(pl, n=16):
    """Points along every chord of a polyline (to check the deviation between samples)."""
    P = np.vstack([pl.points, pl.points[:1]]) if pl.closed else pl.points
    s = np.linspace(0, 1, n)[None, :, None]
    return (P[:-1, None] * (1 - s) + P[1:, None] * s).reshape(-1, 3)


def radial_error(pl, center_xy, radius):
    Q = chord_points(pl)
    return np.abs(np.linalg.norm(Q[:, :2] - np.asarray(center_xy), axis=1) - radius).max()


def test_box_edges(solids):
    es = E.extract(solids.box((0, 0, 0), (1, 2, 3)), "all")
    # every corner joins three edges, so nothing chains
    assert len(es.polylines) == 12 and not any(p.closed for p in es.polylines)
    assert sorted(round(p.length, 9) for p in es.polylines) == [1.0] * 4 + [2.0] * 4 + [3.0] * 4
    assert len(E.extract(solids.box(), "shared").polylines) == 12
    assert E.extract(solids.box(), "naked").polylines == []


def test_box_creases_flatten_to_one_outline(solids):
    es = E.extract(solids.box((0, 0, 0), (1, 2, 3)), "creases", project="top")
    # vertical edges collapse, top and bottom rectangles coincide, the rest chains
    assert len(es.polylines) == 1
    pl = es.polylines[0]
    assert pl.closed and abs(E._area_xy(pl.points)) == pytest.approx(2.0)
    assert np.allclose(pl.points[:, 2], 1.5)
    assert len(pl.edges) == 8  # provenance: the top and bottom rectangles' edges
    low = E.extract(solids.box((0, 0, 0), (1, 2, 3)), "creases", project="bottom").polylines
    assert np.allclose(low[0].points[:, 2], -1.5)


def test_naked_edges_of_a_sheet_chain_into_a_loop(kernel, solids):
    model, edges = kernel.module("topology.model"), kernel.module("topology.edges")
    box = solids.box((0, 0, 0), (1, 1, 1))
    sheet = model.Body()
    sheet.add_face(next(iter(box.faces.values())).surface)
    edges.stitch(sheet)
    es = E.extract(sheet, "naked")
    assert len(es.polylines) == 1 and es.polylines[0].closed
    assert es.polylines[0].length == pytest.approx(4.0)
    assert len(es.polylines[0].points) >= 4


@pytest.mark.parametrize("tol", [1e-4, 1e-5])
def test_cylinder_rims_are_creases_within_tolerance(solids, tol):
    R, H = 0.5, 1.0
    cyl = solids.cylinder((0, 0, 0), (0, 0, 1), R, H)
    es = E.extract(cyl, "creases", tol)
    assert len(es.polylines) == 2 and all(p.closed for p in es.polylines)
    for pl in es.polylines:
        assert radial_error(pl, (0, 0), R) <= tol
        assert np.allclose(pl.points[:, 2], pl.points[0, 2])
    # the seam is tangent, the rims are 90 degree edges
    angles = sorted(E.edge_angle(cyl, e) for e in cyl.edges.values())
    assert angles[0] < 1.0 and angles[-1] == pytest.approx(90.0, abs=0.5)


def test_sphere_has_no_creases(solids):
    assert E.extract(solids.sphere((0, 0, 0), 1.0), "creases").polylines == []


def test_finer_tolerance_gives_more_points(solids):
    cyl = solids.cylinder((0, 0, 0), (0, 0, 1), 0.5, 1.0)
    coarse = E.extract(cyl, "creases", 1e-3).stats()["points"]
    fine = E.extract(cyl, "creases", 1e-5).stats()["points"]
    assert fine > coarse


def test_holes(plate):
    tol = 1e-5
    es = E.extract(plate, "holes", tol)
    holes = sorted(es.holes, key=lambda h: -h["radius"])
    assert len(holes) == 3 == len(es.polylines)  # through holes count once
    a, b, c = holes
    for h, xy, r in (
        (a, (0.02, 0.0), 0.004),
        (b, (-0.03, -0.01), 0.003),
        (c, (-0.02, 0.015), 0.0025),
    ):
        assert h["circular"] and h["vertical"]
        assert h["radius"] == pytest.approx(r, abs=tol)
        assert np.allclose(h["center"], (*xy, 0.005), atol=tol)
        assert h["z_top"] == pytest.approx(0.005)
    assert a["kind"] == b["kind"] == "through"
    assert a["depth"] == pytest.approx(0.01) and a["z_bottom"] == pytest.approx(-0.005)
    bottom = face_named(plate, "Box.-Z")
    assert [e[0] for e in a["other_ends"]] == [bottom]
    assert c["kind"] == "blind" and c["other_ends"] == []
    assert c["depth"] == pytest.approx(0.005) and c["z_bottom"] == pytest.approx(0.0, abs=1e-12)
    # entry loops come from the top face; the polyline is the exact hole edge
    top = face_named(plate, "Box.+Z")
    for pl in es.polylines:
        assert pl.loop[0] == top and pl.hole is not None
        h = pl.hole
        assert radial_error(pl, h["center"][:2], h["radius"]) <= tol
        # Fabex drills at the middle of a spline's bounding box: the hole centre
        mid = 0.5 * (pl.points.min(0) + pl.points.max(0))
        assert np.allclose(mid[:2], h["center"][:2], atol=tol)


def test_holes_at_the_bottom(plate):
    es = E.extract(plate, "holes", project="bottom")
    z = sorted(round(float(p.points[0, 2]), 9) for p in es.polylines)
    assert z == [-0.005, -0.005, 0.0]  # through holes: plate bottom; blind: its floor
    assert all(np.ptp(p.points[:, 2]) == 0 for p in es.polylines)


def test_holes_of_selected_faces(plate):
    bottom = face_named(plate, "Box.-Z")
    es = E.extract(plate, "holes", faces=[bottom])
    assert len(es.holes) == 2 and all(h["face"] == bottom for h in es.holes)


def test_outer_loop_of_one_face(plate):
    top = face_named(plate, "Box.+Z")
    es = E.extract(plate, "outer_loops", faces=[top], project="bottom")
    assert len(es.polylines) == 1
    pl = es.polylines[0]
    assert pl.closed and abs(E._area_xy(pl.points)) == pytest.approx(0.1 * 0.06)
    assert np.allclose(pl.points[:, 2], 0.005)  # one face: its own (flat) Z
    with pytest.raises(KeyError):
        E.extract(plate, "outer_loops", faces=[9999])


def test_matrix_and_world_tolerance(plate):
    tol = 1e-5
    M = np.diag([2.0, 2.0, 2.0, 1.0])
    M[:3, 3] = (1.0, -2.0, 0.5)
    es = E.extract(plate, "holes", tol, matrix=M)
    big = max(es.holes, key=lambda h: h["radius"])
    assert big["radius"] == pytest.approx(0.008, abs=tol)
    assert np.allclose(big["center"], (1.04, -2.0, 0.51), atol=tol)
    assert big["depth"] == pytest.approx(0.02)
    pl = next(p for p in es.polylines if p.loop == (big["face"], big["loop"]))
    assert radial_error(pl, big["center"][:2], big["radius"]) <= tol


def test_chain_joins_shuffled_segments():
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    segs = [E.Polyline(np.array([sq[i], sq[(i + 1) % 4]]), False, [i]) for i in range(4)]
    segs[1].points = segs[1].points[::-1]  # one reversed
    segs = [segs[2], segs[0], segs[3], segs[1]]
    out = E.chain(segs, 1e-9)
    assert len(out) == 1 and out[0].closed and len(out[0].points) == 4
    assert out[0].edges == [0, 1, 2, 3]
    # a T junction stops chaining: three ends meet at (1, 0)
    t = [
        E.Polyline(np.array([[0, 0, 0], [1, 0, 0.0]]), False),
        E.Polyline(np.array([[1, 0, 0], [2, 0, 0.0]]), False),
        E.Polyline(np.array([[1, 0, 0], [1, 1, 0.0]]), False),
    ]
    assert len(E.chain(t, 1e-9)) == 3
    # an open chain stays open and keeps its order
    line = E.chain([t[0], t[1]], 1e-9)
    assert len(line) == 1 and not line[0].closed and len(line[0].points) == 3


def test_fit_circle():
    a = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    P = np.column_stack([1 + 2 * np.cos(a), -1 + 2 * np.sin(a), np.full_like(a, 3.0)])
    c = E.fit_circle(P)
    assert np.allclose(c["center"], (1, -1, 3)) and c["radius"] == pytest.approx(2)
    assert np.allclose(c["axis"], (0, 0, 1)) and c["residual"] < 1e-12


def test_bad_arguments(plate):
    for kw in ({"kind": "nope"}, {"project": "side"}, {"tolerance": 0}):
        with pytest.raises(ValueError):
            E.extract(plate, **kw)


def test_summary_is_plain(plate):
    es = E.extract(plate, "holes")
    s = es.polylines[0].summary()
    assert isinstance(s["length"], float) and isinstance(s["edges"], list)
    assert "_points" not in es.holes[0]
    assert es.stats()["splines"] == 3


def test_a_boss_is_not_a_hole(kernel, solids):
    boolean = kernel.module("ops.boolean").boolean
    body, _ = boolean(
        solids.box((0, 0, 0), (0.1, 0.06, 0.01)),
        solids.cylinder((0.0, 0.0, 0.0), (0, 0, 1), 0.01, 0.02),  # stands 15 mm proud
        "union",
    )
    assert E.extract(body, "holes").holes == []
    found, _ = E.find_holes(body, 1e-5)
    assert [h["kind"] for h in found] == ["boss"]
    assert found[0]["depth"] == pytest.approx(0.015) and found[0]["radius"] == pytest.approx(0.01)


def test_tilted_sheet_outline_survives_flattening(kernel, solids):
    model, edges = kernel.module("topology.model"), kernel.module("topology.edges")
    box = solids.box((0, 0, 0), (1, 1, 1))
    sheet = model.Body()
    sheet.add_face(
        box.faces[next(f for f, x in box.faces.items() if x.name.endswith("+Z"))].surface
    )
    edges.stitch(sheet)
    a = np.radians(30)
    tilt = np.eye(4)
    tilt[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    # its four edges sit at different heights: chained before flattening, so still one loop
    es = E.extract(sheet, "naked", matrix=tilt, project="top")
    assert len(es.polylines) == 1 and es.polylines[0].closed
    assert abs(E._area_xy(es.polylines[0].points)) == pytest.approx(np.cos(a))


def test_project_to_a_z_level(plate):
    es = E.extract(plate, "holes", project=-0.02)
    assert len(es.polylines) == 3 and all(np.all(p.points[:, 2] == -0.02) for p in es.polylines)
    es = E.extract(plate, "creases", project=0.5)
    # outline + three rims, the top and bottom copies kept once
    assert len(es.polylines) == 4 and all(p.closed for p in es.polylines)
    for bad in (float("nan"), True, "middle"):
        with pytest.raises(ValueError):
            E.extract(plate, "holes", project=bad)
