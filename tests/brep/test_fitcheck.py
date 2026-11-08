"""brepcam.fitcheck: concave radii, wall radii and sharp inside corners vs a cutter."""

import numpy as np
import pytest

from brepcam import fitcheck as FC

ROOT2 = np.sqrt(2.0) - 1.0  # material left in a 90 degree corner by a tool of radius 1


@pytest.fixture(scope="module")
def solids(kernel):
    return kernel.module("ops.solids")


@pytest.fixture(scope="module")
def boolean(kernel):
    return kernel.module("ops.boolean").boolean


@pytest.fixture(scope="module")
def pocket(solids, boolean):
    """3 x 3 x 2 block (z -1..1) with a 1.6 x 1.4 pocket, floor at z = 0 (sharp corners)."""
    body, _ = boolean(
        solids.box((0, 0, 0), (3, 3, 2)), solids.box((0, 0, 0.6), (1.6, 1.4, 1.2)), "difference"
    )
    return body


@pytest.fixture(scope="module")
def rounded_pocket(kernel, pocket):
    """The pocket with every inside edge filleted at r = 0.2 (spherical corners)."""
    ids = [
        e.id
        for e in pocket.edges.values()
        if e.is_shared and all(pocket.faces[x.face].name.startswith("B.") for x in e.coedges)
    ]
    body, _ = kernel.module("ops.fillet").fillet_edges(pocket, ids, 0.2)
    return body


@pytest.fixture(scope="module")
def rounded(rounded_pocket):
    return FC.analyze(rounded_pocket)


@pytest.fixture(scope="module")
def plate(solids, boolean):
    """100 x 60 x 10 mm plate: through holes r 4 and 3 mm, a 5 mm deep blind hole r 2.5 mm."""
    body = solids.box((0, 0, 0), (0.1, 0.06, 0.01))
    for c, r, z0, h in (
        ((0.02, 0.0), 0.004, -0.01, 0.03),
        ((-0.03, -0.01), 0.003, -0.01, 0.03),
        ((-0.02, 0.015), 0.0025, 0.0, 0.02),
    ):
        body, _ = boolean(body, solids.cylinder((*c, z0), (0, 0, 1), r, h), "difference")
    return body


def test_convex_solids_set_no_limits(solids):
    for body in (solids.sphere((0, 0, 0), 0.5), solids.box(), solids.cylinder()):
        s = FC.analyze(body).summary()
        assert s["min_concave_radius"]["radius"] is None
        assert s["min_wall_radius"]["radius"] is None
        assert s["sharp_corners"] == []
        r = FC.analyze(body).check(FC.Cutter("BALLNOSE", 10.0))
        assert r["fits"] is True and r["max_tip_radius"] is None


def test_sharp_pocket_corners(pocket):
    fa = FC.analyze(pocket)
    kinds = sorted(c["kind"] for c in fa.corners)
    assert kinds == ["floor"] * 4 + ["vertical"] * 4  # the rim of the opening is convex
    assert all(c["angle"] == pytest.approx(90.0) for c in fa.corners)
    vertical = next(c for c in fa.corners if c["kind"] == "vertical")
    assert vertical["length"] == pytest.approx(1.0)  # pocket floor z = 0, block top z = 1
    assert (vertical["z_min"], vertical["z_max"]) == pytest.approx((0.0, 1.0))
    floor = next(c for c in fa.corners if c["kind"] == "floor")
    assert (floor["z_min"], floor["z_max"]) == pytest.approx((0.0, 0.0))
    # planes only: nothing limits the cutter, but corners keep material
    end = fa.check(FC.Cutter("END", 0.2))
    assert end["fits"] is True and end["problems"] == []
    rest = {c["kind"]: c["rest"] for c in end["sharp_corners"]}
    assert rest["vertical"] == pytest.approx(0.1 * ROOT2) and rest["floor"] == 0.0
    # upside down, the pocket's floor faces away from the tool: its corners are out of reach
    flipped = FC.analyze(pocket, np.diag([1.0, -1.0, -1.0, 1.0]))
    assert sorted(c["kind"] for c in flipped.corners) == ["vertical"] * 4
    ball = fa.check(FC.Cutter("BALLNOSE", 0.2))
    assert {c["kind"]: c["rest"] for c in ball["sharp_corners"]}["floor"] == pytest.approx(
        0.1 * ROOT2
    )
    assert ball["corner_rest_max"] == pytest.approx(0.1 * ROOT2)


def test_rounded_pocket_radii(rounded):
    s = rounded.summary()
    conc, wall = s["min_concave_radius"], s["min_wall_radius"]
    assert conc["radius"] == pytest.approx(0.2, rel=1e-6)
    assert conc["face_name"] in ("Fillet", "Corner")
    # walls steeper than 80 degrees: the spherical corners' contour lines are tightest at
    # their lowest wall point, R cos(10 degrees)
    assert wall["radius"] == pytest.approx(0.2 * np.sin(np.radians(80)), rel=1e-4)
    assert wall["face_name"] == "Corner"
    assert s["sharp_corners"] == []
    fillets = [f for f in s["faces"] if f["name"] == "Fillet"]
    vertical = [f for f in fillets if f["wall_radius"] is not None]
    assert len(fillets) == 8 and len(vertical) == 4
    assert all(f["wall_radius"] == pytest.approx(0.2, rel=1e-9) for f in vertical)
    assert all(f["concave_fraction"] == 1.0 for f in fillets)
    planes = [f for f in s["faces"] if f["name"] not in ("Fillet", "Corner")]
    assert all(f["concave_radius"] is None and f["concave_fraction"] == 0.0 for f in planes)
    bottom = next(f for f in s["faces"] if f["name"] == "Box.-Z")
    assert bottom["undercut_fraction"] == 1.0


@pytest.mark.parametrize(
    "cutter,skin,fits,checks",
    [
        (FC.Cutter("BALLNOSE", 0.4), 0.0, True, set()),  # a ball fits its own radius
        (FC.Cutter("BALLNOSE", 0.41), 0.0, False, {"concave_radius"}),
        (FC.Cutter("BALLNOSE", 0.38), 0.011, False, {"concave_radius"}),  # skin counts
        (FC.Cutter("END", 0.39), 0.0, True, set()),
        # a flat end can't follow the spherical corners below their equator
        (FC.Cutter("END", 0.4), 0.0, False, {"wall_radius"}),
        (FC.Cutter("BULLNOSE", 0.4, 0.2), 0.0, True, set()),  # = a ball
        (FC.Cutter("BULLNOSE", 1.0, 0.2), 0.0, False, {"wall_radius"}),
        (FC.Cutter("BALLCONE", 1.0, 0.1), 0.0, True, set()),
    ],
)
def test_cutters_in_the_rounded_pocket(rounded, cutter, skin, fits, checks):
    r = rounded.check(cutter, skin)
    assert r["fits"] is fits
    assert {p["check"] for p in r["problems"]} == checks
    for p in r["problems"]:
        assert p["radius"] < p["need"] and 0.0 < p["tight_fraction"] <= 1.0
        assert len(p["at"]) == 3
    assert r["max_tip_radius"] == pytest.approx(0.2 - skin, rel=1e-6)


def test_ball_misfit_reports_the_fillets(rounded):
    r = rounded.check(FC.Cutter("BALLNOSE", 0.5))
    names = {p["face_name"] for p in r["problems"]}
    assert names == {"Fillet", "Corner"}
    fillet = next(p for p in r["problems"] if p["face_name"] == "Fillet")
    assert fillet["radius"] == pytest.approx(0.2) and fillet["need"] == 0.25
    assert fillet["tight_fraction"] == 1.0


def test_holes_limit_the_cutter(plate):
    fa = FC.analyze(plate)
    s = fa.summary()
    assert s["min_concave_radius"]["radius"] == pytest.approx(0.0025, rel=1e-6)
    assert s["min_wall_radius"]["radius"] == pytest.approx(0.0025, rel=1e-6)
    at = np.array(s["min_wall_radius"]["at"])
    assert np.hypot(at[0] + 0.02, at[1] - 0.015) == pytest.approx(0.0025, rel=1e-6)
    # only the blind hole has an inside corner (wall meets floor); through-hole rims are convex
    assert [c["kind"] for c in s["sharp_corners"]] == ["floor"]
    corner = s["sharp_corners"][0]
    assert corner["length"] == pytest.approx(2 * np.pi * 0.0025, rel=1e-3)
    assert corner["z_min"] == pytest.approx(0.0)
    assert fa.check(FC.Cutter("END", 0.005))["fits"] is True  # exactly the hole
    r = fa.check(FC.Cutter("END", 0.006))
    assert r["fits"] is False
    assert [p["radius"] for p in r["problems"]] == pytest.approx([0.0025])
    assert r["max_cutter_radius"] == pytest.approx(0.0025, rel=1e-6)
    assert fa.check(FC.Cutter("END", 0.004), skin=0.0005)["fits"] is True  # 2 + 0.5 mm
    assert fa.check(FC.Cutter("END", 0.004), skin=0.0006)["fits"] is False


def test_world_matrix(rounded_pocket):
    th = 0.3
    M = np.eye(4)
    M[:3, :3] = 2.0 * np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    )
    M[:3, 3] = (1.0, 2.0, 3.0)
    s = FC.analyze(rounded_pocket, M).summary()
    assert s["min_concave_radius"]["radius"] == pytest.approx(0.4, rel=1e-6)
    assert s["min_wall_radius"]["radius"] == pytest.approx(0.4 * np.sin(np.radians(80)), rel=1e-4)
    assert 3.0 - 1e-9 < s["min_concave_radius"]["at"][2] < 3.0 + 2 * 0.2 + 1e-9  # floor fillets
    mirror = FC.analyze(rounded_pocket, np.diag([-1.0, 1.0, 1.0, 1.0])).summary()
    assert mirror["min_concave_radius"]["radius"] == pytest.approx(0.2, rel=1e-6)
    upside_down = FC.analyze(rounded_pocket, np.diag([1.0, -1.0, -1.0, 1.0])).summary()
    # the pocket opens downwards: the floor fillets face down (undercuts, not checked); the
    # vertical ones are still walls (a local check: it doesn't know what's above them)
    fillets = [f for f in upside_down["faces"] if f["name"] == "Fillet"]
    floor = [f for f in fillets if f["wall_radius"] is None]
    assert len(floor) == 4
    assert all(f["undercut_fraction"] == 1.0 and f["concave_radius"] is None for f in floor)
    assert all(f["concave_radius"] == pytest.approx(0.2) for f in fillets if f not in floor)


def test_dish_and_undercut(solids, boolean):
    dish, _ = boolean(
        solids.box((0, 0, 0), (2, 2, 1)), solids.sphere((0, 0, 0.5), 0.8), "difference"
    )
    s = FC.analyze(dish).summary()
    assert s["min_concave_radius"]["radius"] == pytest.approx(0.8, rel=1e-6)
    sphere = next(f for f in s["faces"] if "Sphere" in f["name"])
    assert sphere["undercut_fraction"] == 0.0 and sphere["concave_fraction"] == 1.0
    deep, _ = boolean(
        solids.box((0, 0, 0), (2, 2, 1)), solids.sphere((0.013, 0.007, 0.27), 0.7), "difference"
    )
    s = FC.analyze(deep).summary()
    sphere = next(f for f in s["faces"] if "Sphere" in f["name"])
    # above the sphere's centre the cavity faces down
    assert 0.1 < sphere["undercut_fraction"] < 0.5
    assert s["min_concave_radius"]["radius"] == pytest.approx(0.7, rel=1e-6)


def test_open_sheets_face_up(kernel, solids):
    model, edges = kernel.module("topology.model"), kernel.module("topology.edges")
    sheet = model.Body()
    bottom = next(f for f in solids.box().faces.values() if f.name == "Box.-Z")
    sheet.add_face(bottom.surface)  # normal points down
    edges.stitch(sheet)
    fa = FC.analyze(sheet)
    assert fa.warnings and "Open body" in fa.warnings[0]
    assert fa.faces[0].undercut_fraction == 0.0


def test_faces_filter_and_bad_arguments(plate):
    hole = next(
        fid
        for fid, f in plate.faces.items()
        if "Cylinder" in f.name
        and np.isclose(FC.analyze(plate, faces=[fid]).faces[0].concave_radius, 0.004)
    )
    s = FC.analyze(plate, faces=[hole]).summary()
    assert [f["face"] for f in s["faces"]] == [hole]
    assert s["min_wall_radius"]["radius"] == pytest.approx(0.004, rel=1e-6)
    assert s["sharp_corners"] == []  # a through hole: its rims are convex
    with pytest.raises(KeyError):
        FC.analyze(plate, faces=[987654])
    with pytest.raises(ValueError):
        FC.analyze(plate, wall_angle=0)
    with pytest.raises(ValueError):
        FC.analyze(plate, corner_angle=180)
    with pytest.raises(ValueError):
        FC.analyze(plate, np.zeros((4, 4)))


def test_cutter_model():
    end, ball = FC.Cutter("END", 6.0), FC.Cutter("BALLNOSE", 6.0)
    bull, cone = FC.Cutter("BULLNOSE", 6.0, 1.0), FC.Cutter("BALLCONE", 6.0, 0.5)
    assert (end.tip_radius, end.side_radius) == (0.0, 3.0)
    assert (ball.tip_radius, ball.side_radius) == (3.0, None)
    assert (bull.tip_radius, bull.side_radius) == (1.0, 3.0)
    assert (cone.tip_radius, cone.side_radius) == (0.5, None)
    # a bull nose touches a less than vertical wall on its corner, inside its full radius
    assert bull.wall_need(1.0) == 3.0
    assert bull.wall_need(0.5) == pytest.approx(2.5)
    assert end.wall_need(0.5, skin=0.2) == pytest.approx(3.1)
    assert end.corner_rest("vertical", 90) == pytest.approx(3 * ROOT2)
    assert end.corner_rest("floor", 90) == 0.0 and end.corner_rest("other", 45) is None
    assert bull.corner_rest("floor", 90) == pytest.approx(ROOT2)
    assert cone.corner_rest("vertical", 90) is None
    for kind in ("VCARVE", "CYLCONE", "LASER", "PLASMA", "CUSTOM"):
        c = FC.Cutter(kind, 6.0)
        assert not c.checked and c.corner_rest("floor", 90) is None
    fa = FC.FitAnalysis([], [], 80.0, 10.0)
    assert fa.check(FC.Cutter("VCARVE", 6.0))["fits"] is None
