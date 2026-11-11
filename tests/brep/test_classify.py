"""brepcam.classify: surface types, roles, edge kinds and openings from the exact geometry."""

import json

import numpy as np
import pytest

from brepcam import classify as C
from brepcam import edges as E


@pytest.fixture(scope="module")
def solids(kernel):
    return kernel.module("ops.solids")


@pytest.fixture(scope="module")
def pr(kernel):
    return kernel.module("nurbs.primitives")


@pytest.fixture(scope="module")
def boolean(kernel):
    return kernel.module("ops.boolean").boolean


@pytest.fixture(scope="module")
def sheet(kernel):
    Body = kernel.module("topology.model").Body
    stitch = kernel.module("topology.edges").stitch

    def make(surface, name="Sheet"):
        body = Body()
        body.add_face(surface, name)
        stitch(body)
        return body

    return make


@pytest.fixture(scope="module")
def plate(solids, boolean):
    """100 x 60 x 10 mm plate (z -5..5 mm): through holes r 4 and 3 mm, a 5 mm deep r 2.5."""
    body = solids.box((0, 0, 0), (0.1, 0.06, 0.01))
    for c, r, z0, h in (
        ((0.02, 0.0), 0.004, -0.01, 0.03),
        ((-0.03, -0.01), 0.003, -0.01, 0.03),
        ((-0.02, 0.015), 0.0025, 0.0, 0.02),
    ):
        body, _ = boolean(body, solids.cylinder((*c, z0), (0, 0, 1), r, h), "difference")
    return body


@pytest.fixture(scope="module")
def pocket(solids, boolean):
    """3 x 3 x 2 block (z -1..1) with a 1.6 x 1.4 pocket, floor at z = 0 (sharp corners)."""
    body, _ = boolean(
        solids.box((0, 0, 0), (3, 3, 2)), solids.box((0, 0, 0.6), (1.6, 1.4, 1.2)), "difference"
    )
    return body


@pytest.fixture(scope="module")
def rounded(kernel, pocket):
    """The pocket with every inside edge filleted at r = 0.2."""
    ids = [
        e.id
        for e in pocket.edges.values()
        if e.is_shared and all(pocket.faces[x.face].name.startswith("B.") for x in e.coedges)
    ]
    body, _ = kernel.module("ops.fillet").fillet_edges(pocket, ids, 0.2)
    return body


@pytest.fixture(scope="module")
def can(kernel, solids):
    """A cylinder r 1, h 2 with its top edge rounded (r 0.2): the side stays a wall."""
    body = solids.cylinder((0, 0, 0), (0, 0, 1), 1.0, 2.0)
    top = next(f.id for f in body.faces.values() if f.name.endswith("top"))
    side = next(f.id for f in body.faces.values() if f.name.endswith("side"))
    rim = [
        e.id
        for e in body.edges.values()
        if e.is_shared and {c.face for c in e.coedges} == {top, side}
    ]
    body, _ = kernel.module("ops.fillet").fillet_edges(body, rim, 0.2)
    return body


def by_name(cl, name):
    return [f for f in cl.faces if f.name == name]


def one(cl, name):
    (f,) = by_name(cl, name)
    return f


def test_box(solids):
    cl = C.classify(solids.box((0, 0, 0), (1, 2, 3)))
    assert cl.closed and not cl.warnings
    assert {f.surface["type"] for f in cl.faces} == {"plane"}
    roles = {f.name: f.role for f in cl.faces}
    assert roles["Box.+Z"] == "top" and roles["Box.-Z"] == "bottom"
    assert all(roles[n] == "wall" for n in ("Box.-X", "Box.+X", "Box.-Y", "Box.+Y"))
    top = one(cl, "Box.+Z")
    assert np.allclose(top.surface["normal"], (0, 0, 1)) and top.surface["point"][2] == 1.5
    assert top.slope_range() == [0.0, 0.0] and top.area == pytest.approx(2.0)
    assert {e["kind"] for e in cl.edges.values()} == {"convex"} and len(cl.edges) == 12
    assert all(len(f.neighbors) == 4 for f in cl.faces)
    assert np.allclose(cl.bbox_min, (-0.5, -1, -1.5)) and np.allclose(cl.bbox_max, (0.5, 1, 1.5))


def test_holes_and_hole_walls(plate):
    cl = C.classify(plate)
    walls = [f for f in cl.faces if f.surface["type"] == "cylinder"]
    assert sorted(round(f.surface["radius"], 9) for f in walls) == [0.0025, 0.003, 0.004]
    for f in walls:
        assert f.role == "hole_wall" and f.curvature == "concave"
        assert np.allclose(np.abs(f.surface["axis"]), (0, 0, 1))
        assert f.slope_range() == [90.0, 90.0] and f.deviation < 1e-12
    assert [h["index"] for h in cl.holes] == [0, 1, 2]
    assert [h["kind"] for h in cl.holes] == ["through", "through", "blind"]
    assert all(h["prismatic"] and h["circular"] for h in cl.holes)
    assert [h["bottom"] for h in cl.holes] == ["through", "through", "flat"]
    assert all(h["tip_angle"] is None for h in cl.holes)
    blind = cl.holes[2]
    floor = cl.face(next(f for f in blind["faces"] if cl.face(f).surface["type"] == "plane"))
    assert floor.role == "floor" and floor.hole == 2 and floor.z_max == pytest.approx(0.0)
    assert one(cl, "Box.+Z").role == "top" and one(cl, "Box.-Z").role == "bottom"
    kinds = [e["kind"] for e in cl.edges.values()]
    assert kinds.count("concave") == 1 and kinds.count("seam") == 3  # the blind hole's floor


def test_sharp_pocket(pocket):
    cl = C.classify(pocket)
    (h,) = cl.holes
    assert h["kind"] == "blind" and not h["circular"] and h["prismatic"]
    inside = [cl.face(f) for f in h["faces"]]
    assert sorted(f.role for f in inside) == ["floor"] + ["wall"] * 4  # not round: walls
    assert [e["kind"] for e in cl.edges.values()].count("concave") == 8


def test_fillets(rounded):
    cl = C.classify(rounded)
    fillets = [f for f in cl.faces if f.name in ("Fillet", "Corner")]
    assert len(fillets) == 12 and all(f.role == "fillet" for f in fillets)
    assert all(f.min_radius == pytest.approx(0.2, rel=1e-9) for f in fillets)
    assert {f.surface["type"] for f in fillets if f.name == "Corner"} == {"sphere"}
    vertical = [f for f in fillets if f.surface["type"] == "cylinder" and f.z_max - f.z_min > 0.5]
    assert len(vertical) == 4 and all(
        abs(f.surface["axis"][2]) == pytest.approx(1) for f in vertical
    )
    (h,) = cl.holes
    assert not h["prismatic"]  # floor fillets: not a 2.5D pocket
    assert set(h["faces"]) >= {f.face for f in fillets}
    assert {e["kind"] for e in cl.edges.values()} == {"tangent", "convex"}


def test_a_rounded_can_side_is_a_wall(can):
    cl = C.classify(can)
    side = next(f for f in cl.faces if f.name.endswith("side"))
    assert side.surface["type"] == "cylinder" and side.role == "wall"
    blend = next(f for f in cl.faces if f.name == "Fillet")
    assert blend.role == "round" and blend.surface["type"] == "torus"
    assert blend.surface["minor_radius"] == pytest.approx(0.2, rel=1e-6)
    assert blend.surface["radius"] == pytest.approx(0.8, rel=1e-6)


def test_analytic_sheets(pr, sheet):
    arc = pr.circle_arc((0, 0, 0), 1.0, (1, 0, 0), (0, 0, 1), np.pi / 6, np.pi / 2)
    dome = C.classify(sheet(pr.revolve(arc, (0, 0, 0), (0, 0, 1)))).faces[0]
    assert dome.surface["type"] == "sphere" and dome.curvature == "convex"
    assert dome.surface["radius"] == pytest.approx(1.0) and np.allclose(dome.surface["center"], 0)
    assert dome.role == "curved"

    arc = pr.circle_arc((0, 0, 1), 1.0, (1, 0, 0), (0, 0, 1), np.pi, 2 * np.pi)
    trough = C.classify(sheet(pr.extrude(arc, (0, 1.0, 0)))).faces[0]
    assert trough.surface["type"] == "cylinder" and trough.curvature == "concave"
    assert np.allclose(np.abs(trough.surface["axis"]), (0, 1, 0))
    assert np.allclose(trough.surface["center"][::2], (0, 1))

    line = pr.line((0.5, 0, 0), (0.2, 0, 0.6))
    cone = C.classify(sheet(pr.revolve(line, (0, 0, 0), (0, 0, 1)))).faces[0]
    assert cone.surface["type"] == "cone"
    assert cone.surface["half_angle"] == pytest.approx(np.degrees(np.arctan(0.5)))
    assert np.allclose(cone.surface["apex"], (0, 0, 1))
    assert cone.slope_range() == pytest.approx([63.4349, 63.4349], abs=1e-3)

    torus = C.classify(sheet(pr.torus((1, 2, 3), (0, 0, 1), 1.0, 0.25))).faces[0]
    assert torus.surface["type"] == "torus"
    assert torus.surface["radius"] == pytest.approx(1.0) and torus.surface["minor_radius"] == (
        pytest.approx(0.25)
    )
    assert np.allclose(torus.surface["center"], (1, 2, 3))
    assert 0.3 < torus.undercut_fraction < 0.7


def test_freeform_and_tolerance(pr, sheet):
    rng = np.random.default_rng(1)
    ctrl = np.zeros((4, 4, 3))
    g = np.linspace(0, 1, 4)
    ctrl[..., 0], ctrl[..., 1] = np.meshgrid(g, g, indexing="ij")
    ctrl[..., 2] = rng.uniform(0, 0.2, (4, 4))
    f = C.classify(sheet(pr.bezier_surface(ctrl))).faces[0]
    assert f.surface == {"type": "freeform"} and f.deviation is None and f.role == "curved"
    # a plane bent by 0.1 mm: freeform at 0.01 mm, a plane at 1 mm
    ctrl[..., 2] = 0.0
    ctrl[1, 1, 2] = 1e-4
    body = sheet(pr.bezier_surface(ctrl))
    assert C.classify(body, tolerance=1e-5).faces[0].surface["type"] == "freeform"
    flat = C.classify(body, tolerance=1e-3).faces[0]
    assert flat.surface["type"] == "plane" and flat.deviation < 1e-4


def test_world_matrix_and_mirror(plate):
    M = np.eye(4)
    M[:3, :3] = 2.0 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    M[:3, 3] = (5, 1, -3)
    cl = C.classify(plate, M)
    radii = sorted(f.surface["radius"] for f in cl.faces if f.surface["type"] == "cylinder")
    assert radii == pytest.approx([0.005, 0.006, 0.008])
    assert one(cl, "Box.+Z").surface["point"][2] == pytest.approx(-3 + 0.01)
    assert cl.holes[0]["center"][:2] == pytest.approx([5.0, 1.04])
    mirror = M.copy()
    mirror[:3, 0] *= -1
    cl = C.classify(plate, mirror)
    assert one(cl, "Box.+Z").role == "top"  # normals still point out of the material
    assert all(f.curvature == "concave" for f in cl.faces if f.surface["type"] == "cylinder")


def test_bare_cap_is_a_boss(solids, boolean):
    """A dome fused onto a block, without a fillet: a boss, not an opening."""
    body, _ = boolean(
        solids.box((0, 0, 0), (0.08, 0.06, 0.02)), solids.sphere((0.0, 0, 0.003), 0.011), "union"
    )
    cl = C.classify(body)
    assert cl.holes == [] and len(cl.bosses) == 1
    dome = next(f for f in cl.faces if f.surface["type"] == "sphere")
    assert dome.role == "curved" and dome.hole is None
    holes, bosses, _ = E.indexed_holes(body, 1e-5)
    assert holes == [] and bosses[0]["index"] == 0


def test_hole_numbers_and_selection(plate):
    holes, _, _ = E.indexed_holes(plate, 1e-5)
    assert [h["index"] for h in holes] == [0, 1, 2]
    es = E.extract(plate, "holes", holes=[2, 0])
    assert sorted(h["index"] for h in es.holes) == [0, 2] and len(es.polylines) == 2
    bottom = next(f.id for f in plate.faces.values() if f.name == "Box.-Z")
    by_face, _, _ = E.indexed_holes(plate, 1e-5, faces=[bottom])
    assert sorted(h["index"] for h in by_face) == [0, 1]  # whole-body numbers
    assert all(h["face"] == bottom for h in by_face)
    with pytest.raises(ValueError, match="no holes"):
        E.extract(plate, "holes", holes=[7])
    with pytest.raises(ValueError, match="kind 'holes'"):
        E.extract(plate, "creases", holes=[0])


def test_faces_filter_and_errors(plate):
    top = next(f.id for f in plate.faces.values() if f.name == "Box.+Z")
    cl = C.classify(plate, faces=[top])
    assert [f.face for f in cl.faces] == [top] and cl.faces[0].role == "top"
    assert all(top in e["faces"] for e in cl.edges.values())
    with pytest.raises(KeyError):
        C.classify(plate, faces=[999])
    with pytest.raises(ValueError):
        C.classify(plate, tolerance=0)
    with pytest.raises(ValueError):
        C.classify(plate, wall_angle=0)
    with pytest.raises(ValueError):
        C.classify(plate, np.zeros((4, 4)))


def test_summary_is_json(rounded):
    s = C.classify(rounded).summary()
    text = json.dumps(s)
    assert "_concave" not in text and "_points" not in text
    assert s["counts"]["role"]["fillet"] == 12 and s["counts"]["holes"] == 1
    assert s["counts"]["edges"]["tangent"] == 28
    assert set(s["faces"][0]) >= {"face", "role", "surface", "slope_range", "neighbors"}


def test_exact_extents(solids, boolean):
    """The bounding box comes from the exact surfaces: a dome's top between samples, a
    tilted cylinder's rims - so a part's top can be put exactly at Z 0."""
    body, _ = boolean(
        solids.box((0, 0, 0), (0.08, 0.06, 0.02)), solids.sphere((0.015, 0, 0.003), 0.011), "union"
    )
    cl = C.classify(body)
    assert cl.bbox_max[2] == pytest.approx(0.014, abs=1e-12)
    assert np.allclose(cl.bbox_min, (-0.04, -0.03, -0.01), atol=1e-12)
    a, b = 0.3, 0.4
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Rx
    cl = C.classify(solids.cylinder((0, 0, 0), (0, 0, 1), 0.01, 0.02), M)
    axis = M[:3, 2]
    rim = 0.01 * np.sqrt(1 - axis**2)
    assert np.allclose(cl.bbox_min, np.minimum(0, 0.02 * axis) - rim, atol=1e-12)
    assert np.allclose(cl.bbox_max, np.maximum(0, 0.02 * axis) + rim, atol=1e-12)


def test_hole_bottom():
    from types import SimpleNamespace as NS

    def face(kind, z_min, nz_min=0.0, **surface):
        return NS(surface={"type": kind, **surface}, z_min=z_min, nz_min=nz_min, nz_max=nz_min)

    infos = {
        1: face("cylinder", -5),
        2: face("cone", -6, half_angle=59.0),
        3: face("plane", -5, nz_min=1.0),
        4: face("sphere", -6),
    }
    hole = {"kind": "blind", "walls": [1], "faces": [1, 2], "z_bottom": -6}
    assert C.hole_bottom(hole, infos, 1e-6) == ("cone", 118.0)
    assert C.hole_bottom({**hole, "faces": [1, 3], "z_bottom": -5}, infos, 1e-6) == ("flat", None)
    assert C.hole_bottom({**hole, "faces": [1, 4]}, infos, 1e-6) == ("other", None)
    assert C.hole_bottom({**hole, "kind": "through"}, infos, 1e-6) == ("through", None)
    assert C.hole_bottom({**hole, "kind": "open"}, infos, 1e-6) == ("open", None)
