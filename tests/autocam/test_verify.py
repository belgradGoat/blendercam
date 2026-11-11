"""The program verifier (camcore/verify.py) on hand-written programs."""

import numpy as np
import pytest
from shapely.geometry import box

from camcore import verify
from parts import Builder

HEAD = """%
O1000 (TEST)
(T1 D10. END 3FL)
(T2 D5. DRILL 2FL)
G21
G0 G17 G40 G49 G80 G90 G94
"""
TOOL1 = """G91 G28 Z0.
G90
T1 M06
G0 G90 G54 X-20. Y-12. S10000 M03
G43 H1 Z5.
"""
FACE = """(FACE)
G1 Z0. F500.
X20.
Y-4.
X-20.
Y4.
X20.
Y12.
X-20.
G0 Z5.
"""
END = "M30\n%\n"


def profile(offset):
    o = 10 + 5 + offset
    return f"""(PROFILE)
X{-o:.3f} Y{-o:.3f}
G1 Z-10.2 F300.
X{o:.3f} F1000.
Y{o:.3f}
X{-o:.3f}
Y{-o:.3f}
G0 Z5.
"""


@pytest.fixture(scope="module")
def block():
    part, tris, tf = Builder("Block", box(-0.01, -0.01, 0.01, 0.01), 0.0, -0.01).build()
    return part, tris * 1000, tf


def run(block, body, **kw):
    _, tris, tf = block
    return verify.verify(HEAD + TOOL1 + body + END, tris, tf, [-12, -12, -16], [12, 12, 0.5],
                         part_bottom=-10, resolution=0.1, **kw)  # fmt: skip


def faces(rep, what):
    return {r["face"]: r for r in rep[what]["faces"]}


def test_clean_program_is_ok(block):
    rep = run(block, FACE + profile(0.0))
    assert rep["ok"]
    assert rep["gouge"]["faces"] == []
    assert rep["leftover"]["max_mm"] <= 0.02
    assert rep["collisions"] == [] and rep["plunges"] == []
    names = [op["name"] for op in rep["operations"]]
    assert names == ["FACE", "PROFILE"]
    face = rep["operations"][0]
    assert face["removed_cm3"] == pytest.approx(24 * 24 * 0.5 / 1000, rel=0.02)


def test_skin_left_on_the_walls(block):
    rep = run(block, FACE + profile(0.2))
    walls = faces(rep, "leftover")[1]
    assert walls["measured"] == "sideways"
    assert walls["max_mm"] == pytest.approx(0.2, abs=0.03)
    assert rep["leftover"]["area_mm2"] == pytest.approx(80 * 0.2, rel=0.15)
    assert rep["ok"]  # leftover is not an error


def test_expected_leftover(block):
    rep = run(
        block,
        FACE + profile(0.2),
        expected_zones=[{"at": [10, 10], "radius": 1.0, "reason": "corner"}],
    )
    assert [r["face"] for r in rep["leftover"]["unexpected"]] == [1]
    assert rep["leftover"]["expected"][0]["reason"] == "corner"
    rep = run(block, FACE + profile(0.2), expected_faces={1: "later"})
    assert rep["leftover"]["unexpected"] == []
    assert rep["leftover"]["expected"][0]["reason"] == "later"


def test_gouge_and_rapid_crash(block):
    body = (
        FACE
        + profile(0.0)
        + "(GOUGE)\nX0. Y0.\nG1 Z-0.3 F300.\nG0 Z5.\nX-5. Y-5.\nG0 Z-1.\nG0 Z5.\n"
    )
    rep = run(block, body)
    assert not rep["ok"]
    top = faces(rep, "gouge")[0]
    assert top["max_mm"] == pytest.approx(1.0, abs=1e-6)
    crash = [c for c in rep["collisions"] if c["kind"] == "rapid"]
    assert (
        len(crash) == 1
        and crash[0]["operation"] == "GOUGE"
        and crash[0]["mm"] == pytest.approx(1.0)
    )


def test_plunge_into_material(block):
    rep = run(block, "(PLUNGE)\nX-11. Y-11.\nG1 Z-6. F200.\nG0 Z5.\n")
    assert len(rep["plunges"]) == 1
    assert rep["plunges"][0]["mm"] == pytest.approx(6.5)


def test_slotting_and_width_of_cut(block):
    _, tris, tf = block
    prog = HEAD + TOOL1 + "(SLOT)\nX-35. Y-20.\nG1 Z-3. F200.\nX35. F800.\nG0 Z5.\n" + END
    rep = verify.verify(prog, tris, tf, [-30, -30, -16], [30, 30, 0.5], part_bottom=-10,
                        resolution=0.1)  # fmt: skip
    op = rep["operations"][0]
    assert op["name"] == "SLOT"
    assert op["slot_mm"] > 20
    assert op["slot_max_depth_mm"] == pytest.approx(3.5, abs=0.01)


def test_shank_and_holder(block):
    tools = {1: {"flute_length": 5.0, "reach": 8.0, "holder_diameter": 30.0}}
    rep = run(block, FACE + profile(0.0), tools=tools)
    kinds = {c["kind"] for c in rep["collisions"]}
    assert {"shank", "holder"} <= kinds
    assert not rep["ok"]  # the holder hits the stock


def test_fixture(block):
    jaws = [{"min": [-40, -14, -20], "max": [40, -12, -11], "name": "jaw"}]
    rep = run(block, FACE + profile(0.0), fixtures=jaws)
    hit = [c for c in rep["collisions"] if c["kind"] == "fixture"]
    assert not hit  # the profile stops at -10.2, above the jaw
    deep = FACE + "(DEEP)\nX0. Y-17.\nG1 Z-11.5 F300.\nX10.\nG0 Z5.\n"
    rep = run(block, deep, fixtures=jaws)
    hit = [c for c in rep["collisions"] if c["kind"] == "fixture"]
    assert hit and hit[0]["note"] == "jaw"


def test_drill_cycle(block):
    part, tris, tf = Builder("Plate", box(-0.01, -0.01, 0.01, 0.01), 0.0, -0.01).hole(0, 0, 0.005).build()  # fmt: skip
    prog = (HEAD + "T2 M06\nG0 G90 G54 X0. Y0. S7000 M03\nG43 H2 Z5.\n"
            "(DRILL)\nG98 G81 X0. Y0. Z-12. R2. F900.\nG80\nM30\n%\n")  # fmt: skip
    rep = verify.verify(prog, tris * 1000, tf, [-12, -12, -16], [12, 12, 0.0],
                        part_bottom=-10, resolution=0.1)  # fmt: skip
    assert rep["gouge"]["faces"] == []
    # the hole is clear: only the outside (never cut) is left
    left = faces(rep, "leftover")
    hole_wall = next(f["face"] for f in part.faces if f["role"] == "hole_wall")
    assert hole_wall not in left
    assert rep["operations"][0]["removed_cm3"] > 0.19  # ~ pi x 2.5^2 x 10 mm3 + the point


def test_moves_parser_arcs_and_cycles():
    prog = (HEAD + "T1 M06\nG0 G90 G54 X0. Y0. S1000 M03\nG43 H1 Z5.\n(ARC)\nG1 Z0. F100.\n"
            "X10.\nG3 X10. Y0. I-10. J0.\nG2 X0. Y10. R10.\nG0 Z5.\n(CYCLE)\n"
            "G99 G81 X1. Y1. Z-3. R1. F100.\nX2.\nG80\nM30\n%\n")  # fmt: skip
    mv, ops = verify.moves(prog)
    assert ops[1:] == ["ARC", "CYCLE"]
    arc = [m for m in mv if m.op == 1 and m.kind == verify.FEED]
    pts = np.array([m.b for m in arc])
    full = pts[(pts[:, 2] == 0)][2:]
    r = np.hypot(full[:, 0], full[:, 1])
    assert np.allclose(r[: len(r) // 2], 10, atol=1e-6)
    cyc = [m for m in mv if m.op == 2]
    feeds = [m for m in cyc if m.kind == verify.FEED]
    assert [(m.a[2], m.b[2]) for m in feeds] == [(1.0, -3.0), (1.0, -3.0)]
    assert all(m.drill for m in cyc)
    assert cyc[-1].b[2] == 1.0  # G99: back to R


def test_raster_part_interpolates():
    g = verify.Grid(0, 0, 1.0, 4, 4)
    tri = np.array([[[0, 0, 0], [4, 0, 4], [0, 4, 0]]], float)
    H, F = verify.raster_part(tri, [7], g)
    assert H[0, 0] == pytest.approx(0.5)  # pixel centre (0.5, 0.5): z = x
    assert F[0, 0] == 7
    assert H[3, 3] == -np.inf and F[3, 3] == -1


def test_tools_table_over_comments(block):
    rep = run(block, FACE, tools={1: {"diameter": 4.0}})
    # a 4 mm tool along the facing lines (8 mm apart) leaves ridges
    assert faces(rep, "leftover")[0]["max_mm"] == pytest.approx(0.5, abs=1e-6)
