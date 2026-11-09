"""The FANUC program writer: golden programs, formatting, errors."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from camcore import check, fanuc, machine, moves
from camcore.errors import CamcoreError
from camcore.program import DrillCycle, OpBlock, ProgramSpec, ToolSpec

GOLDEN = Path(__file__).resolve().parent / "golden"
UPDATE = os.environ.get("FABEX_UPDATE_REFERENCE") == "1"

EM6 = ToolSpec(1, "END", 6.0, flutes=3, description="EM6")
DR5 = ToolSpec(2, "DRILL", 5.0, flutes=2, description="DR5")
BM3 = ToolSpec(3, "BALL", 3.0, corner_radius=1.5, flutes=2)
SP10 = ToolSpec(4, "SPOT", 10.0, flutes=2)
TAP6 = ToolSpec(5, "TAP", 6.0, pitch=1.0, description="M6")


def path_block(name, tool, pts_mm, rpm=10000, feed=1200.0, plunge=300.0, free=5.0, **kw):
    pts = np.asarray(pts_mm, float)
    kinds = moves.classify(pts, free, math.pi / 6)
    return OpBlock(
        name, tool, rpm, feed=feed, plunge_feed=plunge, clearance=free,
        points=pts[1:], kinds=kinds[1:], **kw,
    )  # fmt: skip


def pocket_points():
    """A square pocket in two layers, entered by a ramp, with a diagonal rapid away."""
    pts = [[0, 0, 5], [10, 10, 5], [10, 10, 1]]
    for z in (-1.0, -2.0):
        pts += [[15, 10, z], [20, 10, z], [20, 20, z], [10, 20, z], [10, 10, z]]
    pts += [[10, 10, 5], [40, 40, 5]]
    return pts


def drill_block(name, tool, cycle, holes, rpm=6000, **kw):
    c = DrillCycle(cycle, holes, r_plane=2.0, feed=400.0, **kw)
    return OpBlock(name, tool, rpm, feed=400.0, plunge_feed=400.0, clearance=5.0, drill=c)


def golden(name, text):
    path = GOLDEN / name
    if UPDATE or not path.exists():
        path.write_text(text)
    assert text == path.read_text(), f"{name} differs (FABEX_UPDATE_REFERENCE=1 rewrites it)"


@pytest.mark.parametrize(
    "x, s",
    [(10, "10."), (0.5, "0.5"), (-0.0004, "0."), (-1.25, "-1.25"), (1.23456, "1.235"), (0, "0.")],
)
def test_numbers_always_have_a_decimal_point(x, s):
    assert fanuc.num(x) == s


def test_comment():
    assert fanuc.comment("Pocket (top) Ø6 – rough ¼in") == "(POCKET [TOP] D6 - ROUGH 1/4IN)"
    assert len(fanuc.comment("x" * 200, 20)) == 22


def test_golden_mill_and_drill():
    m = machine.MachineProfile()
    spec = ProgramSpec(
        1001,
        "Bracket OP1",
        m,
        [
            path_block("Pocket 1", EM6, pocket_points(), strategy="POCKET"),
            path_block("Pocket 2", EM6, pocket_points(), rpm=9000, coolant="MIST"),
            drill_block(
                "Drill 5", DR5, "G83", [(40, 40, -12), (50, 40, -12), (50, 50, -8)], peck=3
            ),
        ],
    )
    text, rep = fanuc.write(spec)
    golden("mill_and_drill.nc", text)
    assert text.count("M06") == 2  # same tool twice: one change
    assert "S9000 M03" in text and "M07" in text
    assert "\nX40. Y40. Z" not in text and "G0 X40. Y40. Z" not in text  # no diagonal rapids
    assert "(POCKET 2)\nS9000 M03\nM09\nM07\nG0 X10. Y10.\n" in text
    assert rep["operations"][2]["holes"] == 3
    assert rep["tools"][0]["operations"] == ["Pocket 1", "Pocket 2"]
    assert rep["minutes"] > 0 and rep["envelope_mm"]["min"][2] == -12.0
    assert check.check(text, m)["ok"]


def test_golden_every_cycle():
    m = machine.MachineProfile()
    holes = [(10, 10, -6), (20, 10, -6)]
    spec = ProgramSpec(
        2000,
        "cycles",
        m,
        [
            drill_block("Spot", SP10, "G82", holes, dwell=0.3),
            drill_block("Drill", DR5, "G81", holes),
            drill_block("Chip break", DR5, "G73", [(10, 10, -15)], peck=2.5),
            drill_block("Peck", DR5, "G83", [(20, 10, -25)], peck=5),
            drill_block("Tap", TAP6, "G84", holes, rpm=500, retract="G99"),
        ],
    )
    text, rep = fanuc.write(spec)
    golden("every_cycle.nc", text)
    assert "G98 G82 X10. Y10. Z-6. R2. P300 F400." in text
    assert "G98 G73 X10. Y10. Z-15. R2. Q2.5 F400." in text
    assert "M29 S500\nG99 G84 X10. Y10. Z-6. R2. F500." in text  # F = S x pitch
    # rigid tapping starts from a stopped spindle: no M03 with the tap
    tap = text[text.index("T5 M06") :]
    assert "M03" not in tap and "G0 G90 G54 X10. Y10.\n" in tap
    assert text.splitlines().count("G80") == 5
    assert "G05.1" not in text  # no AICC around cycles
    assert check.check(text, m)["ok"]


def test_golden_ball_finish_without_aicc():
    m = machine.MachineProfile.from_dict({"aicc": False, "work_offset": "G55"})
    t = np.linspace(0, 2 * math.pi, 50)
    ring = [[30 + 10 * math.cos(a), 30 + 10 * math.sin(a), -0.5 * math.sin(a)] for a in t]
    pts = [[0, 0, 5], [40, 30, 5], *ring, [40, 30, 5]]
    text, _ = fanuc.write(ProgramSpec(3, "finish", m, [path_block("Finish", BM3, pts)]))
    golden("ball_finish_no_aicc.nc", text)
    assert "G05.1" not in text and "G0 G90 G55 X40. Y30. S10000 M03" in text
    assert text.startswith("%\nO0003 (FINISH)\n")
    assert check.check(text, m)["ok"]


def test_aicc_switched_off_before_tool_change_and_end():
    m = machine.MachineProfile()
    text, _ = fanuc.write(
        ProgramSpec(
            1,
            "a",
            m,
            [
                path_block("A", EM6, pocket_points()),
                drill_block("B", DR5, "G81", [(5, 5, -3)]),
                path_block("C", EM6, pocket_points()),
            ],
        )
    )
    lines = text.splitlines()
    on = off = 0
    for i, ln in enumerate(lines):
        on += ln == "G05.1 Q1"
        off += ln == "G05.1 Q0"
        if "M06" in ln or ln == "M30" or ln.startswith("G98 G81"):
            assert on == off, f"AICC still on at line {i}: {ln}"
    assert on == 2
    assert check.check(text, m)["ok"]


def test_feed_and_speed_clamped_with_warnings():
    m = machine.MachineProfile()
    text, rep = fanuc.write(
        ProgramSpec(1, "a", m, [path_block("A", EM6, pocket_points(), rpm=30000, feed=50000)])
    )
    assert "S10000" in text and "F30000." in text and "F50000." not in text
    assert len(rep["warnings"]) == 2


@pytest.mark.parametrize(
    "blocks, profile, code",
    [
        (
            [
                path_block("A", EM6, pocket_points()),
                path_block("B", ToolSpec(1, "END", 8.0), pocket_points()),
            ],
            {},
            "TOOL_CONFLICT",
        ),
        ([path_block("A", ToolSpec(22, "END", 6.0), pocket_points())], {}, "MACHINE_LIMIT"),
        (
            [
                path_block(f"A{i}", ToolSpec(i + 1, "END", 6.0 + i), pocket_points())
                for i in range(3)
            ],
            {"magazine": 2},
            "MACHINE_LIMIT",
        ),
        ([drill_block("D", DR5, "G83", [(0, 0, -5)])], {}, "INVALID_VALUE"),  # no peck
        ([path_block("A", EM6, pocket_points(), coolant="THROUGH")], {}, "MACHINE_LIMIT"),
        ([path_block("A", EM6, pocket_points(), rpm=0)], {}, "INVALID_VALUE"),
        ([], {}, "INVALID_VALUE"),
    ],
)
def test_errors(blocks, profile, code):
    with pytest.raises(CamcoreError) as e:
        fanuc.write(ProgramSpec(1, "a", machine.MachineProfile.from_dict(profile), blocks))
    assert e.value.code == code


def test_r_plane_and_hole_checks():
    m = machine.MachineProfile()
    b = drill_block("D", DR5, "G81", [(0, 0, -5)])
    b.drill.r_plane = 8.0  # above the initial level (5)
    with pytest.raises(CamcoreError):
        fanuc.write(ProgramSpec(1, "a", m, [b]))
    b = drill_block("D", DR5, "G81", [(0, 0, 3)])  # bottom above R
    with pytest.raises(CamcoreError):
        fanuc.write(ProgramSpec(1, "a", m, [b]))


def test_through_coolant_with_profile_code():
    m = machine.MachineProfile.from_dict({"coolant": {"THROUGH": "M88"}})
    text, _ = fanuc.write(
        ProgramSpec(1, "a", m, [path_block("A", EM6, pocket_points(), coolant="THROUGH")])
    )
    assert "M88" in text and text.count("M09") == 1


def test_simplify_removes_collinear_points():
    m = machine.MachineProfile()
    pts = [[0, 0, 5], [0, 0, 5], [0, 0, -1]] + [[x, 0, -1] for x in np.linspace(0, 50, 101)]
    pts += [[50, 0, 5]]
    text, rep = fanuc.write(ProgramSpec(1, "a", m, [path_block("A", EM6, pts)]))
    op = rep["operations"][0]
    assert op["points_written"] < 10 < op["points_in"]
    assert "X50." in text
    text2, rep2 = fanuc.write(ProgramSpec(1, "a", m, [path_block("A", EM6, pts)], simplify=0))
    assert rep2["operations"][0]["points_written"] == rep2["operations"][0]["points_in"]
