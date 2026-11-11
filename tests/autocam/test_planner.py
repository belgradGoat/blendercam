"""The 2.5D planner (camcore/planner.py) on synthetic parts (tests/autocam/parts.py)."""

import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, box

from camcore import planner
from camcore.errors import CamcoreError
from camcore.machine import MachineProfile
from camcore.part import PartModel
from camcore.tools import Tool, complete, default_library
from parts import Builder, bracket

PRESETS = Path(__file__).resolve().parents[2] / "fabex" / "presets" / "cam_cutters"
LIB = default_library(PRESETS)
M = MachineProfile()


def ops_by(p, phase):
    return [op for op in p["operations"] if op["phase"] == phase]


@pytest.fixture(scope="module")
def bracket_plan():
    part, _, _ = bracket()
    return part, planner.plan(part, LIB, M, {})


def test_bracket_stock_and_placement(bracket_plan):
    part, p = bracket_plan
    assert p["version"] == planner.PLAN_VERSION
    assert p["placement"]["translation"] == pytest.approx([-0.1, -0.05, 0.0])
    assert p["stock"]["size"] == pytest.approx([0.064, 0.044, 0.0165])
    assert p["stock"]["min"] == pytest.approx([-0.032, -0.022, -0.016])
    assert p["stock"]["max"][2] == pytest.approx(0.0005)
    assert p["z_end"] == pytest.approx(-0.0102)
    assert p["fixture"]["jaw_top"] == pytest.approx(-0.0122)
    assert p["clearance"] == pytest.approx(0.0055)
    json.dumps(p)  # JSON-able, geometry included


def test_bracket_operations_and_order(bracket_plan):
    _, p = bracket_plan
    phases = [op["phase"] for op in p["operations"]]
    assert phases == sorted(phases, key=["face", "rough", "spot", "drill", "floor", "walls"].index)
    assert phases[0] == "face"
    assert len(ops_by(p, "rough")) == 3  # outside, pocket, the 4 mm hole
    assert len(ops_by(p, "floor")) == 2 and len(ops_by(p, "walls")) == 3
    assert p["operations"][-1]["phase"] == "walls"  # the outside last
    names = [op["name"] for op in p["operations"]]
    assert len(set(names)) == len(names) and names[0].startswith("Bracket 01")
    for op in p["operations"]:
        assert op["props"]["cutter_id"] == op["number"]
        assert op["id"] in p["geometry"]
        assert op["why"]
    # rough before the finish of the same depth range, top down
    rough = ops_by(p, "rough")
    assert [r["z_top"] for r in rough] == sorted([r["z_top"] for r in rough], reverse=True)
    outside = next(r for r in rough if r["region"]["open"])
    assert outside["z_bottom"] == pytest.approx(-0.0102)
    assert outside["props"]["max_z"] == pytest.approx(0.0)
    assert outside["props"]["skin"] == pytest.approx(0.0002)
    assert outside["props"]["movement.helix_enter"] is True


def test_bracket_tools(bracket_plan):
    _, p = bracket_plan
    tools = {t["id"]: t for t in p["tools"]}
    assert set(tools) == {"EM10", "EM4", "EM2", "SP6", "DR5"}
    assert sorted(t["number"] for t in p["tools"]) == [1, 2, 3, 4, 5]
    assert tools["EM10"]["number"] == 1  # faces first
    first_use = []
    for op in p["operations"]:
        if op["number"] not in first_use:
            first_use.append(op["number"])
    assert first_use == sorted(first_use)


def test_bracket_holes(bracket_plan):
    part, p = bracket_plan
    rows = {r["hole"]: r for r in p["holes"]}
    assert rows[1]["action"] == rows[2]["action"] == "drill"
    assert rows[3]["action"] == "mill" and "flat" in rows[3]["why"]
    drill = ops_by(p, "drill")[0]
    assert drill["strategy"] == "DRILL" and drill["holes"] == [1, 2]
    pts = np.asarray(p["geometry"][drill["id"]]["points"])
    tip = 0.0025 / np.tan(np.radians(70))
    assert pts[:, 2] == pytest.approx(-0.01 - tip - 0.0005, abs=1e-7)
    assert sorted(pts[:, 0].tolist()) == pytest.approx([-0.02, 0.02])
    assert drill["props"]["max_z"] == pytest.approx(0.0)
    spot = ops_by(p, "spot")[0]
    spts = np.asarray(p["geometry"][spot["id"]]["points"])
    assert spts[:, 2] == pytest.approx(-0.0025)  # 90 deg spot to the 5 mm diameter
    assert any("M6 tap drill" in w for w in p["warnings"])


def test_bracket_coverage_and_rest(bracket_plan):
    part, p = bracket_plan
    uncovered = p["coverage"]["uncovered"]
    assert [u["role"] for u in uncovered] == ["bottom"]
    covered = {int(k) for k in p["coverage"]["faces"]}
    assert covered == {f["face"] for f in part.faces if f["role"] != "bottom"}
    # the pocket's sharp corners keep the 4 mm end mill's radius
    assert len(p["rest"]) == 4
    assert all(r["radius_left"] == pytest.approx(0.002) for r in p["rest"])
    pocket_walls = [op for op in ops_by(p, "walls") if op["tool"] == "EM4"]
    assert len(pocket_walls) == 1


def test_summary_drops_geometry(bracket_plan):
    _, p = bracket_plan
    s = planner.summary(p)
    assert "geometry" not in s and s["operations"] == p["operations"]


def test_turret_map_numbers():
    part, _, _ = bracket()
    lib = [complete(t) if t.id != "DR5" else complete(Tool(**{**t.to_dict(), "number": 1})) for t in LIB]  # fmt: skip
    p = planner.plan(part, lib, M, {})
    numbers = {t["id"]: t["number"] for t in p["tools"]}
    assert numbers["DR5"] == 1
    assert sorted(numbers.values()) == [1, 2, 3, 4, 5]
    assert numbers["EM10"] == 2


def test_no_matching_drill_mills_the_holes():
    part, _, _ = bracket()
    lib = [t for t in LIB if t.id != "DR5"]
    p = planner.plan(part, lib, M, {})
    assert not ops_by(p, "drill") and not ops_by(p, "spot")
    rows = {r["hole"]: r for r in p["holes"]}
    assert rows[1]["action"] == "mill" and "no Ø5 mm drill" in rows[1]["why"]
    assert len(ops_by(p, "rough")) == 5  # two more round regions
    assert [u["role"] for u in p["coverage"]["uncovered"]] == ["bottom"]


def test_stock_size_and_corner_origin():
    part, _, _ = bracket()
    p = planner.plan(part, LIB, M, {"stock": [0.07, 0.05, 0.02], "origin": "stock_corner"})
    assert p["stock"]["min"][:2] == pytest.approx([0.0, 0.0])
    assert p["stock"]["size"] == pytest.approx([0.07, 0.05, 0.02])
    assert p["stock"]["allowance_top"] == pytest.approx(0.0005)
    assert p["stock"]["grip"] == pytest.approx(0.0095)
    assert p["placement"]["translation"][:2] == pytest.approx([-0.1 + 0.035, -0.05 + 0.025])
    with pytest.raises(CamcoreError):
        planner.plan(part, LIB, M, {"stock": [0.05, 0.05, 0.02]})


def test_options_are_checked():
    part, _, _ = bracket()
    with pytest.raises(CamcoreError) as e:
        planner.plan(part, LIB, M, {"grip": 0.001})
    assert e.value.code == "INVALID_VALUE"
    with pytest.raises(CamcoreError) as e:
        planner.plan(part, LIB, M, {"alowance_top": 0.001})
    assert e.value.code == "INVALID_PROPERTY"


def test_step_part():
    """An L-shaped step: the top over half the part, a step floor 5 mm down."""
    b = Builder("Step", box(0, 0, 0.04, 0.02), 0.0, -0.012)
    b.pocket(box(0.02, -0.001, 0.041, 0.021), -0.005)  # open to three sides: a step
    part, _, _ = b.build()
    # the 'pocket' reaches past the outline: clip it to the part
    part.levels[1].polygon = part.levels[1].polygon.intersection(box(0, 0, 0.04, 0.02))
    part.levels[0].polygon = box(0, 0, 0.02, 0.02)
    p = planner.plan(part, LIB, M, {})
    floors = ops_by(p, "floor")
    assert len(floors) == 1 and floors[0]["z_bottom"] == pytest.approx(-0.005)
    rough = ops_by(p, "rough")
    assert any(r["z_bottom"] == pytest.approx(-0.005) for r in rough)
    assert any(r["z_bottom"] == pytest.approx(-0.0122) for r in rough)


def test_obstacle_band_and_uncovered_fillet():
    b = Builder("Fillet", box(0, 0, 0.04, 0.03), 0.0, -0.01)
    pocket = box(0.01, 0.01, 0.03, 0.02)
    b.pocket(pocket, -0.006)
    ring = pocket.difference(pocket.buffer(-0.002, join_style="mitre"))
    b.obstacle(ring, -0.006, -0.004)
    part, _, _ = b.build()
    p = planner.plan(part, LIB, M, {})
    fillet = next(f["face"] for f in part.faces if f["role"] == "fillet")
    unc = {u["face"]: u for u in p["coverage"]["uncovered"]}
    assert fillet in unc and "3D" in unc[fillet]["reason"]
    # the pocket's walls stop at the fillet's top; below it the region is smaller
    walls = [op for op in ops_by(p, "walls") if not op["props"].get("cut_type") == "OUTSIDE"]
    assert any(op["z_bottom"] == pytest.approx(-0.004) for op in walls)


def test_region_no_tool_fits():
    b = Builder("Slot", box(0, 0, 0.03, 0.02), 0.0, -0.005)
    b.pocket(box(0.01, 0.005, 0.02, 0.0055), -0.002)  # 0.5 mm wide
    part, _, _ = b.build()
    p = planner.plan(part, LIB, M, {})
    assert any("fits no end mill" in w for w in p["warnings"])
    unc = {u["role"] for u in p["coverage"]["uncovered"]}
    assert "floor" in unc


def test_through_window_and_part_model_roundtrip():
    b = Builder("Frame", box(0, 0, 0.05, 0.04), 0.0, -0.008)
    b.pocket(box(0.015, 0.012, 0.035, 0.028), -0.008)  # down to the bottom: a window
    part, _, _ = b.build()
    part.levels = [lv for lv in part.levels if lv.role != "floor"]
    again = PartModel.from_dict(json.loads(json.dumps(part.to_dict())))
    assert again.outline().area == pytest.approx(part.outline().area)
    p = planner.plan(again, LIB, M, {})
    rough = [op for op in ops_by(p, "rough") if not op["region"]["open"]]
    assert len(rough) == 1 and rough[0]["z_bottom"] == pytest.approx(-0.0082)
    assert not ops_by(p, "floor")


def test_tap_drill_and_cone_bottom():
    b = Builder("Tapped", box(0, 0, 0.03, 0.03), 0.0, -0.015)
    b.hole(0.015, 0.015, 0.0068, z_bottom=-0.012, bottom="cone", tip_angle=118.0)
    part, _, _ = b.build()
    p = planner.plan(part, LIB, M, {})
    assert any("M8 tap drill" in w for w in p["warnings"])
    drill = ops_by(p, "drill")[0]
    assert drill["tool"] == "DR6.8"
    assert any("118" in n for n in drill["notes"])
    pts = p["geometry"][drill["id"]]["points"]
    assert pts[0][2] == pytest.approx(-0.012)


def test_spot_drill_clears_walls():
    """A 6.8 mm hole 4 mm from a step's wall: the 10 mm spot drill would cut the wall."""
    b = Builder("Near", box(0, 0, 0.04, 0.03), 0.0, -0.012)
    b.pocket(box(-0.001, 0.015, 0.041, 0.031), -0.004)
    b.hole(0.02, 0.019, 0.0068, z_top=-0.004)
    part, _, _ = b.build()
    part.levels[1].polygon = part.levels[1].polygon.intersection(box(0, 0, 0.04, 0.03))
    p = planner.plan(part, LIB, M, {})
    spot = ops_by(p, "spot")[0]
    assert spot["tool"] == "SP6" and "smaller" in spot["why"]
    z = p["geometry"][spot["id"]]["points"][0][2]
    assert z == pytest.approx(-0.004 - 0.8 * 0.006 / 2, abs=1e-7)
    far = Builder("Far", box(0, 0, 0.04, 0.03), 0.0, -0.012).hole(0.02, 0.015, 0.0068).build()[0]
    assert ops_by(planner.plan(far, LIB, M, {}), "spot")[0]["tool"] == "SP10"
