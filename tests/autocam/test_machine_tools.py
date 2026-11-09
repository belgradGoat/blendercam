"""Machine profile and tool library."""

from pathlib import Path

import pytest

from camcore import machine, tools
from camcore.errors import CamcoreError

PRESETS = Path(__file__).resolve().parents[2] / "fabex" / "presets" / "cam_cutters"


# ---- machine profile


def test_default_profile_is_a_robodrill():
    p = machine.MachineProfile()
    p.validate()
    assert "Robodrill" in p.name
    assert p.magazine == 21 and p.spindle_max == 10000 and p.travel == [0.5, 0.4, 0.33]
    assert p.units_block == "G21" and "G80" in p.safe_start


def test_profile_round_trip_and_overlay():
    p = machine.MachineProfile.from_dict({"spindle_max": 24000, "coolant": {"THROUGH": "M88"}})
    assert p.spindle_max == 24000
    assert p.coolant["THROUGH"] == "M88" and p.coolant["FLOOD"] == "M08"  # defaults kept
    q = machine.MachineProfile.from_dict(p.to_dict())
    assert q == p
    r = machine.MachineProfile.from_dict({"magazine": 14}, base=p)
    assert r.spindle_max == 24000 and r.magazine == 14


def test_unknown_field_lists_close_matches():
    with pytest.raises(CamcoreError) as e:
        machine.MachineProfile.from_dict({"spindel_max": 1})
    assert e.value.code == "INVALID_PROPERTY"
    assert "spindle_max" in e.value.details["close_matches"]["spindel_max"]


@pytest.mark.parametrize(
    "bad",
    [
        {"travel": [0.5, -1, 0.3]},
        {"travel": [0.5, 0.4]},
        {"work_offset": "G60"},
        {"program_number": 0},
        {"spindle_min": 20000},
        {"decimals": 2},
        {"coolant": {"STEAM": "M99"}},
        {"magazine": 0},
    ],
)
def test_invalid_profiles(bad):
    with pytest.raises(CamcoreError) as e:
        machine.MachineProfile.from_dict(bad)
    assert e.value.code == "INVALID_VALUE"


def test_work_offsets_accepted():
    for wo in ("G54", "G59", "G54.1 P1", "G54.1P48"):
        machine.MachineProfile.from_dict({"work_offset": wo})


def test_presets():
    assert machine.preset("robodrill_d14sib5").magazine == 14
    assert machine.preset("robodrill_d21lib5").travel[0] == 0.7
    with pytest.raises(CamcoreError) as e:
        machine.preset("haas")
    assert "robodrill_d21mib5" in e.value.details["presets"]


# ---- tools


def test_default_library_from_fabex_presets():
    lib = tools.default_library(PRESETS)
    ids = [t.id for t in lib]
    assert len(ids) == len(set(ids))
    ends = [t for t in lib if t.type == "END"]
    balls = [t for t in lib if t.type == "BALL"]
    assert len(ends) >= 15 and len(balls) >= 15
    em6 = next(t for t in lib if t.id == "EM6")
    assert em6.diameter == pytest.approx(0.006)
    assert em6.flute_length == pytest.approx(0.013) and em6.reach > em6.flute_length
    assert em6.flutes == 3 and em6.source.endswith("end_cyl_6.00mm.py")
    bm3 = next(t for t in lib if t.id == "BM3")
    assert bm3.corner_radius == pytest.approx(0.0015)
    drills = {round(t.diameter * 1000, 3) for t in lib if t.type == "DRILL"}
    assert {3.3, 5.0, 6.8, 8.5, 10.2}.issubset(drills)
    assert all(t.tip_angle == 140 for t in lib if t.type == "DRILL")
    assert {t.id for t in lib if t.type == "SPOT"} == {"SP6", "SP10"}
    assert all(t.number is None for t in lib)


def test_library_without_presets_has_drills_only():
    lib = tools.default_library(None)
    assert {t.type for t in lib} == {"DRILL", "SPOT"}


def test_parse_preset():
    text = 'import bpy\nd = x\nd.cutter_type = "BALLNOSE"\nd.cutter_diameter = 0.0030\n'
    assert tools.parse_preset(text) == ("BALLNOSE", 0.003)
    assert tools.parse_preset("nothing here") is None


def test_user_library_turret_map():
    lib = tools.library_from_list(
        [
            {"id": "EM10", "type": "end", "diameter": 0.010, "number": 1},
            {"id": "BULL8", "type": "BULL", "diameter": 0.008, "corner_radius": 0.001, "number": 2},
            {"id": "M6", "type": "TAP", "diameter": 0.006, "pitch": 0.001, "number": 3},
        ]
    )
    assert [t.number for t in lib] == [1, 2, 3]
    assert lib[0].type == "END" and lib[0].source == "user"
    assert lib[0].flute_length > 0 and lib[0].holder_diameter > lib[0].diameter


@pytest.mark.parametrize(
    "items, code",
    [
        ([], "INVALID_VALUE"),
        ([{"type": "END", "diameter": 0.006}], "INVALID_VALUE"),
        ([{"id": "a", "type": "END", "diameter": 0.006, "colour": "red"}], "INVALID_PROPERTY"),
        ([{"id": "a", "type": "SAW", "diameter": 0.006}], "INVALID_VALUE"),
        ([{"id": "a", "type": "END", "diameter": 0}], "INVALID_VALUE"),
        ([{"id": "a", "type": "TAP", "diameter": 0.006}], "INVALID_VALUE"),
        ([{"id": "a", "type": "BULL", "diameter": 0.006, "corner_radius": 0.004}], "INVALID_VALUE"),
        (
            [
                {"id": "a", "type": "END", "diameter": 0.006, "number": 1},
                {"id": "b", "type": "END", "diameter": 0.008, "number": 1},
            ],
            "TOOL_CONFLICT",
        ),
        (
            [
                {"id": "a", "type": "END", "diameter": 0.006},
                {"id": "a", "type": "END", "diameter": 0.008},
            ],
            "INVALID_VALUE",
        ),
        ([{"id": "a", "type": "END", "diameter": 0.006, "reach": 0.001}], "INVALID_VALUE"),
    ],
)
def test_invalid_libraries(items, code):
    with pytest.raises(CamcoreError) as e:
        tools.library_from_list(items)
    assert e.value.code == code


def test_fabex_props():
    lib = {t.id: t for t in tools.default_library(PRESETS)}
    assert tools.to_fabex_props(lib["EM6"]) == {
        "cutter_type": "END",
        "cutter_diameter": 0.006,
        "cutter_flutes": 3,
        "cutter_description": "6 mm carbide end mill",
    }
    assert tools.to_fabex_props(lib["BM3"])["cutter_type"] == "BALLNOSE"
    spot = tools.to_fabex_props(lib["SP10"])
    assert spot["cutter_type"] == "VCARVE" and spot["cutter_tip_angle"] == 90
    t = tools.library_from_list(
        [{"id": "B", "type": "BULL", "diameter": 0.01, "corner_radius": 0.001, "number": 7}]
    )[0]
    props = tools.to_fabex_props(t)
    assert props["cutter_id"] == 7 and props["bull_corner_radius"] == 0.001
