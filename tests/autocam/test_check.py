"""The G-code checker: one program per rule."""

import pytest

from camcore import check, machine

HEAD = ["%", "O1000 (T)", "(T1 D6. END 3FL)", "G21", "G0 G17 G40 G49 G80 G90 G94"]
TOOL = ["G91 G28 Z0.", "G90", "T1 M06", "G0 G90 G54 X0. Y0. S8000 M03", "G43 H1 Z5.", "M08"]
CUT = ["G1 Z-1. F300.", "X10. F1000.", "G0 Z5."]
TAIL = ["M09", "M05", "G91 G28 Z0.", "G28 Y0.", "G90", "M30", "%"]


def prog(*body, head=HEAD, tail=TAIL):
    return "\n".join(head + list(body) + tail) + "\n"


def rules(result, level="errors"):
    return {e["rule"] for e in result[level]}


def test_valid_program():
    r = check.check(prog(*TOOL, *CUT))
    assert r["ok"], r["errors"]
    assert r["warnings"] == []
    assert r["tools"] == [1]
    assert r["envelope_mm"] == {"min": [0.0, 0.0, -1.0], "max": [10.0, 0.0, 5.0]}


def test_aicc_mode_is_not_a_length():
    r = check.check(prog(*TOOL, "G05.1 Q1", *CUT, "G05.1 Q0"))
    assert r["ok"], r["errors"]


@pytest.mark.parametrize(
    "text, rule",
    [
        (prog(*TOOL, *CUT, head=HEAD[1:]), "FRAMING"),  # no leading %
        (prog(*TOOL, *CUT, head=["%"] + HEAD[2:]), "FRAMING"),  # no O-number
        (prog(*TOOL, *CUT, tail=TAIL[:-2] + ["%"]), "FRAMING"),  # no M30
        (prog(*TOOL, *CUT, tail=TAIL[:-1] + ["G0 X1."] + ["%"]), "FRAMING"),  # after M30
        (prog(*TOOL, *CUT, head=[h for h in HEAD if h != "G21"]), "UNITS"),
        (prog(*TOOL, "G1 Z-1 F300."), "DECIMAL_POINT"),
        (prog("G0 X0. Y0. S8000 M03", "G1 Z-1. F300."), "TOOL"),
        (prog(*TOOL[:4], "G0 Z5.", *CUT), "LENGTH_OFFSET"),  # G0 Z before G43
        (prog(*TOOL[:4], "G1 Z-1. F300."), "LENGTH_OFFSET"),  # no G43 at all
        (prog(*TOOL[:4], "G43 H2 Z5.", *CUT), "LENGTH_OFFSET"),  # H != T
        (prog(*TOOL, "M05", *CUT), "SPINDLE"),
        (prog(*TOOL, "S12000 M03", *CUT), "SPINDLE"),
        (prog(*TOOL, "G1 Z-1. F40000."), "FEED"),
        (prog(*TOOL, "G98 G81 X1. Y1. Z-3. R2. F300.", "G91 G28 Z0.", "G90", "T1 M06"), "CYCLE"),
        (prog(*TOOL, "G05.1 Q1", *CUT), "AICC"),  # still on at M30
        (prog(*TOOL, "G91", "G1 X1. F300.", "G90"), "INCREMENTAL"),
        (prog(*TOOL, "G0 X50. Y50. Z-1."), "RAPID"),
        (prog(*TOOL, "G1 X600. F1000."), "ENVELOPE"),
        (prog(*TOOL, "G98 G81 X1. Y1. Z-3. F300.", "G80"), "CYCLE"),  # no R
    ],
)
def test_errors(text, rule):
    r = check.check(text)
    assert not r["ok"]
    assert rule in rules(r), r["errors"]


@pytest.mark.parametrize(
    "text, rule",
    [
        (prog(*TOOL, "G0 X50. Y50. Z20."), "RAPID"),  # across while going up
        (prog(*TOOL, *CUT, "G91 G28 Z0.", "G90", "T1 M06"), "SPINDLE"),  # no M05 before M06
        (prog(*TOOL, "G1 Z-8. F300."), "PLUNGE"),  # 8 mm into the stock with a 6 mm end mill
        (prog(*TOOL, "G05.1 Q1", "G98 G81 X1. Y1. Z-3. R2. F300.", "G80", "G05.1 Q0"), "AICC"),
    ],
)
def test_warnings(text, rule):
    r = check.check(text)
    assert rule in rules(r, "warnings"), r


def test_size_warning():
    m = machine.MachineProfile.from_dict({"program_memory": 1})
    body = [f"X{i}.5" for i in range(200)]
    r = check.check(prog(*TOOL, "G1 Z-1. F300.", *body), m)
    assert r["ok"] and "SIZE" in rules(r, "warnings")


def test_rigid_tapping_from_a_stopped_spindle():
    tool = TOOL[:3] + ["G0 G90 G54 X0. Y0.", "G43 H1 Z5.", "M08"]
    r = check.check(prog(*tool, "M29 S500", "G99 G84 X1. Y1. Z-6. R2. F500.", "G80"))
    assert r["ok"], r["errors"]
    r = check.check(prog(*tool, "G99 G84 X1. Y1. Z-6. R2. F500.", "G80"))
    assert "SPINDLE" in rules(r)


def test_cycles_expand_into_the_envelope():
    r = check.check(prog(*TOOL, "G98 G83 X20. Y30. Z-12. R2. Q3. F300.", "X40.", "G80"))
    assert r["ok"], r["errors"]
    assert r["envelope_mm"]["min"] == [0.0, 0.0, -12.0]
    assert r["envelope_mm"]["max"] == [40.0, 30.0, 5.0]


def test_repeated_errors_are_capped():
    body = ["G1 Z-1 F300."] + [f"X{i}" for i in range(50)]
    r = check.check(prog(*TOOL, *body))
    assert len(r["errors"]) == 20 and r["suppressed"]["error:DECIMAL_POINT"] > 0
